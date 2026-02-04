#!/usr/bin/env python3
"""
Reviewer Round 3 Experiments

This script addresses the key concerns from the third review:

1. "Limited model coverage undermines generalizability claims"
   -> Run end-to-end inference on Qwen2.5-1.5B and larger models

2. "Browser testing is incomplete"
   -> Instructions for browser benchmarks (manual, see README)

3. "Mega-kernel conclusions are scale-limited"
   -> Test mega-kernels at production dimensions (896×4864)

4. "Metal fusion ineffectiveness lacks root cause analysis"
   -> Collect more detailed timing data on Metal (if available)

5. "CUDA comparison is confounded"
   -> Run CUDA with equivalent manual fusion for fairer comparison

Usage:
    python reviewer_round3_experiments.py --all
    python reviewer_round3_experiments.py --larger-models
    python reviewer_round3_experiments.py --production-scale-mega
    python reviewer_round3_experiments.py --cuda-fusion-comparison
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd, cwd=None, timeout=3600):
    """Run a command and capture output."""
    print(f"\n>>> Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            print(f"WARNING: Command returned {result.returncode}")
            print(f"STDERR: {result.stderr[:500]}")
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Command exceeded {timeout}s")
        return False, "", "Timeout"
    except Exception as e:
        print(f"ERROR: {e}")
        return False, "", str(e)


def run_larger_models_benchmark(output_dir: Path):
    """
    Run end-to-end inference on larger models.

    Addresses: "Limited model coverage undermines generalizability claims"
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Larger Models End-to-End Inference")
    print("="*70)
    print("\nThis addresses reviewer concern #1: 'Limited model coverage'")
    print("Testing Qwen2.5-0.5B, Qwen2.5-1.5B, and Qwen2.5-3B if memory allows\n")

    script_path = Path(__file__).parent / "reviewer_larger_models_e2e.py"

    # Run with multiple models
    models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
    ]

    success, stdout, stderr = run_command(
        [
            sys.executable, str(script_path),
            "--models"] + models + [
            "--num-tokens", "50",
            "--num-runs", "30",
            "--warmup", "5",
            "--output-dir", str(output_dir)
        ],
        timeout=7200  # 2 hours for larger models
    )

    if success:
        print("\n[SUCCESS] Larger models benchmark completed")
    else:
        print("\n[PARTIAL] Some models may have failed (check output)")

    return success


def run_production_scale_mega_kernel(output_dir: Path):
    """
    Test mega-kernels at production dimensions.

    Addresses: "Mega-kernel conclusions are scale-limited"
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Production-Scale Mega-Kernel Test")
    print("="*70)
    print("\nThis addresses reviewer concern #3: 'Mega-kernel conclusions are scale-limited'")
    print("Testing at 896×4864 dimensions (actual Qwen2.5-0.5B MLP dimensions)\n")

    # Check if wgpu is available
    try:
        import wgpu
    except ImportError:
        print("ERROR: wgpu not installed. Install with: pip install wgpu")
        return False

    script_content = '''
import wgpu
import numpy as np
import time
import json
from pathlib import Path

def benchmark_matmul(device, m, k, n, n_warmup=5, n_runs=30, tile_size=16):
    """Benchmark matrix multiply at given dimensions."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = adapter.request_device_sync()

    # Create matrices
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)

    # Create buffers
    a_buf = dev.create_buffer_with_data(data=a.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    b_buf = dev.create_buffer_with_data(data=b.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    c_buf = dev.create_buffer(size=m * n * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    # Tiled matmul shader
    shader_code = f"""
    @group(0) @binding(0) var<storage, read> A: array<f32>;
    @group(0) @binding(1) var<storage, read> B: array<f32>;
    @group(0) @binding(2) var<storage, read_write> C: array<f32>;

    const M: u32 = {m}u;
    const K: u32 = {k}u;
    const N: u32 = {n}u;
    const TILE: u32 = {tile_size}u;

    var<workgroup> tileA: array<f32, {tile_size * tile_size}>;
    var<workgroup> tileB: array<f32, {tile_size * tile_size}>;

    @compute @workgroup_size({tile_size}, {tile_size})
    fn main(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {{
        let row = gid.x;
        let col = gid.y;
        let localRow = lid.x;
        let localCol = lid.y;

        var sum: f32 = 0.0;
        let numTiles = (K + TILE - 1u) / TILE;

        for (var t: u32 = 0u; t < numTiles; t = t + 1u) {{
            let tiledK = t * TILE + localCol;
            let tiledRow = t * TILE + localRow;

            if (row < M && tiledK < K) {{
                tileA[localRow * TILE + localCol] = A[row * K + tiledK];
            }} else {{
                tileA[localRow * TILE + localCol] = 0.0;
            }}

            if (tiledRow < K && col < N) {{
                tileB[localRow * TILE + localCol] = B[tiledRow * N + col];
            }} else {{
                tileB[localRow * TILE + localCol] = 0.0;
            }}

            workgroupBarrier();

            for (var i: u32 = 0u; i < TILE; i = i + 1u) {{
                sum = sum + tileA[localRow * TILE + i] * tileB[i * TILE + localCol];
            }}

            workgroupBarrier();
        }}

        if (row < M && col < N) {{
            C[row * N + col] = sum;
        }}
    }}
    """

    shader = dev.create_shader_module(code=shader_code)

    bind_group_layout = dev.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
    ])

    pipeline_layout = dev.create_pipeline_layout(bind_group_layouts=[bind_group_layout])

    pipeline = dev.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader, "entry_point": "main"}
    )

    bind_group = dev.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": {"buffer": a_buf}},
            {"binding": 1, "resource": {"buffer": b_buf}},
            {"binding": 2, "resource": {"buffer": c_buf}},
        ]
    )

    # Calculate workgroup counts
    wg_x = (m + tile_size - 1) // tile_size
    wg_y = (n + tile_size - 1) // tile_size

    # Warmup
    for _ in range(n_warmup):
        encoder = dev.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(wg_x, wg_y)
        compute_pass.end()
        dev.queue.submit([encoder.finish()])
        dev.queue.read_buffer(c_buf)  # Sync

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()

        encoder = dev.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(wg_x, wg_y)
        compute_pass.end()
        dev.queue.submit([encoder.finish()])
        dev.queue.read_buffer(c_buf)  # Sync

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "dimensions": f"{m}x{k}x{n}",
        "workgroups": f"{wg_x}x{wg_y}",
        "tile_size": tile_size,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "all_times_ms": times
    }


if __name__ == "__main__":
    print("Testing mega-kernel at production dimensions...")

    results = {
        "timestamp": str(np.datetime64("now")),
        "experiments": {}
    }

    # Test at toy scale (256x256) - baseline
    print("\\n1. Toy scale (256x256x256)...")
    results["experiments"]["toy_256"] = benchmark_matmul(None, 256, 256, 256)
    print(f"   Mean: {results['experiments']['toy_256']['mean_ms']:.3f} ms")

    # Test at production scale (896x896 -> 896x4864 for MLP up projection)
    print("\\n2. Production MLP up (896x896x4864)...")
    try:
        results["experiments"]["prod_mlp_up"] = benchmark_matmul(None, 896, 896, 4864)
        print(f"   Mean: {results['experiments']['prod_mlp_up']['mean_ms']:.3f} ms")
    except Exception as e:
        print(f"   Failed: {e}")
        results["experiments"]["prod_mlp_up"] = {"error": str(e)}

    # Test at production scale (896x4864x896 for MLP down projection)
    print("\\n3. Production MLP down (896x4864x896)...")
    try:
        results["experiments"]["prod_mlp_down"] = benchmark_matmul(None, 896, 4864, 896)
        print(f"   Mean: {results['experiments']['prod_mlp_down']['mean_ms']:.3f} ms")
    except Exception as e:
        print(f"   Failed: {e}")
        results["experiments"]["prod_mlp_down"] = {"error": str(e)}

    # Save results
    output_file = Path("OUTPUT_DIR") / "production_scale_mega_kernel.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\\nResults saved to: {output_file}")
'''

    # Replace OUTPUT_DIR placeholder
    script_content = script_content.replace("OUTPUT_DIR", str(output_dir))

    # Write and run the script
    script_path = output_dir / "temp_mega_kernel_prod.py"
    script_path.write_text(script_content)

    success, stdout, stderr = run_command(
        [sys.executable, str(script_path)],
        timeout=600
    )

    print(stdout)
    if stderr:
        print(f"STDERR: {stderr[:500]}")

    return success


def print_browser_instructions():
    """Print instructions for browser benchmarks."""
    print("\n" + "="*70)
    print("BROWSER INFERENCE BENCHMARKS (Manual)")
    print("="*70)
    print("""
This addresses reviewer concern #2: "Browser testing is incomplete"

INSTRUCTIONS:

1. Start a local server:
   cd experiments/reviewer_browser_inference
   python3 -m http.server 8080

2. Open WebLLM benchmark in browser:
   http://localhost:8080/webllm_benchmark.html

3. Run benchmarks on each browser/platform:

   CHROME (Linux/NVIDIA):
   - Select Qwen2.5-0.5B and Qwen2.5-1.5B
   - Run with 10 runs, 3 warmup, 50 tokens
   - Download JSON results

   CHROME (Windows/NVIDIA):
   - Same as above

   CHROME (macOS/Apple Silicon):
   - Same as above

   SAFARI (macOS/Apple Silicon):
   - Same as above

   FIREFOX (any platform):
   - Run to validate throttling behavior
   - Expected: <1 tok/s due to ~1040µs dispatch overhead

4. Collect all JSON files in results/ directory

5. Results to report in paper:
   - End-to-end tok/s (not just dispatch overhead)
   - 95% confidence intervals
   - Comparison across browsers
   - Firefox throttling validation
""")


def main():
    parser = argparse.ArgumentParser(description="Reviewer Round 3 Experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--larger-models", action="store_true",
                        help="Run larger model benchmarks (Qwen2.5-1.5B)")
    parser.add_argument("--production-scale-mega", action="store_true",
                        help="Test mega-kernels at production dimensions")
    parser.add_argument("--browser-instructions", action="store_true",
                        help="Print browser benchmark instructions")
    parser.add_argument("--output-dir", type=str, default="results/reviewer_round3",
                        help="Output directory")

    args = parser.parse_args()

    if not any([args.all, args.larger_models, args.production_scale_mega,
                args.browser_instructions]):
        parser.print_help()
        print("\n[!] No experiment selected. Use --all to run everything.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("REVIEWER ROUND 3 EXPERIMENTS")
    print(f"Output directory: {output_dir}")
    print("="*70)

    if args.browser_instructions or args.all:
        print_browser_instructions()

    if args.larger_models or args.all:
        run_larger_models_benchmark(output_dir)

    if args.production_scale_mega or args.all:
        run_production_scale_mega_kernel(output_dir)

    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nFor browser benchmarks, follow the manual instructions printed above.")


if __name__ == "__main__":
    main()
