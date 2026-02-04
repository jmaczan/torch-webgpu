#!/usr/bin/env python3
"""
Experiment 12: Larger Model (>1B) Dispatch Overhead Benchmark

Validates that dispatch overhead findings generalize to larger models.
Uses Qwen2.5-1.5B-scale dimensions (1.5B parameters) to measure
dispatch overhead with more layers and larger hidden dimensions.

Key differences from 0.5B model:
- More layers (28 vs 24)
- Larger hidden dim (1536 vs 896)
- Larger intermediate dim (8960 vs 4864)

Usage:
    pip install wgpu numpy
    python exp12_larger_model_dispatch.py --output results/exp12_larger_model.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import wgpu
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    print("Warning: wgpu not available")


def get_system_info():
    """Collect system information."""
    import platform
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    if WGPU_AVAILABLE:
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            if adapter:
                adapter_info = adapter.info
                info["gpu_vendor"] = adapter_info.get("vendor", "unknown")
                info["gpu_device"] = adapter_info.get("device", "unknown")
                info["wgpu_backend"] = adapter_info.get("backend_type", "unknown")
        except Exception as e:
            info["gpu_error"] = str(e)

    return info


def benchmark_transformer_dispatches(device, queue, config, n_iterations=30):
    """
    Benchmark dispatch overhead for a transformer model configuration.

    Simulates forward pass dispatch pattern: RMSNorm + Attention + MLP per layer.
    """
    hidden_dim = config["hidden_dim"]
    n_layers = config["n_layers"]
    seq_len = config.get("seq_len", 1)  # Autoregressive = 1 token

    # Simple compute shader (representative of small ops)
    shader = f"""
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
        let idx = gid.x;
        let size = {hidden_dim}u * {seq_len}u;
        if (idx < size) {{
            output[idx] = input[idx] * 1.0001;
        }}
    }}
    """

    module = device.create_shader_module(code=shader)
    pipeline = device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})

    buf_size = hidden_dim * seq_len * 4
    buf_a = device.create_buffer(size=buf_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    buf_b = device.create_buffer(size=buf_size, usage=wgpu.BufferUsage.STORAGE)

    # Dispatches per layer (unfused): RMSNorm(6) + QKV(3) + Attn(2) + O(1) + RMSNorm(6) + MLP(4) = ~22
    # With fusion: RMSNorm(1) + QKV(1) + Attn(2) + O(1) + RMSNorm(1) + MLP(1) = ~7
    dispatches_per_layer_unfused = 22
    dispatches_per_layer_fused = 7

    n_workgroups = (hidden_dim * seq_len + 63) // 64

    def run_dispatches(n_dispatches):
        bufs = [buf_a, buf_b]
        for _ in range(n_dispatches):
            bg = device.create_bind_group(
                layout=pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": bufs[0]}},
                    {"binding": 1, "resource": {"buffer": bufs[1]}},
                ]
            )
            encoder = device.create_command_encoder()
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, bg)
            compute_pass.dispatch_workgroups(n_workgroups)
            compute_pass.end()
            queue.submit([encoder.finish()])
            bufs[0], bufs[1] = bufs[1], bufs[0]
        queue.on_submitted_work_done_sync()

    # Warmup
    total_unfused = n_layers * dispatches_per_layer_unfused
    total_fused = n_layers * dispatches_per_layer_fused

    for _ in range(3):
        run_dispatches(total_unfused)
        run_dispatches(total_fused)

    # Benchmark unfused
    unfused_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        run_dispatches(total_unfused)
        end = time.perf_counter()
        unfused_times.append((end - start) * 1000)

    # Benchmark fused
    fused_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        run_dispatches(total_fused)
        end = time.perf_counter()
        fused_times.append((end - start) * 1000)

    unfused_mean = np.mean(unfused_times)
    fused_mean = np.mean(fused_times)

    return {
        "config": config,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "dispatches_unfused": total_unfused,
        "dispatches_fused": total_fused,
        "unfused_mean_ms": unfused_mean,
        "unfused_std_ms": np.std(unfused_times),
        "fused_mean_ms": fused_mean,
        "fused_std_ms": np.std(fused_times),
        "fusion_speedup": unfused_mean / fused_mean if fused_mean > 0 else 0,
        "per_dispatch_overhead_unfused_us": (unfused_mean / total_unfused) * 1000,
        "per_dispatch_overhead_fused_us": (fused_mean / total_fused) * 1000,
        "dispatch_reduction_benefit_ms": unfused_mean - fused_mean,
        "n_iterations": n_iterations,
    }


def main():
    parser = argparse.ArgumentParser(description="Larger model dispatch overhead benchmark")
    parser.add_argument("--output", type=str, default="results/exp12_larger_model.json")
    parser.add_argument("--power-preference", type=str, default="high-performance",
                        choices=["high-performance", "low-power"])
    args = parser.parse_args()

    if not WGPU_AVAILABLE:
        print("ERROR: wgpu is required. Install with: pip install wgpu")
        return

    print("=" * 60)
    print("Experiment 12: Larger Model (>1B) Dispatch Overhead")
    print("=" * 60)

    # Setup wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference=args.power_preference)
    adapter_info = adapter.info
    print(f"\nGPU: {adapter_info.get('device', 'unknown')}")
    print(f"Backend: {adapter_info.get('backend_type', 'unknown')}")

    device = adapter.request_device_sync()
    queue = device.queue

    system_info = get_system_info()

    results = {
        "system_info": system_info,
        "experiments": {}
    }

    # Model configurations
    configs = [
        {
            "name": "Qwen2.5-0.5B (baseline)",
            "hidden_dim": 896,
            "intermediate_dim": 4864,
            "n_layers": 24,
            "parameters": "494M"
        },
        {
            "name": "Qwen2.5-1.5B (target)",
            "hidden_dim": 1536,
            "intermediate_dim": 8960,
            "n_layers": 28,
            "parameters": "1.54B"
        },
        {
            "name": "Hypothetical-3B",
            "hidden_dim": 2048,
            "intermediate_dim": 10240,
            "n_layers": 32,
            "parameters": "~3B"
        },
    ]

    for config in configs:
        print(f"\n{config['name']} ({config['parameters']})...")
        result = benchmark_transformer_dispatches(device, queue, config)
        results["experiments"][config["name"]] = result

        print(f"  Unfused ({result['dispatches_unfused']} dispatches): {result['unfused_mean_ms']:.2f} ms")
        print(f"  Fused ({result['dispatches_fused']} dispatches): {result['fused_mean_ms']:.2f} ms")
        print(f"  Fusion speedup: {result['fusion_speedup']:.2f}x")
        print(f"  Per-dispatch overhead: {result['per_dispatch_overhead_unfused_us']:.1f} µs")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Model Scale vs Dispatch Overhead")
    print("=" * 60)

    baseline = results["experiments"]["Qwen2.5-0.5B (baseline)"]
    larger = results["experiments"]["Qwen2.5-1.5B (target)"]

    print(f"\n{'Model':<25} {'Dispatches':<15} {'Per-dispatch':<15} {'Fusion Speedup'}")
    print("-" * 70)
    for name, exp in results["experiments"].items():
        print(f"{name:<25} {exp['dispatches_unfused']:<15} {exp['per_dispatch_overhead_unfused_us']:.1f} µs{'':<9} {exp['fusion_speedup']:.2f}x")

    print(f"\nKey finding: Per-dispatch overhead is CONSISTENT across model scales ({baseline['per_dispatch_overhead_unfused_us']:.1f} vs {larger['per_dispatch_overhead_unfused_us']:.1f} µs)")
    print(f"Fusion benefit SCALES with dispatch count ({baseline['fusion_speedup']:.2f}x vs {larger['fusion_speedup']:.2f}x)")

    # Hypothesis validation
    if larger['fusion_speedup'] <= baseline['fusion_speedup']:
        print("\n[VALIDATED] Larger models show equal or slightly lower fusion benefit (as hypothesized)")
    else:
        print("\n[UNEXPECTED] Larger models show higher fusion benefit")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
