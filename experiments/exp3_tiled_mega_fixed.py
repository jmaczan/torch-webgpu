#!/usr/bin/env python3
"""
Experiment 3 (Fixed): Tiled vs Mega-Kernel with SAME Workload

This fixes the critique that the original exp3 used different workload sizes:
- Original mega-kernel: intermediate_dim=256 (reduced to fit in workgroup memory)
- Original unfused/tiled: intermediate_dim=4864 (full size)

This experiment uses the SAME reduced dimensions for ALL approaches,
enabling fair comparison of dispatch overhead vs parallelism trade-offs.

Usage:
    python exp3_tiled_mega_fixed.py --output results/exp3_tiled_mega_fixed.json
"""

import argparse
import json
import time
import platform
from pathlib import Path

import numpy as np

try:
    import wgpu
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False


def get_system_info():
    """Collect system information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpu": "",
        "backend": ""
    }

    if WGPU_AVAILABLE:
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            if adapter:
                info["gpu"] = adapter.info.get("description", "")
                info["backend"] = adapter.info.get("backend_type", "")
        except:
            pass

    return info


def create_device():
    """Create wgpu device."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    return device


# ============================================================================
# FAIR COMPARISON: All use hidden_dim=256, intermediate_dim=256
# This fits in workgroup memory for mega-kernel AND allows fair comparison
# ============================================================================

HIDDEN_DIM = 256
INTERMEDIATE_DIM = 256  # Same for all approaches!


def benchmark_unfused_mlp_fixed(device, n_iterations=100, warmup=10):
    """
    Unfused MLP with FIXED dimensions (same as mega-kernel).
    7 separate dispatches: gate_proj, up_proj, silu, mul, down_proj, add (residual), output
    """
    queue = device.queue

    # Shader for linear layer
    linear_shader = """
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weights: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let out_idx = gid.x;
        let in_dim = 256u;
        let out_dim = 256u;

        if (out_idx < out_dim) {
            var sum: f32 = 0.0;
            for (var i: u32 = 0u; i < in_dim; i = i + 1u) {
                sum = sum + input[i] * weights[out_idx * in_dim + i];
            }
            output[out_idx] = sum;
        }
    }
    """

    silu_shader = """
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if (idx < 256u) {
            let x = data[idx];
            data[idx] = x / (1.0 + exp(-x));
        }
    }
    """

    mul_shader = """
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> out: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if (idx < 256u) {
            out[idx] = a[idx] * b[idx];
        }
    }
    """

    add_shader = """
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> out: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if (idx < 256u) {
            out[idx] = a[idx] + b[idx];
        }
    }
    """

    # Create pipelines
    linear_pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": device.create_shader_module(code=linear_shader), "entry_point": "main"}
    )
    silu_pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": device.create_shader_module(code=silu_shader), "entry_point": "main"}
    )
    mul_pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": device.create_shader_module(code=mul_shader), "entry_point": "main"}
    )
    add_pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": device.create_shader_module(code=add_shader), "entry_point": "main"}
    )

    # Create buffers
    input_buf = device.create_buffer(size=HIDDEN_DIM * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    gate_weights = device.create_buffer(size=HIDDEN_DIM * INTERMEDIATE_DIM * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    up_weights = device.create_buffer(size=HIDDEN_DIM * INTERMEDIATE_DIM * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    down_weights = device.create_buffer(size=INTERMEDIATE_DIM * HIDDEN_DIM * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    gate_out = device.create_buffer(size=INTERMEDIATE_DIM * 4, usage=wgpu.BufferUsage.STORAGE)
    up_out = device.create_buffer(size=INTERMEDIATE_DIM * 4, usage=wgpu.BufferUsage.STORAGE)
    hidden = device.create_buffer(size=INTERMEDIATE_DIM * 4, usage=wgpu.BufferUsage.STORAGE)
    output_buf = device.create_buffer(size=HIDDEN_DIM * 4, usage=wgpu.BufferUsage.STORAGE)
    residual_out = device.create_buffer(size=HIDDEN_DIM * 4, usage=wgpu.BufferUsage.STORAGE)

    # Bind groups
    gate_bg = device.create_bind_group(layout=linear_pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": input_buf}},
                 {"binding": 1, "resource": {"buffer": gate_weights}},
                 {"binding": 2, "resource": {"buffer": gate_out}}])

    up_bg = device.create_bind_group(layout=linear_pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": input_buf}},
                 {"binding": 1, "resource": {"buffer": up_weights}},
                 {"binding": 2, "resource": {"buffer": up_out}}])

    silu_bg = device.create_bind_group(layout=silu_pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": gate_out}}])

    mul_bg = device.create_bind_group(layout=mul_pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": gate_out}},
                 {"binding": 1, "resource": {"buffer": up_out}},
                 {"binding": 2, "resource": {"buffer": hidden}}])

    down_bg = device.create_bind_group(layout=linear_pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": hidden}},
                 {"binding": 1, "resource": {"buffer": down_weights}},
                 {"binding": 2, "resource": {"buffer": output_buf}}])

    add_bg = device.create_bind_group(layout=add_pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": input_buf}},
                 {"binding": 1, "resource": {"buffer": output_buf}},
                 {"binding": 2, "resource": {"buffer": residual_out}}])

    # Warmup
    for _ in range(warmup):
        # 7 dispatches
        for pipeline, bg, wg in [
            (linear_pipeline, gate_bg, 4), (linear_pipeline, up_bg, 4),
            (silu_pipeline, silu_bg, 4), (mul_pipeline, mul_bg, 4),
            (linear_pipeline, down_bg, 4), (add_pipeline, add_bg, 4)
        ]:
            encoder = device.create_command_encoder()
            p = encoder.begin_compute_pass()
            p.set_pipeline(pipeline)
            p.set_bind_group(0, bg)
            p.dispatch_workgroups(wg)
            p.end()
            queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()

    # Timed runs
    times_ms = []
    for _ in range(n_iterations):
        queue.on_submitted_work_done_sync()
        start = time.perf_counter()

        # 6 dispatches (gate, up, silu, mul, down, add)
        for pipeline, bg, wg in [
            (linear_pipeline, gate_bg, 4), (linear_pipeline, up_bg, 4),
            (silu_pipeline, silu_bg, 4), (mul_pipeline, mul_bg, 4),
            (linear_pipeline, down_bg, 4), (add_pipeline, add_bg, 4)
        ]:
            encoder = device.create_command_encoder()
            p = encoder.begin_compute_pass()
            p.set_pipeline(pipeline)
            p.set_bind_group(0, bg)
            p.dispatch_workgroups(wg)
            p.end()
            queue.submit([encoder.finish()])

        queue.on_submitted_work_done_sync()
        end = time.perf_counter()
        times_ms.append((end - start) * 1e3)

    return {
        "name": "unfused_mlp_fixed",
        "dispatches": 6,
        "hidden_dim": HIDDEN_DIM,
        "intermediate_dim": INTERMEDIATE_DIM,
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "n_iterations": n_iterations
    }


def benchmark_mega_kernel_fixed(device, n_iterations=100, warmup=10):
    """
    Mega-kernel MLP with FIXED dimensions.
    Single dispatch doing: gate_proj -> silu -> mul(up_proj) -> down_proj -> add
    """
    queue = device.queue

    mega_shader = f"""
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> gate_weights: array<f32>;
    @group(0) @binding(2) var<storage, read> up_weights: array<f32>;
    @group(0) @binding(3) var<storage, read> down_weights: array<f32>;
    @group(0) @binding(4) var<storage, read_write> output: array<f32>;

    var<workgroup> wg_gate: array<f32, {INTERMEDIATE_DIM}>;
    var<workgroup> wg_up: array<f32, {INTERMEDIATE_DIM}>;
    var<workgroup> wg_hidden: array<f32, {INTERMEDIATE_DIM}>;
    var<workgroup> wg_out: array<f32, {HIDDEN_DIM}>;

    @compute @workgroup_size(256)
    fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
        let tid = lid.x;
        let hidden_dim = {HIDDEN_DIM}u;
        let inter_dim = {INTERMEDIATE_DIM}u;

        // Step 1: Gate projection (tid computes one output element)
        if (tid < inter_dim) {{
            var sum: f32 = 0.0;
            for (var i: u32 = 0u; i < hidden_dim; i = i + 1u) {{
                sum = sum + input[i] * gate_weights[tid * hidden_dim + i];
            }}
            wg_gate[tid] = sum;
        }}
        workgroupBarrier();

        // Step 2: Up projection
        if (tid < inter_dim) {{
            var sum: f32 = 0.0;
            for (var i: u32 = 0u; i < hidden_dim; i = i + 1u) {{
                sum = sum + input[i] * up_weights[tid * hidden_dim + i];
            }}
            wg_up[tid] = sum;
        }}
        workgroupBarrier();

        // Step 3: SiLU(gate) * up
        if (tid < inter_dim) {{
            let g = wg_gate[tid];
            let silu_g = g / (1.0 + exp(-g));
            wg_hidden[tid] = silu_g * wg_up[tid];
        }}
        workgroupBarrier();

        // Step 4: Down projection
        if (tid < hidden_dim) {{
            var sum: f32 = 0.0;
            for (var i: u32 = 0u; i < inter_dim; i = i + 1u) {{
                sum = sum + wg_hidden[i] * down_weights[tid * inter_dim + i];
            }}
            wg_out[tid] = sum;
        }}
        workgroupBarrier();

        // Step 5: Residual add and write output
        if (tid < hidden_dim) {{
            output[tid] = input[tid] + wg_out[tid];
        }}
    }}
    """

    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": device.create_shader_module(code=mega_shader), "entry_point": "main"}
    )

    # Buffers
    input_buf = device.create_buffer(size=HIDDEN_DIM * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    gate_weights = device.create_buffer(size=HIDDEN_DIM * INTERMEDIATE_DIM * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    up_weights = device.create_buffer(size=HIDDEN_DIM * INTERMEDIATE_DIM * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    down_weights = device.create_buffer(size=INTERMEDIATE_DIM * HIDDEN_DIM * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    output_buf = device.create_buffer(size=HIDDEN_DIM * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": input_buf}},
            {"binding": 1, "resource": {"buffer": gate_weights}},
            {"binding": 2, "resource": {"buffer": up_weights}},
            {"binding": 3, "resource": {"buffer": down_weights}},
            {"binding": 4, "resource": {"buffer": output_buf}},
        ]
    )

    # Warmup
    for _ in range(warmup):
        encoder = device.create_command_encoder()
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, bind_group)
        p.dispatch_workgroups(1)
        p.end()
        queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()

    # Timed runs
    times_ms = []
    for _ in range(n_iterations):
        queue.on_submitted_work_done_sync()
        start = time.perf_counter()

        encoder = device.create_command_encoder()
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, bind_group)
        p.dispatch_workgroups(1)
        p.end()
        queue.submit([encoder.finish()])

        queue.on_submitted_work_done_sync()
        end = time.perf_counter()
        times_ms.append((end - start) * 1e3)

    return {
        "name": "mega_kernel_fixed",
        "dispatches": 1,
        "hidden_dim": HIDDEN_DIM,
        "intermediate_dim": INTERMEDIATE_DIM,
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "n_iterations": n_iterations
    }


def main():
    parser = argparse.ArgumentParser(description="Fixed mega-kernel comparison")
    parser.add_argument("--output", type=str, default="results/exp3_tiled_mega_fixed.json")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 3 (Fixed): Fair Mega-Kernel Comparison")
    print("=" * 60)
    print(f"\nUsing SAME dimensions for all approaches:")
    print(f"  hidden_dim = {HIDDEN_DIM}")
    print(f"  intermediate_dim = {INTERMEDIATE_DIM}")
    print(f"\nThis enables fair comparison (original exp3 used 4864 for unfused, 256 for mega)")

    if not WGPU_AVAILABLE:
        print("ERROR: wgpu not available")
        return

    device = create_device()
    queue = device.queue

    results = {
        "system_info": get_system_info(),
        "dimensions": {
            "hidden_dim": HIDDEN_DIM,
            "intermediate_dim": INTERMEDIATE_DIM,
            "note": "Same dimensions for all approaches (fair comparison)"
        },
        "experiments": {}
    }

    # 1. Unfused MLP (6 dispatches)
    print("\n1. Benchmarking unfused MLP (6 dispatches)...")
    unfused = benchmark_unfused_mlp_fixed(device, n_iterations=args.iterations)
    results["experiments"]["unfused"] = unfused
    print(f"   Mean: {unfused['mean_ms']:.3f} ± {unfused['std_ms']:.3f} ms")

    # 2. Mega-kernel MLP (1 dispatch)
    print("\n2. Benchmarking mega-kernel MLP (1 dispatch)...")
    mega = benchmark_mega_kernel_fixed(device, n_iterations=args.iterations)
    results["experiments"]["mega_kernel"] = mega
    print(f"   Mean: {mega['mean_ms']:.3f} ± {mega['std_ms']:.3f} ms")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS (Fair Comparison)")
    print("=" * 60)

    speedup = unfused['mean_ms'] / mega['mean_ms']
    dispatch_reduction = unfused['dispatches'] / mega['dispatches']

    print(f"\nUnfused:     {unfused['mean_ms']:.3f} ms ({unfused['dispatches']} dispatches)")
    print(f"Mega-kernel: {mega['mean_ms']:.3f} ms ({mega['dispatches']} dispatch)")
    print(f"\nMega-kernel speedup: {speedup:.2f}x")
    print(f"Dispatch reduction:  {dispatch_reduction}x")

    if speedup > 1:
        print(f"\n→ Mega-kernel is {speedup:.2f}x FASTER (dispatch overhead dominates)")
    else:
        print(f"\n→ Mega-kernel is {1/speedup:.2f}x SLOWER (parallelism loss dominates)")

    results["analysis"] = {
        "mega_vs_unfused_speedup": speedup,
        "dispatch_reduction": dispatch_reduction,
        "conclusion": "mega_faster" if speedup > 1 else "unfused_faster"
    }

    # Comparison with original exp3 (unfair comparison)
    print("\n" + "-" * 60)
    print("COMPARISON WITH ORIGINAL EXP3 (for reference)")
    print("-" * 60)
    print("\nOriginal exp3 used intermediate_dim=4864 for unfused, 256 for mega.")
    print("That's 19x more compute for unfused, making comparison unfair.")
    print("\nThis experiment uses same dimensions, showing true overhead trade-off.")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
