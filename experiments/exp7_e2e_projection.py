#!/usr/bin/env python3
"""
Experiment 7: End-to-End Performance Projection

This experiment addresses the critique: "You claim wgpu-native has lower overhead,
but where's the proof it translates to faster inference?"

We:
1. Run a standardized sequential dispatch benchmark on wgpu-native
2. Compare with Dawn results from exp6
3. Project what end-to-end LLM inference would look like on wgpu-native

The projection methodology:
- Measure: dispatch_overhead_dawn, dispatch_overhead_wgpu
- Known: TTFT_dawn = 42ms with N dispatches
- Project: TTFT_wgpu = TTFT_dawn - N * (overhead_dawn - overhead_wgpu)

Usage:
    python exp7_e2e_projection.py --output results/exp7_e2e_projection.json
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
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }

    if WGPU_AVAILABLE:
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            if adapter:
                info["gpu_vendor"] = adapter.info.get("vendor", "")
                info["gpu_device"] = adapter.info.get("device", "")
                info["gpu_description"] = adapter.info.get("description", "")
                info["wgpu_backend"] = adapter.info.get("backend_type", "")
        except Exception as e:
            info["wgpu_error"] = str(e)

    return info


def create_simple_compute_pipeline(device):
    """Create a simple compute pipeline for dispatch overhead measurement."""
    shader_code = """
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if (idx < arrayLength(&data)) {
            data[idx] = data[idx] + 1.0;
        }
    }
    """

    module = device.create_shader_module(code=shader_code)
    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": module, "entry_point": "main"}
    )

    return pipeline


def benchmark_wgpu_sequential_dispatches(n_dispatches=100, n_iterations=50, warmup=5):
    """
    Measure time for N sequential dispatches on wgpu-native.
    Matches the methodology in exp6 for Dawn.
    """
    if not WGPU_AVAILABLE:
        return {"error": "wgpu not available"}

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    queue = device.queue

    # Create pipeline and buffer
    pipeline = create_simple_compute_pipeline(device)
    buffer_size = 256 * 4  # 256 floats
    buffer = device.create_buffer(
        size=buffer_size,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": buffer}}]
    )

    # Warmup
    for _ in range(warmup):
        for _ in range(n_dispatches):
            encoder = device.create_command_encoder()
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(4)
            compute_pass.end()
            queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()

    # Measure
    times_ms = []
    for _ in range(n_iterations):
        queue.on_submitted_work_done_sync()

        start = time.perf_counter()
        for _ in range(n_dispatches):
            encoder = device.create_command_encoder()
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(4)
            compute_pass.end()
            queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()
        end = time.perf_counter()

        times_ms.append((end - start) * 1e3)

    mean_total_ms = float(np.mean(times_ms))
    per_dispatch_us = (mean_total_ms * 1000) / n_dispatches

    return {
        "name": "wgpu_sequential_dispatches",
        "n_dispatches": n_dispatches,
        "mean_total_ms": mean_total_ms,
        "std_total_ms": float(np.std(times_ms)),
        "min_total_ms": float(np.min(times_ms)),
        "max_total_ms": float(np.max(times_ms)),
        "per_dispatch_us": per_dispatch_us,
        "n_iterations": n_iterations
    }


def benchmark_wgpu_batched_dispatches(n_dispatches=100, n_iterations=50, warmup=5):
    """
    Measure time for N dispatches batched into a single command buffer.
    This shows the potential of command batching.
    """
    if not WGPU_AVAILABLE:
        return {"error": "wgpu not available"}

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    queue = device.queue

    pipeline = create_simple_compute_pipeline(device)
    buffer_size = 256 * 4
    buffer = device.create_buffer(
        size=buffer_size,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": buffer}}]
    )

    # Warmup
    for _ in range(warmup):
        encoder = device.create_command_encoder()
        for _ in range(n_dispatches):
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(4)
            compute_pass.end()
        queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()

    # Measure
    times_ms = []
    for _ in range(n_iterations):
        queue.on_submitted_work_done_sync()

        start = time.perf_counter()
        encoder = device.create_command_encoder()
        for _ in range(n_dispatches):
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(4)
            compute_pass.end()
        queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()
        end = time.perf_counter()

        times_ms.append((end - start) * 1e3)

    mean_total_ms = float(np.mean(times_ms))
    per_dispatch_us = (mean_total_ms * 1000) / n_dispatches

    return {
        "name": "wgpu_batched_dispatches",
        "n_dispatches": n_dispatches,
        "mean_total_ms": mean_total_ms,
        "std_total_ms": float(np.std(times_ms)),
        "per_dispatch_us": per_dispatch_us,
        "n_iterations": n_iterations
    }


def project_e2e_performance(dawn_sequential, wgpu_sequential, dawn_e2e_ttft_ms=42.0, dawn_dispatches=560):
    """
    Project end-to-end performance on wgpu-native based on dispatch overhead difference.

    Methodology:
    - Assume compute time is identical (same GPU)
    - Difference is purely dispatch overhead
    - Project: TTFT_wgpu = TTFT_dawn - dispatches * (overhead_dawn - overhead_wgpu)
    """
    if "error" in dawn_sequential or "error" in wgpu_sequential:
        return {"error": "Missing data for projection"}

    dawn_per_dispatch_us = dawn_sequential["per_dispatch_us"]
    wgpu_per_dispatch_us = wgpu_sequential["per_dispatch_us"]

    overhead_diff_us = dawn_per_dispatch_us - wgpu_per_dispatch_us
    overhead_diff_ms = overhead_diff_us / 1000

    total_savings_ms = dawn_dispatches * overhead_diff_ms
    projected_ttft_ms = dawn_e2e_ttft_ms - total_savings_ms

    # Sanity check
    if projected_ttft_ms < 0:
        # This would mean dispatch overhead is > total TTFT, which is impossible
        # Adjust to indicate compute-bound scenario
        projected_ttft_ms = dawn_e2e_ttft_ms * (wgpu_per_dispatch_us / dawn_per_dispatch_us)

    projected_toks = 50 / (projected_ttft_ms / 1000 + 49 * (projected_ttft_ms / 1000))  # Rough estimate

    return {
        "dawn_e2e_ttft_ms": dawn_e2e_ttft_ms,
        "dawn_dispatches": dawn_dispatches,
        "dawn_per_dispatch_us": dawn_per_dispatch_us,
        "wgpu_per_dispatch_us": wgpu_per_dispatch_us,
        "overhead_diff_per_dispatch_us": overhead_diff_us,
        "total_overhead_savings_ms": total_savings_ms,
        "projected_wgpu_ttft_ms": projected_ttft_ms,
        "speedup_factor": dawn_e2e_ttft_ms / projected_ttft_ms if projected_ttft_ms > 0 else float('inf'),
        "note": "Projection assumes compute time is identical; only dispatch overhead differs"
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end performance projection")
    parser.add_argument("--output", type=str, default="results/exp7_e2e_projection.json")
    parser.add_argument("--dawn-results", type=str, default="results/exp6_dawn_direct.json",
                        help="Path to Dawn results from exp6")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 7: End-to-End Performance Projection")
    print("=" * 60)

    results = {
        "system_info": get_system_info(),
        "experiments": {}
    }

    # 1. wgpu sequential dispatches (matches exp6 methodology)
    print("\n1. Measuring wgpu-native sequential dispatches (100 ops)...")
    wgpu_sequential = benchmark_wgpu_sequential_dispatches(n_dispatches=100, n_iterations=50)
    results["experiments"]["wgpu_sequential"] = wgpu_sequential
    if "error" not in wgpu_sequential:
        print(f"   Total: {wgpu_sequential['mean_total_ms']:.2f} ms for {wgpu_sequential['n_dispatches']} dispatches")
        print(f"   Per-dispatch: {wgpu_sequential['per_dispatch_us']:.1f} µs")

    # 2. wgpu batched dispatches (shows batching potential)
    print("\n2. Measuring wgpu-native batched dispatches (100 ops in 1 submit)...")
    wgpu_batched = benchmark_wgpu_batched_dispatches(n_dispatches=100, n_iterations=50)
    results["experiments"]["wgpu_batched"] = wgpu_batched
    if "error" not in wgpu_batched:
        print(f"   Total: {wgpu_batched['mean_total_ms']:.2f} ms for {wgpu_batched['n_dispatches']} dispatches")
        print(f"   Per-dispatch: {wgpu_batched['per_dispatch_us']:.1f} µs")

    if "error" not in wgpu_sequential and "error" not in wgpu_batched:
        batching_speedup = wgpu_sequential['mean_total_ms'] / wgpu_batched['mean_total_ms']
        print(f"   Batching speedup: {batching_speedup:.2f}x")
        results["experiments"]["batching_speedup"] = batching_speedup

    # 3. Load Dawn results for comparison
    print("\n3. Loading Dawn results for comparison...")
    dawn_sequential = None
    try:
        with open(args.dawn_results) as f:
            dawn_data = json.load(f)
            if "experiments" in dawn_data and "sequential_dispatches" in dawn_data["experiments"]:
                dawn_sequential = dawn_data["experiments"]["sequential_dispatches"]
                print(f"   Dawn per-dispatch: {dawn_sequential['per_dispatch_us']:.1f} µs")
                results["dawn_results"] = dawn_sequential
    except FileNotFoundError:
        print(f"   WARNING: Dawn results not found at {args.dawn_results}")
        print("   Run exp6_dawn_direct.py first, or provide --dawn-results path")
        # Use derived estimate
        dawn_sequential = {
            "per_dispatch_us": 95.0,  # Derived from TTFT differences
            "note": "Estimated from TTFT differences (not directly measured)"
        }
        print(f"   Using estimated Dawn overhead: {dawn_sequential['per_dispatch_us']} µs")
        results["dawn_results"] = dawn_sequential

    # 4. Project end-to-end performance
    print("\n4. Projecting end-to-end performance...")
    projection = project_e2e_performance(
        dawn_sequential=dawn_sequential,
        wgpu_sequential=wgpu_sequential,
        dawn_e2e_ttft_ms=42.0,  # From our benchmarks
        dawn_dispatches=560     # With fusion
    )
    results["projection"] = projection

    if "error" not in projection:
        print(f"\n   Dawn TTFT (measured):     {projection['dawn_e2e_ttft_ms']:.1f} ms")
        print(f"   Dawn per-dispatch:        {projection['dawn_per_dispatch_us']:.1f} µs")
        print(f"   wgpu per-dispatch:        {projection['wgpu_per_dispatch_us']:.1f} µs")
        print(f"   Overhead savings:         {projection['total_overhead_savings_ms']:.1f} ms")
        print(f"   Projected wgpu TTFT:      {projection['projected_wgpu_ttft_ms']:.1f} ms")
        print(f"   Projected speedup:        {projection['speedup_factor']:.2f}x")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if "error" not in wgpu_sequential and dawn_sequential:
        wgpu_us = wgpu_sequential['per_dispatch_us']
        dawn_us = dawn_sequential['per_dispatch_us']
        print(f"\nPer-dispatch overhead:")
        print(f"  Dawn:        {dawn_us:.1f} µs")
        print(f"  wgpu-native: {wgpu_us:.1f} µs")
        print(f"  Ratio:       {dawn_us / wgpu_us:.2f}x faster on wgpu")

    if "error" not in projection:
        print(f"\nProjected end-to-end improvement:")
        print(f"  Dawn TTFT:      {projection['dawn_e2e_ttft_ms']:.1f} ms → {50/projection['dawn_e2e_ttft_ms']*1000:.1f} tok/s")
        print(f"  wgpu TTFT:      {projection['projected_wgpu_ttft_ms']:.1f} ms → {50/projection['projected_wgpu_ttft_ms']*1000:.1f} tok/s (projected)")

    print("=" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
