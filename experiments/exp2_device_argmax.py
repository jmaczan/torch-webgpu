#!/usr/bin/env python3
"""
Experiment 2: Device-Side Argmax Implementation

This experiment implements and benchmarks device-side argmax to measure
whether it can reduce the ~11ms sync overhead observed in autoregressive generation.

Usage:
    python exp2_device_argmax.py --output results/exp2_device_argmax.json

Measures:
1. Full logits readback (current approach): Map entire 600KB+ buffer
2. Device-side argmax: Compute argmax on GPU, read only 4 bytes
3. Buffer map/unmap microbenchmark: Isolate mapping overhead
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
    print("Warning: wgpu not available. Install with: pip install wgpu")


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
                info["gpu"] = adapter.info.get("description", "unknown")
                info["backend"] = adapter.info.get("backend_type", "unknown")
        except:
            pass
    return info


def benchmark_full_readback(device, queue, vocab_size=151936, n_iterations=100):
    """
    Benchmark reading full logits tensor from GPU (current approach).
    vocab_size=151936 is Qwen2.5's vocabulary size.
    """
    buffer_size = vocab_size * 4  # float32

    # Create GPU buffer with logits
    gpu_buffer = device.create_buffer(
        size=buffer_size,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    )

    # Create staging buffer for readback
    staging_buffer = device.create_buffer(
        size=buffer_size,
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )

    # Initialize with random data
    logits = np.random.randn(vocab_size).astype(np.float32)
    queue.write_buffer(gpu_buffer, 0, logits.tobytes())

    # Warmup
    for _ in range(3):
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(gpu_buffer, 0, staging_buffer, 0, buffer_size)
        queue.submit([encoder.finish()])
        staging_buffer.map_sync(mode=wgpu.MapMode.READ)
        _ = staging_buffer.read_mapped()
        staging_buffer.unmap()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()

        # Copy from GPU to staging
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(gpu_buffer, 0, staging_buffer, 0, buffer_size)
        queue.submit([encoder.finish()])

        # Map and read
        staging_buffer.map_sync(mode=wgpu.MapMode.READ)
        data = staging_buffer.read_mapped()
        staging_buffer.unmap()

        # CPU-side argmax (what we currently do)
        logits_array = np.frombuffer(data, dtype=np.float32)
        token_id = np.argmax(logits_array)

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "name": "full_readback",
        "buffer_size_kb": buffer_size / 1024,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "n_iterations": n_iterations
    }


def benchmark_device_argmax(device, queue, vocab_size=151936, n_iterations=100):
    """
    Benchmark device-side argmax: compute on GPU, read only 4 bytes.
    """

    # Argmax shader using parallel reduction
    shader_code = f"""
    @group(0) @binding(0) var<storage, read> logits: array<f32>;
    @group(0) @binding(1) var<storage, read_write> result: array<u32>;

    var<workgroup> wg_max: array<f32, 256>;
    var<workgroup> wg_idx: array<u32, 256>;

    @compute @workgroup_size(256)
    fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
        let tid = lid.x;
        let vocab_size = {vocab_size}u;
        let elements_per_thread = (vocab_size + 255u) / 256u;

        // Each thread finds max in its chunk
        var local_max: f32 = -1e30;
        var local_idx: u32 = 0u;

        for (var i = 0u; i < elements_per_thread; i++) {{
            let idx = tid * elements_per_thread + i;
            if (idx < vocab_size) {{
                let val = logits[idx];
                if (val > local_max) {{
                    local_max = val;
                    local_idx = idx;
                }}
            }}
        }}

        wg_max[tid] = local_max;
        wg_idx[tid] = local_idx;
        workgroupBarrier();

        // Parallel reduction to find global max
        for (var s = 128u; s > 0u; s = s >> 1u) {{
            if (tid < s) {{
                if (wg_max[tid + s] > wg_max[tid]) {{
                    wg_max[tid] = wg_max[tid + s];
                    wg_idx[tid] = wg_idx[tid + s];
                }}
            }}
            workgroupBarrier();
        }}

        // Thread 0 writes result
        if (tid == 0u) {{
            result[0] = wg_idx[0];
        }}
    }}
    """

    module = device.create_shader_module(code=shader_code)
    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": module, "entry_point": "main"}
    )

    # Create buffers
    logits_buffer = device.create_buffer(
        size=vocab_size * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )
    result_buffer = device.create_buffer(
        size=4,  # Single u32
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )
    staging_buffer = device.create_buffer(
        size=4,
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )

    # Initialize logits
    logits = np.random.randn(vocab_size).astype(np.float32)
    queue.write_buffer(logits_buffer, 0, logits.tobytes())

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": logits_buffer}},
            {"binding": 1, "resource": {"buffer": result_buffer}}
        ]
    )

    # Warmup
    for _ in range(3):
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        encoder.copy_buffer_to_buffer(result_buffer, 0, staging_buffer, 0, 4)
        queue.submit([encoder.finish()])
        staging_buffer.map_sync(mode=wgpu.MapMode.READ)
        _ = staging_buffer.read_mapped()
        staging_buffer.unmap()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()

        # Run argmax shader
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()

        # Copy result (4 bytes only)
        encoder.copy_buffer_to_buffer(result_buffer, 0, staging_buffer, 0, 4)
        queue.submit([encoder.finish()])

        # Map and read
        staging_buffer.map_sync(mode=wgpu.MapMode.READ)
        data = staging_buffer.read_mapped()
        token_id = np.frombuffer(data, dtype=np.uint32)[0]
        staging_buffer.unmap()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    # Verify correctness
    expected_token_id = np.argmax(logits)
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(1)
    compute_pass.end()
    encoder.copy_buffer_to_buffer(result_buffer, 0, staging_buffer, 0, 4)
    queue.submit([encoder.finish()])
    staging_buffer.map_sync(mode=wgpu.MapMode.READ)
    actual_token_id = np.frombuffer(staging_buffer.read_mapped(), dtype=np.uint32)[0]
    staging_buffer.unmap()

    return {
        "name": "device_argmax",
        "buffer_size_bytes": 4,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "n_iterations": n_iterations,
        "correctness": bool(actual_token_id == expected_token_id),
        "expected_token": int(expected_token_id),
        "actual_token": int(actual_token_id)
    }


def benchmark_buffer_map_unmap(device, queue, buffer_sizes=[4, 1024, 4096, 16384, 65536, 262144, 600*1024], n_iterations=100):
    """
    Microbenchmark: measure buffer map/unmap overhead for different sizes.
    """
    results = []

    for size in buffer_sizes:
        gpu_buffer = device.create_buffer(
            size=size,
            usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        )
        staging_buffer = device.create_buffer(
            size=size,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
        )

        # Initialize
        data = np.random.bytes(size)
        queue.write_buffer(gpu_buffer, 0, data)

        # Warmup
        for _ in range(3):
            encoder = device.create_command_encoder()
            encoder.copy_buffer_to_buffer(gpu_buffer, 0, staging_buffer, 0, size)
            queue.submit([encoder.finish()])
            staging_buffer.map_sync(mode=wgpu.MapMode.READ)
            _ = staging_buffer.read_mapped()
            staging_buffer.unmap()

        # Timed runs
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()

            encoder = device.create_command_encoder()
            encoder.copy_buffer_to_buffer(gpu_buffer, 0, staging_buffer, 0, size)
            queue.submit([encoder.finish()])
            staging_buffer.map_sync(mode=wgpu.MapMode.READ)
            _ = staging_buffer.read_mapped()
            staging_buffer.unmap()

            end = time.perf_counter()
            times.append((end - start) * 1000)

        results.append({
            "size_bytes": size,
            "size_kb": size / 1024,
            "mean_ms": np.mean(times),
            "std_ms": np.std(times)
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Device-side argmax experiment")
    parser.add_argument("--output", type=str, default="results/exp2_device_argmax.json")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not WGPU_AVAILABLE:
        print("ERROR: wgpu required. Install with: pip install wgpu")
        return

    print("=" * 60)
    print("Experiment 2: Device-Side Argmax")
    print("=" * 60)

    system_info = get_system_info()
    print(f"GPU: {system_info.get('gpu', 'unknown')}")

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    queue = device.queue

    results = {
        "system_info": system_info,
        "vocab_size": 151936,
        "experiments": {}
    }

    # 1. Full readback benchmark
    print("\n1. Benchmarking full logits readback (current approach)...")
    full_readback = benchmark_full_readback(device, queue, n_iterations=args.iterations)
    results["experiments"]["full_readback"] = full_readback
    print(f"   Time: {full_readback['mean_ms']:.2f} ± {full_readback['std_ms']:.2f} ms")
    print(f"   Buffer size: {full_readback['buffer_size_kb']:.1f} KB")

    # 2. Device-side argmax
    print("\n2. Benchmarking device-side argmax...")
    device_argmax = benchmark_device_argmax(device, queue, n_iterations=args.iterations)
    results["experiments"]["device_argmax"] = device_argmax
    print(f"   Time: {device_argmax['mean_ms']:.2f} ± {device_argmax['std_ms']:.2f} ms")
    print(f"   Correctness: {'PASS' if device_argmax['correctness'] else 'FAIL'}")

    # Calculate improvement
    improvement = full_readback['mean_ms'] - device_argmax['mean_ms']
    improvement_pct = improvement / full_readback['mean_ms'] * 100
    print(f"   Improvement: {improvement:.2f} ms ({improvement_pct:.1f}%)")
    results["experiments"]["improvement_ms"] = improvement
    results["experiments"]["improvement_pct"] = improvement_pct

    # 3. Buffer map/unmap microbenchmark
    print("\n3. Buffer map/unmap microbenchmark...")
    map_unmap_results = benchmark_buffer_map_unmap(device, queue, n_iterations=args.iterations)
    results["experiments"]["buffer_map_unmap"] = map_unmap_results

    print("   Size (KB)    Time (ms)")
    for r in map_unmap_results:
        print(f"   {r['size_kb']:>8.1f}    {r['mean_ms']:.2f} ± {r['std_ms']:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Full readback (600KB):  {full_readback['mean_ms']:.2f} ms")
    print(f"Device argmax (4B):     {device_argmax['mean_ms']:.2f} ms")
    print(f"Improvement:            {improvement:.2f} ms ({improvement_pct:.1f}%)")
    print(f"Minimum map/unmap (4B): {map_unmap_results[0]['mean_ms']:.2f} ms")
    print("=" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
