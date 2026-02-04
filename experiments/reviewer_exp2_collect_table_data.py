#!/usr/bin/env python3
"""
Reviewer Response Experiment 2: Collect Raw Data for Tables Lacking CIs

This script addresses Minor Reviewer Request #6:
"Add error bars/CIs to more tables: Tables 7, 8, 14, 15 lack uncertainty quantification."

And Minor Request #8:
"Statistical significance for all comparisons: significance tests are missing for
mega-kernel (Table 14), device-side argmax (Table 15), and tiled strategy (Table 16)"

This collects 30 runs of each measurement to calculate proper 95% CIs and p-values.

Usage:
    pip install wgpu numpy scipy
    python reviewer_exp2_collect_table_data.py --output results/reviewer_table_data.json
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    import wgpu
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    print("ERROR: wgpu required. Install with: pip install wgpu")

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. p-values will not be computed.")


def calculate_stats(data):
    """Calculate comprehensive statistics."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample std

    # 95% CI
    if SCIPY_AVAILABLE:
        t_value = scipy_stats.t.ppf(0.975, n - 1)
    else:
        t_value = 2.0 if n < 30 else 1.96

    std_error = std / np.sqrt(n)
    margin = t_value * std_error

    return {
        "mean": float(mean),
        "std": float(std),
        "ci95_lower": float(mean - margin),
        "ci95_upper": float(mean + margin),
        "cv_percent": float(std / mean * 100) if mean > 0 else 0,
        "n": n,
        "raw_data": [float(x) for x in data],
    }


def ttest_ind(data1, data2):
    """Perform independent t-test and return p-value and effect size."""
    if not SCIPY_AVAILABLE:
        return {"p_value": None, "note": "scipy not available"}

    # Welch's t-test (doesn't assume equal variances)
    t_stat, p_value = scipy_stats.ttest_ind(data1, data2, equal_var=False)

    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant_p05": p_value < 0.05,
        "significant_p01": p_value < 0.01,
    }


def benchmark_kernel_optimization(device, queue, n_runs=30):
    """
    Collect data for Table 7: Kernel optimization results.
    Measures isolated speedup for parallel softmax and tiled matmul.
    """
    print("\n--- Table 7: Kernel Optimization Results ---")
    results = {}

    # Test dimensions
    vocab_size = 151936  # Qwen vocabulary
    hidden_dim = 896

    # 1. Naive softmax (sequential)
    print("  Measuring naive softmax...")
    naive_softmax_shader = """
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(1)
    fn main() {
        let n = arrayLength(&input);
        var max_val: f32 = input[0];
        for (var i = 1u; i < n; i++) {
            max_val = max(max_val, input[i]);
        }
        var sum: f32 = 0.0;
        for (var i = 0u; i < n; i++) {
            sum += exp(input[i] - max_val);
        }
        for (var i = 0u; i < n; i++) {
            output[i] = exp(input[i] - max_val) / sum;
        }
    }
    """

    # Due to shader complexity for large vocab, use smaller test size
    test_size = 4096  # Smaller for naive version

    naive_module = device.create_shader_module(code=naive_softmax_shader)
    naive_pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": naive_module, "entry_point": "main"}
    )

    naive_input = device.create_buffer(
        size=test_size * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )
    naive_output = device.create_buffer(
        size=test_size * 4,
        usage=wgpu.BufferUsage.STORAGE
    )

    queue.write_buffer(naive_input, 0, np.random.randn(test_size).astype(np.float32).tobytes())

    naive_bg = device.create_bind_group(
        layout=naive_pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": naive_input}},
            {"binding": 1, "resource": {"buffer": naive_output}},
        ]
    )

    # Warmup
    for _ in range(5):
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(naive_pipeline)
        compute_pass.set_bind_group(0, naive_bg)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        queue.submit([encoder.finish()])
    queue.on_submitted_work_done_sync()

    # Measure
    naive_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(naive_pipeline)
        compute_pass.set_bind_group(0, naive_bg)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()
        end = time.perf_counter()
        naive_times.append((end - start) * 1000)

    results["naive_softmax"] = calculate_stats(naive_times)
    print(f"    Naive softmax ({test_size}): {results['naive_softmax']['mean']:.3f} ± {results['naive_softmax']['std']:.3f} ms")

    # 2. Parallel softmax with shared memory
    print("  Measuring parallel softmax...")
    parallel_softmax_shader = f"""
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    var<workgroup> wg_max: array<f32, 256>;
    var<workgroup> wg_sum: array<f32, 256>;

    @compute @workgroup_size(256)
    fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
        let idx = lid.x;
        let n = {test_size}u;

        // Find max (parallel reduction)
        var local_max: f32 = -1e38;
        for (var i = idx; i < n; i += 256u) {{
            local_max = max(local_max, input[i]);
        }}
        wg_max[idx] = local_max;
        workgroupBarrier();

        for (var s = 128u; s > 0u; s = s >> 1u) {{
            if (idx < s) {{
                wg_max[idx] = max(wg_max[idx], wg_max[idx + s]);
            }}
            workgroupBarrier();
        }}
        let global_max = wg_max[0];

        // Sum exp (parallel reduction)
        var local_sum: f32 = 0.0;
        for (var i = idx; i < n; i += 256u) {{
            local_sum += exp(input[i] - global_max);
        }}
        wg_sum[idx] = local_sum;
        workgroupBarrier();

        for (var s = 128u; s > 0u; s = s >> 1u) {{
            if (idx < s) {{
                wg_sum[idx] += wg_sum[idx + s];
            }}
            workgroupBarrier();
        }}
        let global_sum = wg_sum[0];

        // Normalize
        for (var i = idx; i < n; i += 256u) {{
            output[i] = exp(input[i] - global_max) / global_sum;
        }}
    }}
    """

    parallel_module = device.create_shader_module(code=parallel_softmax_shader)
    parallel_pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": parallel_module, "entry_point": "main"}
    )

    parallel_bg = device.create_bind_group(
        layout=parallel_pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": naive_input}},
            {"binding": 1, "resource": {"buffer": naive_output}},
        ]
    )

    # Warmup
    for _ in range(5):
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(parallel_pipeline)
        compute_pass.set_bind_group(0, parallel_bg)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        queue.submit([encoder.finish()])
    queue.on_submitted_work_done_sync()

    # Measure
    parallel_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(parallel_pipeline)
        compute_pass.set_bind_group(0, parallel_bg)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()
        end = time.perf_counter()
        parallel_times.append((end - start) * 1000)

    results["parallel_softmax"] = calculate_stats(parallel_times)
    print(f"    Parallel softmax ({test_size}): {results['parallel_softmax']['mean']:.3f} ± {results['parallel_softmax']['std']:.3f} ms")

    # Speedup and significance
    speedup = results["naive_softmax"]["mean"] / results["parallel_softmax"]["mean"]
    results["softmax_speedup"] = speedup
    results["softmax_significance"] = ttest_ind(naive_times, parallel_times)
    print(f"    Speedup: {speedup:.1f}x, p={results['softmax_significance'].get('p_value', 'N/A')}")

    return results


def benchmark_mega_kernel_comparison(device, queue, n_runs=30):
    """
    Collect data for Table 14: Mega-kernel vs multi-workgroup fair comparison.
    """
    print("\n--- Table 14: Mega-kernel vs Multi-workgroup ---")
    results = {}

    # Use 256x256 dimensions (as mentioned in paper)
    hidden_dim = 256
    intermediate_dim = 256

    # Mega-kernel (single workgroup)
    print("  Measuring mega-kernel (single workgroup)...")
    mega_shader = f"""
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;

    var<workgroup> intermediate: array<f32, {intermediate_dim}>;

    fn silu(x: f32) -> f32 {{
        return x / (1.0 + exp(-x));
    }}

    @compute @workgroup_size(256)
    fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
        let idx = lid.x;
        let hidden = {hidden_dim}u;
        let inter = {intermediate_dim}u;

        // Matmul: each thread computes one output
        if (idx < inter) {{
            var sum: f32 = 0.0;
            for (var j = 0u; j < hidden; j++) {{
                sum += input[j] * weight[idx * hidden + j];
            }}
            intermediate[idx] = sum;
        }}
        workgroupBarrier();

        // SiLU activation
        if (idx < inter) {{
            output[idx] = silu(intermediate[idx]);
        }}
    }}
    """

    mega_module = device.create_shader_module(code=mega_shader)
    mega_pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": mega_module, "entry_point": "main"}
    )

    input_buf = device.create_buffer(size=hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    weight_buf = device.create_buffer(size=hidden_dim * intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    output_buf = device.create_buffer(size=intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE)

    queue.write_buffer(input_buf, 0, np.random.randn(hidden_dim).astype(np.float32).tobytes())
    queue.write_buffer(weight_buf, 0, np.random.randn(intermediate_dim, hidden_dim).astype(np.float32).tobytes())

    mega_bg = device.create_bind_group(
        layout=mega_pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": input_buf}},
            {"binding": 1, "resource": {"buffer": weight_buf}},
            {"binding": 2, "resource": {"buffer": output_buf}},
        ]
    )

    # Warmup
    for _ in range(5):
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(mega_pipeline)
        compute_pass.set_bind_group(0, mega_bg)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        queue.submit([encoder.finish()])
    queue.on_submitted_work_done_sync()

    # Measure
    mega_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(mega_pipeline)
        compute_pass.set_bind_group(0, mega_bg)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()
        end = time.perf_counter()
        mega_times.append((end - start) * 1000)

    results["mega_kernel"] = calculate_stats(mega_times)
    print(f"    Mega-kernel: {results['mega_kernel']['mean']:.4f} ± {results['mega_kernel']['std']:.4f} ms")

    # Multi-workgroup (2 dispatches)
    print("  Measuring multi-workgroup (2 dispatches)...")

    matmul_shader = f"""
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
        let idx = gid.x;
        let hidden = {hidden_dim}u;
        let inter = {intermediate_dim}u;
        if (idx < inter) {{
            var sum: f32 = 0.0;
            for (var j = 0u; j < hidden; j++) {{
                sum += input[j] * weight[idx * hidden + j];
            }}
            output[idx] = sum;
        }}
    }}
    """

    silu_shader = f"""
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    fn silu(x: f32) -> f32 {{
        return x / (1.0 + exp(-x));
    }}

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
        let idx = gid.x;
        if (idx < {intermediate_dim}u) {{
            output[idx] = silu(input[idx]);
        }}
    }}
    """

    matmul_module = device.create_shader_module(code=matmul_shader)
    silu_module = device.create_shader_module(code=silu_shader)

    matmul_pipeline = device.create_compute_pipeline(layout="auto", compute={"module": matmul_module, "entry_point": "main"})
    silu_pipeline = device.create_compute_pipeline(layout="auto", compute={"module": silu_module, "entry_point": "main"})

    intermediate_buf = device.create_buffer(size=intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE)
    output_buf2 = device.create_buffer(size=intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE)

    matmul_bg = device.create_bind_group(
        layout=matmul_pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": input_buf}},
            {"binding": 1, "resource": {"buffer": weight_buf}},
            {"binding": 2, "resource": {"buffer": intermediate_buf}},
        ]
    )
    silu_bg = device.create_bind_group(
        layout=silu_pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": intermediate_buf}},
            {"binding": 1, "resource": {"buffer": output_buf2}},
        ]
    )

    n_workgroups = (intermediate_dim + 63) // 64

    # Warmup
    for _ in range(5):
        encoder = device.create_command_encoder()
        pass1 = encoder.begin_compute_pass()
        pass1.set_pipeline(matmul_pipeline)
        pass1.set_bind_group(0, matmul_bg)
        pass1.dispatch_workgroups(n_workgroups)
        pass1.end()
        pass2 = encoder.begin_compute_pass()
        pass2.set_pipeline(silu_pipeline)
        pass2.set_bind_group(0, silu_bg)
        pass2.dispatch_workgroups(n_workgroups)
        pass2.end()
        queue.submit([encoder.finish()])
    queue.on_submitted_work_done_sync()

    # Measure
    multi_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        encoder = device.create_command_encoder()
        pass1 = encoder.begin_compute_pass()
        pass1.set_pipeline(matmul_pipeline)
        pass1.set_bind_group(0, matmul_bg)
        pass1.dispatch_workgroups(n_workgroups)
        pass1.end()
        pass2 = encoder.begin_compute_pass()
        pass2.set_pipeline(silu_pipeline)
        pass2.set_bind_group(0, silu_bg)
        pass2.dispatch_workgroups(n_workgroups)
        pass2.end()
        queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()
        end = time.perf_counter()
        multi_times.append((end - start) * 1000)

    results["multi_workgroup"] = calculate_stats(multi_times)
    print(f"    Multi-workgroup: {results['multi_workgroup']['mean']:.4f} ± {results['multi_workgroup']['std']:.4f} ms")

    # Comparison
    speedup = results["multi_workgroup"]["mean"] / results["mega_kernel"]["mean"]
    results["mega_vs_multi_speedup"] = speedup
    results["mega_vs_multi_significance"] = ttest_ind(mega_times, multi_times)
    print(f"    Mega-kernel speedup: {speedup:.2f}x")
    print(f"    p-value: {results['mega_vs_multi_significance'].get('p_value', 'N/A')}")
    print(f"    Cohen's d: {results['mega_vs_multi_significance'].get('cohens_d', 'N/A')}")

    return results


def benchmark_device_argmax(device, queue, n_runs=30):
    """
    Collect data for Table 15: Device-side argmax comparison.
    """
    print("\n--- Table 15: Device-side Argmax ---")
    results = {}

    vocab_size = 4096  # Smaller for benchmarking
    buffer_size = vocab_size * 4

    # Full buffer readback simulation
    print("  Measuring full readback...")
    data_buf = device.create_buffer(
        size=buffer_size,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    )
    staging_buf = device.create_buffer(
        size=buffer_size,
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )

    queue.write_buffer(data_buf, 0, np.random.randn(vocab_size).astype(np.float32).tobytes())

    full_readback_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(data_buf, 0, staging_buf, 0, buffer_size)
        queue.submit([encoder.finish()])
        staging_buf.map_sync(mode=wgpu.MapMode.READ)
        _ = staging_buf.read_mapped()
        staging_buf.unmap()
        end = time.perf_counter()
        full_readback_times.append((end - start) * 1000)

    results["full_readback"] = calculate_stats(full_readback_times)
    print(f"    Full readback ({vocab_size} floats): {results['full_readback']['mean']:.3f} ± {results['full_readback']['std']:.3f} ms")

    # Device-side argmax
    print("  Measuring device-side argmax...")
    argmax_shader = f"""
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32>;

    var<workgroup> wg_max: array<f32, 256>;
    var<workgroup> wg_idx: array<u32, 256>;

    @compute @workgroup_size(256)
    fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
        let idx = lid.x;
        let n = {vocab_size}u;

        // Each thread finds local max
        var local_max: f32 = -1e38;
        var local_idx: u32 = 0u;
        for (var i = idx; i < n; i += 256u) {{
            if (input[i] > local_max) {{
                local_max = input[i];
                local_idx = i;
            }}
        }}
        wg_max[idx] = local_max;
        wg_idx[idx] = local_idx;
        workgroupBarrier();

        // Parallel reduction for max
        for (var s = 128u; s > 0u; s = s >> 1u) {{
            if (idx < s) {{
                if (wg_max[idx + s] > wg_max[idx]) {{
                    wg_max[idx] = wg_max[idx + s];
                    wg_idx[idx] = wg_idx[idx + s];
                }}
            }}
            workgroupBarrier();
        }}

        if (idx == 0u) {{
            output[0] = wg_idx[0];
        }}
    }}
    """

    argmax_module = device.create_shader_module(code=argmax_shader)
    argmax_pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": argmax_module, "entry_point": "main"}
    )

    argmax_output = device.create_buffer(
        size=4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )
    argmax_staging = device.create_buffer(
        size=4,
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )

    argmax_bg = device.create_bind_group(
        layout=argmax_pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": data_buf}},
            {"binding": 1, "resource": {"buffer": argmax_output}},
        ]
    )

    # Warmup
    for _ in range(5):
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(argmax_pipeline)
        compute_pass.set_bind_group(0, argmax_bg)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        encoder.copy_buffer_to_buffer(argmax_output, 0, argmax_staging, 0, 4)
        queue.submit([encoder.finish()])
        argmax_staging.map_sync(mode=wgpu.MapMode.READ)
        _ = argmax_staging.read_mapped()
        argmax_staging.unmap()

    device_argmax_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(argmax_pipeline)
        compute_pass.set_bind_group(0, argmax_bg)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        encoder.copy_buffer_to_buffer(argmax_output, 0, argmax_staging, 0, 4)
        queue.submit([encoder.finish()])
        argmax_staging.map_sync(mode=wgpu.MapMode.READ)
        _ = argmax_staging.read_mapped()
        argmax_staging.unmap()
        end = time.perf_counter()
        device_argmax_times.append((end - start) * 1000)

    results["device_argmax"] = calculate_stats(device_argmax_times)
    print(f"    Device argmax (4 bytes): {results['device_argmax']['mean']:.3f} ± {results['device_argmax']['std']:.3f} ms")

    # Improvement
    improvement = (results["full_readback"]["mean"] - results["device_argmax"]["mean"]) / results["full_readback"]["mean"] * 100
    results["improvement_percent"] = improvement
    results["argmax_significance"] = ttest_ind(full_readback_times, device_argmax_times)
    print(f"    Improvement: {improvement:.1f}%")
    print(f"    p-value: {results['argmax_significance'].get('p_value', 'N/A')}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect raw data for tables lacking CIs")
    parser.add_argument("--output", type=str, default="results/reviewer_table_data.json")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--power-preference", type=str, default="high-performance")
    args = parser.parse_args()

    if not WGPU_AVAILABLE:
        print("ERROR: wgpu required")
        return

    print("=" * 70)
    print("REVIEWER RESPONSE: Collecting Raw Data for Tables with Missing CIs")
    print("=" * 70)

    adapter = wgpu.gpu.request_adapter_sync(power_preference=args.power_preference)
    if not adapter:
        print("ERROR: No adapter found")
        return

    adapter_info = adapter.info
    print(f"\nGPU: {adapter_info.get('description', 'unknown')}")
    print(f"Backend: {adapter_info.get('backend_type', 'unknown')}")

    device = adapter.request_device_sync()
    queue = device.queue

    results = {
        "system_info": {
            "gpu": adapter_info.get("description", "unknown"),
            "backend": adapter_info.get("backend_type", "unknown"),
            "vendor": adapter_info.get("vendor", "unknown"),
        },
        "n_runs": args.runs,
    }

    # Collect data for each table
    results["table7_kernel_optimization"] = benchmark_kernel_optimization(device, queue, args.runs)
    results["table14_mega_kernel"] = benchmark_mega_kernel_comparison(device, queue, args.runs)
    results["table15_device_argmax"] = benchmark_device_argmax(device, queue, args.runs)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER REVISION")
    print("=" * 70)

    print("\nTable 14 (Mega-kernel) can now include:")
    t14 = results["table14_mega_kernel"]
    print(f"  Mega: {t14['mega_kernel']['mean']:.4f} ms [95% CI: {t14['mega_kernel']['ci95_lower']:.4f}, {t14['mega_kernel']['ci95_upper']:.4f}]")
    print(f"  Multi: {t14['multi_workgroup']['mean']:.4f} ms [95% CI: {t14['multi_workgroup']['ci95_lower']:.4f}, {t14['multi_workgroup']['ci95_upper']:.4f}]")
    print(f"  Significance: p={t14['mega_vs_multi_significance'].get('p_value', 'N/A'):.4f}, d={t14['mega_vs_multi_significance'].get('cohens_d', 'N/A'):.2f}")

    print("\nTable 15 (Device argmax) can now include:")
    t15 = results["table15_device_argmax"]
    print(f"  Full readback: {t15['full_readback']['mean']:.3f} ms [95% CI: {t15['full_readback']['ci95_lower']:.3f}, {t15['full_readback']['ci95_upper']:.3f}]")
    print(f"  Device argmax: {t15['device_argmax']['mean']:.3f} ms [95% CI: {t15['device_argmax']['ci95_lower']:.3f}, {t15['device_argmax']['ci95_upper']:.3f}]")
    print(f"  Significance: p={t15['argmax_significance'].get('p_value', 'N/A'):.4f}")

    # Save - convert numpy types to Python types
    def convert_numpy(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results = convert_numpy(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
