#!/usr/bin/env python3
"""
Experiment 1: Cross-GPU WebGPU Validation

This script benchmarks WebGPU compute performance on any GPU using wgpu-py.
Run this on different machines (Apple M2, AMD, Intel) to validate cross-vendor findings.

Usage:
    pip install wgpu numpy scipy
    python exp1_cross_gpu_webgpu.py --output results/exp1_apple_m2_webgpu.json

This measures:
1. Per-dispatch overhead (encoder creation, bind group, submission)
2. Kernel execution time for representative operations (matmul, reduction)
3. Unfused vs fused RMSNorm performance
4. Single-workgroup mega-kernel vs multi-workgroup approach
"""

import argparse
import json
import math
import platform
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
                adapter_info = adapter.info
                info["gpu_vendor"] = adapter_info.get("vendor", "unknown")
                info["gpu_architecture"] = adapter_info.get("architecture", "unknown")
                info["gpu_device"] = adapter_info.get("device", "unknown")
                info["gpu_description"] = adapter_info.get("description", "unknown")
                info["wgpu_backend"] = adapter_info.get("backend_type", "unknown")
        except Exception as e:
            info["gpu_error"] = str(e)

    return info


def measure_dispatch_overhead(device, queue, n_iterations=1000):
    """Measure per-dispatch overhead components."""

    # Simple shader that does minimal work
    shader_code = """
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        data[gid.x] = data[gid.x] + 1.0;
    }
    """

    shader_module = device.create_shader_module(code=shader_code)

    # Create pipeline
    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": shader_module, "entry_point": "main"}
    )

    # Create buffer
    buffer_size = 256 * 4  # 256 floats
    buffer = device.create_buffer(
        size=buffer_size,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    )

    # Create bind group
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": buffer}}]
    )

    # Warmup
    for _ in range(10):
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(4)
        compute_pass.end()
        queue.submit([encoder.finish()])

    # Wait for warmup to complete
    queue.on_submitted_work_done_sync()

    # Measure individual dispatch times
    dispatch_times = []

    for _ in range(n_iterations):
        start = time.perf_counter()

        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(4)
        compute_pass.end()
        queue.submit([encoder.finish()])

        end = time.perf_counter()
        dispatch_times.append((end - start) * 1e6)  # Convert to microseconds

    # Wait for all work to complete
    queue.on_submitted_work_done_sync()

    return {
        "mean_dispatch_us": np.mean(dispatch_times),
        "std_dispatch_us": np.std(dispatch_times),
        "min_dispatch_us": np.min(dispatch_times),
        "max_dispatch_us": np.max(dispatch_times),
        "n_iterations": n_iterations,
        "all_times_us": dispatch_times[:100]  # First 100 for analysis
    }


def benchmark_sequential_dispatches(device, queue, n_dispatches=100, n_iterations=50, warmup=5):
    """
    Measure TRUE dispatch overhead (sync only at end, not after each op).
    This avoids inflating overhead with sync costs.
    """
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

    buffer = device.create_buffer(
        size=256 * 4,
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
        "name": "sequential_dispatches",
        "n_dispatches": n_dispatches,
        "mean_total_ms": mean_total_ms,
        "std_total_ms": float(np.std(times_ms)),
        "per_dispatch_us": per_dispatch_us,
        "n_iterations": n_iterations,
        "note": "TRUE dispatch overhead (sync only at end)"
    }


def benchmark_rmsnorm_unfused(device, queue, hidden_dim=896, n_iterations=100):
    """Benchmark unfused RMSNorm (6 separate dispatches)."""

    # Shader for each RMSNorm component
    shaders = {
        "square": """
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let idx = gid.x;
                if (idx < arrayLength(&input)) {
                    output[idx] = input[idx] * input[idx];
                }
            }
        """,
        "reduce_sum": """
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            var<workgroup> wg_data: array<f32, 256>;
            @compute @workgroup_size(256)
            fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
                let idx = lid.x;
                wg_data[idx] = input[idx];
                workgroupBarrier();
                // Simple reduction
                for (var s = 128u; s > 0u; s = s >> 1u) {
                    if (idx < s) {
                        wg_data[idx] += wg_data[idx + s];
                    }
                    workgroupBarrier();
                }
                if (idx == 0u) {
                    output[wid.x] = wg_data[0];
                }
            }
        """,
        "rsqrt": """
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            @compute @workgroup_size(1)
            fn main() {
                let mean = input[0] / 896.0;
                output[0] = 1.0 / sqrt(mean + 1e-6);
            }
        """,
        "normalize": """
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read> scale: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let idx = gid.x;
                if (idx < arrayLength(&input)) {
                    output[idx] = input[idx] * scale[0];
                }
            }
        """,
        "weight_mul": """
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read> weight: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let idx = gid.x;
                if (idx < arrayLength(&input)) {
                    output[idx] = input[idx] * weight[idx];
                }
            }
        """
    }

    # Create all pipelines
    pipelines = {}
    for name, code in shaders.items():
        module = device.create_shader_module(code=code)
        pipelines[name] = device.create_compute_pipeline(
            layout="auto",
            compute={"module": module, "entry_point": "main"}
        )

    # Create buffers
    buffer_size = hidden_dim * 4
    input_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    squared_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE)
    sum_buf = device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE)
    rsqrt_buf = device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE)
    normalized_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE)
    weight_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    output_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    # Initialize buffers
    input_data = np.random.randn(hidden_dim).astype(np.float32)
    weight_data = np.ones(hidden_dim, dtype=np.float32)
    queue.write_buffer(input_buf, 0, input_data.tobytes())
    queue.write_buffer(weight_buf, 0, weight_data.tobytes())

    # Create bind groups
    bg_square = device.create_bind_group(
        layout=pipelines["square"].get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": input_buf}},
            {"binding": 1, "resource": {"buffer": squared_buf}}
        ]
    )
    bg_reduce = device.create_bind_group(
        layout=pipelines["reduce_sum"].get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": squared_buf}},
            {"binding": 1, "resource": {"buffer": sum_buf}}
        ]
    )
    bg_rsqrt = device.create_bind_group(
        layout=pipelines["rsqrt"].get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": sum_buf}},
            {"binding": 1, "resource": {"buffer": rsqrt_buf}}
        ]
    )
    bg_normalize = device.create_bind_group(
        layout=pipelines["normalize"].get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": input_buf}},
            {"binding": 1, "resource": {"buffer": rsqrt_buf}},
            {"binding": 2, "resource": {"buffer": normalized_buf}}
        ]
    )
    bg_weight = device.create_bind_group(
        layout=pipelines["weight_mul"].get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": normalized_buf}},
            {"binding": 1, "resource": {"buffer": weight_buf}},
            {"binding": 2, "resource": {"buffer": output_buf}}
        ]
    )

    # Warmup
    for _ in range(5):
        encoder = device.create_command_encoder()

        # 6 separate dispatches
        pass1 = encoder.begin_compute_pass()
        pass1.set_pipeline(pipelines["square"])
        pass1.set_bind_group(0, bg_square)
        pass1.dispatch_workgroups((hidden_dim + 255) // 256)
        pass1.end()

        pass2 = encoder.begin_compute_pass()
        pass2.set_pipeline(pipelines["reduce_sum"])
        pass2.set_bind_group(0, bg_reduce)
        pass2.dispatch_workgroups(1)
        pass2.end()

        pass3 = encoder.begin_compute_pass()
        pass3.set_pipeline(pipelines["rsqrt"])
        pass3.set_bind_group(0, bg_rsqrt)
        pass3.dispatch_workgroups(1)
        pass3.end()

        pass4 = encoder.begin_compute_pass()
        pass4.set_pipeline(pipelines["normalize"])
        pass4.set_bind_group(0, bg_normalize)
        pass4.dispatch_workgroups((hidden_dim + 255) // 256)
        pass4.end()

        pass5 = encoder.begin_compute_pass()
        pass5.set_pipeline(pipelines["weight_mul"])
        pass5.set_bind_group(0, bg_weight)
        pass5.dispatch_workgroups((hidden_dim + 255) // 256)
        pass5.end()

        queue.submit([encoder.finish()])

    queue.on_submitted_work_done_sync()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()

        encoder = device.create_command_encoder()

        pass1 = encoder.begin_compute_pass()
        pass1.set_pipeline(pipelines["square"])
        pass1.set_bind_group(0, bg_square)
        pass1.dispatch_workgroups((hidden_dim + 255) // 256)
        pass1.end()

        pass2 = encoder.begin_compute_pass()
        pass2.set_pipeline(pipelines["reduce_sum"])
        pass2.set_bind_group(0, bg_reduce)
        pass2.dispatch_workgroups(1)
        pass2.end()

        pass3 = encoder.begin_compute_pass()
        pass3.set_pipeline(pipelines["rsqrt"])
        pass3.set_bind_group(0, bg_rsqrt)
        pass3.dispatch_workgroups(1)
        pass3.end()

        pass4 = encoder.begin_compute_pass()
        pass4.set_pipeline(pipelines["normalize"])
        pass4.set_bind_group(0, bg_normalize)
        pass4.dispatch_workgroups((hidden_dim + 255) // 256)
        pass4.end()

        pass5 = encoder.begin_compute_pass()
        pass5.set_pipeline(pipelines["weight_mul"])
        pass5.set_bind_group(0, bg_weight)
        pass5.dispatch_workgroups((hidden_dim + 255) // 256)
        pass5.end()

        queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "name": "rmsnorm_unfused",
        "dispatches": 5,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "n_iterations": n_iterations
    }


def benchmark_rmsnorm_fused(device, queue, hidden_dim=896, n_iterations=100):
    """Benchmark fused RMSNorm (1 dispatch)."""

    shader_code = f"""
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;

    var<workgroup> wg_data: array<f32, 256>;

    @compute @workgroup_size(256)
    fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
        let idx = lid.x;
        let hidden_dim = {hidden_dim}u;

        // Each thread accumulates multiple elements
        var sum_sq: f32 = 0.0;
        for (var i = idx; i < hidden_dim; i += 256u) {{
            let val = input[i];
            sum_sq += val * val;
        }}

        wg_data[idx] = sum_sq;
        workgroupBarrier();

        // Parallel reduction
        for (var s = 128u; s > 0u; s = s >> 1u) {{
            if (idx < s) {{
                wg_data[idx] += wg_data[idx + s];
            }}
            workgroupBarrier();
        }}

        // Compute rsqrt
        let mean = wg_data[0] / f32(hidden_dim);
        let rsqrt_val = 1.0 / sqrt(mean + 1e-6);

        // Normalize and apply weight
        for (var i = idx; i < hidden_dim; i += 256u) {{
            output[i] = input[i] * rsqrt_val * weight[i];
        }}
    }}
    """

    module = device.create_shader_module(code=shader_code)
    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": module, "entry_point": "main"}
    )

    buffer_size = hidden_dim * 4
    input_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    weight_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    output_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    input_data = np.random.randn(hidden_dim).astype(np.float32)
    weight_data = np.ones(hidden_dim, dtype=np.float32)
    queue.write_buffer(input_buf, 0, input_data.tobytes())
    queue.write_buffer(weight_buf, 0, weight_data.tobytes())

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": input_buf}},
            {"binding": 1, "resource": {"buffer": weight_buf}},
            {"binding": 2, "resource": {"buffer": output_buf}}
        ]
    )

    # Warmup
    for _ in range(5):
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        queue.submit([encoder.finish()])
    queue.on_submitted_work_done_sync()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()

        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        "name": "rmsnorm_fused",
        "dispatches": 1,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "n_iterations": n_iterations
    }


def benchmark_mega_kernel_single_workgroup(device, queue, hidden_dim=896, intermediate_dim=256, n_iterations=100):
    """
    Benchmark mega-kernel approach (single workgroup).
    Simplified MLP: input -> matmul -> silu -> output
    Limited to 256 threads due to workgroup limit.
    """

    shader_code = f"""
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

        // Step 1: Matmul (input @ weight^T) - each thread computes one output
        if (idx < inter) {{
            var sum: f32 = 0.0;
            for (var j = 0u; j < hidden; j++) {{
                sum += input[j] * weight[idx * hidden + j];
            }}
            intermediate[idx] = sum;
        }}
        workgroupBarrier();

        // Step 2: SiLU activation
        if (idx < inter) {{
            output[idx] = silu(intermediate[idx]);
        }}
    }}
    """

    module = device.create_shader_module(code=shader_code)
    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": module, "entry_point": "main"}
    )

    input_buf = device.create_buffer(size=hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    weight_buf = device.create_buffer(size=hidden_dim * intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    output_buf = device.create_buffer(size=intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    input_data = np.random.randn(hidden_dim).astype(np.float32)
    weight_data = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32)
    queue.write_buffer(input_buf, 0, input_data.tobytes())
    queue.write_buffer(weight_buf, 0, weight_data.tobytes())

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": input_buf}},
            {"binding": 1, "resource": {"buffer": weight_buf}},
            {"binding": 2, "resource": {"buffer": output_buf}}
        ]
    )

    # Warmup
    for _ in range(5):
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(1)  # Single workgroup!
        compute_pass.end()
        queue.submit([encoder.finish()])
    queue.on_submitted_work_done_sync()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()

        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        "name": "mega_kernel_single_workgroup",
        "dispatches": 1,
        "threads": 256,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "n_iterations": n_iterations
    }


def benchmark_multi_workgroup(device, queue, hidden_dim=896, intermediate_dim=256, n_iterations=100):
    """
    Benchmark multi-workgroup approach (2 dispatches, many workgroups each).
    """

    # Shader 1: Matmul with many workgroups
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

    # Shader 2: SiLU with many workgroups
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

    input_buf = device.create_buffer(size=hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    weight_buf = device.create_buffer(size=hidden_dim * intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    intermediate_buf = device.create_buffer(size=intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE)
    output_buf = device.create_buffer(size=intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    input_data = np.random.randn(hidden_dim).astype(np.float32)
    weight_data = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32)
    queue.write_buffer(input_buf, 0, input_data.tobytes())
    queue.write_buffer(weight_buf, 0, weight_data.tobytes())

    matmul_bg = device.create_bind_group(
        layout=matmul_pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": input_buf}},
            {"binding": 1, "resource": {"buffer": weight_buf}},
            {"binding": 2, "resource": {"buffer": intermediate_buf}}
        ]
    )
    silu_bg = device.create_bind_group(
        layout=silu_pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": intermediate_buf}},
            {"binding": 1, "resource": {"buffer": output_buf}}
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

    # Timed runs
    times = []
    for _ in range(n_iterations):
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
        times.append((end - start) * 1000)

    return {
        "name": "multi_workgroup",
        "dispatches": 2,
        "threads": n_workgroups * 64,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "n_iterations": n_iterations
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-GPU WebGPU benchmark")
    parser.add_argument("--output", type=str, default="results/exp1_webgpu.json", help="Output JSON file")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--power-preference", type=str, default="high-performance",
                        choices=["high-performance", "low-power"],
                        help="GPU power preference (high-performance for discrete, low-power for integrated)")
    args = parser.parse_args()

    if not WGPU_AVAILABLE:
        print("ERROR: wgpu is required. Install with: pip install wgpu")
        return

    print("=" * 60)
    print("Experiment 1: Cross-GPU WebGPU Validation")
    print("=" * 60)

    # Request specific adapter
    adapter = wgpu.gpu.request_adapter_sync(power_preference=args.power_preference)
    if not adapter:
        print(f"ERROR: No adapter found for power_preference={args.power_preference}")
        return

    adapter_info = adapter.info
    print(f"\nSelected GPU: {adapter_info.get('device', 'unknown')}")
    print(f"Vendor: {adapter_info.get('vendor', 'unknown')}")
    print(f"Backend: {adapter_info.get('backend_type', 'unknown')}")
    print(f"Adapter type: {adapter_info.get('adapter_type', 'unknown')}")

    # Get system info
    system_info = get_system_info()
    # Override with selected adapter info
    system_info["gpu_vendor"] = adapter_info.get("vendor", "unknown")
    system_info["gpu_device"] = adapter_info.get("device", "unknown")
    system_info["gpu_description"] = adapter_info.get("description", "unknown")
    system_info["wgpu_backend"] = adapter_info.get("backend_type", "unknown")
    system_info["adapter_type"] = adapter_info.get("adapter_type", "unknown")
    system_info["power_preference"] = args.power_preference

    # Request device
    device = adapter.request_device_sync()
    queue = device.queue

    print(f"\nRunning benchmarks with {args.iterations} iterations each...")

    results = {
        "system_info": system_info,
        "experiments": {}
    }

    # 1. Dispatch overhead (single op with implicit sync)
    print("\n1. Measuring dispatch overhead (single op)...")
    overhead_results = measure_dispatch_overhead(device, queue, n_iterations=1000)
    results["experiments"]["dispatch_overhead"] = overhead_results
    print(f"   Per-dispatch overhead: {overhead_results['mean_dispatch_us']:.1f} ± {overhead_results['std_dispatch_us']:.1f} µs")
    print(f"   (Note: includes implicit sync overhead)")

    # 1b. Sequential dispatches (TRUE dispatch overhead)
    print("\n1b. Measuring TRUE dispatch overhead (100 sequential ops, sync at end)...")
    sequential_results = benchmark_sequential_dispatches(device, queue, n_dispatches=100, n_iterations=50)
    results["experiments"]["sequential_dispatches"] = sequential_results
    print(f"   Total: {sequential_results['mean_total_ms']:.2f} ms for {sequential_results['n_dispatches']} dispatches")
    print(f"   TRUE per-dispatch overhead: {sequential_results['per_dispatch_us']:.1f} µs")

    # 2. RMSNorm unfused vs fused
    print("\n2. Benchmarking RMSNorm unfused (5 dispatches)...")
    rmsnorm_unfused = benchmark_rmsnorm_unfused(device, queue, n_iterations=args.iterations)
    results["experiments"]["rmsnorm_unfused"] = rmsnorm_unfused
    print(f"   Time: {rmsnorm_unfused['mean_ms']:.3f} ± {rmsnorm_unfused['std_ms']:.3f} ms")

    print("\n3. Benchmarking RMSNorm fused (1 dispatch)...")
    rmsnorm_fused = benchmark_rmsnorm_fused(device, queue, n_iterations=args.iterations)
    results["experiments"]["rmsnorm_fused"] = rmsnorm_fused
    print(f"   Time: {rmsnorm_fused['mean_ms']:.3f} ± {rmsnorm_fused['std_ms']:.3f} ms")

    fusion_speedup = rmsnorm_unfused['mean_ms'] / rmsnorm_fused['mean_ms']
    print(f"   Fusion speedup: {fusion_speedup:.2f}x")
    results["experiments"]["rmsnorm_fusion_speedup"] = fusion_speedup

    # 3. Mega-kernel vs multi-workgroup
    print("\n4. Benchmarking mega-kernel (single workgroup, 256 threads)...")
    mega_single = benchmark_mega_kernel_single_workgroup(device, queue, n_iterations=args.iterations)
    results["experiments"]["mega_kernel_single_wg"] = mega_single
    print(f"   Time: {mega_single['mean_ms']:.3f} ± {mega_single['std_ms']:.3f} ms")

    print("\n5. Benchmarking multi-workgroup (2 dispatches, many threads)...")
    multi_wg = benchmark_multi_workgroup(device, queue, n_iterations=args.iterations)
    results["experiments"]["multi_workgroup"] = multi_wg
    print(f"   Time: {multi_wg['mean_ms']:.3f} ± {multi_wg['std_ms']:.3f} ms")

    mega_slowdown = mega_single['mean_ms'] / multi_wg['mean_ms']
    print(f"   Mega-kernel slowdown: {mega_slowdown:.1f}x slower than multi-workgroup")
    results["experiments"]["mega_kernel_slowdown"] = mega_slowdown

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"GPU: {system_info.get('gpu_description', 'unknown')}")
    print(f"Backend: {system_info.get('wgpu_backend', 'unknown')}")
    print(f"Per-dispatch overhead: {overhead_results['mean_dispatch_us']:.1f} µs")
    print(f"RMSNorm fusion speedup: {fusion_speedup:.2f}x")
    print(f"Mega-kernel slowdown: {mega_slowdown:.1f}x")
    print("=" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
