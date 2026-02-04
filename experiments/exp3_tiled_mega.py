#!/usr/bin/env python3
"""
Experiment 3: Multi-Dispatch Tiled Mega Strategy

This experiment implements and benchmarks the middle-ground approach:
- Split fused operations into 2-4 passes
- Each pass uses many workgroups (preserves parallelism)
- Compare to: (a) single-workgroup mega-kernel, (b) fully separate dispatches

Usage:
    python exp3_tiled_mega.py --output results/exp3_tiled_mega.json

This tests the MLP block: RMSNorm -> Gate+Up -> SiLU*Up -> Down
- Approach A: 7 separate dispatches (current unfused)
- Approach B: 3 tiled dispatches (proposed middle-ground)
- Approach C: 1 mega-kernel dispatch (single workgroup - known to be slow)
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


def get_system_info():
    import platform
    info = {"platform": platform.platform()}
    if WGPU_AVAILABLE:
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            if adapter:
                info["gpu"] = adapter.info.get("description", "unknown")
                info["backend"] = adapter.info.get("backend_type", "unknown")
        except:
            pass
    return info


def benchmark_unfused_mlp(device, queue, hidden_dim=896, intermediate_dim=4864, n_iterations=100):
    """
    Approach A: Fully unfused MLP (7 separate dispatches)
    1. RMSNorm (simplified to just normalize)
    2. Gate projection (matmul)
    3. Up projection (matmul)
    4. SiLU on gate
    5. Multiply gate * up
    6. Down projection (matmul)
    7. Residual add
    """

    # For simplicity, we'll simulate the key operations
    shaders = {
        "rmsnorm": f"""
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            var<workgroup> wg_data: array<f32, 256>;

            @compute @workgroup_size(256)
            fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
                let tid = lid.x;
                let hidden = {hidden_dim}u;
                var sum_sq: f32 = 0.0;
                for (var i = tid; i < hidden; i += 256u) {{
                    sum_sq += input[i] * input[i];
                }}
                wg_data[tid] = sum_sq;
                workgroupBarrier();
                for (var s = 128u; s > 0u; s >>= 1u) {{
                    if (tid < s) {{ wg_data[tid] += wg_data[tid + s]; }}
                    workgroupBarrier();
                }}
                let rsqrt = 1.0 / sqrt(wg_data[0] / f32(hidden) + 1e-6);
                for (var i = tid; i < hidden; i += 256u) {{
                    output[i] = input[i] * rsqrt;
                }}
            }}
        """,
        "matmul_gate": f"""
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
        """,
        "matmul_up": f"""
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
        """,
        "silu": f"""
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let idx = gid.x;
                if (idx < {intermediate_dim}u) {{
                    let x = input[idx];
                    output[idx] = x / (1.0 + exp(-x));
                }}
            }}
        """,
        "mul": f"""
            @group(0) @binding(0) var<storage, read> a: array<f32>;
            @group(0) @binding(1) var<storage, read> b: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let idx = gid.x;
                if (idx < {intermediate_dim}u) {{
                    output[idx] = a[idx] * b[idx];
                }}
            }}
        """,
        "matmul_down": f"""
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read> weight: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let idx = gid.x;
                let hidden = {hidden_dim}u;
                let inter = {intermediate_dim}u;
                if (idx < hidden) {{
                    var sum: f32 = 0.0;
                    for (var j = 0u; j < inter; j++) {{
                        sum += input[j] * weight[idx * inter + j];
                    }}
                    output[idx] = sum;
                }}
            }}
        """,
        "add": f"""
            @group(0) @binding(0) var<storage, read> a: array<f32>;
            @group(0) @binding(1) var<storage, read> b: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
                let idx = gid.x;
                if (idx < {hidden_dim}u) {{
                    output[idx] = a[idx] + b[idx];
                }}
            }}
        """
    }

    # Create pipelines
    pipelines = {}
    for name, code in shaders.items():
        module = device.create_shader_module(code=code)
        pipelines[name] = device.create_compute_pipeline(
            layout="auto",
            compute={"module": module, "entry_point": "main"}
        )

    # Create buffers
    input_buf = device.create_buffer(size=hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    norm_buf = device.create_buffer(size=hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE)
    gate_w_buf = device.create_buffer(size=hidden_dim * intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    up_w_buf = device.create_buffer(size=hidden_dim * intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    down_w_buf = device.create_buffer(size=intermediate_dim * hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    gate_buf = device.create_buffer(size=intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE)
    up_buf = device.create_buffer(size=intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE)
    silu_buf = device.create_buffer(size=intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE)
    mul_buf = device.create_buffer(size=intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE)
    down_buf = device.create_buffer(size=hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE)
    output_buf = device.create_buffer(size=hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    # Initialize (random data)
    queue.write_buffer(input_buf, 0, np.random.randn(hidden_dim).astype(np.float32).tobytes())
    queue.write_buffer(gate_w_buf, 0, np.random.randn(intermediate_dim, hidden_dim).astype(np.float32).tobytes())
    queue.write_buffer(up_w_buf, 0, np.random.randn(intermediate_dim, hidden_dim).astype(np.float32).tobytes())
    queue.write_buffer(down_w_buf, 0, np.random.randn(hidden_dim, intermediate_dim).astype(np.float32).tobytes())

    # Create bind groups
    bg_rmsnorm = device.create_bind_group(layout=pipelines["rmsnorm"].get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": input_buf}}, {"binding": 1, "resource": {"buffer": norm_buf}}])
    bg_gate = device.create_bind_group(layout=pipelines["matmul_gate"].get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": norm_buf}}, {"binding": 1, "resource": {"buffer": gate_w_buf}}, {"binding": 2, "resource": {"buffer": gate_buf}}])
    bg_up = device.create_bind_group(layout=pipelines["matmul_up"].get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": norm_buf}}, {"binding": 1, "resource": {"buffer": up_w_buf}}, {"binding": 2, "resource": {"buffer": up_buf}}])
    bg_silu = device.create_bind_group(layout=pipelines["silu"].get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": gate_buf}}, {"binding": 1, "resource": {"buffer": silu_buf}}])
    bg_mul = device.create_bind_group(layout=pipelines["mul"].get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": silu_buf}}, {"binding": 1, "resource": {"buffer": up_buf}}, {"binding": 2, "resource": {"buffer": mul_buf}}])
    bg_down = device.create_bind_group(layout=pipelines["matmul_down"].get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": mul_buf}}, {"binding": 1, "resource": {"buffer": down_w_buf}}, {"binding": 2, "resource": {"buffer": down_buf}}])
    bg_add = device.create_bind_group(layout=pipelines["add"].get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": input_buf}}, {"binding": 1, "resource": {"buffer": down_buf}}, {"binding": 2, "resource": {"buffer": output_buf}}])

    wg_inter = (intermediate_dim + 63) // 64
    wg_hidden = (hidden_dim + 63) // 64

    def run_unfused():
        encoder = device.create_command_encoder()

        p = encoder.begin_compute_pass()
        p.set_pipeline(pipelines["rmsnorm"])
        p.set_bind_group(0, bg_rmsnorm)
        p.dispatch_workgroups(1)
        p.end()

        p = encoder.begin_compute_pass()
        p.set_pipeline(pipelines["matmul_gate"])
        p.set_bind_group(0, bg_gate)
        p.dispatch_workgroups(wg_inter)
        p.end()

        p = encoder.begin_compute_pass()
        p.set_pipeline(pipelines["matmul_up"])
        p.set_bind_group(0, bg_up)
        p.dispatch_workgroups(wg_inter)
        p.end()

        p = encoder.begin_compute_pass()
        p.set_pipeline(pipelines["silu"])
        p.set_bind_group(0, bg_silu)
        p.dispatch_workgroups(wg_inter)
        p.end()

        p = encoder.begin_compute_pass()
        p.set_pipeline(pipelines["mul"])
        p.set_bind_group(0, bg_mul)
        p.dispatch_workgroups(wg_inter)
        p.end()

        p = encoder.begin_compute_pass()
        p.set_pipeline(pipelines["matmul_down"])
        p.set_bind_group(0, bg_down)
        p.dispatch_workgroups(wg_hidden)
        p.end()

        p = encoder.begin_compute_pass()
        p.set_pipeline(pipelines["add"])
        p.set_bind_group(0, bg_add)
        p.dispatch_workgroups(wg_hidden)
        p.end()

        queue.submit([encoder.finish()])

    # Warmup
    for _ in range(5):
        run_unfused()
    queue.on_submitted_work_done_sync()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        run_unfused()
        queue.on_submitted_work_done_sync()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        "name": "unfused_mlp",
        "dispatches": 7,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "n_iterations": n_iterations
    }


def benchmark_tiled_mlp(device, queue, hidden_dim=896, intermediate_dim=4864, n_iterations=100):
    """
    Approach B: Tiled MLP (3 dispatches with many workgroups each)
    Pass 1: RMSNorm
    Pass 2: Gate+Up projections + SiLU + Multiply (fused, many workgroups)
    Pass 3: Down projection + Residual add
    """

    # Pass 1: RMSNorm (same as before)
    rmsnorm_shader = f"""
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        var<workgroup> wg_data: array<f32, 256>;

        @compute @workgroup_size(256)
        fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
            let tid = lid.x;
            let hidden = {hidden_dim}u;
            var sum_sq: f32 = 0.0;
            for (var i = tid; i < hidden; i += 256u) {{
                sum_sq += input[i] * input[i];
            }}
            wg_data[tid] = sum_sq;
            workgroupBarrier();
            for (var s = 128u; s > 0u; s >>= 1u) {{
                if (tid < s) {{ wg_data[tid] += wg_data[tid + s]; }}
                workgroupBarrier();
            }}
            let rsqrt = 1.0 / sqrt(wg_data[0] / f32(hidden) + 1e-6);
            for (var i = tid; i < hidden; i += 256u) {{
                output[i] = input[i] * rsqrt;
            }}
        }}
    """

    # Pass 2: Fused Gate+Up+SiLU+Mul (many workgroups)
    fused_gate_up_shader = f"""
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read> gate_w: array<f32>;
        @group(0) @binding(2) var<storage, read> up_w: array<f32>;
        @group(0) @binding(3) var<storage, read_write> output: array<f32>;

        fn silu(x: f32) -> f32 {{
            return x / (1.0 + exp(-x));
        }}

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
            let idx = gid.x;
            let hidden = {hidden_dim}u;
            let inter = {intermediate_dim}u;

            if (idx < inter) {{
                // Compute gate projection
                var gate_sum: f32 = 0.0;
                for (var j = 0u; j < hidden; j++) {{
                    gate_sum += input[j] * gate_w[idx * hidden + j];
                }}

                // Compute up projection
                var up_sum: f32 = 0.0;
                for (var j = 0u; j < hidden; j++) {{
                    up_sum += input[j] * up_w[idx * hidden + j];
                }}

                // SiLU(gate) * up
                output[idx] = silu(gate_sum) * up_sum;
            }}
        }}
    """

    # Pass 3: Down projection + residual (many workgroups)
    down_residual_shader = f"""
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read> down_w: array<f32>;
        @group(0) @binding(2) var<storage, read> residual: array<f32>;
        @group(0) @binding(3) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
            let idx = gid.x;
            let hidden = {hidden_dim}u;
            let inter = {intermediate_dim}u;

            if (idx < hidden) {{
                var sum: f32 = 0.0;
                for (var j = 0u; j < inter; j++) {{
                    sum += input[j] * down_w[idx * inter + j];
                }}
                output[idx] = sum + residual[idx];
            }}
        }}
    """

    # Create pipelines
    rmsnorm_pipeline = device.create_compute_pipeline(layout="auto",
        compute={"module": device.create_shader_module(code=rmsnorm_shader), "entry_point": "main"})
    fused_pipeline = device.create_compute_pipeline(layout="auto",
        compute={"module": device.create_shader_module(code=fused_gate_up_shader), "entry_point": "main"})
    down_pipeline = device.create_compute_pipeline(layout="auto",
        compute={"module": device.create_shader_module(code=down_residual_shader), "entry_point": "main"})

    # Create buffers
    input_buf = device.create_buffer(size=hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    norm_buf = device.create_buffer(size=hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE)
    gate_w_buf = device.create_buffer(size=hidden_dim * intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    up_w_buf = device.create_buffer(size=hidden_dim * intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    down_w_buf = device.create_buffer(size=intermediate_dim * hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    inter_buf = device.create_buffer(size=intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE)
    output_buf = device.create_buffer(size=hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    # Initialize
    queue.write_buffer(input_buf, 0, np.random.randn(hidden_dim).astype(np.float32).tobytes())
    queue.write_buffer(gate_w_buf, 0, np.random.randn(intermediate_dim, hidden_dim).astype(np.float32).tobytes())
    queue.write_buffer(up_w_buf, 0, np.random.randn(intermediate_dim, hidden_dim).astype(np.float32).tobytes())
    queue.write_buffer(down_w_buf, 0, np.random.randn(hidden_dim, intermediate_dim).astype(np.float32).tobytes())

    # Create bind groups
    bg_rmsnorm = device.create_bind_group(layout=rmsnorm_pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": input_buf}}, {"binding": 1, "resource": {"buffer": norm_buf}}])
    bg_fused = device.create_bind_group(layout=fused_pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": norm_buf}}, {"binding": 1, "resource": {"buffer": gate_w_buf}},
                 {"binding": 2, "resource": {"buffer": up_w_buf}}, {"binding": 3, "resource": {"buffer": inter_buf}}])
    bg_down = device.create_bind_group(layout=down_pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": inter_buf}}, {"binding": 1, "resource": {"buffer": down_w_buf}},
                 {"binding": 2, "resource": {"buffer": input_buf}}, {"binding": 3, "resource": {"buffer": output_buf}}])

    wg_inter = (intermediate_dim + 63) // 64
    wg_hidden = (hidden_dim + 63) // 64

    def run_tiled():
        encoder = device.create_command_encoder()

        # Pass 1: RMSNorm
        p = encoder.begin_compute_pass()
        p.set_pipeline(rmsnorm_pipeline)
        p.set_bind_group(0, bg_rmsnorm)
        p.dispatch_workgroups(1)
        p.end()

        # Pass 2: Fused gate+up+silu+mul
        p = encoder.begin_compute_pass()
        p.set_pipeline(fused_pipeline)
        p.set_bind_group(0, bg_fused)
        p.dispatch_workgroups(wg_inter)
        p.end()

        # Pass 3: Down + residual
        p = encoder.begin_compute_pass()
        p.set_pipeline(down_pipeline)
        p.set_bind_group(0, bg_down)
        p.dispatch_workgroups(wg_hidden)
        p.end()

        queue.submit([encoder.finish()])

    # Warmup
    for _ in range(5):
        run_tiled()
    queue.on_submitted_work_done_sync()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        run_tiled()
        queue.on_submitted_work_done_sync()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        "name": "tiled_mlp",
        "dispatches": 3,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "n_iterations": n_iterations
    }


def benchmark_mega_mlp(device, queue, hidden_dim=896, intermediate_dim=256, n_iterations=100):
    """
    Approach C: Single-workgroup mega-kernel (known to be slow)
    Note: We use smaller intermediate_dim to fit in workgroup memory
    """

    shader_code = f"""
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read> gate_w: array<f32>;
        @group(0) @binding(2) var<storage, read> up_w: array<f32>;
        @group(0) @binding(3) var<storage, read> down_w: array<f32>;
        @group(0) @binding(4) var<storage, read_write> output: array<f32>;

        var<workgroup> wg_sum: array<f32, 256>;
        var<workgroup> intermediate: array<f32, {intermediate_dim}>;

        fn silu(x: f32) -> f32 {{
            return x / (1.0 + exp(-x));
        }}

        @compute @workgroup_size(256)
        fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
            let tid = lid.x;
            let hidden = {hidden_dim}u;
            let inter = {intermediate_dim}u;

            // Step 1: RMSNorm
            var sum_sq: f32 = 0.0;
            for (var i = tid; i < hidden; i += 256u) {{
                sum_sq += input[i] * input[i];
            }}
            wg_sum[tid] = sum_sq;
            workgroupBarrier();
            for (var s = 128u; s > 0u; s >>= 1u) {{
                if (tid < s) {{ wg_sum[tid] += wg_sum[tid + s]; }}
                workgroupBarrier();
            }}
            let rsqrt_val = 1.0 / sqrt(wg_sum[0] / f32(hidden) + 1e-6);
            workgroupBarrier();

            // Step 2: Gate+Up+SiLU+Mul (serial for elements > 256)
            if (tid < inter) {{
                var gate_sum: f32 = 0.0;
                var up_sum: f32 = 0.0;
                for (var j = 0u; j < hidden; j++) {{
                    let normed = input[j] * rsqrt_val;
                    gate_sum += normed * gate_w[tid * hidden + j];
                    up_sum += normed * up_w[tid * hidden + j];
                }}
                intermediate[tid] = silu(gate_sum) * up_sum;
            }}
            workgroupBarrier();

            // Step 3: Down projection + residual
            for (var i = tid; i < hidden; i += 256u) {{
                var sum: f32 = 0.0;
                for (var j = 0u; j < inter; j++) {{
                    sum += intermediate[j] * down_w[i * inter + j];
                }}
                output[i] = sum + input[i];
            }}
        }}
    """

    pipeline = device.create_compute_pipeline(layout="auto",
        compute={"module": device.create_shader_module(code=shader_code), "entry_point": "main"})

    input_buf = device.create_buffer(size=hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    gate_w_buf = device.create_buffer(size=hidden_dim * intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    up_w_buf = device.create_buffer(size=hidden_dim * intermediate_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    down_w_buf = device.create_buffer(size=intermediate_dim * hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    output_buf = device.create_buffer(size=hidden_dim * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    queue.write_buffer(input_buf, 0, np.random.randn(hidden_dim).astype(np.float32).tobytes())
    queue.write_buffer(gate_w_buf, 0, np.random.randn(intermediate_dim, hidden_dim).astype(np.float32).tobytes())
    queue.write_buffer(up_w_buf, 0, np.random.randn(intermediate_dim, hidden_dim).astype(np.float32).tobytes())
    queue.write_buffer(down_w_buf, 0, np.random.randn(hidden_dim, intermediate_dim).astype(np.float32).tobytes())

    bg = device.create_bind_group(layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": input_buf}}, {"binding": 1, "resource": {"buffer": gate_w_buf}},
                 {"binding": 2, "resource": {"buffer": up_w_buf}}, {"binding": 3, "resource": {"buffer": down_w_buf}},
                 {"binding": 4, "resource": {"buffer": output_buf}}])

    def run_mega():
        encoder = device.create_command_encoder()
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, bg)
        p.dispatch_workgroups(1)
        p.end()
        queue.submit([encoder.finish()])

    # Warmup
    for _ in range(5):
        run_mega()
    queue.on_submitted_work_done_sync()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        run_mega()
        queue.on_submitted_work_done_sync()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        "name": "mega_kernel_mlp",
        "dispatches": 1,
        "intermediate_dim": intermediate_dim,
        "note": "Reduced intermediate_dim to fit in workgroup memory",
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "n_iterations": n_iterations
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-dispatch tiled mega strategy experiment")
    parser.add_argument("--output", type=str, default="results/exp3_tiled_mega.json")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not WGPU_AVAILABLE:
        print("ERROR: wgpu required")
        return

    print("=" * 60)
    print("Experiment 3: Multi-Dispatch Tiled Mega Strategy")
    print("=" * 60)

    system_info = get_system_info()
    print(f"GPU: {system_info.get('gpu', 'unknown')}")

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    queue = device.queue

    results = {
        "system_info": system_info,
        "hidden_dim": 896,
        "intermediate_dim": 4864,
        "experiments": {}
    }

    print("\n1. Benchmarking unfused MLP (7 dispatches)...")
    unfused = benchmark_unfused_mlp(device, queue, n_iterations=args.iterations)
    results["experiments"]["unfused"] = unfused
    print(f"   Time: {unfused['mean_ms']:.3f} ± {unfused['std_ms']:.3f} ms")

    print("\n2. Benchmarking tiled MLP (3 dispatches)...")
    tiled = benchmark_tiled_mlp(device, queue, n_iterations=args.iterations)
    results["experiments"]["tiled"] = tiled
    print(f"   Time: {tiled['mean_ms']:.3f} ± {tiled['std_ms']:.3f} ms")

    print("\n3. Benchmarking mega-kernel MLP (1 dispatch, 256 threads)...")
    mega = benchmark_mega_mlp(device, queue, n_iterations=args.iterations)
    results["experiments"]["mega_kernel"] = mega
    print(f"   Time: {mega['mean_ms']:.3f} ± {mega['std_ms']:.3f} ms")
    print(f"   Note: {mega['note']}")

    # Analysis
    tiled_vs_unfused = unfused['mean_ms'] / tiled['mean_ms']
    mega_vs_tiled = mega['mean_ms'] / tiled['mean_ms']

    results["analysis"] = {
        "tiled_speedup_vs_unfused": tiled_vs_unfused,
        "mega_slowdown_vs_tiled": mega_vs_tiled
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Unfused (7 dispatches):  {unfused['mean_ms']:.3f} ms")
    print(f"Tiled (3 dispatches):    {tiled['mean_ms']:.3f} ms  ({tiled_vs_unfused:.2f}x faster than unfused)")
    print(f"Mega-kernel (1 dispatch):{mega['mean_ms']:.3f} ms  ({mega_vs_tiled:.1f}x slower than tiled)")
    print("=" * 60)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
