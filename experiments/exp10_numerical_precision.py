#!/usr/bin/env python3
"""
Experiment 10: Numerical Precision Validation

Verifies that WebGPU inference produces outputs consistent with CPU/CUDA baselines.
Reports any numerical precision differences.

Usage:
    python exp10_numerical_precision.py --output results/exp10_precision.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available")

try:
    import wgpu
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False


def compare_rmsnorm_precision():
    """Compare RMSNorm implementation between CPU and WebGPU."""
    if not WGPU_AVAILABLE:
        return None

    # Setup wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    queue = device.queue

    hidden_dim = 896

    # WebGPU fused RMSNorm shader
    shader_code = f"""
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;

    var<workgroup> wg_data: array<f32, 256>;

    @compute @workgroup_size(256)
    fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
        let idx = lid.x;
        let hidden_dim = {hidden_dim}u;

        var sum_sq: f32 = 0.0;
        for (var i = idx; i < hidden_dim; i += 256u) {{
            let val = input[i];
            sum_sq += val * val;
        }}

        wg_data[idx] = sum_sq;
        workgroupBarrier();

        for (var s = 128u; s > 0u; s = s >> 1u) {{
            if (idx < s) {{ wg_data[idx] += wg_data[idx + s]; }}
            workgroupBarrier();
        }}

        let mean = wg_data[0] / f32(hidden_dim);
        let rsqrt_val = 1.0 / sqrt(mean + 1e-6);

        for (var i = idx; i < hidden_dim; i += 256u) {{
            output[i] = input[i] * rsqrt_val * weight[i];
        }}
    }}
    """

    module = device.create_shader_module(code=shader_code)
    pipeline = device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})

    # Create test data
    np.random.seed(42)
    input_data = np.random.randn(hidden_dim).astype(np.float32)
    weight_data = np.random.randn(hidden_dim).astype(np.float32)

    # CPU reference implementation
    variance = np.mean(input_data ** 2)
    rsqrt = 1.0 / np.sqrt(variance + 1e-6)
    cpu_output = input_data * rsqrt * weight_data

    # WebGPU implementation
    buffer_size = hidden_dim * 4
    input_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    weight_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    output_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    staging_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)

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

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(1)
    compute_pass.end()
    encoder.copy_buffer_to_buffer(output_buf, 0, staging_buf, 0, buffer_size)
    queue.submit([encoder.finish()])
    queue.on_submitted_work_done_sync()

    # Read back WebGPU output via staging buffer
    staging_buf.map_sync(mode=wgpu.MapMode.READ)
    webgpu_output = np.frombuffer(staging_buf.read_mapped(), dtype=np.float32).copy()
    staging_buf.unmap()

    # Compare
    abs_diff = np.abs(cpu_output - webgpu_output)
    rel_diff = abs_diff / (np.abs(cpu_output) + 1e-10)

    return {
        "operation": "RMSNorm",
        "hidden_dim": hidden_dim,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
        "cpu_output_sample": cpu_output[:5].tolist(),
        "webgpu_output_sample": webgpu_output[:5].tolist(),
        "match_within_1e-5": bool(np.allclose(cpu_output, webgpu_output, rtol=1e-5, atol=1e-5)),
        "match_within_1e-4": bool(np.allclose(cpu_output, webgpu_output, rtol=1e-4, atol=1e-4)),
        "match_within_1e-3": bool(np.allclose(cpu_output, webgpu_output, rtol=1e-3, atol=1e-3)),
    }


def compare_matmul_precision():
    """Compare matrix multiplication precision between CPU and WebGPU."""
    if not WGPU_AVAILABLE:
        return None

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    queue = device.queue

    M, K, N = 256, 896, 256

    matmul_shader = f"""
    @group(0) @binding(0) var<storage, read> A: array<f32>;
    @group(0) @binding(1) var<storage, read> B: array<f32>;
    @group(0) @binding(2) var<storage, read_write> C: array<f32>;

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
        let row = gid.y;
        let col = gid.x;
        let M = {M}u;
        let K = {K}u;
        let N = {N}u;

        if (row < M && col < N) {{
            var sum: f32 = 0.0;
            for (var k = 0u; k < K; k++) {{
                sum += A[row * K + k] * B[k * N + col];
            }}
            C[row * N + col] = sum;
        }}
    }}
    """

    module = device.create_shader_module(code=matmul_shader)
    pipeline = device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})

    # Create test data
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    # CPU reference
    cpu_output = A @ B

    # WebGPU
    A_buf = device.create_buffer(size=A.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    B_buf = device.create_buffer(size=B.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    C_buf = device.create_buffer(size=cpu_output.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    staging_buf = device.create_buffer(size=cpu_output.nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)

    queue.write_buffer(A_buf, 0, A.tobytes())
    queue.write_buffer(B_buf, 0, B.tobytes())

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": A_buf}},
            {"binding": 1, "resource": {"buffer": B_buf}},
            {"binding": 2, "resource": {"buffer": C_buf}}
        ]
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((N + 15) // 16, (M + 15) // 16)
    compute_pass.end()
    encoder.copy_buffer_to_buffer(C_buf, 0, staging_buf, 0, cpu_output.nbytes)
    queue.submit([encoder.finish()])
    queue.on_submitted_work_done_sync()

    staging_buf.map_sync(mode=wgpu.MapMode.READ)
    webgpu_output = np.frombuffer(staging_buf.read_mapped(), dtype=np.float32).copy().reshape(M, N)
    staging_buf.unmap()

    # Compare
    abs_diff = np.abs(cpu_output - webgpu_output)
    rel_diff = abs_diff / (np.abs(cpu_output) + 1e-10)

    return {
        "operation": "MatMul",
        "dimensions": f"{M}x{K} @ {K}x{N}",
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
        "cpu_output_sample": cpu_output[0, :5].tolist(),
        "webgpu_output_sample": webgpu_output[0, :5].tolist(),
        "match_within_1e-5": bool(np.allclose(cpu_output, webgpu_output, rtol=1e-5, atol=1e-5)),
        "match_within_1e-4": bool(np.allclose(cpu_output, webgpu_output, rtol=1e-4, atol=1e-4)),
        "match_within_1e-3": bool(np.allclose(cpu_output, webgpu_output, rtol=1e-3, atol=1e-3)),
    }


def compare_softmax_precision():
    """Compare softmax precision between CPU and WebGPU."""
    if not WGPU_AVAILABLE:
        return None

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    queue = device.queue

    vocab_size = 1024

    softmax_shader = f"""
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    var<workgroup> wg_max: f32;
    var<workgroup> wg_sum: f32;
    var<workgroup> wg_shared: array<f32, 256>;

    @compute @workgroup_size(256)
    fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
        let idx = lid.x;
        let vocab_size = {vocab_size}u;

        // Find max (for numerical stability)
        var local_max: f32 = -3.4028235e+38;
        for (var i = idx; i < vocab_size; i += 256u) {{
            local_max = max(local_max, input[i]);
        }}
        wg_shared[idx] = local_max;
        workgroupBarrier();

        for (var s = 128u; s > 0u; s = s >> 1u) {{
            if (idx < s) {{
                wg_shared[idx] = max(wg_shared[idx], wg_shared[idx + s]);
            }}
            workgroupBarrier();
        }}

        if (idx == 0u) {{ wg_max = wg_shared[0]; }}
        workgroupBarrier();

        // Compute exp and sum
        var local_sum: f32 = 0.0;
        for (var i = idx; i < vocab_size; i += 256u) {{
            local_sum += exp(input[i] - wg_max);
        }}
        wg_shared[idx] = local_sum;
        workgroupBarrier();

        for (var s = 128u; s > 0u; s = s >> 1u) {{
            if (idx < s) {{
                wg_shared[idx] += wg_shared[idx + s];
            }}
            workgroupBarrier();
        }}

        if (idx == 0u) {{ wg_sum = wg_shared[0]; }}
        workgroupBarrier();

        // Normalize
        for (var i = idx; i < vocab_size; i += 256u) {{
            output[i] = exp(input[i] - wg_max) / wg_sum;
        }}
    }}
    """

    module = device.create_shader_module(code=softmax_shader)
    pipeline = device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})

    # Create test data
    np.random.seed(42)
    input_data = np.random.randn(vocab_size).astype(np.float32) * 10  # Scale up for numerical challenge

    # CPU reference (stable softmax)
    shifted = input_data - np.max(input_data)
    exp_shifted = np.exp(shifted)
    cpu_output = exp_shifted / np.sum(exp_shifted)

    # WebGPU
    buffer_size = vocab_size * 4
    input_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    output_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    staging_buf = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)

    queue.write_buffer(input_buf, 0, input_data.tobytes())

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": input_buf}},
            {"binding": 1, "resource": {"buffer": output_buf}}
        ]
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(1)
    compute_pass.end()
    encoder.copy_buffer_to_buffer(output_buf, 0, staging_buf, 0, buffer_size)
    queue.submit([encoder.finish()])
    queue.on_submitted_work_done_sync()

    staging_buf.map_sync(mode=wgpu.MapMode.READ)
    webgpu_output = np.frombuffer(staging_buf.read_mapped(), dtype=np.float32).copy()
    staging_buf.unmap()

    # Compare
    abs_diff = np.abs(cpu_output - webgpu_output)
    rel_diff = abs_diff / (np.abs(cpu_output) + 1e-10)

    # Check probability sum
    cpu_sum = np.sum(cpu_output)
    webgpu_sum = np.sum(webgpu_output)

    return {
        "operation": "Softmax",
        "vocab_size": vocab_size,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
        "cpu_probability_sum": float(cpu_sum),
        "webgpu_probability_sum": float(webgpu_sum),
        "cpu_output_sample": cpu_output[:5].tolist(),
        "webgpu_output_sample": webgpu_output[:5].tolist(),
        "match_within_1e-5": bool(np.allclose(cpu_output, webgpu_output, rtol=1e-5, atol=1e-5)),
        "match_within_1e-4": bool(np.allclose(cpu_output, webgpu_output, rtol=1e-4, atol=1e-4)),
        "match_within_1e-3": bool(np.allclose(cpu_output, webgpu_output, rtol=1e-3, atol=1e-3)),
    }


def main():
    parser = argparse.ArgumentParser(description="Numerical precision validation")
    parser.add_argument("--output", type=str, default="results/exp10_precision.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 10: Numerical Precision Validation")
    print("=" * 60)

    results = {"experiments": {}}

    # RMSNorm
    print("\n1. RMSNorm precision...")
    rmsnorm_result = compare_rmsnorm_precision()
    if rmsnorm_result:
        results["experiments"]["rmsnorm"] = rmsnorm_result
        print(f"   Max absolute diff: {rmsnorm_result['max_abs_diff']:.2e}")
        print(f"   Match within 1e-3: {rmsnorm_result['match_within_1e-3']}")

    # MatMul
    print("\n2. MatMul precision...")
    matmul_result = compare_matmul_precision()
    if matmul_result:
        results["experiments"]["matmul"] = matmul_result
        print(f"   Max absolute diff: {matmul_result['max_abs_diff']:.2e}")
        print(f"   Match within 1e-3: {matmul_result['match_within_1e-3']}")

    # Softmax
    print("\n3. Softmax precision...")
    softmax_result = compare_softmax_precision()
    if softmax_result:
        results["experiments"]["softmax"] = softmax_result
        print(f"   Max absolute diff: {softmax_result['max_abs_diff']:.2e}")
        print(f"   CPU prob sum: {softmax_result['cpu_probability_sum']:.6f}")
        print(f"   WebGPU prob sum: {softmax_result['webgpu_probability_sum']:.6f}")
        print(f"   Match within 1e-3: {softmax_result['match_within_1e-3']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_match = True
    for name, exp in results["experiments"].items():
        match = exp.get('match_within_1e-3', False)
        all_match = all_match and match
        status = "PASS" if match else "FAIL"
        print(f"{name}: {status} (max diff: {exp.get('max_abs_diff', 'N/A'):.2e})")

    print(f"\nOverall: {'ALL PASS' if all_match else 'SOME FAILURES'}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
