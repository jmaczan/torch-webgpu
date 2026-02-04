#!/usr/bin/env python3
"""
Experiment 5: CUDA Fusion Comparison

This experiment compares WebGPU dispatch overhead with CUDA kernel launch overhead,
and tests whether CUDA benefits from manual kernel fusion (it shouldn't, significantly).

Usage:
    python exp5_cuda_fusion.py --output results/exp5_cuda_fusion.json

This measures:
1. CUDA kernel launch overhead (minimum achievable)
2. CUDA unfused vs fused RMSNorm (to show fusion benefit is minimal in CUDA)
3. Comparison with WebGPU per-dispatch overhead
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import wgpu
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False


def get_system_info():
    import platform
    info = {"platform": platform.platform()}

    if CUDA_AVAILABLE:
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["torch_version"] = torch.__version__

    if WGPU_AVAILABLE:
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            if adapter:
                info["webgpu_device"] = adapter.info.get("description", "unknown")
        except:
            pass

    return info


def benchmark_cuda_launch_overhead(n_iterations=10000):
    """Measure minimum CUDA kernel launch overhead."""
    if not CUDA_AVAILABLE:
        return {"error": "CUDA not available"}

    # Create a minimal kernel using torch
    x = torch.zeros(1, device='cuda')

    # Warmup
    for _ in range(100):
        y = x + 1
    torch.cuda.synchronize()

    # Measure individual kernel launches
    launch_times = []
    for _ in range(n_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        y = x + 1  # Minimal kernel
        torch.cuda.synchronize()
        end = time.perf_counter()
        launch_times.append((end - start) * 1e6)  # microseconds

    return {
        "name": "cuda_launch_overhead",
        "mean_us": np.mean(launch_times),
        "std_us": np.std(launch_times),
        "min_us": np.min(launch_times),
        "max_us": np.max(launch_times),
        "n_iterations": n_iterations
    }


def benchmark_cuda_unfused_rmsnorm(hidden_dim=896, n_iterations=1000):
    """CUDA unfused RMSNorm: separate operations."""
    if not CUDA_AVAILABLE:
        return {"error": "CUDA not available"}

    x = torch.randn(1, hidden_dim, device='cuda')
    weight = torch.ones(hidden_dim, device='cuda')
    eps = 1e-6

    # Warmup
    for _ in range(10):
        squared = x.pow(2)
        mean = squared.mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(mean + eps)
        normalized = x * rsqrt
        output = normalized * weight
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        squared = x.pow(2)
        mean = squared.mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(mean + eps)
        normalized = x * rsqrt
        output = normalized * weight

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)

    return {
        "name": "cuda_unfused_rmsnorm",
        "operations": 5,
        "mean_us": np.mean(times),
        "std_us": np.std(times),
        "n_iterations": n_iterations
    }


def benchmark_cuda_fused_rmsnorm(hidden_dim=896, n_iterations=1000):
    """CUDA fused RMSNorm using torch operations (PyTorch may internally fuse)."""
    if not CUDA_AVAILABLE:
        return {"error": "CUDA not available"}

    x = torch.randn(1, hidden_dim, device='cuda')
    weight = torch.ones(hidden_dim, device='cuda')
    eps = 1e-6

    def fused_rmsnorm(x, weight, eps):
        # This formulation may be optimized by PyTorch
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight

    # Warmup
    for _ in range(10):
        output = fused_rmsnorm(x, weight, eps)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = fused_rmsnorm(x, weight, eps)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)

    return {
        "name": "cuda_fused_rmsnorm",
        "mean_us": np.mean(times),
        "std_us": np.std(times),
        "n_iterations": n_iterations
    }


def benchmark_cuda_compiled_rmsnorm(hidden_dim=896, n_iterations=1000):
    """CUDA RMSNorm with torch.compile."""
    if not CUDA_AVAILABLE:
        return {"error": "CUDA not available"}

    x = torch.randn(1, hidden_dim, device='cuda')
    weight = torch.ones(hidden_dim, device='cuda')
    eps = 1e-6

    @torch.compile
    def compiled_rmsnorm(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight

    # Warmup (includes compilation)
    for _ in range(10):
        output = compiled_rmsnorm(x, weight, eps)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = compiled_rmsnorm(x, weight, eps)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)

    return {
        "name": "cuda_compiled_rmsnorm",
        "mean_us": np.mean(times),
        "std_us": np.std(times),
        "n_iterations": n_iterations
    }


def benchmark_webgpu_dispatch_overhead(n_iterations=1000):
    """Measure WebGPU per-dispatch overhead for comparison."""
    if not WGPU_AVAILABLE:
        return {"error": "wgpu not available"}

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    queue = device.queue

    shader_code = """
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        data[gid.x] = data[gid.x] + 1.0;
    }
    """

    module = device.create_shader_module(code=shader_code)
    pipeline = device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})
    buffer = device.create_buffer(size=256*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    bind_group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": buffer}}])

    # Warmup
    for _ in range(10):
        encoder = device.create_command_encoder()
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, bind_group)
        p.dispatch_workgroups(4)
        p.end()
        queue.submit([encoder.finish()])
    queue.on_submitted_work_done_sync()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()

        encoder = device.create_command_encoder()
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, bind_group)
        p.dispatch_workgroups(4)
        p.end()
        queue.submit([encoder.finish()])

        end = time.perf_counter()
        times.append((end - start) * 1e6)

    queue.on_submitted_work_done_sync()

    return {
        "name": "webgpu_dispatch_overhead",
        "mean_us": np.mean(times),
        "std_us": np.std(times),
        "min_us": np.min(times),
        "max_us": np.max(times),
        "n_iterations": n_iterations
    }


def main():
    parser = argparse.ArgumentParser(description="CUDA fusion comparison experiment")
    parser.add_argument("--output", type=str, default="results/exp5_cuda_fusion.json")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 5: CUDA Fusion Comparison")
    print("=" * 60)

    system_info = get_system_info()
    print(f"CUDA: {system_info.get('cuda_device', 'not available')}")
    print(f"WebGPU: {system_info.get('webgpu_device', 'not available')}")

    results = {
        "system_info": system_info,
        "experiments": {}
    }

    # 1. CUDA launch overhead
    print("\n1. Measuring CUDA kernel launch overhead...")
    cuda_launch = benchmark_cuda_launch_overhead(n_iterations=10000)
    results["experiments"]["cuda_launch"] = cuda_launch
    if "error" not in cuda_launch:
        print(f"   CUDA launch overhead: {cuda_launch['mean_us']:.1f} ± {cuda_launch['std_us']:.1f} µs")
    else:
        print(f"   {cuda_launch['error']}")

    # 2. WebGPU dispatch overhead
    print("\n2. Measuring WebGPU dispatch overhead...")
    webgpu_dispatch = benchmark_webgpu_dispatch_overhead(n_iterations=1000)
    results["experiments"]["webgpu_dispatch"] = webgpu_dispatch
    if "error" not in webgpu_dispatch:
        print(f"   WebGPU dispatch overhead: {webgpu_dispatch['mean_us']:.1f} ± {webgpu_dispatch['std_us']:.1f} µs")
    else:
        print(f"   {webgpu_dispatch['error']}")

    # 3. CUDA unfused RMSNorm
    print("\n3. Benchmarking CUDA unfused RMSNorm...")
    cuda_unfused = benchmark_cuda_unfused_rmsnorm(n_iterations=args.iterations)
    results["experiments"]["cuda_unfused_rmsnorm"] = cuda_unfused
    if "error" not in cuda_unfused:
        print(f"   Time: {cuda_unfused['mean_us']:.1f} ± {cuda_unfused['std_us']:.1f} µs")
    else:
        print(f"   {cuda_unfused['error']}")

    # 4. CUDA fused RMSNorm
    print("\n4. Benchmarking CUDA fused RMSNorm...")
    cuda_fused = benchmark_cuda_fused_rmsnorm(n_iterations=args.iterations)
    results["experiments"]["cuda_fused_rmsnorm"] = cuda_fused
    if "error" not in cuda_fused:
        print(f"   Time: {cuda_fused['mean_us']:.1f} ± {cuda_fused['std_us']:.1f} µs")
    else:
        print(f"   {cuda_fused['error']}")

    # 5. CUDA compiled RMSNorm
    print("\n5. Benchmarking CUDA compiled RMSNorm...")
    cuda_compiled = benchmark_cuda_compiled_rmsnorm(n_iterations=args.iterations)
    results["experiments"]["cuda_compiled_rmsnorm"] = cuda_compiled
    if "error" not in cuda_compiled:
        print(f"   Time: {cuda_compiled['mean_us']:.1f} ± {cuda_compiled['std_us']:.1f} µs")
    else:
        print(f"   {cuda_compiled['error']}")

    # Analysis
    analysis = {}

    if "error" not in cuda_launch and "error" not in webgpu_dispatch:
        overhead_ratio = webgpu_dispatch['mean_us'] / cuda_launch['mean_us']
        analysis["webgpu_vs_cuda_overhead_ratio"] = overhead_ratio
        print(f"\n   WebGPU dispatch is {overhead_ratio:.1f}x slower than CUDA launch")

    if "error" not in cuda_unfused and "error" not in cuda_fused:
        cuda_fusion_benefit = cuda_unfused['mean_us'] / cuda_fused['mean_us']
        analysis["cuda_fusion_speedup"] = cuda_fusion_benefit
        print(f"   CUDA fusion speedup: {cuda_fusion_benefit:.2f}x")

    results["analysis"] = analysis

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "error" not in cuda_launch:
        print(f"CUDA kernel launch overhead:    {cuda_launch['mean_us']:.1f} µs")
    if "error" not in webgpu_dispatch:
        print(f"WebGPU dispatch overhead:       {webgpu_dispatch['mean_us']:.1f} µs")
    if "webgpu_vs_cuda_overhead_ratio" in analysis:
        print(f"Overhead ratio (WebGPU/CUDA):   {analysis['webgpu_vs_cuda_overhead_ratio']:.1f}x")
    if "cuda_fusion_speedup" in analysis:
        print(f"CUDA fusion speedup:            {analysis['cuda_fusion_speedup']:.2f}x")
        if analysis['cuda_fusion_speedup'] < 1.5:
            print("  → Minimal benefit from fusion in CUDA (as expected)")
    print("=" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
