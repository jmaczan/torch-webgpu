#!/usr/bin/env python3
"""
Experiment 6: Dawn Direct Dispatch Overhead Measurement

This experiment measures Dawn's dispatch overhead using the SAME methodology
as exp1_cross_gpu_webgpu.py (which uses wgpu-native), enabling fair comparison.

The key difference from our derived ~95µs figure:
- Derived: (TTFT_unfused - TTFT_fused) / dispatches_saved (includes framework overhead)
- Direct: Measure individual dispatch times in isolation (pure WebGPU overhead)

Usage:
    python exp6_dawn_direct.py --output results/exp6_dawn_direct.json

Requires: torch-webgpu with Dawn backend installed
"""

import argparse
import json
import time
import platform
from pathlib import Path

import numpy as np

# Try to import torch-webgpu
try:
    import torch
    import torch_webgpu
    TORCH_WEBGPU_AVAILABLE = True
except ImportError:
    TORCH_WEBGPU_AVAILABLE = False
    print("WARNING: torch-webgpu not available. This experiment requires Dawn.")


def get_system_info():
    """Collect system information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }

    if TORCH_WEBGPU_AVAILABLE:
        info["torch_version"] = torch.__version__
        info["backend"] = "Dawn"
        try:
            info["webgpu_device"] = "Dawn (via torch-webgpu)"
        except:
            pass

    return info


def sync_webgpu(tensor):
    """
    Force synchronization by accessing tensor data.
    torch-webgpu doesn't have explicit sync, but accessing values forces completion.
    """
    _ = tensor.view(-1)[0].item()


def benchmark_dawn_dispatch_overhead_via_torch(n_iterations=1000, warmup=100):
    """
    Measure Dawn dispatch overhead using torch-webgpu operations.
    """
    if not TORCH_WEBGPU_AVAILABLE:
        return {"error": "torch-webgpu not available"}

    device = torch.device("webgpu")
    x = torch.ones(64).to(device)

    # Warmup
    for _ in range(warmup):
        y = x + 1.0
    sync_webgpu(y)

    # Measure individual operations
    times_us = []
    for _ in range(n_iterations):
        sync_webgpu(x)
        start = time.perf_counter()
        y = x + 1.0
        sync_webgpu(y)
        end = time.perf_counter()
        times_us.append((end - start) * 1e6)

    return {
        "name": "dawn_dispatch_overhead",
        "method": "torch_webgpu_elementwise",
        "mean_us": float(np.mean(times_us)),
        "std_us": float(np.std(times_us)),
        "min_us": float(np.min(times_us)),
        "max_us": float(np.max(times_us)),
        "median_us": float(np.median(times_us)),
        "p95_us": float(np.percentile(times_us, 95)),
        "n_iterations": n_iterations,
        "all_times_us": times_us[:100]
    }


def benchmark_dawn_matmul_overhead(n_iterations=500, warmup=50):
    """
    Measure Dawn dispatch overhead for matrix multiplication.
    """
    if not TORCH_WEBGPU_AVAILABLE:
        return {"error": "torch-webgpu not available"}

    device = torch.device("webgpu")
    # Create on CPU, then move to WebGPU (randn not supported on webgpu)
    a = torch.randn(64, 64).to(device)
    b = torch.randn(64, 64).to(device)

    # Warmup
    for _ in range(warmup):
        c = torch.matmul(a, b)
    sync_webgpu(c)

    # Measure
    times_us = []
    for _ in range(n_iterations):
        sync_webgpu(a)
        start = time.perf_counter()
        c = torch.matmul(a, b)
        sync_webgpu(c)
        end = time.perf_counter()
        times_us.append((end - start) * 1e6)

    return {
        "name": "dawn_matmul_overhead",
        "matrix_size": 64,
        "mean_us": float(np.mean(times_us)),
        "std_us": float(np.std(times_us)),
        "min_us": float(np.min(times_us)),
        "max_us": float(np.max(times_us)),
        "median_us": float(np.median(times_us)),
        "n_iterations": n_iterations
    }


def benchmark_dawn_rmsnorm_unfused(hidden_dim=896, n_iterations=100, warmup=10):
    """
    Measure Dawn RMSNorm unfused (multiple dispatches).
    """
    if not TORCH_WEBGPU_AVAILABLE:
        return {"error": "torch-webgpu not available"}

    device = torch.device("webgpu")
    x = torch.randn(1, hidden_dim).to(device)
    weight = torch.ones(hidden_dim).to(device)
    eps = 1e-6

    def rmsnorm_unfused(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps)
        return x_norm * weight

    # Warmup
    for _ in range(warmup):
        y = rmsnorm_unfused(x, weight, eps)
    sync_webgpu(y)

    # Measure
    times_ms = []
    for _ in range(n_iterations):
        sync_webgpu(x)
        start = time.perf_counter()
        y = rmsnorm_unfused(x, weight, eps)
        sync_webgpu(y)
        end = time.perf_counter()
        times_ms.append((end - start) * 1e3)

    return {
        "name": "dawn_rmsnorm_unfused",
        "dispatches": 5,
        "hidden_dim": hidden_dim,
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "n_iterations": n_iterations
    }


def benchmark_dawn_sequential_dispatches(n_dispatches=100, n_iterations=50, warmup=5):
    """
    Measure time for N sequential dispatches.
    """
    if not TORCH_WEBGPU_AVAILABLE:
        return {"error": "torch-webgpu not available"}

    device = torch.device("webgpu")

    # Warmup
    for _ in range(warmup):
        x = torch.ones(256).to(device)
        for _ in range(n_dispatches):
            x = x + 1.0
        sync_webgpu(x)

    # Measure
    times_ms = []
    for _ in range(n_iterations):
        x = torch.ones(256).to(device)
        sync_webgpu(x)

        start = time.perf_counter()
        for _ in range(n_dispatches):
            x = x + 1.0
        sync_webgpu(x)
        end = time.perf_counter()

        times_ms.append((end - start) * 1e3)

    mean_total_ms = float(np.mean(times_ms))
    per_dispatch_us = (mean_total_ms * 1000) / n_dispatches

    return {
        "name": "dawn_sequential_dispatches",
        "n_dispatches": n_dispatches,
        "mean_total_ms": mean_total_ms,
        "std_total_ms": float(np.std(times_ms)),
        "per_dispatch_us": per_dispatch_us,
        "n_iterations": n_iterations
    }


def main():
    parser = argparse.ArgumentParser(description="Dawn direct dispatch overhead measurement")
    parser.add_argument("--output", type=str, default="results/exp6_dawn_direct.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 6: Dawn Direct Dispatch Overhead")
    print("=" * 60)

    if not TORCH_WEBGPU_AVAILABLE:
        print("ERROR: torch-webgpu not available. Cannot run this experiment.")
        return

    results = {
        "system_info": get_system_info(),
        "experiments": {}
    }

    # 1. Basic dispatch overhead
    print("\n1. Measuring Dawn dispatch overhead (elementwise)...")
    dispatch_overhead = benchmark_dawn_dispatch_overhead_via_torch(n_iterations=1000)
    results["experiments"]["dispatch_overhead"] = dispatch_overhead
    if "error" not in dispatch_overhead:
        print(f"   Mean: {dispatch_overhead['mean_us']:.1f} ± {dispatch_overhead['std_us']:.1f} µs")
        print(f"   Min: {dispatch_overhead['min_us']:.1f} µs, Median: {dispatch_overhead['median_us']:.1f} µs")

    # 2. Matmul overhead
    print("\n2. Measuring Dawn matmul overhead...")
    matmul_overhead = benchmark_dawn_matmul_overhead(n_iterations=500)
    results["experiments"]["matmul_overhead"] = matmul_overhead
    if "error" not in matmul_overhead:
        print(f"   Mean: {matmul_overhead['mean_us']:.1f} ± {matmul_overhead['std_us']:.1f} µs")

    # 3. RMSNorm unfused
    print("\n3. Measuring Dawn RMSNorm unfused...")
    rmsnorm = benchmark_dawn_rmsnorm_unfused(n_iterations=100)
    results["experiments"]["rmsnorm_unfused"] = rmsnorm
    if "error" not in rmsnorm:
        print(f"   Mean: {rmsnorm['mean_ms']:.3f} ± {rmsnorm['std_ms']:.3f} ms")

    # 4. Sequential dispatches
    print("\n4. Measuring Dawn sequential dispatches (100 ops)...")
    sequential = benchmark_dawn_sequential_dispatches(n_dispatches=100, n_iterations=50)
    results["experiments"]["sequential_dispatches"] = sequential
    if "error" not in sequential:
        print(f"   Total: {sequential['mean_total_ms']:.2f} ms for {sequential['n_dispatches']} dispatches")
        print(f"   Per-dispatch: {sequential['per_dispatch_us']:.1f} µs")

    # Analysis
    print("\n" + "=" * 60)
    print("COMPARISON WITH WGPU-NATIVE (from exp1)")
    print("=" * 60)

    if "error" not in dispatch_overhead:
        dawn_overhead = dispatch_overhead['mean_us']
        wgpu_vulkan_overhead = 25.2  # From exp1
        wgpu_metal_overhead = 49.6   # From exp1

        print(f"\nPer-dispatch overhead (single op):")
        print(f"  Dawn (this experiment):  {dawn_overhead:.1f} µs")
        print(f"  wgpu/Vulkan (exp1):      {wgpu_vulkan_overhead:.1f} µs")
        print(f"  wgpu/Metal (exp1):       {wgpu_metal_overhead:.1f} µs")
        print(f"\nRatios:")
        print(f"  Dawn / wgpu-Vulkan:      {dawn_overhead / wgpu_vulkan_overhead:.2f}x")
        print(f"  Dawn / wgpu-Metal:       {dawn_overhead / wgpu_metal_overhead:.2f}x")

        results["comparison"] = {
            "dawn_overhead_us": dawn_overhead,
            "wgpu_vulkan_overhead_us": wgpu_vulkan_overhead,
            "wgpu_metal_overhead_us": wgpu_metal_overhead,
            "dawn_vs_wgpu_vulkan_ratio": dawn_overhead / wgpu_vulkan_overhead,
            "dawn_vs_wgpu_metal_ratio": dawn_overhead / wgpu_metal_overhead
        }

    if "error" not in sequential:
        dawn_seq = sequential['per_dispatch_us']
        print(f"\nPer-dispatch overhead (sequential context):")
        print(f"  Dawn: {dawn_seq:.1f} µs")

        results["comparison"]["dawn_sequential_per_dispatch_us"] = dawn_seq

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
