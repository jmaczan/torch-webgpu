#!/usr/bin/env python
"""
Profile WebGPU dispatch overhead to get accurate measurements.

This script measures the actual time spent on different components
of WebGPU dispatch overhead during torch-webgpu inference.
"""

import time
import torch
import json
from pathlib import Path

# Import torch_webgpu to register the backend
import torch_webgpu
from torch_webgpu.compiler.webgpu_compiler import webgpu_backend


def measure_single_op_overhead(op_name, op_func, input_tensors, n_runs=100):
    """Measure overhead for a single operation type."""
    # Warmup
    for _ in range(10):
        result = op_func(*input_tensors)
        if hasattr(result, 'cpu'):
            result.cpu()  # Force sync

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = op_func(*input_tensors)
        if hasattr(result, 'cpu'):
            result.cpu()  # Force sync
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        "op": op_name,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "n_runs": n_runs,
    }


def profile_overhead():
    """Profile different sources of WebGPU overhead."""
    results = {}

    print("=" * 60)
    print("WebGPU Dispatch Overhead Profiling")
    print("=" * 60)
    print()

    # Test different operation sizes
    sizes = [
        (64, 64),
        (256, 256),
        (512, 512),
        (896, 896),  # Qwen hidden size
        (1024, 1024),
    ]

    for m, n in sizes:
        print(f"\nTesting matrix size {m}x{n}...")

        # Create test tensors
        x = torch.randn(1, m, dtype=torch.float32)
        w = torch.randn(m, n, dtype=torch.float32)

        # Compile a simple matmul
        @torch.compile(backend=webgpu_backend)
        def matmul_fn(a, b):
            return torch.matmul(a, b)

        # Measure
        result = measure_single_op_overhead(
            f"matmul_{m}x{n}",
            matmul_fn,
            (x, w),
            n_runs=50
        )
        results[f"matmul_{m}x{n}"] = result
        print(f"  Mean: {result['mean_ms']:.3f}ms, Min: {result['min_ms']:.3f}ms, Max: {result['max_ms']:.3f}ms")

    # Test elementwise operations (should show pure dispatch overhead)
    print("\nTesting elementwise operations (pure dispatch overhead)...")

    for size in [1024, 4096, 16384, 65536]:
        x = torch.randn(size, dtype=torch.float32)

        @torch.compile(backend=webgpu_backend)
        def add_fn(a):
            return a + 1.0

        result = measure_single_op_overhead(
            f"add_scalar_{size}",
            add_fn,
            (x,),
            n_runs=100
        )
        results[f"add_scalar_{size}"] = result
        print(f"  add_scalar_{size}: Mean: {result['mean_ms']:.3f}ms")

    # Measure sequence of operations (forward pass simulation)
    print("\nSimulating forward pass dispatch overhead...")

    # Create a sequence that mimics transformer layer
    hidden_size = 896
    intermediate_size = 4864

    x = torch.randn(1, 10, hidden_size, dtype=torch.float32)  # (batch, seq, hidden)
    w1 = torch.randn(hidden_size, intermediate_size, dtype=torch.float32)
    w2 = torch.randn(intermediate_size, hidden_size, dtype=torch.float32)

    @torch.compile(backend=webgpu_backend)
    def mlp_block(x, w1, w2):
        h = torch.matmul(x, w1)
        h = torch.relu(h)
        return torch.matmul(h, w2)

    # Warmup
    for _ in range(5):
        _ = mlp_block(x, w1, w2).cpu()

    # Measure MLP block (multiple dispatches)
    times = []
    for _ in range(20):
        start = time.perf_counter()
        _ = mlp_block(x, w1, w2).cpu()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    results["mlp_block_3ops"] = {
        "op": "mlp_block (matmul+relu+matmul)",
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "n_runs": 20,
        "note": "3 operations in sequence"
    }
    print(f"  MLP block (3 ops): Mean: {results['mlp_block_3ops']['mean_ms']:.3f}ms")

    # Calculate per-dispatch overhead estimate
    single_op_overhead = results["add_scalar_1024"]["mean_ms"]
    mlp_overhead = results["mlp_block_3ops"]["mean_ms"]
    estimated_per_op = mlp_overhead / 3  # Rough estimate assuming 3 dispatches

    print("\n" + "=" * 60)
    print("OVERHEAD ANALYSIS")
    print("=" * 60)
    print(f"Single elementwise op (add_scalar_1024): {single_op_overhead:.3f}ms")
    print(f"MLP block (3 ops): {mlp_overhead:.3f}ms")
    print(f"Estimated per-dispatch overhead: ~{estimated_per_op:.3f}ms")
    print()
    print("Note: Per-dispatch overhead includes:")
    print("  - Command encoder creation")
    print("  - Bind group creation/lookup")
    print("  - Buffer management")
    print("  - Pipeline state setup")
    print("  - Validation")
    print("  - Queue submission")

    # Save results
    output_path = Path(__file__).parent / "overhead_profile.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def profile_forward_pass_breakdown():
    """Profile actual forward pass to count dispatches and measure overhead."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 60)
    print("FORWARD PASS PROFILING")
    print("=" * 60)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model.eval()

    # Compile model
    print("Compiling model with WebGPU backend...")
    compiled_model = torch.compile(model, backend=webgpu_backend)

    # Prepare input
    inputs = tokenizer("Hello", return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = compiled_model(input_ids)

    # Measure forward pass
    print("Measuring forward pass...")
    times = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = compiled_model(input_ids)
        # Force sync
        _ = outputs.logits.cpu()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean_time = sum(times) / len(times)

    print(f"\nForward pass time: {mean_time:.2f}ms")
    print(f"  (Baseline from optimization log: ~75-100ms)")

    # Estimate dispatch count based on model architecture
    # Qwen2.5-0.5B has 24 layers, each with:
    # - QKV projection (1-3 ops depending on fusion)
    # - Attention scores (1 matmul)
    # - Softmax (1 op)
    # - Attention output (1 matmul)
    # - Output projection (1 linear)
    # - MLP gate/up (1-2 linear)
    # - MLP activation (1 op)
    # - MLP down (1 linear)
    # - 2x RMSNorm (6 ops each if not fused, 1 if fused)
    # Plus embedding, final norm, LM head

    estimated_dispatches = 24 * (3 + 1 + 1 + 1 + 1 + 2 + 1 + 1 + 12) + 3 + 6 + 1
    # Simplified: 24 * ~23 + 10 = ~562 theoretical ops
    # With some fusion: ~200-300 actual dispatches

    estimated_overhead_per_dispatch = mean_time / 200  # Assuming ~200 dispatches

    print(f"\nEstimated breakdown (assuming ~200 dispatches):")
    print(f"  Total time: {mean_time:.2f}ms")
    print(f"  Per-dispatch overhead: ~{estimated_overhead_per_dispatch:.3f}ms")

    return {
        "forward_pass_ms": mean_time,
        "times": times,
        "estimated_dispatches": "~200",
        "estimated_per_dispatch_ms": estimated_overhead_per_dispatch,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Profile WebGPU overhead")
    parser.add_argument("--full", action="store_true", help="Include full model forward pass profiling")
    args = parser.parse_args()

    results = profile_overhead()

    if args.full:
        forward_results = profile_forward_pass_breakdown()
        results["forward_pass"] = forward_results

        # Update saved results
        output_path = Path(__file__).parent / "overhead_profile.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
