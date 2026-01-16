#!/usr/bin/env python
"""Test that softmax optimization is working correctly."""

import sys
import time
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch_webgpu  # noqa


def test_softmax():
    device = torch.device("webgpu")

    # Test different sizes
    sizes = [
        (1, 100),      # Small (should use simple kernel)
        (1, 1000),     # Medium (should use simple kernel)
        (1, 10000),    # Large (should use parallel kernel)
        (1, 151936),   # Vocab size (should use parallel kernel)
        (10, 151936),  # Batch + vocab (should use parallel kernel)
    ]

    print("Testing softmax correctness and performance")
    print("=" * 60)

    for shape in sizes:
        # Create input on CPU, move to WebGPU
        x_cpu = torch.randn(*shape)
        x_gpu = x_cpu.to(device)

        # Compute on CPU (reference)
        expected = torch.softmax(x_cpu, dim=-1)

        # Warmup
        for _ in range(3):
            _ = torch.softmax(x_gpu, dim=-1)

        # Time it
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result_gpu = torch.softmax(x_gpu, dim=-1)
            # Sync
            _ = result_gpu.to("cpu")
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = sum(times) / len(times)

        # Copy back and compare
        result_cpu = result_gpu.to("cpu")

        # Check correctness
        max_diff = (result_cpu - expected).abs().max().item()
        is_close = torch.allclose(result_cpu, expected, rtol=1e-4, atol=1e-4)

        status = "OK" if is_close else "MISMATCH"
        print(f"Shape {str(shape):15s}: {avg_time:8.3f}ms, max_diff={max_diff:.6f}, {status}")

    print()
    print("If large shapes (>1024) are fast (~0.5ms), parallel kernel is working")
    print("If large shapes are slow (~45ms), something is wrong")


if __name__ == "__main__":
    test_softmax()
