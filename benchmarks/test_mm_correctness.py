#!/usr/bin/env python
"""Test matmul correctness, especially with transposed/strided matrices."""

import sys
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch_webgpu  # noqa


def test_mm():
    device = torch.device("webgpu")

    print("Testing mm correctness")
    print("=" * 60)

    test_cases = [
        # (M, K, N, description)
        (4, 4, 4, "Small square"),
        (16, 16, 16, "Tile-size square"),
        (17, 17, 17, "Non-tile-aligned square"),
        (32, 64, 32, "Rectangular"),
        (1, 896, 896, "Single row (like seq_len=1 linear)"),
        (10, 896, 896, "Small batch (like seq_len=10 linear)"),
        (10, 896, 4864, "MLP up projection shape"),
        (10, 4864, 896, "MLP down projection shape"),
    ]

    all_passed = True

    for M, K, N, desc in test_cases:
        # Create contiguous inputs
        A_cpu = torch.randn(M, K)
        B_cpu = torch.randn(K, N)

        A_gpu = A_cpu.to(device)
        B_gpu = B_cpu.to(device)

        # Compute reference on CPU
        expected = torch.mm(A_cpu, B_cpu)

        # Compute on WebGPU
        result_gpu = torch.mm(A_gpu, B_gpu)
        result_cpu = result_gpu.to("cpu")

        max_diff = (result_cpu - expected).abs().max().item()
        is_close = torch.allclose(result_cpu, expected, rtol=1e-4, atol=1e-4)

        status = "PASS" if is_close else "FAIL"
        print(f"{desc} ({M}x{K} @ {K}x{N}): max_diff={max_diff:.6f} {status}")

        if not is_close:
            all_passed = False

    print()
    print("Testing mm with transposed B (like linear)")
    print("-" * 60)

    for M, K, N, desc in test_cases:
        # Create A contiguous, B transposed
        A_cpu = torch.randn(M, K)
        B_original_cpu = torch.randn(N, K)  # [out_features, in_features]
        B_cpu = B_original_cpu.t()  # [in_features, out_features] - transposed view

        A_gpu = A_cpu.to(device)
        B_gpu = B_original_cpu.to(device).t()  # Create transposed view on GPU

        # Verify B is actually transposed (not contiguous)
        assert not B_gpu.is_contiguous(), "B should be non-contiguous (transposed)"

        # Compute reference on CPU
        expected = torch.mm(A_cpu, B_cpu)

        # Compute on WebGPU
        result_gpu = torch.mm(A_gpu, B_gpu)
        result_cpu = result_gpu.to("cpu")

        max_diff = (result_cpu - expected).abs().max().item()
        is_close = torch.allclose(result_cpu, expected, rtol=1e-4, atol=1e-4)

        status = "PASS" if is_close else "FAIL"
        print(f"{desc} transposed ({M}x{K} @ {K}x{N}): max_diff={max_diff:.6f} {status}")

        if not is_close:
            all_passed = False

    print()
    print("Testing F.linear (simulates transformer linear layer)")
    print("-" * 60)

    linear_cases = [
        (1, 896, 896, "Attention QKV seq_len=1"),
        (10, 896, 896, "Attention QKV seq_len=10"),
        (10, 896, 4864, "MLP up"),
        (10, 4864, 896, "MLP down"),
        (1, 896, 151936, "LM head seq_len=1"),
    ]

    for batch, in_features, out_features, desc in linear_cases:
        input_cpu = torch.randn(batch, in_features)
        weight_cpu = torch.randn(out_features, in_features)

        input_gpu = input_cpu.to(device)
        weight_gpu = weight_cpu.to(device)

        expected = torch.nn.functional.linear(input_cpu, weight_cpu)
        result_gpu = torch.nn.functional.linear(input_gpu, weight_gpu)
        result_cpu = result_gpu.to("cpu")

        max_diff = (result_cpu - expected).abs().max().item()
        is_close = torch.allclose(result_cpu, expected, rtol=1e-4, atol=1e-4)

        status = "PASS" if is_close else "FAIL"
        print(f"{desc} ({batch}x{in_features} @ {in_features}x{out_features}): max_diff={max_diff:.6f} {status}")

        if not is_close:
            all_passed = False

    print()
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")

    return all_passed


if __name__ == "__main__":
    test_mm()
