#!/usr/bin/env python
"""
Profile individual ops in torch-webgpu to identify slowest operations.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Initialize torch-webgpu
import torch_webgpu  # noqa


def profile_op(op_fn, name, warmup=3, runs=10):
    """Profile a single operation."""
    # Warmup
    for _ in range(warmup):
        op_fn()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        op_fn()
        # Force sync by copying to CPU
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    return {"name": name, "avg_ms": avg, "min_ms": min_t, "max_ms": max_t}


def sync_webgpu():
    """Force GPU sync by doing a small copy."""
    t = torch.empty(1, device="webgpu")
    _ = t.to("cpu")


def to_webgpu(*shapes):
    """Create random tensors on CPU and move to WebGPU."""
    device = torch.device("webgpu")
    return [torch.randn(shape).to(device) for shape in shapes]


def main():
    device = torch.device("webgpu")

    print("=" * 70)
    print("Profiling WebGPU Operations")
    print("=" * 70)

    results = []

    # ========================================
    # Matrix Multiplication (the biggest bottleneck for LLMs)
    # ========================================
    print("\n--- Matrix Multiplication ---")

    # Qwen2.5-0.5B-Instruct layer dimensions (approximate):
    # hidden_size = 896
    # intermediate_size = 4864
    # num_attention_heads = 14
    # head_dim = 64
    # vocab_size = 151936

    shapes = [
        # (M, K, N) -> A(M,K) @ B(K,N) = C(M,N)
        # Attention Q/K/V projection: (seq_len, hidden_size) @ (hidden_size, hidden_size)
        (1, 896, 896, "attn_qkv_proj_seq1"),
        (10, 896, 896, "attn_qkv_proj_seq10"),
        (50, 896, 896, "attn_qkv_proj_seq50"),
        (100, 896, 896, "attn_qkv_proj_seq100"),

        # MLP up projection: (seq_len, hidden_size) @ (hidden_size, intermediate_size)
        (1, 896, 4864, "mlp_up_proj_seq1"),
        (10, 896, 4864, "mlp_up_proj_seq10"),
        (50, 896, 4864, "mlp_up_proj_seq50"),

        # MLP down projection: (seq_len, intermediate_size) @ (intermediate_size, hidden_size)
        (1, 4864, 896, "mlp_down_proj_seq1"),
        (10, 4864, 896, "mlp_down_proj_seq10"),
        (50, 4864, 896, "mlp_down_proj_seq50"),

        # LM head: (seq_len, hidden_size) @ (hidden_size, vocab_size)
        (1, 896, 151936, "lm_head_seq1"),
        (10, 896, 151936, "lm_head_seq10"),

        # Standard benchmark shapes
        (128, 128, 128, "mm_128x128x128"),
        (256, 256, 256, "mm_256x256x256"),
        (512, 512, 512, "mm_512x512x512"),
        (1024, 1024, 1024, "mm_1024x1024x1024"),
    ]

    for M, K, N, name in shapes:
        A = torch.randn(M, K).to(device)
        B = torch.randn(K, N).to(device)

        def mm_op():
            return torch.mm(A, B)

        # Warmup
        for _ in range(3):
            mm_op()
        sync_webgpu()

        times = []
        for _ in range(10):
            start = time.perf_counter()
            C = mm_op()
            sync_webgpu()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg = sum(times) / len(times)
        flops = 2 * M * K * N
        gflops = flops / (avg / 1000) / 1e9

        result = {
            "name": name,
            "shape": f"({M}, {K}) @ ({K}, {N})",
            "avg_ms": avg,
            "gflops": gflops,
        }
        results.append(result)
        print(f"{name:25s}: {avg:8.3f}ms, {gflops:8.2f} GFLOPS")

    # ========================================
    # Softmax
    # ========================================
    print("\n--- Softmax ---")

    softmax_shapes = [
        (1, 14, 10, 10, "softmax_attn_seq10"),
        (1, 14, 50, 50, "softmax_attn_seq50"),
        (1, 14, 100, 100, "softmax_attn_seq100"),
        (1, 151936, "softmax_lm_head"),
        (10, 151936, "softmax_lm_head_seq10"),
    ]

    for shape_info in softmax_shapes:
        if len(shape_info) == 3:
            shape = shape_info[:2]
            name = shape_info[2]
        else:
            shape = shape_info[:4]
            name = shape_info[4]

        x = torch.randn(*shape).to(device)

        def softmax_op():
            return torch.softmax(x, dim=-1)

        # Warmup
        for _ in range(3):
            softmax_op()
        sync_webgpu()

        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = softmax_op()
            sync_webgpu()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg = sum(times) / len(times)
        print(f"{name:25s}: {avg:8.3f}ms (shape={shape})")

    # ========================================
    # Elementwise operations
    # ========================================
    print("\n--- Elementwise Ops ---")

    sizes = [
        (1, 896, "elem_hidden_seq1"),
        (10, 896, "elem_hidden_seq10"),
        (50, 896, "elem_hidden_seq50"),
        (10, 4864, "elem_intermediate_seq10"),
    ]

    for shape_info in sizes:
        shape = shape_info[:2]
        name = shape_info[2]

        a = torch.randn(*shape).to(device)
        b = torch.randn(*shape).to(device)

        # Test add
        for _ in range(3):
            _ = a + b
        sync_webgpu()

        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = a + b
            sync_webgpu()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg = sum(times) / len(times)
        print(f"add_{name:20s}: {avg:8.3f}ms")

        # Test mul
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = a * b
            sync_webgpu()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg = sum(times) / len(times)
        print(f"mul_{name:20s}: {avg:8.3f}ms")

    # ========================================
    # Activation functions
    # ========================================
    print("\n--- Activations ---")

    x = torch.randn(10, 4864).to(device)

    activations = [
        ("relu", lambda: torch.relu(x)),
        ("gelu", lambda: torch.nn.functional.gelu(x)),
        ("silu", lambda: torch.nn.functional.silu(x)),
    ]

    for act_name, act_fn in activations:
        for _ in range(3):
            act_fn()
        sync_webgpu()

        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = act_fn()
            sync_webgpu()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg = sum(times) / len(times)
        print(f"{act_name:10s} (10, 4864): {avg:8.3f}ms")

    # ========================================
    # RMSNorm components (manual)
    # ========================================
    print("\n--- RMSNorm components ---")

    x = torch.randn(10, 896).to(device)
    weight = torch.randn(896).to(device)

    def rmsnorm():
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + 1e-6)
        return x_normed * weight

    for _ in range(3):
        rmsnorm()
    sync_webgpu()

    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = rmsnorm()
        sync_webgpu()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg = sum(times) / len(times)
    print(f"rmsnorm (10, 896): {avg:8.3f}ms")

    # ========================================
    # Embedding lookup
    # ========================================
    print("\n--- Embedding ---")

    vocab_size = 151936
    hidden_size = 896
    embed_weight = torch.randn(vocab_size, hidden_size).to(device)

    for seq_len in [1, 10, 50]:
        indices = torch.randint(0, vocab_size, (1, seq_len)).to(device)

        def embed():
            return torch.nn.functional.embedding(indices, embed_weight)

        for _ in range(3):
            embed()
        sync_webgpu()

        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = embed()
            sync_webgpu()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg = sum(times) / len(times)
        print(f"embedding seq_len={seq_len:3d}: {avg:8.3f}ms")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("Summary: Top operations by time (for model-relevant shapes)")
    print("=" * 70)

    model_ops = [r for r in results if "seq" in r["name"]]
    model_ops.sort(key=lambda x: x["avg_ms"], reverse=True)

    for i, op in enumerate(model_ops[:10]):
        print(f"{i+1:2d}. {op['name']:30s}: {op['avg_ms']:8.3f}ms, {op.get('gflops', 0):8.2f} GFLOPS")


if __name__ == "__main__":
    main()
