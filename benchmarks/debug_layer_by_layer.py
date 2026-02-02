#!/usr/bin/env python
"""Debug script to trace through Qwen layer by layer and find numerical divergence."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch_webgpu  # noqa

from transformers import AutoModelForCausalLM, AutoTokenizer


def compare_tensors(name, cpu_tensor, gpu_tensor, rtol=1e-3, atol=1e-3):
    """Compare CPU and GPU tensors and report differences."""
    gpu_cpu = gpu_tensor.to("cpu")
    diff = (cpu_tensor - gpu_cpu).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    is_close = torch.allclose(cpu_tensor, gpu_cpu, rtol=rtol, atol=atol)
    status = "OK" if is_close else "DIVERGED"

    print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status}]")
    print(f"    CPU: min={cpu_tensor.min():.4f}, max={cpu_tensor.max():.4f}, mean={cpu_tensor.mean():.4f}")
    print(f"    GPU: min={gpu_cpu.min():.4f}, max={gpu_cpu.max():.4f}, mean={gpu_cpu.mean():.4f}")

    if not is_close:
        # Find where the divergence is worst
        idx = diff.argmax()
        idx_tuple = tuple(
            (idx // diff.shape[i+1:].numel() if i+1 < len(diff.shape) else idx) % diff.shape[i]
            for i in range(len(diff.shape))
        )
        print(f"    Worst at index {idx_tuple}: CPU={cpu_tensor.flatten()[idx]:.4f}, GPU={gpu_cpu.flatten()[idx]:.4f}")

    return is_close


def test_single_rmsnorm():
    """Test a single RMSNorm in isolation."""
    print("\n" + "=" * 60)
    print("Testing single RMSNorm in isolation")
    print("=" * 60)

    device = torch.device("webgpu")

    # Simple RMSNorm
    hidden_size = 896
    eps = 1e-6
    weight_cpu = torch.randn(hidden_size)
    x_cpu = torch.randn(1, 10, hidden_size)

    weight_gpu = weight_cpu.to(device)
    x_gpu = x_cpu.to(device)

    # CPU computation
    variance_cpu = x_cpu.pow(2).mean(dim=-1, keepdim=True)
    x_norm_cpu = x_cpu * torch.rsqrt(variance_cpu + eps)
    out_cpu = x_norm_cpu * weight_cpu

    # GPU computation
    variance_gpu = x_gpu.pow(2).mean(dim=-1, keepdim=True)
    x_norm_gpu = x_gpu * torch.rsqrt(variance_gpu + eps)
    out_gpu = x_norm_gpu * weight_gpu

    print("\nStep-by-step comparison:")
    compare_tensors("x.pow(2)", x_cpu.pow(2), x_gpu.pow(2))
    compare_tensors("variance", variance_cpu, variance_gpu)
    compare_tensors("rsqrt(var+eps)", torch.rsqrt(variance_cpu + eps), torch.rsqrt(variance_gpu + eps))
    compare_tensors("x_norm", x_norm_cpu, x_norm_gpu)
    compare_tensors("output", out_cpu, out_gpu)


def test_attention_components():
    """Test attention components step by step."""
    print("\n" + "=" * 60)
    print("Testing attention components")
    print("=" * 60)

    device = torch.device("webgpu")

    # Load model for weights
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
    )
    model.eval()

    layer = model.model.layers[0]

    # Create test input
    batch, seq_len, hidden = 1, 10, 896
    x_cpu = torch.randn(batch, seq_len, hidden)
    x_gpu = x_cpu.to(device)

    print(f"\nInput shape: {x_cpu.shape}")

    # Test input_layernorm
    print("\n1. Testing input_layernorm (RMSNorm):")
    norm_weight_cpu = layer.input_layernorm.weight.data
    norm_weight_gpu = norm_weight_cpu.to(device)

    # Manual RMSNorm on CPU
    variance_cpu = x_cpu.pow(2).mean(dim=-1, keepdim=True)
    normed_cpu = x_cpu * torch.rsqrt(variance_cpu + 1e-6) * norm_weight_cpu

    # Manual RMSNorm on GPU
    variance_gpu = x_gpu.pow(2).mean(dim=-1, keepdim=True)
    normed_gpu = x_gpu * torch.rsqrt(variance_gpu + 1e-6) * norm_weight_gpu

    compare_tensors("input_layernorm", normed_cpu, normed_gpu)

    # Test Q projection
    print("\n2. Testing Q projection:")
    q_weight_cpu = layer.self_attn.q_proj.weight.data
    q_weight_gpu = q_weight_cpu.to(device)

    q_cpu = F.linear(normed_cpu, q_weight_cpu)
    q_gpu = F.linear(normed_gpu, q_weight_gpu)

    compare_tensors("Q projection", q_cpu, q_gpu)

    # Test K projection
    print("\n3. Testing K projection:")
    k_weight_cpu = layer.self_attn.k_proj.weight.data
    k_weight_gpu = k_weight_cpu.to(device)

    k_cpu = F.linear(normed_cpu, k_weight_cpu)
    k_gpu = F.linear(normed_gpu, k_weight_gpu)

    compare_tensors("K projection", k_cpu, k_gpu)

    # Test V projection
    print("\n4. Testing V projection:")
    v_weight_cpu = layer.self_attn.v_proj.weight.data
    v_weight_gpu = v_weight_cpu.to(device)

    v_cpu = F.linear(normed_cpu, v_weight_cpu)
    v_gpu = F.linear(normed_gpu, v_weight_gpu)

    compare_tensors("V projection", v_cpu, v_gpu)


def test_full_layer_manual():
    """Manually step through a full transformer layer."""
    print("\n" + "=" * 60)
    print("Testing full transformer layer - MANUAL STEP BY STEP")
    print("=" * 60)

    device = torch.device("webgpu")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Get actual embedding
    inputs = tokenizer("Hello, world!", return_tensors="pt")

    with torch.no_grad():
        # Get embeddings
        embed_cpu = model.model.embed_tokens(inputs["input_ids"])
        print(f"\nEmbedding: shape={embed_cpu.shape}")

        embed_gpu = embed_cpu.to(device)

        layer = model.model.layers[0]

        # Step 1: Input layernorm
        print("\n--- Step 1: Input LayerNorm ---")
        weight = layer.input_layernorm.weight.data
        eps = layer.input_layernorm.variance_epsilon

        # CPU
        var_cpu = embed_cpu.pow(2).mean(dim=-1, keepdim=True)
        normed_cpu = embed_cpu * torch.rsqrt(var_cpu + eps) * weight

        # GPU
        weight_gpu = weight.to(device)
        var_gpu = embed_gpu.pow(2).mean(dim=-1, keepdim=True)
        normed_gpu = embed_gpu * torch.rsqrt(var_gpu + eps) * weight_gpu

        compare_tensors("input_layernorm", normed_cpu, normed_gpu)

        # Step 2: Q, K, V projections
        print("\n--- Step 2: Q, K, V Projections ---")
        q_w = layer.self_attn.q_proj.weight.to(device)
        k_w = layer.self_attn.k_proj.weight.to(device)
        v_w = layer.self_attn.v_proj.weight.to(device)

        q_cpu = F.linear(normed_cpu, layer.self_attn.q_proj.weight)
        k_cpu = F.linear(normed_cpu, layer.self_attn.k_proj.weight)
        v_cpu = F.linear(normed_cpu, layer.self_attn.v_proj.weight)

        q_gpu = F.linear(normed_gpu, q_w)
        k_gpu = F.linear(normed_gpu, k_w)
        v_gpu = F.linear(normed_gpu, v_w)

        compare_tensors("Q projection", q_cpu, q_gpu)
        compare_tensors("K projection", k_cpu, k_gpu)
        compare_tensors("V projection", v_cpu, v_gpu)

        # Step 3: Reshape for attention
        print("\n--- Step 3: Reshape for multi-head attention ---")
        num_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        head_dim = layer.self_attn.head_dim

        bsz, seq_len, _ = q_cpu.shape

        q_cpu_reshaped = q_cpu.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k_cpu_reshaped = k_cpu.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v_cpu_reshaped = v_cpu.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        q_gpu_reshaped = q_gpu.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k_gpu_reshaped = k_gpu.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v_gpu_reshaped = v_gpu.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        compare_tensors("Q reshaped", q_cpu_reshaped, q_gpu_reshaped)
        compare_tensors("K reshaped", k_cpu_reshaped, k_gpu_reshaped)
        compare_tensors("V reshaped", v_cpu_reshaped, v_gpu_reshaped)

        # Step 4: Attention scores
        print("\n--- Step 4: Attention scores (Q @ K^T) ---")
        scale = head_dim ** -0.5

        # Repeat KV for GQA
        num_key_value_groups = num_heads // num_kv_heads
        k_cpu_expanded = k_cpu_reshaped.repeat_interleave(num_key_value_groups, dim=1)
        v_cpu_expanded = v_cpu_reshaped.repeat_interleave(num_key_value_groups, dim=1)
        k_gpu_expanded = k_gpu_reshaped.repeat_interleave(num_key_value_groups, dim=1)
        v_gpu_expanded = v_gpu_reshaped.repeat_interleave(num_key_value_groups, dim=1)

        attn_scores_cpu = torch.matmul(q_cpu_reshaped, k_cpu_expanded.transpose(-2, -1)) * scale
        attn_scores_gpu = torch.matmul(q_gpu_reshaped, k_gpu_expanded.transpose(-2, -1)) * scale

        compare_tensors("Attention scores", attn_scores_cpu, attn_scores_gpu)

        # Step 5: Softmax
        print("\n--- Step 5: Softmax ---")
        attn_weights_cpu = F.softmax(attn_scores_cpu, dim=-1)
        attn_weights_gpu = F.softmax(attn_scores_gpu, dim=-1)

        compare_tensors("Attention weights", attn_weights_cpu, attn_weights_gpu)

        # Step 6: Attention output
        print("\n--- Step 6: Attention output (weights @ V) ---")
        attn_out_cpu = torch.matmul(attn_weights_cpu, v_cpu_expanded)
        attn_out_gpu = torch.matmul(attn_weights_gpu, v_gpu_expanded)

        compare_tensors("Attention output", attn_out_cpu, attn_out_gpu)

        # Step 7: Reshape and output projection
        print("\n--- Step 7: Output projection ---")
        attn_out_cpu_flat = attn_out_cpu.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_out_gpu_flat = attn_out_gpu.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        o_w = layer.self_attn.o_proj.weight.to(device)

        proj_out_cpu = F.linear(attn_out_cpu_flat, layer.self_attn.o_proj.weight)
        proj_out_gpu = F.linear(attn_out_gpu_flat, o_w)

        compare_tensors("Output projection", proj_out_cpu, proj_out_gpu)

        # Step 8: Residual
        print("\n--- Step 8: Residual connection ---")
        hidden_cpu = embed_cpu + proj_out_cpu
        hidden_gpu = embed_gpu + proj_out_gpu

        compare_tensors("After attention residual", hidden_cpu, hidden_gpu)

        # Step 9: Post attention layernorm
        print("\n--- Step 9: Post attention LayerNorm ---")
        post_norm_weight = layer.post_attention_layernorm.weight.to(device)

        var_post_cpu = hidden_cpu.pow(2).mean(dim=-1, keepdim=True)
        normed_post_cpu = hidden_cpu * torch.rsqrt(var_post_cpu + eps) * layer.post_attention_layernorm.weight

        var_post_gpu = hidden_gpu.pow(2).mean(dim=-1, keepdim=True)
        normed_post_gpu = hidden_gpu * torch.rsqrt(var_post_gpu + eps) * post_norm_weight

        compare_tensors("Post attention layernorm", normed_post_cpu, normed_post_gpu)

        # Step 10: MLP gate projection
        print("\n--- Step 10: MLP Gate projection ---")
        gate_w = layer.mlp.gate_proj.weight.to(device)

        gate_cpu = F.linear(normed_post_cpu, layer.mlp.gate_proj.weight)
        gate_gpu = F.linear(normed_post_gpu, gate_w)

        compare_tensors("MLP gate", gate_cpu, gate_gpu)

        # Step 11: MLP up projection
        print("\n--- Step 11: MLP Up projection ---")
        up_w = layer.mlp.up_proj.weight.to(device)

        up_cpu = F.linear(normed_post_cpu, layer.mlp.up_proj.weight)
        up_gpu = F.linear(normed_post_gpu, up_w)

        compare_tensors("MLP up", up_cpu, up_gpu)

        # Step 12: SiLU and multiply
        print("\n--- Step 12: SiLU(gate) * up ---")
        mlp_intermediate_cpu = F.silu(gate_cpu) * up_cpu
        mlp_intermediate_gpu = F.silu(gate_gpu) * up_gpu

        compare_tensors("SiLU(gate) * up", mlp_intermediate_cpu, mlp_intermediate_gpu)

        # Step 13: MLP down projection
        print("\n--- Step 13: MLP Down projection ---")
        down_w = layer.mlp.down_proj.weight.to(device)

        mlp_out_cpu = F.linear(mlp_intermediate_cpu, layer.mlp.down_proj.weight)
        mlp_out_gpu = F.linear(mlp_intermediate_gpu, down_w)

        compare_tensors("MLP output", mlp_out_cpu, mlp_out_gpu)

        # Step 14: Final residual
        print("\n--- Step 14: Final residual ---")
        final_cpu = hidden_cpu + mlp_out_cpu
        final_gpu = hidden_gpu + mlp_out_gpu

        compare_tensors("Layer output", final_cpu, final_gpu)

        print("\n" + "=" * 60)
        print("Manual layer test complete!")


def test_layer_forward_vs_manual():
    """Compare layer.forward() to manual computation."""
    print("\n" + "=" * 60)
    print("Comparing layer.forward() to manual computation")
    print("=" * 60)

    device = torch.device("webgpu")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    inputs = tokenizer("Hello, world!", return_tensors="pt")

    with torch.no_grad():
        embed_cpu = model.model.embed_tokens(inputs["input_ids"])

        # Get position embeddings
        position_ids = torch.arange(embed_cpu.shape[1]).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(embed_cpu, position_ids)

        layer_cpu = model.model.layers[0]

        # Run layer forward on CPU
        out_cpu, = layer_cpu(embed_cpu, position_ids=position_ids, position_embeddings=position_embeddings)

        print(f"Layer forward CPU output: shape={out_cpu.shape}, max={out_cpu.max():.4f}, min={out_cpu.min():.4f}")

        # Now move layer to GPU and run
        layer_gpu = model.model.layers[0].to(device)
        embed_gpu = embed_cpu.to(device)
        position_ids_gpu = position_ids.to(device)
        position_embeddings_gpu = (position_embeddings[0].to(device), position_embeddings[1].to(device))

        out_gpu, = layer_gpu(embed_gpu, position_ids=position_ids_gpu, position_embeddings=position_embeddings_gpu)
        out_gpu_cpu = out_gpu.to("cpu")

        print(f"Layer forward GPU output: shape={out_gpu_cpu.shape}, max={out_gpu_cpu.max():.4f}, min={out_gpu_cpu.min():.4f}")

        compare_tensors("Layer forward", out_cpu, out_gpu)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["rmsnorm", "attention", "manual", "forward", "all"], default="all")
    args = parser.parse_args()

    if args.test in ["rmsnorm", "all"]:
        test_single_rmsnorm()

    if args.test in ["attention", "all"]:
        test_attention_components()

    if args.test in ["manual", "all"]:
        test_full_layer_manual()

    if args.test in ["forward", "all"]:
        test_layer_forward_vs_manual()
