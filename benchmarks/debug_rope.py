#!/usr/bin/env python
"""Debug rotary position embeddings (RoPE) on WebGPU."""

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

    return is_close


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embedding to Q and K tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def test_rotary_embeddings():
    """Test rotary embeddings step by step."""
    print("\n" + "=" * 60)
    print("Testing Rotary Position Embeddings")
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
        print(f"\nEmbedding: shape={embed_cpu.shape}")

        layer = model.model.layers[0]

        # Get config values
        num_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        head_dim = layer.self_attn.head_dim

        bsz, seq_len, _ = embed_cpu.shape

        # Step 1: Get rotary embeddings
        print("\n--- Step 1: Get rotary embeddings (cos, sin) ---")
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cos_cpu, sin_cpu = model.model.rotary_emb(embed_cpu, position_ids)

        print(f"cos shape: {cos_cpu.shape}, sin shape: {sin_cpu.shape}")
        print(f"cos: min={cos_cpu.min():.4f}, max={cos_cpu.max():.4f}")
        print(f"sin: min={sin_cpu.min():.4f}, max={sin_cpu.max():.4f}")

        # Move to GPU
        cos_gpu = cos_cpu.to(device)
        sin_gpu = sin_cpu.to(device)

        compare_tensors("cos", cos_cpu, cos_gpu)
        compare_tensors("sin", sin_cpu, sin_gpu)

        # Step 2: Apply input layernorm
        print("\n--- Step 2: Input LayerNorm ---")
        weight = layer.input_layernorm.weight.data
        eps = layer.input_layernorm.variance_epsilon

        var_cpu = embed_cpu.pow(2).mean(dim=-1, keepdim=True)
        normed_cpu = embed_cpu * torch.rsqrt(var_cpu + eps) * weight

        embed_gpu = embed_cpu.to(device)
        weight_gpu = weight.to(device)
        var_gpu = embed_gpu.pow(2).mean(dim=-1, keepdim=True)
        normed_gpu = embed_gpu * torch.rsqrt(var_gpu + eps) * weight_gpu

        compare_tensors("input_layernorm", normed_cpu, normed_gpu)

        # Step 3: Q, K projections
        print("\n--- Step 3: Q, K Projections ---")
        q_cpu = F.linear(normed_cpu, layer.self_attn.q_proj.weight)
        k_cpu = F.linear(normed_cpu, layer.self_attn.k_proj.weight)

        q_w_gpu = layer.self_attn.q_proj.weight.to(device)
        k_w_gpu = layer.self_attn.k_proj.weight.to(device)

        q_gpu = F.linear(normed_gpu, q_w_gpu)
        k_gpu = F.linear(normed_gpu, k_w_gpu)

        compare_tensors("Q projection", q_cpu, q_gpu)
        compare_tensors("K projection", k_cpu, k_gpu)

        # Step 4: Reshape Q, K for attention
        print("\n--- Step 4: Reshape for attention ---")
        q_cpu_reshaped = q_cpu.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k_cpu_reshaped = k_cpu.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        q_gpu_reshaped = q_gpu.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k_gpu_reshaped = k_gpu.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        compare_tensors("Q reshaped", q_cpu_reshaped, q_gpu_reshaped)
        compare_tensors("K reshaped", k_cpu_reshaped, k_gpu_reshaped)

        # Step 5: Apply rotary embeddings
        print("\n--- Step 5: Apply rotary embeddings ---")

        # Step 5a: rotate_half
        print("\nStep 5a: rotate_half(Q)")
        q_rot_half_cpu = rotate_half(q_cpu_reshaped)
        q_rot_half_gpu = rotate_half(q_gpu_reshaped)
        compare_tensors("rotate_half(Q)", q_rot_half_cpu, q_rot_half_gpu)

        print("\nStep 5b: rotate_half(K)")
        k_rot_half_cpu = rotate_half(k_cpu_reshaped)
        k_rot_half_gpu = rotate_half(k_gpu_reshaped)
        compare_tensors("rotate_half(K)", k_rot_half_cpu, k_rot_half_gpu)

        # Step 5c: Unsqueeze cos/sin
        print("\nStep 5c: Unsqueeze cos/sin")
        cos_cpu_unsq = cos_cpu.unsqueeze(1)
        sin_cpu_unsq = sin_cpu.unsqueeze(1)
        cos_gpu_unsq = cos_gpu.unsqueeze(1)
        sin_gpu_unsq = sin_gpu.unsqueeze(1)

        print(f"cos unsqueezed shape: {cos_cpu_unsq.shape}")
        compare_tensors("cos unsqueezed", cos_cpu_unsq, cos_gpu_unsq)
        compare_tensors("sin unsqueezed", sin_cpu_unsq, sin_gpu_unsq)

        # Step 5d: Q * cos
        print("\nStep 5d: Q * cos")
        q_times_cos_cpu = q_cpu_reshaped * cos_cpu_unsq
        q_times_cos_gpu = q_gpu_reshaped * cos_gpu_unsq
        compare_tensors("Q * cos", q_times_cos_cpu, q_times_cos_gpu)

        # Step 5e: rotate_half(Q) * sin
        print("\nStep 5e: rotate_half(Q) * sin")
        q_rot_times_sin_cpu = q_rot_half_cpu * sin_cpu_unsq
        q_rot_times_sin_gpu = q_rot_half_gpu * sin_gpu_unsq
        compare_tensors("rotate_half(Q) * sin", q_rot_times_sin_cpu, q_rot_times_sin_gpu)

        # Step 5f: Q_embed = Q * cos + rotate_half(Q) * sin
        print("\nStep 5f: Q_embed = Q * cos + rotate_half(Q) * sin")
        q_embed_cpu = q_times_cos_cpu + q_rot_times_sin_cpu
        q_embed_gpu = q_times_cos_gpu + q_rot_times_sin_gpu
        compare_tensors("Q_embed", q_embed_cpu, q_embed_gpu)

        # Step 5g: Same for K
        print("\nStep 5g: K_embed")
        k_embed_cpu = (k_cpu_reshaped * cos_cpu_unsq) + (k_rot_half_cpu * sin_cpu_unsq)
        k_embed_gpu = (k_gpu_reshaped * cos_gpu_unsq) + (k_rot_half_gpu * sin_gpu_unsq)
        compare_tensors("K_embed", k_embed_cpu, k_embed_gpu)

        print("\n" + "=" * 60)
        print("Rotary embedding test complete!")


def test_attention_with_rope():
    """Test full attention with rotary embeddings."""
    print("\n" + "=" * 60)
    print("Testing Full Attention with RoPE")
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
        print(f"\nEmbedding: shape={embed_cpu.shape}")

        layer = model.model.layers[0]

        # Get config values
        num_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        head_dim = layer.self_attn.head_dim
        num_kv_groups = num_heads // num_kv_heads

        bsz, seq_len, _ = embed_cpu.shape

        # Get rotary embeddings
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cos_cpu, sin_cpu = model.model.rotary_emb(embed_cpu, position_ids)

        # Input layernorm
        weight = layer.input_layernorm.weight.data
        eps = layer.input_layernorm.variance_epsilon
        var_cpu = embed_cpu.pow(2).mean(dim=-1, keepdim=True)
        normed_cpu = embed_cpu * torch.rsqrt(var_cpu + eps) * weight

        # Q, K, V projections
        q_cpu = F.linear(normed_cpu, layer.self_attn.q_proj.weight)
        k_cpu = F.linear(normed_cpu, layer.self_attn.k_proj.weight)
        v_cpu = F.linear(normed_cpu, layer.self_attn.v_proj.weight)

        # Reshape
        q_cpu = q_cpu.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k_cpu = k_cpu.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v_cpu = v_cpu.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos_unsq = cos_cpu.unsqueeze(1)
        sin_unsq = sin_cpu.unsqueeze(1)
        q_cpu = (q_cpu * cos_unsq) + (rotate_half(q_cpu) * sin_unsq)
        k_cpu = (k_cpu * cos_unsq) + (rotate_half(k_cpu) * sin_unsq)

        # Repeat KV for GQA
        k_cpu = k_cpu.repeat_interleave(num_kv_groups, dim=1)
        v_cpu = v_cpu.repeat_interleave(num_kv_groups, dim=1)

        # Attention
        scale = head_dim ** -0.5
        attn_scores_cpu = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) * scale
        attn_weights_cpu = F.softmax(attn_scores_cpu, dim=-1)
        attn_out_cpu = torch.matmul(attn_weights_cpu, v_cpu)

        # Output projection
        attn_out_cpu = attn_out_cpu.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        proj_out_cpu = F.linear(attn_out_cpu, layer.self_attn.o_proj.weight)

        # Residual
        hidden_cpu = embed_cpu + proj_out_cpu

        print(f"Attention output (CPU): min={hidden_cpu.min():.4f}, max={hidden_cpu.max():.4f}")

        # Now GPU
        embed_gpu = embed_cpu.to(device)
        cos_gpu = cos_cpu.to(device)
        sin_gpu = sin_cpu.to(device)

        weight_gpu = weight.to(device)
        var_gpu = embed_gpu.pow(2).mean(dim=-1, keepdim=True)
        normed_gpu = embed_gpu * torch.rsqrt(var_gpu + eps) * weight_gpu

        q_w_gpu = layer.self_attn.q_proj.weight.to(device)
        k_w_gpu = layer.self_attn.k_proj.weight.to(device)
        v_w_gpu = layer.self_attn.v_proj.weight.to(device)
        o_w_gpu = layer.self_attn.o_proj.weight.to(device)

        q_gpu = F.linear(normed_gpu, q_w_gpu)
        k_gpu = F.linear(normed_gpu, k_w_gpu)
        v_gpu = F.linear(normed_gpu, v_w_gpu)

        q_gpu = q_gpu.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k_gpu = k_gpu.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v_gpu = v_gpu.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        cos_unsq_gpu = cos_gpu.unsqueeze(1)
        sin_unsq_gpu = sin_gpu.unsqueeze(1)
        q_gpu = (q_gpu * cos_unsq_gpu) + (rotate_half(q_gpu) * sin_unsq_gpu)
        k_gpu = (k_gpu * cos_unsq_gpu) + (rotate_half(k_gpu) * sin_unsq_gpu)

        k_gpu = k_gpu.repeat_interleave(num_kv_groups, dim=1)
        v_gpu = v_gpu.repeat_interleave(num_kv_groups, dim=1)

        attn_scores_gpu = torch.matmul(q_gpu, k_gpu.transpose(-2, -1)) * scale
        attn_weights_gpu = F.softmax(attn_scores_gpu, dim=-1)
        attn_out_gpu = torch.matmul(attn_weights_gpu, v_gpu)

        attn_out_gpu = attn_out_gpu.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        proj_out_gpu = F.linear(attn_out_gpu, o_w_gpu)

        hidden_gpu = embed_gpu + proj_out_gpu

        print(f"Attention output (GPU): min={hidden_gpu.to('cpu').min():.4f}, max={hidden_gpu.to('cpu').max():.4f}")

        compare_tensors("Full attention with RoPE", hidden_cpu, hidden_gpu)


if __name__ == "__main__":
    test_rotary_embeddings()
    test_attention_with_rope()
