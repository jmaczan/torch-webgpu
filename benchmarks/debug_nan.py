#!/usr/bin/env python
"""Debug script to find where NaN is introduced in the model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch_webgpu  # noqa

from transformers import AutoModelForCausalLM, AutoTokenizer


def check_tensor(name, tensor):
    """Check if tensor contains NaN or Inf."""
    if tensor is None:
        return
    if hasattr(tensor, 'to'):
        t = tensor.to('cpu') if tensor.device.type != 'cpu' else tensor
    else:
        return

    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    max_val = t.max().item()
    min_val = t.min().item()

    status = ""
    if has_nan:
        status += " [NaN!]"
    if has_inf:
        status += " [Inf!]"

    print(f"  {name}: min={min_val:.4f}, max={max_val:.4f}{status}")
    return has_nan or has_inf


def debug_layer_0(model, device, tokenizer):
    """Debug layer 0 step by step."""
    inputs = tokenizer("Hello", return_tensors="pt")
    seq_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        input_ids = inputs['input_ids'].to(device)
        layer = model.model.layers[0]

        # Get embedding
        hidden = model.model.embed_tokens(input_ids)
        print(f"embed_tokens: shape={hidden.shape}")
        check_tensor("embed_tokens", hidden)

        # Get rotary embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = model.model.rotary_emb(hidden, position_ids)
        print(f"cos: shape={cos.shape}, sin: shape={sin.shape}")

        # Input layernorm
        normed = layer.input_layernorm(hidden)
        print(f"\ninput_layernorm: shape={normed.shape}")
        check_tensor("input_layernorm", normed)

        # Q, K, V projections
        q = layer.self_attn.q_proj(normed)
        k = layer.self_attn.k_proj(normed)
        v = layer.self_attn.v_proj(normed)
        print(f"\nQ: shape={q.shape}")
        check_tensor("Q projection", q)
        print(f"K: shape={k.shape}")
        check_tensor("K projection", k)
        print(f"V: shape={v.shape}")
        check_tensor("V projection", v)

        # Reshape for attention
        num_heads = model.config.num_attention_heads  # 14
        num_kv_heads = model.config.num_key_value_heads  # 2
        head_dim = layer.self_attn.head_dim  # 64

        bsz = q.shape[0]
        q_reshaped = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k_reshaped = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v_reshaped = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        print(f"\nQ reshaped: {q_reshaped.shape}")
        print(f"K reshaped: {k_reshaped.shape}")
        print(f"cos: {cos.shape}, sin: {sin.shape}")

        # Apply rotary embeddings (manually, avoiding transformers' function)
        # cos/sin are [batch, seq_len, head_dim] = [1, 1, 64]
        # Need to unsqueeze to [batch, 1, seq_len, head_dim] for broadcasting
        cos_unsq = cos.unsqueeze(1)  # [1, 1, 1, 64]
        sin_unsq = sin.unsqueeze(1)  # [1, 1, 1, 64]
        print(f"cos_unsq: {cos_unsq.shape}")

        # Q: [1, 14, 1, 64] * [1, 1, 1, 64] should broadcast fine
        print("\nApplying rotary embeddings manually...")

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q_reshaped * cos_unsq) + (rotate_half(q_reshaped) * sin_unsq)
        k_embed = (k_reshaped * cos_unsq) + (rotate_half(k_reshaped) * sin_unsq)

        print(f"Q after RoPE: {q_embed.shape}")
        check_tensor("Q after RoPE", q_embed)
        print(f"K after RoPE: {k_embed.shape}")
        check_tensor("K after RoPE", k_embed)

        # Repeat KV for GQA
        num_kv_groups = num_heads // num_kv_heads
        k_exp = k_embed.repeat_interleave(num_kv_groups, dim=1)
        v_exp = v_reshaped.repeat_interleave(num_kv_groups, dim=1)
        print(f"\nK expanded: {k_exp.shape}")

        # Attention scores
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q_embed, k_exp.transpose(-2, -1)) * scale
        print(f"\nAttn scores: {attn_scores.shape}")
        check_tensor("attn_scores", attn_scores)

        # Softmax
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        check_tensor("attn_weights", attn_weights)

        # Attention output
        attn_out = torch.matmul(attn_weights, v_exp)
        check_tensor("attn_out", attn_out)

        # Reshape and project
        attn_out_flat = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        proj_out = layer.self_attn.o_proj(attn_out_flat)
        check_tensor("o_proj", proj_out)

        # Residual
        hidden_after_attn = hidden + proj_out
        check_tensor("after_attn_residual", hidden_after_attn)

        # Post attention layernorm
        post_normed = layer.post_attention_layernorm(hidden_after_attn)
        check_tensor("post_attention_layernorm", post_normed)

        # MLP
        gate = layer.mlp.gate_proj(post_normed)
        up = layer.mlp.up_proj(post_normed)
        check_tensor("gate_proj", gate)
        check_tensor("up_proj", up)

        mlp_intermediate = torch.nn.functional.silu(gate) * up
        check_tensor("silu(gate) * up", mlp_intermediate)

        mlp_out = layer.mlp.down_proj(mlp_intermediate)
        check_tensor("down_proj", mlp_out)

        # Final residual
        hidden_final = hidden_after_attn + mlp_out
        check_tensor("layer_output", hidden_final)


def main():
    device = torch.device("webgpu")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Move model to GPU
    model_gpu = model.to(device)

    print("\n--- Debugging Layer 0 step by step ---")
    debug_layer_0(model_gpu, device, tokenizer)


def debug_layer(layer, input_before_layer, position_embeddings, layer_idx, device, seq_len):
    """Debug a single layer in detail."""
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # Get input (need to get it from the previous layer)
    # Actually we can't easily get the input before this layer
    # Let's just trace what's happening inside
    pass


if __name__ == "__main__":
    main()
