#!/usr/bin/env python
"""Test different attention implementations on WebGPU."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch_webgpu  # noqa

from transformers import AutoModelForCausalLM, AutoTokenizer


def compare_tensors(name, cpu_tensor, gpu_tensor, rtol=1e-3, atol=1e-3):
    """Compare CPU and GPU tensors."""
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


def test_layer_with_attn_impl(attn_implementation=None):
    """Test layer forward with specific attention implementation."""
    impl_name = attn_implementation or "default"
    print(f"\n{'=' * 60}")
    print(f"Testing layer.forward() with attn_implementation={impl_name}")
    print("=" * 60)

    device = torch.device("webgpu")

    print("Loading model...")
    kwargs = {"torch_dtype": torch.float32}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        **kwargs
    )
    model.eval()

    print(f"Actual attn_implementation: {model.config._attn_implementation}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    inputs = tokenizer("Hello, world!", return_tensors="pt")

    with torch.no_grad():
        embed_cpu = model.model.embed_tokens(inputs["input_ids"])

        # Get position embeddings
        position_ids = torch.arange(embed_cpu.shape[1]).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(embed_cpu, position_ids)

        layer_cpu = model.model.layers[0]

        # Run layer forward on CPU
        layer_output = layer_cpu(embed_cpu, position_ids=position_ids, position_embeddings=position_embeddings)
        out_cpu = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        print(f"Layer forward CPU output: shape={out_cpu.shape}, max={out_cpu.max():.4f}, min={out_cpu.min():.4f}")

        # Now move layer to GPU and run
        layer_gpu = model.model.layers[0].to(device)
        embed_gpu = embed_cpu.to(device)
        position_ids_gpu = position_ids.to(device)
        position_embeddings_gpu = (position_embeddings[0].to(device), position_embeddings[1].to(device))

        layer_output_gpu = layer_gpu(embed_gpu, position_ids=position_ids_gpu, position_embeddings=position_embeddings_gpu)
        out_gpu = layer_output_gpu[0] if isinstance(layer_output_gpu, tuple) else layer_output_gpu
        out_gpu_cpu = out_gpu.to("cpu")

        print(f"Layer forward GPU output: shape={out_gpu_cpu.shape}, max={out_gpu_cpu.max():.4f}, min={out_gpu_cpu.min():.4f}")

        is_close = compare_tensors("Layer forward", out_cpu, out_gpu)
        return is_close


def test_full_model_with_attn_impl(attn_implementation=None):
    """Test full model forward with specific attention implementation."""
    impl_name = attn_implementation or "default"
    print(f"\n{'=' * 60}")
    print(f"Testing full model.forward() with attn_implementation={impl_name}")
    print("=" * 60)

    device = torch.device("webgpu")

    print("Loading model...")
    kwargs = {"torch_dtype": torch.float32}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        **kwargs
    )
    model.eval()

    print(f"Actual attn_implementation: {model.config._attn_implementation}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    inputs = tokenizer("Hello, world!", return_tensors="pt")

    with torch.no_grad():
        # CPU forward
        out_cpu = model(inputs["input_ids"])
        logits_cpu = out_cpu.logits

        print(f"Model forward CPU logits: shape={logits_cpu.shape}, max={logits_cpu.max():.4f}, min={logits_cpu.min():.4f}")

        # GPU forward
        model_gpu = model.to(device)
        input_ids_gpu = inputs["input_ids"].to(device)

        out_gpu = model_gpu(input_ids_gpu)
        logits_gpu = out_gpu.logits.to("cpu")

        print(f"Model forward GPU logits: shape={logits_gpu.shape}, max={logits_gpu.max():.4f}, min={logits_gpu.min():.4f}")

        is_close = compare_tensors("Model forward logits", logits_cpu, logits_gpu)

        # Check if we get same predictions
        pred_cpu = logits_cpu[0, -1].argmax().item()
        pred_gpu = logits_gpu[0, -1].argmax().item()
        print(f"\n  Predicted token (CPU): {pred_cpu} = '{tokenizer.decode([pred_cpu])}'")
        print(f"  Predicted token (GPU): {pred_gpu} = '{tokenizer.decode([pred_gpu])}'")

        return is_close


def test_generate_with_attn_impl(attn_implementation=None):
    """Test model.generate() with specific attention implementation."""
    impl_name = attn_implementation or "default"
    print(f"\n{'=' * 60}")
    print(f"Testing model.generate() with attn_implementation={impl_name}")
    print("=" * 60)

    device = torch.device("webgpu")

    print("Loading model...")
    kwargs = {"torch_dtype": torch.float32}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        **kwargs
    )
    model.eval()

    print(f"Actual attn_implementation: {model.config._attn_implementation}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        # CPU generate
        output_cpu = model.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        text_cpu = tokenizer.decode(output_cpu[0], skip_special_tokens=True)
        print(f"CPU generation: {text_cpu}")

        # GPU generate
        model_gpu = model.to(device)
        input_ids_gpu = inputs["input_ids"].to(device)

        output_gpu = model_gpu.generate(
            input_ids_gpu,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        text_gpu = tokenizer.decode(output_gpu[0].to("cpu"), skip_special_tokens=True)
        print(f"GPU generation: {text_gpu}")

        is_correct = text_cpu == text_gpu
        print(f"\nGenerations match: {is_correct}")

        return is_correct


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["default", "eager", "sdpa"], default="eager")
    parser.add_argument("--test", choices=["layer", "model", "generate", "all"], default="all")
    args = parser.parse_args()

    attn_impl = None if args.impl == "default" else args.impl

    results = {}

    if args.test in ["layer", "all"]:
        results["layer"] = test_layer_with_attn_impl(attn_impl)

    if args.test in ["model", "all"]:
        results["model"] = test_full_model_with_attn_impl(attn_impl)

    if args.test in ["generate", "all"]:
        results["generate"] = test_generate_with_attn_impl(attn_impl)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for test, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test}: {status}")
