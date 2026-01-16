#!/usr/bin/env python
"""
Profile Qwen2.5-0.5B-Instruct inference to identify bottlenecks.
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_webgpu.compiler.webgpu_compiler import webgpu_backend


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def profile_single_forward(model, input_ids, n_runs=5):
    """Profile a single forward pass."""
    times = []

    # Warmup
    for _ in range(2):
        with torch.no_grad():
            _ = model(input_ids)

    # Profile
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "avg_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def profile_generation_loop(model, tokenizer, prompt, n_tokens=10):
    """Profile each step in the generation loop."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    step_times = []
    seq_lengths = []

    generated_ids = input_ids.clone()

    with torch.no_grad():
        for i in range(n_tokens):
            seq_len = generated_ids.shape[1]

            step_start = time.perf_counter()
            outputs = model(generated_ids)
            forward_end = time.perf_counter()

            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            argmax_end = time.perf_counter()

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            cat_end = time.perf_counter()

            step_times.append({
                "seq_len": seq_len,
                "forward_ms": (forward_end - step_start) * 1000,
                "argmax_ms": (argmax_end - forward_end) * 1000,
                "cat_ms": (cat_end - argmax_end) * 1000,
                "total_ms": (cat_end - step_start) * 1000,
            })
            seq_lengths.append(seq_len)

    return step_times


def analyze_model_structure(model):
    """Analyze model structure and parameter counts."""
    layer_params = defaultdict(int)
    total_params = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel

        # Categorize by layer type
        if "embed" in name.lower():
            layer_params["embedding"] += numel
        elif "lm_head" in name.lower():
            layer_params["lm_head"] += numel
        elif "self_attn" in name.lower() or "attention" in name.lower():
            layer_params["attention"] += numel
        elif "mlp" in name.lower() or "ffn" in name.lower():
            layer_params["mlp"] += numel
        elif "norm" in name.lower() or "ln" in name.lower():
            layer_params["norm"] += numel
        else:
            layer_params["other"] += numel

    return {
        "total_params": total_params,
        "layer_breakdown": dict(layer_params),
    }


def main():
    print("=" * 60)
    print("Profiling Qwen2.5-0.5B-Instruct")
    print("=" * 60)
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    # Analyze model structure
    print("\n--- Model Structure ---")
    structure = analyze_model_structure(model)
    print(f"Total parameters: {structure['total_params']/1e6:.2f}M")
    print("Parameter breakdown:")
    for layer_type, count in sorted(structure['layer_breakdown'].items(), key=lambda x: -x[1]):
        pct = count / structure['total_params'] * 100
        print(f"  {layer_type}: {count/1e6:.2f}M ({pct:.1f}%)")

    # Compile model
    print("\n--- Compiling with WebGPU backend ---")
    compiled_model = torch.compile(model, backend=webgpu_backend, dynamic=False)

    # Test different input lengths
    print("\n--- Forward Pass Profiling (varying sequence length) ---")
    for seq_len in [5, 10, 20, 50, 100]:
        # Create dummy input
        dummy_input = torch.randint(0, 1000, (1, seq_len))

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = compiled_model(dummy_input)

        # Profile
        times = []
        for _ in range(5):
            start = time.perf_counter()
            with torch.no_grad():
                _ = compiled_model(dummy_input)
            times.append(time.perf_counter() - start)

        avg_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        max_ms = max(times) * 1000
        print(f"Seq len {seq_len:3d}: avg={avg_ms:7.2f}ms, min={min_ms:7.2f}ms, max={max_ms:7.2f}ms")

    # Profile generation loop
    print("\n--- Generation Loop Profiling ---")
    prompt = "The capital of France is"
    step_times = profile_generation_loop(compiled_model, tokenizer, prompt, n_tokens=20)

    print(f"\nGeneration steps (prompt: '{prompt}'):")
    print(f"{'Step':>4} | {'Seq Len':>7} | {'Forward':>10} | {'Argmax':>8} | {'Cat':>8} | {'Total':>10}")
    print("-" * 60)

    forward_total = 0
    argmax_total = 0
    cat_total = 0

    for i, st in enumerate(step_times):
        forward_total += st["forward_ms"]
        argmax_total += st["argmax_ms"]
        cat_total += st["cat_ms"]
        print(f"{i:4d} | {st['seq_len']:7d} | {st['forward_ms']:8.2f}ms | {st['argmax_ms']:6.2f}ms | {st['cat_ms']:6.2f}ms | {st['total_ms']:8.2f}ms")

    print("-" * 60)
    total_time = sum(s["total_ms"] for s in step_times)
    print(f"\nTime breakdown:")
    print(f"  Forward pass: {forward_total:.2f}ms ({forward_total/total_time*100:.1f}%)")
    print(f"  Argmax:       {argmax_total:.2f}ms ({argmax_total/total_time*100:.1f}%)")
    print(f"  Cat:          {cat_total:.2f}ms ({cat_total/total_time*100:.1f}%)")
    print(f"  Total:        {total_time:.2f}ms")
    print(f"\nAvg tokens/sec: {len(step_times) / (total_time/1000):.2f}")

    # Scaling analysis
    print("\n--- Scaling Analysis ---")
    print("Forward time vs sequence length:")
    for st in step_times[:10]:
        print(f"  seq_len={st['seq_len']:3d} -> forward={st['forward_ms']:7.2f}ms")


if __name__ == "__main__":
    main()
