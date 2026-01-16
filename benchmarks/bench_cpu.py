#!/usr/bin/env python
"""Benchmark Qwen2.5-0.5B-Instruct on CPU (PyTorch eager mode)."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_cpu_info():
    """Get CPU information."""
    try:
        result = subprocess.run(
            ["lscpu"], capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        info = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
        return {
            "model": info.get("Model name", "Unknown"),
            "cores": info.get("CPU(s)", "Unknown"),
            "threads_per_core": info.get("Thread(s) per core", "Unknown"),
        }
    except Exception:
        return {"model": "Unknown", "cores": "Unknown"}


def benchmark_cpu(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "The capital of France is",
    n_tokens: int = 32,
    warmup: int = 3,
    runs: int = 10,
    output: str = None,
    quiet: bool = False,
):
    """Run CPU benchmark."""

    if not quiet:
        print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_tokens = input_ids.shape[1]

    if not quiet:
        print(f'Input: "{prompt}" ({input_tokens} tokens)')
        print(f"Generating {n_tokens} tokens per run, {runs} runs after {warmup} warmup")
        print()

    # Warmup
    if not quiet:
        print("Warming up...")
    for i in range(warmup):
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        if not quiet:
            print(f"  Warmup {i+1}/{warmup} done")

    if not quiet:
        print()
        print("Benchmarking...")

    # Benchmark runs
    times = []
    tokens_generated = []

    for i in range(runs):
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        end = time.perf_counter()

        elapsed = end - start
        gen_tokens = outputs.shape[1] - input_tokens
        tok_per_sec = gen_tokens / elapsed

        times.append(elapsed)
        tokens_generated.append(gen_tokens)

        if not quiet:
            # Estimate TTFT (first token time)
            ttft_ms = elapsed / gen_tokens * 1000
            print(f"  Run {i+1}/{runs}: {gen_tokens} tokens in {elapsed:.3f}s = {tok_per_sec:.2f} tok/s, TTFT: {ttft_ms:.2f}ms")

    # Calculate statistics
    total_tokens = sum(tokens_generated)
    total_time = sum(times)
    avg_tokens_per_sec = total_tokens / total_time
    std_tokens_per_sec = torch.tensor([t/e for t, e in zip(tokens_generated, times)]).std().item()
    avg_ttft_ms = sum([e/t*1000 for e, t in zip(times, tokens_generated)]) / len(times)

    results = {
        "tokens_per_second": avg_tokens_per_sec,
        "tokens_per_second_std": std_tokens_per_sec,
        "avg_time_per_run_s": total_time / runs,
        "time_to_first_token_ms": avg_ttft_ms,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "n_tokens_requested": n_tokens,
        "avg_tokens_generated": sum(tokens_generated) / len(tokens_generated),
        "runs": runs,
        "warmup": warmup,
        "input_tokens": input_tokens,
        "model": model_name,
        "backend": "cpu-eager",
        "hardware": get_cpu_info(),
        "prompt": prompt,
    }

    if not quiet:
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Tokens/second:       {avg_tokens_per_sec:.2f} (+/- {std_tokens_per_sec:.2f})")
        print(f"Time to first token: {avg_ttft_ms:.2f} ms")
        print(f"Avg time per run:    {total_time/runs:.3f} s")
        print(f"Total tokens:        {total_tokens}")
        print(f"Total time:          {total_time:.3f} s")
        print("=" * 60)

    # Save results
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        if not quiet:
            print(f"\nResults saved to: {output}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Qwen on CPU")
    parser.add_argument("--output", type=str, default="benchmarks/results_cpu.json",
                        help="Output JSON file")
    parser.add_argument("--n-tokens", type=int, default=32,
                        help="Number of tokens to generate per run")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of benchmark runs")
    parser.add_argument("--prompt", type=str, default="The capital of France is",
                        help="Input prompt")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output")

    args = parser.parse_args()
    benchmark_cpu(
        prompt=args.prompt,
        n_tokens=args.n_tokens,
        warmup=args.warmup,
        runs=args.runs,
        output=args.output,
        quiet=args.quiet,
    )
