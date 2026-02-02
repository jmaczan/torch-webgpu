#!/usr/bin/env python
"""
Benchmark script for Qwen2.5-0.5B-Instruct on WebGPU backend.

Measures:
- Tokens/second (generation throughput)
- Time to first token (latency)
- Peak memory usage
"""

import argparse
import json
import time
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for torch_webgpu import
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch_webgpu  # noqa - registers WebGPU device
import torch.nn as nn


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class FusedRMSNorm(nn.Module):
    """Fused RMSNorm that uses the optimized WebGPU kernel."""
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x):
        return torch.ops.webgpu.rms_norm(x, self.weight, self.eps)


class FusedMLP(nn.Module):
    """Fused MLP that combines gate+up projections with SiLU activation."""
    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        self.gate_weight = gate_proj.weight
        self.up_weight = up_proj.weight
        self.down_proj = down_proj

    def forward(self, x):
        # Fused: silu(x @ gate_weight.T) * (x @ up_weight.T)
        hidden = torch.ops.webgpu.fused_gate_up_silu(x, self.gate_weight, self.up_weight)
        return self.down_proj(hidden)


def optimize_model_for_webgpu(model):
    """Replace decomposed ops with fused WebGPU kernels for maximum performance."""
    rmsnorm_count = 0
    mlp_count = 0

    # Replace RMSNorm layers
    for name, module in list(model.named_modules()):
        if 'RMSNorm' in type(module).__name__:
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            fused = FusedRMSNorm(module.weight, eps=module.variance_epsilon)
            setattr(parent, parts[-1], fused)
            rmsnorm_count += 1

    # Replace MLP layers
    for name, module in list(model.named_modules()):
        if 'MLP' in type(module).__name__ and hasattr(module, 'gate_proj'):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            fused = FusedMLP(module.gate_proj, module.up_proj, module.down_proj)
            setattr(parent, parts[-1], fused)
            mlp_count += 1

    return model, rmsnorm_count + mlp_count


def get_gpu_info():
    """Get GPU information if available."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {
                "gpu": parts[0] if len(parts) > 0 else "Unknown",
                "driver": parts[1] if len(parts) > 1 else "Unknown",
                "memory_total": parts[2] if len(parts) > 2 else "Unknown",
            }
    except Exception:
        pass
    return {"gpu": "Unknown", "driver": "Unknown", "memory_total": "Unknown"}


def benchmark_inference(
    model,
    tokenizer,
    prompt: str,
    n_tokens: int = 50,
    warmup: int = 3,
    runs: int = 10,
    verbose: bool = True,
    device: str = "webgpu",
):
    """Benchmark token generation with the given model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    input_length = input_ids.shape[1]

    if verbose:
        print(f"Input prompt: '{prompt}'")
        print(f"Input tokens: {input_length}")
        print(f"Generating {n_tokens} tokens per run, {runs} runs after {warmup} warmup")
        print()

    # Warmup runs
    if verbose:
        print("Warming up...")
    for i in range(warmup):
        generated_ids = input_ids.clone()
        with torch.no_grad():
            for _ in range(min(5, n_tokens)):  # Short warmup generation
                outputs = model(generated_ids)
                next_token_logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
        if verbose:
            print(f"  Warmup {i+1}/{warmup} done")

    # Timed runs
    times = []
    ttft_times = []  # Time to first token
    tokens_generated = []

    if verbose:
        print("\nBenchmarking...")

    for run_idx in range(runs):
        generated_ids = input_ids.clone()
        tokens_this_run = 0

        run_start = time.perf_counter()
        first_token_time = None

        with torch.no_grad():
            for tok_idx in range(n_tokens):
                outputs = model(generated_ids)
                next_token_logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_this_run += 1

                # Check for EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

        run_end = time.perf_counter()

        run_time = run_end - run_start
        ttft = first_token_time - run_start if first_token_time else run_time

        times.append(run_time)
        ttft_times.append(ttft)
        tokens_generated.append(tokens_this_run)

        if verbose:
            tps = tokens_this_run / run_time
            print(f"  Run {run_idx+1}/{runs}: {tokens_this_run} tokens in {run_time:.3f}s = {tps:.2f} tok/s, TTFT: {ttft*1000:.2f}ms")

    # Calculate statistics
    total_tokens = sum(tokens_generated)
    total_time = sum(times)
    avg_tokens_per_run = total_tokens / runs

    tokens_per_second = total_tokens / total_time
    avg_time_per_run = total_time / runs
    avg_ttft = sum(ttft_times) / len(ttft_times)

    # Calculate std
    tps_per_run = [t / tm for t, tm in zip(tokens_generated, times)]
    tps_std = (sum((x - tokens_per_second)**2 for x in tps_per_run) / len(tps_per_run)) ** 0.5

    results = {
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_std": tps_std,
        "avg_time_per_run_s": avg_time_per_run,
        "time_to_first_token_ms": avg_ttft * 1000,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "n_tokens_requested": n_tokens,
        "avg_tokens_generated": avg_tokens_per_run,
        "runs": runs,
        "warmup": warmup,
        "input_tokens": input_length,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen2.5-0.5B-Instruct on WebGPU")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--n-tokens", type=int, default=50, help="Number of tokens to generate per run")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Input prompt")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print("Qwen2.5-0.5B-Instruct WebGPU Benchmark")
        print("=" * 60)
        print()

    # Get hardware info
    hw_info = get_gpu_info()
    if verbose:
        print(f"Hardware: {hw_info['gpu']}")
        print(f"Driver: {hw_info['driver']}")
        print()

    # Load model and tokenizer
    if verbose:
        print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    if verbose:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {total_params/1e9:.2f}B parameters")
        print()

    # Move model to WebGPU device
    if verbose:
        print("Moving model to WebGPU device...")
    webgpu_device = torch.device("webgpu")
    model = model.to(webgpu_device)

    # Optimize model by replacing decomposed ops with fused WebGPU kernels
    if verbose:
        print("Optimizing model with fused WebGPU kernels...")
    model, replaced = optimize_model_for_webgpu(model)
    if verbose:
        print(f"  Replaced {replaced} RMSNorm layers with fused version")
        print()

    # Run benchmark
    results = benchmark_inference(
        model,
        tokenizer,
        prompt=args.prompt,
        n_tokens=args.n_tokens,
        warmup=args.warmup,
        runs=args.runs,
        verbose=verbose,
        device=webgpu_device,
    )

    # Add metadata
    results["model"] = MODEL_NAME
    results["backend"] = "torch-webgpu"
    results["hardware"] = hw_info
    results["prompt"] = args.prompt

    # Print summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Tokens/second:       {results['tokens_per_second']:.2f} (+/- {results['tokens_per_second_std']:.2f})")
    print(f"Time to first token: {results['time_to_first_token_ms']:.2f} ms")
    print(f"Avg time per run:    {results['avg_time_per_run_s']:.3f} s")
    print(f"Total tokens:        {results['total_tokens']}")
    print(f"Total time:          {results['total_time_s']:.3f} s")
    print("=" * 60)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
