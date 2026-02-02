#!/usr/bin/env python
"""
Benchmark script for Qwen2.5-0.5B-Instruct using ONNX Runtime.

This benchmarks the ONNX model with:
- CPU ExecutionProvider (baseline)
- CUDA ExecutionProvider (if available)

Note: WebGPU ExecutionProvider is currently not available in the Python
ONNX Runtime package - it's only available in the JavaScript/WASM version.
"""

import argparse
import json
import time
import sys
from pathlib import Path

try:
    from optimum.onnxruntime import ORTModelForCausalLM
    HAS_OPTIMUM = True
except ImportError:
    HAS_OPTIMUM = False
    print("Warning: optimum not installed. Run: pip install optimum[onnxruntime]")

import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ONNX_MODEL_DIR = Path(__file__).parent / "qwen_onnx"


def get_available_providers():
    """Get list of available ONNX Runtime execution providers."""
    return ort.get_available_providers()


def get_hardware_info():
    """Get hardware information."""
    info = {}

    # CPU info
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu"] = line.split(":")[1].strip()
                    break
    except:
        info["cpu"] = "Unknown"

    # GPU info (if CUDA available)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            info["gpu"] = parts[0] if len(parts) > 0 else "Unknown"
            info["driver"] = parts[1] if len(parts) > 1 else "Unknown"
            info["memory_total"] = parts[2] if len(parts) > 2 else "Unknown"
    except:
        pass

    return info


def benchmark_onnx(
    model,
    tokenizer,
    prompt: str,
    n_tokens: int = 32,
    warmup: int = 3,
    runs: int = 10,
    verbose: bool = True,
):
    """Benchmark ONNX model token generation."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
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
        attn_mask = attention_mask.clone()
        for _ in range(min(5, n_tokens)):
            outputs = model(generated_ids, attention_mask=attn_mask)
            next_token_logits = outputs.logits[0, -1, :]
            if isinstance(next_token_logits, torch.Tensor):
                next_token = next_token_logits.argmax().unsqueeze(0).unsqueeze(0).cpu()
            else:
                import numpy as np
                next_token = torch.tensor([[np.argmax(next_token_logits)]])
            generated_ids = torch.cat([generated_ids.cpu(), next_token.cpu()], dim=1)
            attn_mask = torch.cat([attn_mask.cpu(), torch.ones((1, 1), dtype=attn_mask.dtype)], dim=1)
        if verbose:
            print(f"  Warmup {i+1}/{warmup} done")

    # Timed runs
    times = []
    ttft_times = []
    tokens_generated = []

    if verbose:
        print("\nBenchmarking...")

    for run_idx in range(runs):
        generated_ids = input_ids.clone()
        attn_mask = attention_mask.clone()
        tokens_this_run = 0

        run_start = time.perf_counter()
        first_token_time = None

        for tok_idx in range(n_tokens):
            outputs = model(generated_ids, attention_mask=attn_mask)
            next_token_logits = outputs.logits[0, -1, :]

            if isinstance(next_token_logits, torch.Tensor):
                next_token = next_token_logits.argmax().unsqueeze(0).unsqueeze(0).cpu()
            else:
                import numpy as np
                next_token = torch.tensor([[np.argmax(next_token_logits)]])

            if first_token_time is None:
                first_token_time = time.perf_counter()

            generated_ids = torch.cat([generated_ids.cpu(), next_token.cpu()], dim=1)
            attn_mask = torch.cat([attn_mask.cpu(), torch.ones((1, 1), dtype=attn_mask.dtype)], dim=1)
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
        "avg_tokens_generated": total_tokens / runs,
        "runs": runs,
        "warmup": warmup,
        "input_tokens": input_length,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen ONNX model")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--n-tokens", type=int, default=32, help="Number of tokens to generate per run")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Input prompt")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--provider", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="ONNX Runtime execution provider")
    parser.add_argument("--model-dir", type=str, default=str(ONNX_MODEL_DIR),
                        help="Path to ONNX model directory")
    args = parser.parse_args()

    verbose = not args.quiet

    if not HAS_OPTIMUM:
        print("Error: optimum package not installed")
        sys.exit(1)

    if verbose:
        print("=" * 60)
        print("Qwen2.5-0.5B-Instruct ONNX Runtime Benchmark")
        print("=" * 60)
        print()

    # Check available providers
    available = get_available_providers()
    if verbose:
        print(f"Available providers: {available}")

    # Select provider
    if args.provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            print("Error: CUDA provider not available")
            sys.exit(1)
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    if verbose:
        print(f"Using provider: {provider}")
        print()

    # Get hardware info
    hw_info = get_hardware_info()
    if verbose:
        print(f"Hardware: {hw_info}")
        print()

    # Check if ONNX model exists, if not export it
    model_dir = Path(args.model_dir)
    if not model_dir.exists() or not list(model_dir.glob("*.onnx")):
        if verbose:
            print(f"ONNX model not found at {model_dir}, exporting...")
        from export_qwen_onnx import export_model
        export_model(model_dir)

    # Load model
    if verbose:
        print(f"Loading ONNX model from {model_dir}...")

    if args.provider == "cuda":
        model = ORTModelForCausalLM.from_pretrained(
            model_dir,
            provider="CUDAExecutionProvider",
        )
    else:
        model = ORTModelForCausalLM.from_pretrained(
            model_dir,
            provider="CPUExecutionProvider",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if verbose:
        print("Model loaded.")
        print()

    # Run benchmark
    results = benchmark_onnx(
        model,
        tokenizer,
        prompt=args.prompt,
        n_tokens=args.n_tokens,
        warmup=args.warmup,
        runs=args.runs,
        verbose=verbose,
    )

    # Add metadata
    results["model"] = MODEL_NAME
    results["backend"] = f"onnxruntime-{args.provider}"
    results["provider"] = provider
    results["hardware"] = hw_info
    results["prompt"] = args.prompt

    # Print summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Provider:            {provider}")
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
