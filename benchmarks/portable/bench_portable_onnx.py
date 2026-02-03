#!/usr/bin/env python
"""
Portable ONNX Runtime benchmark for Qwen2.5-0.5B-Instruct.

Supports multiple execution providers:
- CUDA
- CPU

Works on any platform with pip-installable packages.

Usage:
    pip install onnxruntime-gpu transformers optimum[onnxruntime]
    python bench_portable_onnx.py --provider cuda --output results_onnx_cuda.json
    python bench_portable_onnx.py --provider cpu --output results_onnx_cpu.json
"""

import argparse
import json
import math
import platform
import time
from pathlib import Path


def get_system_info():
    """Get system information for reproducibility."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    try:
        import onnxruntime as ort
        info["onnxruntime_version"] = ort.__version__
        info["available_providers"] = ort.get_available_providers()
    except ImportError:
        info["onnxruntime_version"] = "Not installed"
        info["available_providers"] = []

    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        info["cpu_brand"] = cpu.get("brand_raw", "Unknown")
    except ImportError:
        info["cpu_brand"] = platform.processor()

    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        info["ram_gb"] = "Unknown"

    return info


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate mean, std, and confidence interval."""
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std = math.sqrt(variance)

    try:
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    except ImportError:
        t_value = 2.0 if n < 30 else 1.96

    std_error = std / math.sqrt(n)
    margin = t_value * std_error

    return {
        "mean": mean,
        "std": std,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
        "n": n,
    }


def benchmark_onnx_inference(
    model,
    tokenizer,
    prompt: str,
    n_tokens: int = 50,
    warmup: int = 3,
    runs: int = 30,
    verbose: bool = True,
):
    """Benchmark ONNX model inference using optimum's generate()."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]

    if verbose:
        print(f"Input prompt: '{prompt}'")
        print(f"Input tokens: {input_length}")
        print(f"Generating {n_tokens} tokens, {runs} runs after {warmup} warmup")
        print()

    # Warmup
    if verbose:
        print("Warming up...")
    for i in range(warmup):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=min(5, n_tokens),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if verbose:
            print(f"  Warmup {i+1}/{warmup} done")

    # Timed runs
    times = []
    ttft_times = []
    tokens_generated = []

    if verbose:
        print("\nBenchmarking...")

    for run_idx in range(runs):
        run_start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        run_end = time.perf_counter()

        # Count generated tokens (excluding input)
        tokens_this_run = outputs.shape[1] - input_length
        run_time = run_end - run_start

        # Estimate TTFT as time / tokens (approximation for generate())
        ttft = run_time / tokens_this_run if tokens_this_run > 0 else run_time

        times.append(run_time)
        ttft_times.append(ttft)
        tokens_generated.append(tokens_this_run)

        if verbose:
            tps = tokens_this_run / run_time
            print(f"  Run {run_idx+1}/{runs}: {tokens_this_run} tokens in {run_time:.3f}s = {tps:.2f} tok/s")

    # Calculate statistics
    tps_per_run = [t / tm for t, tm in zip(tokens_generated, times)]
    tps_stats = calculate_confidence_interval(tps_per_run)
    ttft_stats = calculate_confidence_interval([t * 1000 for t in ttft_times])

    total_tokens = sum(tokens_generated)
    total_time = sum(times)

    results = {
        "tokens_per_second": tps_stats["mean"],
        "tokens_per_second_std": tps_stats["std"],
        "tokens_per_second_ci95": [tps_stats["ci_lower"], tps_stats["ci_upper"]],
        "coefficient_of_variation": (tps_stats["std"] / tps_stats["mean"] * 100) if tps_stats["mean"] > 0 else 0,
        "time_to_first_token_ms": ttft_stats["mean"],
        "time_to_first_token_std_ms": ttft_stats["std"],
        "time_to_first_token_ci95_ms": [ttft_stats["ci_lower"], ttft_stats["ci_upper"]],
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "n_tokens_requested": n_tokens,
        "runs": runs,
        "warmup": warmup,
        "input_tokens": input_length,
        "all_tps": tps_per_run,
        "all_ttft_ms": [t * 1000 for t in ttft_times],
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Portable ONNX Runtime benchmark")
    parser.add_argument("--output", type=str, default="results_onnx_portable.json", help="Output JSON file")
    parser.add_argument("--provider", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="ONNX Runtime execution provider")
    parser.add_argument("--n-tokens", type=int, default=50, help="Tokens to generate per run")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=30, help="Benchmark runs")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Input prompt")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    verbose = not args.quiet
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    if verbose:
        print("=" * 60)
        print(f"Qwen2.5-0.5B-Instruct ONNX Benchmark ({args.provider.upper()})")
        print("=" * 60)
        print()

    # Get system info
    system_info = get_system_info()
    if verbose:
        print(f"Platform: {system_info['platform']}")
        print(f"CPU: {system_info['cpu_brand']}")
        print(f"ONNX Runtime: {system_info['onnxruntime_version']}")
        print(f"Available providers: {system_info['available_providers']}")
        print()

    # Import here after args parsing
    global torch
    import torch
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer

    # Load model with optimum (handles export automatically)
    if verbose:
        print(f"Loading {model_name} with ONNX Runtime...")

    # Set provider
    if args.provider == "cuda":
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    try:
        model = ORTModelForCausalLM.from_pretrained(
            model_name,
            export=True,
            provider=provider,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nMake sure you have installed:")
        print("  pip install optimum[onnxruntime] onnxruntime-gpu")
        return None

    if verbose:
        print(f"Model loaded with provider: {provider}")
        print()

    # Run benchmark
    results = benchmark_onnx_inference(
        model,
        tokenizer,
        prompt=args.prompt,
        n_tokens=args.n_tokens,
        warmup=args.warmup,
        runs=args.runs,
        verbose=verbose,
    )

    # Add metadata
    results["model"] = model_name
    results["backend"] = f"onnxruntime-{args.provider}"
    results["system_info"] = system_info
    results["prompt"] = args.prompt

    # Print summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Tokens/second:       {results['tokens_per_second']:.2f} +/- {results['tokens_per_second_std']:.2f}")
    ci = results.get('tokens_per_second_ci95', [])
    if ci:
        print(f"  95% CI:            [{ci[0]:.2f}, {ci[1]:.2f}]")
    print(f"CV:                  {results.get('coefficient_of_variation', 0):.1f}%")
    print(f"Time to first token: {results['time_to_first_token_ms']:.2f} ms")
    print("=" * 60)

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
