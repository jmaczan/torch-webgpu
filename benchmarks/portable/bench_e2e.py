#!/usr/bin/env python
"""
Portable end-to-end LLM benchmark for Qwen2.5-0.5B/1.5B-Instruct.

Auto-detects available backends (CUDA, MPS, CPU) and runs benchmarks
on each. Works on Windows, macOS, and Linux.

Usage:
    pip install torch transformers scipy
    python bench_e2e.py                          # All backends, 0.5B
    python bench_e2e.py --model 1.5B             # All backends, 1.5B
    python bench_e2e.py --backends cuda cpu       # Specific backends
    python bench_e2e.py --output-dir results/     # Custom output directory

Output: JSON files per backend matching paper methodology (30 runs, 50 tokens).
"""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
}


def get_system_info():
    """Get system information for reproducibility."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_count": torch.get_num_threads(),
    }

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

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
        )
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            info["nvidia_driver"] = result.stdout.strip()
        except Exception:
            info["nvidia_driver"] = "Unknown"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_available"] = True

    return info


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate mean, std, and confidence interval using t-distribution."""
    n = len(data)
    if n < 2:
        mean = data[0] if data else 0
        return {"mean": mean, "std": 0, "ci_lower": mean, "ci_upper": mean, "n": n}

    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)  # Bessel's correction
    std = math.sqrt(variance)

    try:
        from scipy import stats

        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    except ImportError:
        t_value = 2.045 if n == 30 else (2.0 if n < 30 else 1.96)

    std_error = std / math.sqrt(n)
    margin = t_value * std_error

    return {
        "mean": mean,
        "std": std,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
        "n": n,
    }


def sync_device(device_type):
    """Synchronize GPU device."""
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def detect_backends():
    """Detect available backends."""
    backends = ["cpu"]
    if torch.cuda.is_available():
        backends.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backends.append("mps")
    return backends


def load_model(model_name, backend, verbose=True):
    """Load model on specified backend."""
    if verbose:
        print(f"Loading {model_name} on {backend}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if backend == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cuda",
            trust_remote_code=True,
        )
    elif backend == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    model.eval()

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model loaded: {total_params / 1e9:.2f}B parameters on {backend}")

    return model, tokenizer


def benchmark_inference(
    model,
    tokenizer,
    prompt,
    n_tokens=50,
    warmup=5,
    runs=30,
    verbose=True,
):
    """Benchmark token-by-token generation with accurate TTFT measurement.

    Uses manual token-by-token generation (not model.generate()) to enable
    precise TTFT measurement and per-token timing consistent with the paper
    methodology.
    """
    device = next(model.parameters()).device
    device_type = device.type

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    input_length = input_ids.shape[1]

    if verbose:
        print(f"  Input: '{prompt}' ({input_length} tokens)")
        print(f"  Generating {n_tokens} tokens, {runs} runs after {warmup} warmup")

    # Warmup
    if verbose:
        print("  Warming up...")
    for i in range(warmup):
        generated_ids = input_ids.clone()
        with torch.no_grad():
            for _ in range(min(5, n_tokens)):
                sync_device(device_type)
                outputs = model(generated_ids)
                next_token_logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
        sync_device(device_type)
        if verbose:
            print(f"    Warmup {i + 1}/{warmup} done")

    # Timed runs
    times = []
    ttft_times = []
    tokens_generated = []

    if verbose:
        print("  Benchmarking...")

    for run_idx in range(runs):
        generated_ids = input_ids.clone()
        tokens_this_run = 0

        sync_device(device_type)
        run_start = time.perf_counter()
        first_token_time = None

        with torch.no_grad():
            for tok_idx in range(n_tokens):
                outputs = model(generated_ids)
                next_token_logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)

                if first_token_time is None:
                    sync_device(device_type)
                    first_token_time = time.perf_counter()

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_this_run += 1

                if next_token.item() == tokenizer.eos_token_id:
                    break

        sync_device(device_type)
        run_end = time.perf_counter()

        run_time = run_end - run_start
        ttft = first_token_time - run_start if first_token_time else run_time

        times.append(run_time)
        ttft_times.append(ttft)
        tokens_generated.append(tokens_this_run)

        if verbose:
            tps = tokens_this_run / run_time
            print(
                f"    Run {run_idx + 1}/{runs}: {tokens_this_run} tok in {run_time:.3f}s = {tps:.2f} tok/s (TTFT: {ttft * 1000:.1f}ms)"
            )

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
        "coefficient_of_variation": (tps_stats["std"] / tps_stats["mean"] * 100)
        if tps_stats["mean"] > 0
        else 0,
        "time_to_first_token_ms": ttft_stats["mean"],
        "time_to_first_token_std_ms": ttft_stats["std"],
        "time_to_first_token_ci95_ms": [ttft_stats["ci_lower"], ttft_stats["ci_upper"]],
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "n_tokens_requested": n_tokens,
        "avg_tokens_generated": sum(tokens_generated) / len(tokens_generated),
        "runs": runs,
        "warmup": warmup,
        "input_tokens": input_length,
        "all_tps": tps_per_run,
        "all_ttft_ms": [t * 1000 for t in ttft_times],
    }

    return results


def run_backend(model_name, backend, args, system_info):
    """Run benchmark for a single backend and return results."""
    warmup = 5 if backend in ("cuda", "mps") else 3

    model, tokenizer = load_model(model_name, backend, verbose=not args.quiet)

    results = benchmark_inference(
        model,
        tokenizer,
        prompt=args.prompt,
        n_tokens=args.n_tokens,
        warmup=warmup,
        runs=args.runs,
        verbose=not args.quiet,
    )

    # Add metadata
    results["model"] = model_name
    results["backend"] = f"pytorch-{backend}"
    results["system_info"] = system_info
    results["prompt"] = args.prompt

    # Print summary
    print()
    print(f"  {'=' * 50}")
    print(f"  RESULTS ({backend.upper()})")
    print(f"  {'=' * 50}")
    print(
        f"  Tokens/second:       {results['tokens_per_second']:.2f} +/- {results['tokens_per_second_std']:.2f}"
    )
    ci = results["tokens_per_second_ci95"]
    print(f"    95% CI:            [{ci[0]:.2f}, {ci[1]:.2f}]")
    print(f"  CV:                  {results['coefficient_of_variation']:.1f}%")
    print(f"  Time to first token: {results['time_to_first_token_ms']:.2f} ms")
    print(f"  {'=' * 50}")

    # Clean up GPU memory
    del model
    del tokenizer
    if backend == "cuda":
        torch.cuda.empty_cache()
    elif backend == "mps":
        torch.mps.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Portable E2E LLM benchmark (CUDA/MPS/CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python bench_e2e.py                           # All backends, 0.5B, 30 runs
    python bench_e2e.py --model 1.5B              # All backends, 1.5B
    python bench_e2e.py --backends cuda cpu        # Only CUDA and CPU
    python bench_e2e.py --runs 10 --n-tokens 32   # Quick test
    python bench_e2e.py --output-dir ./results     # Custom output directory
""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="0.5B",
        choices=["0.5B", "1.5B"],
        help="Model size (default: 0.5B)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Backends to benchmark (default: auto-detect all available)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for JSON results (default: current dir)",
    )
    parser.add_argument(
        "--n-tokens", type=int, default=50, help="Tokens per run (default: 50)"
    )
    parser.add_argument(
        "--runs", type=int, default=30, help="Benchmark runs (default: 30)"
    )
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    model_name = MODELS[args.model]
    model_tag = args.model.replace(".", "").lower()

    # Detect backends
    available = detect_backends()
    if args.backends:
        backends = [b for b in args.backends if b in available]
        skipped = [b for b in args.backends if b not in available]
        if skipped:
            print(f"WARNING: Backends not available: {skipped}")
    else:
        backends = available

    if not backends:
        print("ERROR: No backends available!")
        return

    print("=" * 60)
    print(f"Qwen2.5-{args.model}-Instruct E2E Benchmark")
    print("=" * 60)
    print(f"Model:    {model_name}")
    print(f"Backends: {backends}")
    print(f"Runs:     {args.runs}")
    print(f"Tokens:   {args.n_tokens}")
    print()

    system_info = get_system_info()
    if not args.quiet:
        print(f"Platform: {system_info['platform']}")
        print(f"CPU:      {system_info.get('cpu_brand', 'Unknown')}")
        if "gpu_name" in system_info:
            print(f"GPU:      {system_info['gpu_name']}")
        print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for backend in backends:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {backend.upper()}")
        print(f"{'=' * 60}")

        results = run_backend(model_name, backend, args, system_info)
        all_results[backend] = results

        # Save individual result
        output_file = output_dir / f"results_{backend}_{model_tag}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to: {output_file}")

    # Print comparison summary
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Backend':<15} {'Tok/s':>10} {'95% CI':>20} {'CV':>8} {'TTFT (ms)':>12}")
    print("-" * 65)

    for backend, results in all_results.items():
        ci = results["tokens_per_second_ci95"]
        print(
            f"{backend:<15} {results['tokens_per_second']:>10.2f} "
            f"[{ci[0]:>8.2f}, {ci[1]:>8.2f}] "
            f"{results['coefficient_of_variation']:>7.1f}% "
            f"{results['time_to_first_token_ms']:>11.1f}"
        )
    print(f"{'=' * 60}")

    # Save combined summary
    summary_file = output_dir / f"summary_{model_tag}.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "model": model_name,
                "model_size": args.model,
                "system_info": system_info,
                "backends": {
                    k: {
                        "tokens_per_second": v["tokens_per_second"],
                        "tokens_per_second_ci95": v["tokens_per_second_ci95"],
                        "coefficient_of_variation": v["coefficient_of_variation"],
                        "time_to_first_token_ms": v["time_to_first_token_ms"],
                    }
                    for k, v in all_results.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
