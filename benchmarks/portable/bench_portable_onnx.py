#!/usr/bin/env python
"""
Portable ONNX Runtime benchmark for Qwen2.5-0.5B-Instruct.

Supports multiple execution providers:
- WebGPU (via DirectML on Windows, Metal on Mac)
- CUDA
- CPU

Works on any platform with pip-installable packages.

Usage:
    # For WebGPU/DirectML (Windows):
    pip install onnxruntime-directml transformers optimum[onnxruntime]

    # For Metal (Mac):
    pip install onnxruntime transformers optimum[onnxruntime]

    # Then run:
    python bench_portable_onnx.py --provider dml --output results_onnx_dml.json
    python bench_portable_onnx.py --provider cpu --output results_onnx_cpu.json
"""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np


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


def export_model_to_onnx(model_name: str, output_dir: Path, verbose: bool = True):
    """Export model to ONNX format using optimum."""
    if verbose:
        print(f"Exporting {model_name} to ONNX...")

    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer

        # Export using optimum
        model = ORTModelForCausalLM.from_pretrained(
            model_name,
            export=True,
        )
        model.save_pretrained(output_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)

        if verbose:
            print(f"Model exported to {output_dir}")

        return True
    except Exception as e:
        if verbose:
            print(f"Export failed: {e}")
        return False


def load_onnx_model(model_path: Path, provider: str, verbose: bool = True):
    """Load ONNX model with specified execution provider."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    # Map provider names to ONNX Runtime provider names
    provider_map = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "dml": "DmlExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
        "webgpu": "WebGPUExecutionProvider",  # Not yet available in most builds
    }

    ort_provider = provider_map.get(provider.lower(), provider)

    available = ort.get_available_providers()
    if ort_provider not in available:
        if verbose:
            print(f"Warning: {ort_provider} not available. Available: {available}")
            print(f"Falling back to CPU")
        ort_provider = "CPUExecutionProvider"

    if verbose:
        print(f"Using execution provider: {ort_provider}")

    # Find model file
    model_file = model_path / "model.onnx"
    if not model_file.exists():
        # Try decoder model for causal LM
        model_file = model_path / "decoder_model.onnx"

    if not model_file.exists():
        raise FileNotFoundError(f"No ONNX model found in {model_path}")

    # Create session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(model_file),
        sess_options,
        providers=[ort_provider],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return session, tokenizer


def benchmark_onnx_inference(
    session,
    tokenizer,
    prompt: str,
    n_tokens: int = 50,
    warmup: int = 3,
    runs: int = 30,
    verbose: bool = True,
):
    """Benchmark ONNX model inference."""
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    input_length = input_ids.shape[1]

    if verbose:
        print(f"Input prompt: '{prompt}'")
        print(f"Input tokens: {input_length}")
        print(f"Generating {n_tokens} tokens, {runs} runs after {warmup} warmup")
        print()

    # Get input/output names
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    def generate_token(current_ids):
        """Generate a single token."""
        feed_dict = {"input_ids": current_ids}

        # Add attention mask if required
        if "attention_mask" in input_names:
            feed_dict["attention_mask"] = np.ones_like(current_ids)

        outputs = session.run(output_names, feed_dict)
        logits = outputs[0]

        # Get next token (greedy)
        next_token = np.argmax(logits[0, -1, :])
        return next_token

    # Warmup
    if verbose:
        print("Warming up...")
    for i in range(warmup):
        current_ids = input_ids.copy()
        for _ in range(min(5, n_tokens)):
            next_token = generate_token(current_ids)
            current_ids = np.concatenate([current_ids, [[next_token]]], axis=1)
        if verbose:
            print(f"  Warmup {i+1}/{warmup} done")

    # Timed runs
    times = []
    ttft_times = []
    tokens_generated = []

    if verbose:
        print("\nBenchmarking...")

    for run_idx in range(runs):
        current_ids = input_ids.copy()
        tokens_this_run = 0

        run_start = time.perf_counter()
        first_token_time = None

        for tok_idx in range(n_tokens):
            next_token = generate_token(current_ids)

            if first_token_time is None:
                first_token_time = time.perf_counter()

            current_ids = np.concatenate([current_ids, [[next_token]]], axis=1)
            tokens_this_run += 1

            if next_token == tokenizer.eos_token_id:
                break

        run_end = time.perf_counter()

        run_time = run_end - run_start
        ttft = first_token_time - run_start if first_token_time else run_time

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
                        choices=["cpu", "cuda", "dml", "coreml", "webgpu"],
                        help="ONNX Runtime execution provider")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Path to ONNX model directory (will export if not exists)")
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

    # Determine model path
    if args.model_dir:
        model_path = Path(args.model_dir)
    else:
        model_path = Path("qwen_onnx")

    # Export if needed
    if not model_path.exists() or not (model_path / "model.onnx").exists():
        if verbose:
            print(f"ONNX model not found at {model_path}, exporting...")
        model_path.mkdir(parents=True, exist_ok=True)
        success = export_model_to_onnx(model_name, model_path, verbose)
        if not success:
            print("Failed to export model. Please install optimum:")
            print("  pip install optimum[onnxruntime]")
            return None

    # Load model
    if verbose:
        print(f"Loading ONNX model from {model_path}...")
    session, tokenizer = load_onnx_model(model_path, args.provider, verbose)

    if verbose:
        print()

    # Run benchmark
    results = benchmark_onnx_inference(
        session,
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
