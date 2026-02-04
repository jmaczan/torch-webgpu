#!/usr/bin/env python3
"""
Reviewer Response Experiment 1: End-to-End LLM Inference on Multiple GPUs

This script addresses Major Reviewer Request #1:
"Conduct end-to-end inference benchmarks on additional GPUs: At minimum,
validate the 20.6 tok/s claim (or equivalent) on AMD and Apple Silicon"

This script attempts to run full Qwen2.5-0.5B inference on:
- Apple M2 (Metal backend)
- AMD GPU (Vulkan backend)
- Any available GPU via wgpu

Usage:
    # On Apple M2
    pip install torch transformers wgpu numpy scipy
    python reviewer_exp1_e2e_inference.py --output results/reviewer_e2e_m2.json

    # On AMD Linux
    python reviewer_exp1_e2e_inference.py --output results/reviewer_e2e_amd.json

IMPORTANT: This requires PyTorch with MPS (Apple) or CPU, plus wgpu for dispatch measurements.
Full WebGPU LLM inference requires torch-webgpu built for the platform.
"""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import wgpu for GPU info
try:
    import wgpu
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def get_system_info():
    """Collect comprehensive system information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    # Check available backends
    info["torch_backends"] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    }

    # Get GPU info via wgpu if available
    if WGPU_AVAILABLE:
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            if adapter:
                adapter_info = adapter.info
                info["wgpu_gpu"] = {
                    "vendor": adapter_info.get("vendor", "unknown"),
                    "device": adapter_info.get("device", "unknown"),
                    "description": adapter_info.get("description", "unknown"),
                    "backend": adapter_info.get("backend_type", "unknown"),
                }
        except Exception as e:
            info["wgpu_error"] = str(e)

    # Try to get CPU info
    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        info["cpu_brand"] = cpu.get("brand_raw", platform.processor())
    except ImportError:
        info["cpu_brand"] = platform.processor()

    return info


def calculate_stats(data):
    """Calculate mean, std, and 95% CI."""
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std = math.sqrt(variance)

    # t-value for 95% CI
    try:
        from scipy import stats
        t_value = stats.t.ppf(0.975, n - 1)
    except ImportError:
        t_value = 2.0 if n < 30 else 1.96

    std_error = std / math.sqrt(n)
    margin = t_value * std_error

    return {
        "mean": mean,
        "std": std,
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
        "cv_percent": (std / mean * 100) if mean > 0 else 0,
        "n": n,
    }


def benchmark_inference(model, tokenizer, device, prompt, n_tokens=50, warmup=5, runs=30, verbose=True):
    """Run LLM inference benchmark."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    input_length = input_ids.shape[1]

    if verbose:
        print(f"Input: '{prompt}' ({input_length} tokens)")
        print(f"Generating {n_tokens} tokens, {runs} runs after {warmup} warmup")

    # Warmup
    if verbose:
        print("Warming up...")
    for i in range(warmup):
        generated_ids = input_ids.clone()
        with torch.no_grad():
            for _ in range(min(5, n_tokens)):
                outputs = model(generated_ids)
                next_token_logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
        if verbose:
            print(f"  Warmup {i+1}/{warmup}")

    # Timed runs
    all_tps = []
    all_ttft = []

    if verbose:
        print("Benchmarking...")

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

                if next_token.item() == tokenizer.eos_token_id:
                    break

        run_end = time.perf_counter()
        run_time = run_end - run_start
        ttft = first_token_time - run_start if first_token_time else run_time

        tps = tokens_this_run / run_time
        all_tps.append(tps)
        all_ttft.append(ttft * 1000)  # ms

        if verbose:
            print(f"  Run {run_idx+1}/{runs}: {tps:.2f} tok/s, TTFT: {ttft*1000:.1f}ms")

    # Calculate statistics
    tps_stats = calculate_stats(all_tps)
    ttft_stats = calculate_stats(all_ttft)

    return {
        "tokens_per_second": tps_stats["mean"],
        "tokens_per_second_std": tps_stats["std"],
        "tokens_per_second_ci95": [tps_stats["ci95_lower"], tps_stats["ci95_upper"]],
        "coefficient_of_variation": tps_stats["cv_percent"],
        "time_to_first_token_ms": ttft_stats["mean"],
        "time_to_first_token_std_ms": ttft_stats["std"],
        "time_to_first_token_ci95_ms": [ttft_stats["ci95_lower"], ttft_stats["ci95_upper"]],
        "runs": runs,
        "warmup": warmup,
        "n_tokens": n_tokens,
        "input_tokens": input_length,
        "all_tps": all_tps,
        "all_ttft_ms": all_ttft,
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end LLM inference benchmark for reviewer response")
    parser.add_argument("--output", type=str, default="results/reviewer_e2e_inference.json")
    parser.add_argument("--n-tokens", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Device to run on (auto will pick best available)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 70)
        print("REVIEWER RESPONSE: End-to-End LLM Inference Benchmark")
        print("=" * 70)

    # Get system info
    system_info = get_system_info()

    if verbose:
        print(f"\nPlatform: {system_info['platform']}")
        print(f"CPU: {system_info.get('cpu_brand', 'Unknown')}")
        if 'wgpu_gpu' in system_info:
            print(f"GPU (wgpu): {system_info['wgpu_gpu'].get('description', 'Unknown')}")
            print(f"Backend: {system_info['wgpu_gpu'].get('backend', 'Unknown')}")

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "mps"
        else:
            device = torch.device("cpu")
            device_name = "cpu"
    else:
        device = torch.device(args.device)
        device_name = args.device

    if verbose:
        print(f"\nUsing device: {device_name}")

    # Load model
    if verbose:
        print(f"\nLoading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    model = model.to(device)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {total_params/1e9:.2f}B parameters")

    # Run benchmark
    results = benchmark_inference(
        model, tokenizer, device,
        prompt=args.prompt,
        n_tokens=args.n_tokens,
        warmup=args.warmup,
        runs=args.runs,
        verbose=verbose,
    )

    # Add metadata
    results["model"] = MODEL_NAME
    results["device"] = device_name
    results["backend"] = f"pytorch-{device_name}"
    results["system_info"] = system_info
    results["prompt"] = args.prompt

    # For Apple M2, note this is MPS (Metal Performance Shaders), not WebGPU
    if device_name == "mps":
        results["note"] = "MPS (Metal Performance Shaders) backend, not WebGPU. For WebGPU comparison, see dispatch overhead measurements in exp1_cross_gpu_webgpu.py"

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Device: {device_name}")
    print(f"Tokens/second: {results['tokens_per_second']:.2f} +/- {results['tokens_per_second_std']:.2f}")
    print(f"  95% CI: [{results['tokens_per_second_ci95'][0]:.2f}, {results['tokens_per_second_ci95'][1]:.2f}]")
    print(f"CV: {results['coefficient_of_variation']:.1f}%")
    print(f"Time to first token: {results['time_to_first_token_ms']:.2f} ms")
    print("=" * 70)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print comparison note
    print()
    print("IMPORTANT NOTE FOR REVIEWER RESPONSE:")
    print("-" * 70)
    if device_name == "mps":
        print("This benchmark uses PyTorch MPS (Metal Performance Shaders), which is")
        print("Apple's optimized GPU backend for PyTorch - NOT WebGPU.")
        print()
        print("For WebGPU comparison on Apple M2:")
        print("1. Dispatch overhead: exp1_cross_gpu_webgpu.py shows ~71 µs (wgpu/Metal)")
        print("2. Full WebGPU inference would require torch-webgpu built for Metal")
        print("3. Current MPS result provides a baseline for M2 GPU performance")
    elif device_name == "cpu":
        print("This is CPU-only inference. For WebGPU inference on this platform,")
        print("torch-webgpu must be built with the appropriate backend.")
        print("Dispatch overhead can be measured with exp1_cross_gpu_webgpu.py")
    print("-" * 70)


if __name__ == "__main__":
    main()
