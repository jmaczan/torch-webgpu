#!/usr/bin/env python3
"""
End-to-End Inference Benchmarks for Larger Models

This script runs full end-to-end inference (not just micro-benchmarks) on larger models
to address reviewer concerns about limited model coverage.

Models tested:
- Qwen2.5-0.5B-Instruct (494M params) - baseline
- Qwen2.5-1.5B-Instruct (1.5B params) - larger model
- Qwen2.5-3B-Instruct (3B params) - if GPU memory allows

Requirements:
- PyTorch with CUDA or MPS
- transformers library
- Sufficient GPU memory (8GB+ for 1.5B, 12GB+ for 3B)
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_system_info():
    """Collect system information."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["device"] = "cuda"
    elif info["mps_available"]:
        info["device"] = "mps"
        info["gpu_name"] = "Apple Silicon (MPS)"
    else:
        info["device"] = "cpu"
        info["gpu_name"] = "CPU"

    return info


def calculate_statistics(values):
    """Calculate comprehensive statistics with 95% CI."""
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    sem = std / np.sqrt(n)

    # 95% CI using t-distribution
    t_critical = stats.t.ppf(0.975, df=n-1)
    ci95_lower = mean - t_critical * sem
    ci95_upper = mean + t_critical * sem

    cv = (std / mean) * 100 if mean > 0 else 0

    return {
        "mean": float(mean),
        "std": float(std),
        "sem": float(sem),
        "ci95_lower": float(ci95_lower),
        "ci95_upper": float(ci95_upper),
        "cv_percent": float(cv),
        "n": n,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "all_values": [float(v) for v in values]
    }


def benchmark_model(model_name: str, device: str, num_tokens: int = 50,
                    num_runs: int = 30, warmup_runs: int = 5):
    """
    Run end-to-end inference benchmark on a model.

    Returns detailed timing information including tokens/sec, TTFT, and per-token latency.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Load model and tokenizer
    print("Loading model...")
    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Determine dtype based on device
    if device == "cuda":
        dtype = torch.float16
    elif device == "mps":
        dtype = torch.float16  # MPS supports float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device != "mps" else None,
        trust_remote_code=True
    )

    if device == "mps":
        model = model.to("mps")

    model.eval()

    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Get model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params / 1e9:.2f}B")

    # Prepare prompt
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device if device != "mps" else "mps")
    input_length = inputs.input_ids.shape[1]

    print(f"\nPrompt: '{prompt}' ({input_length} tokens)")
    print(f"Generating {num_tokens} tokens per run")

    # Warmup runs
    print(f"\nWarmup ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        print(f"  Warmup {i+1}/{warmup_runs} complete")

    # Synchronize before timed runs
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    print(f"\nTimed runs ({num_runs})...")
    run_data = []

    for i in range(num_runs):
        # Synchronize before timing
        if device == "cuda":
            torch.cuda.synchronize()

        # Measure TTFT using streaming
        ttft = None
        tokens_generated = 0
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

        # Synchronize after generation
        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # Calculate metrics
        total_time = end_time - start_time
        output_length = outputs.sequences.shape[1]
        tokens_generated = output_length - input_length
        tokens_per_second = tokens_generated / total_time

        # Estimate TTFT (first forward pass)
        # For autoregressive generation, TTFT ≈ total_time / tokens_generated for first token
        # This is an approximation; true TTFT requires callback support
        estimated_ttft_ms = (total_time / tokens_generated) * 1000

        run_data.append({
            "run": i + 1,
            "total_time_s": total_time,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_second,
            "estimated_ttft_ms": estimated_ttft_ms
        })

        print(f"  Run {i+1:2d}/{num_runs}: {tokens_per_second:6.2f} tok/s "
              f"({tokens_generated} tokens in {total_time:.3f}s)")

    # Calculate statistics
    tps_values = [r["tokens_per_second"] for r in run_data]
    time_values = [r["total_time_s"] for r in run_data]
    ttft_values = [r["estimated_ttft_ms"] for r in run_data]

    tps_stats = calculate_statistics(tps_values)
    time_stats = calculate_statistics(time_values)
    ttft_stats = calculate_statistics(ttft_values)

    # Decode a sample output
    sample_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    results = {
        "model": model_name,
        "num_parameters": num_params,
        "num_parameters_billions": num_params / 1e9,
        "device": device,
        "dtype": str(dtype),
        "config": {
            "prompt": prompt,
            "input_tokens": input_length,
            "max_new_tokens": num_tokens,
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
            "do_sample": False
        },
        "results": {
            "tokens_per_second": tps_stats,
            "total_time_seconds": time_stats,
            "estimated_ttft_ms": ttft_stats,
            "model_load_time_seconds": load_time
        },
        "sample_output": sample_output[:200] + "..." if len(sample_output) > 200 else sample_output,
        "all_runs": run_data
    }

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Tokens/sec: {tps_stats['mean']:.2f} ± {tps_stats['std']:.2f}")
    print(f"95% CI: [{tps_stats['ci95_lower']:.2f}, {tps_stats['ci95_upper']:.2f}]")
    print(f"CV: {tps_stats['cv_percent']:.1f}%")
    print(f"Min/Max: {tps_stats['min']:.2f} / {tps_stats['max']:.2f}")

    # Cleanup
    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="End-to-end LLM inference benchmark for larger models")
    parser.add_argument("--models", nargs="+", default=[
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct"
    ], help="Models to benchmark")
    parser.add_argument("--num-tokens", type=int, default=50, help="Tokens to generate")
    parser.add_argument("--num-runs", type=int, default=30, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/mps/cpu, auto-detected if not specified)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")

    args = parser.parse_args()

    # Detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("="*60)
    print("LARGER MODEL END-TO-END INFERENCE BENCHMARK")
    print("="*60)

    # Collect system info
    system_info = get_system_info()
    print(f"\nSystem Information:")
    print(f"  Platform: {system_info['platform']}")
    print(f"  Device: {device}")
    print(f"  GPU: {system_info.get('gpu_name', 'N/A')}")
    if 'gpu_memory_gb' in system_info:
        print(f"  GPU Memory: {system_info['gpu_memory_gb']:.1f} GB")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks for each model
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info,
        "benchmark_config": {
            "num_tokens": args.num_tokens,
            "num_runs": args.num_runs,
            "warmup_runs": args.warmup,
            "device": device
        },
        "models": {}
    }

    for model_name in args.models:
        try:
            results = benchmark_model(
                model_name=model_name,
                device=device,
                num_tokens=args.num_tokens,
                num_runs=args.num_runs,
                warmup_runs=args.warmup
            )
            all_results["models"][model_name] = results

        except Exception as e:
            print(f"\nERROR benchmarking {model_name}: {e}")
            all_results["models"][model_name] = {"error": str(e)}
            import traceback
            traceback.print_exc()
            continue

    # Save results
    output_file = output_dir / f"reviewer_larger_models_{device}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<40} {'Params':>10} {'Tok/s':>10} {'95% CI':>20} {'CV':>8}")
    print("-"*80)

    for model_name, results in all_results["models"].items():
        if "error" in results:
            print(f"{model_name:<40} {'ERROR':>10}")
            continue

        params = f"{results['num_parameters_billions']:.2f}B"
        tps = results["results"]["tokens_per_second"]
        ci = f"[{tps['ci95_lower']:.1f}, {tps['ci95_upper']:.1f}]"
        cv = f"{tps['cv_percent']:.1f}%"
        print(f"{model_name:<40} {params:>10} {tps['mean']:>10.2f} {ci:>20} {cv:>8}")

    print("="*80)


if __name__ == "__main__":
    main()
