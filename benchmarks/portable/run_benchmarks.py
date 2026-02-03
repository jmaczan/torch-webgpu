#!/usr/bin/env python
"""
Unified benchmark runner for portable benchmarks.

Runs all available benchmarks and generates comparison summary.

Usage:
    python run_benchmarks.py --output-dir results/
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_benchmark(script: str, output_file: str, extra_args: list = None):
    """Run a benchmark script and return success status."""
    cmd = [sys.executable, script, "--output", output_file]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)

    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"Script not found: {script}")
        return False


def load_results(filepath: Path):
    """Load benchmark results from JSON file."""
    if not filepath.exists():
        return None
    with open(filepath) as f:
        return json.load(f)


def generate_summary(results_dir: Path):
    """Generate comparison summary from all benchmark results."""
    results = {}

    # Load all result files
    for f in results_dir.glob("*.json"):
        data = load_results(f)
        if data and "backend" in data:
            results[data["backend"]] = data

    if not results:
        print("No results found to summarize")
        return

    # Print comparison table
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON SUMMARY")
    print("="*80)
    print()
    print(f"{'Backend':<25} {'Tok/s':>10} {'95% CI':>20} {'CV%':>8} {'TTFT(ms)':>10}")
    print("-"*80)

    # Sort by tokens/second
    sorted_results = sorted(results.items(),
                           key=lambda x: x[1].get('tokens_per_second', 0),
                           reverse=True)

    baseline_tps = None
    for backend, data in sorted_results:
        tps = data.get('tokens_per_second', 0)
        tps_std = data.get('tokens_per_second_std', 0)
        ci = data.get('tokens_per_second_ci95', [0, 0])
        cv = data.get('coefficient_of_variation', 0)
        ttft = data.get('time_to_first_token_ms', 0)

        if baseline_tps is None:
            baseline_tps = tps

        ci_str = f"[{ci[0]:.1f}, {ci[1]:.1f}]"

        print(f"{backend:<25} {tps:>10.2f} {ci_str:>20} {cv:>7.1f}% {ttft:>10.1f}")

    print("-"*80)

    # Speed comparison
    if len(sorted_results) > 1:
        print("\nSpeed Comparison (vs fastest):")
        for backend, data in sorted_results:
            tps = data.get('tokens_per_second', 0)
            ratio = tps / baseline_tps if baseline_tps else 0
            print(f"  {backend}: {ratio:.2f}x")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "backends": {b: {
            "tokens_per_second": d.get('tokens_per_second'),
            "tokens_per_second_ci95": d.get('tokens_per_second_ci95'),
            "coefficient_of_variation": d.get('coefficient_of_variation'),
            "time_to_first_token_ms": d.get('time_to_first_token_ms'),
            "system_info": d.get('system_info'),
        } for b, d in results.items()}
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all portable benchmarks")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Only run CPU benchmark")
    parser.add_argument("--skip-cpu", action="store_true",
                        help="Skip CPU benchmark")
    parser.add_argument("--onnx-provider", type=str, default="cpu",
                        choices=["cpu", "cuda", "dml", "coreml"],
                        help="ONNX Runtime execution provider")
    parser.add_argument("--n-tokens", type=int, default=50,
                        help="Tokens to generate per run")
    parser.add_argument("--runs", type=int, default=30,
                        help="Number of benchmark runs")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent
    extra_args = ["--n-tokens", str(args.n_tokens), "--runs", str(args.runs)]

    benchmarks_run = []

    # Run CPU benchmark
    if not args.skip_cpu:
        cpu_script = script_dir / "bench_portable_cpu.py"
        cpu_output = output_dir / "results_cpu.json"
        if run_benchmark(str(cpu_script), str(cpu_output), extra_args):
            benchmarks_run.append("pytorch-cpu")

    if args.cpu_only:
        generate_summary(output_dir)
        return

    # Run ONNX benchmark
    try:
        import onnxruntime
        onnx_script = script_dir / "bench_portable_onnx.py"
        onnx_output = output_dir / f"results_onnx_{args.onnx_provider}.json"
        onnx_args = extra_args + ["--provider", args.onnx_provider]
        if run_benchmark(str(onnx_script), str(onnx_output), onnx_args):
            benchmarks_run.append(f"onnxruntime-{args.onnx_provider}")
    except ImportError:
        print("\nSkipping ONNX benchmark (onnxruntime not installed)")
        print("Install with: pip install onnxruntime")

    # Generate summary
    generate_summary(output_dir)

    print(f"\n{'='*60}")
    print("BENCHMARKS COMPLETE")
    print(f"{'='*60}")
    print(f"Ran benchmarks: {', '.join(benchmarks_run)}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
