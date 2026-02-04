#!/usr/bin/env python3
"""
Analyze dispatch profiler traces from torch-webgpu.

Usage:
    python analyze_profiler_trace.py profiler_trace.json

This script reads the JSON output from torch_webgpu._C.get_profile_stats()
and produces summary statistics and optional visualizations.
"""

import argparse
import json
import sys
from pathlib import Path


def analyze_trace(trace_path: str) -> dict:
    """Analyze a profiler trace file."""
    with open(trace_path) as f:
        trace = json.load(f)

    dispatches = trace.get("dispatches", [])
    if not dispatches:
        print("No dispatches found in trace")
        return {}

    # Extract timing components
    encoder_times = [d.get("encoder_us", 0) for d in dispatches]
    bindgroup_times = [d.get("bindgroup_us", 0) for d in dispatches]
    submit_times = [d.get("submit_us", 0) for d in dispatches]

    # Calculate statistics
    def stats(times):
        if not times:
            return {"mean": 0, "min": 0, "max": 0, "total": 0}
        return {
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "total": sum(times)
        }

    results = {
        "total_dispatches": len(dispatches),
        "wall_clock_ms": trace.get("wall_clock_ms", 0),
        "encoder_us": stats(encoder_times),
        "bindgroup_us": stats(bindgroup_times),
        "submit_us": stats(submit_times),
        "total_cpu_overhead_us": sum(encoder_times) + sum(bindgroup_times) + sum(submit_times),
    }

    # Calculate derived metrics
    if results["wall_clock_ms"] > 0:
        cpu_overhead_ms = results["total_cpu_overhead_us"] / 1000
        results["cpu_overhead_fraction"] = cpu_overhead_ms / results["wall_clock_ms"]
        results["implied_gpu_time_ms"] = results["wall_clock_ms"] - cpu_overhead_ms
        results["per_dispatch_overhead_us"] = results["total_cpu_overhead_us"] / len(dispatches)

    return results


def print_summary(results: dict):
    """Print a human-readable summary."""
    print("=" * 60)
    print("Dispatch Profiler Analysis")
    print("=" * 60)
    print(f"Total dispatches:        {results.get('total_dispatches', 0)}")
    print(f"Wall clock time:         {results.get('wall_clock_ms', 0):.2f} ms")
    print()
    print("CPU-side overhead breakdown:")
    print(f"  Encoder creation:      {results.get('encoder_us', {}).get('total', 0) / 1000:.2f} ms")
    print(f"  Bind group creation:   {results.get('bindgroup_us', {}).get('total', 0) / 1000:.2f} ms")
    print(f"  Queue submission:      {results.get('submit_us', {}).get('total', 0) / 1000:.2f} ms")
    print(f"  Total CPU overhead:    {results.get('total_cpu_overhead_us', 0) / 1000:.2f} ms")
    print()
    print("Per-dispatch averages:")
    print(f"  Encoder:               {results.get('encoder_us', {}).get('mean', 0):.1f} us")
    print(f"  Bind group:            {results.get('bindgroup_us', {}).get('mean', 0):.1f} us")
    print(f"  Submit:                {results.get('submit_us', {}).get('mean', 0):.1f} us")
    print(f"  Total per dispatch:    {results.get('per_dispatch_overhead_us', 0):.1f} us")
    print()

    if "cpu_overhead_fraction" in results:
        print("Analysis:")
        print(f"  CPU overhead fraction: {results['cpu_overhead_fraction'] * 100:.1f}%")
        print(f"  Implied GPU time:      {results['implied_gpu_time_ms']:.2f} ms")
        print()
        print("Note: CPU overhead > wall clock suggests GPU/CPU overlap (pipelining).")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze dispatch profiler traces")
    parser.add_argument("trace_file", help="Path to profiler trace JSON file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not Path(args.trace_file).exists():
        print(f"Error: File not found: {args.trace_file}")
        sys.exit(1)

    results = analyze_trace(args.trace_file)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
