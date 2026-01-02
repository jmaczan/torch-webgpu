#!/usr/bin/env python
import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None


def percentiles(samples: List[float], ps=(1, 50, 90, 95, 99)) -> Dict[str, float]:
    arr = np.array(samples, dtype=np.float64)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    return data.get("benchmarks", [])


def _to_ms(entry: Dict[str, Any]) -> float:
    value = float(entry["real_time"])
    unit = entry.get("time_unit", "ns")
    scale = {
        "ns": 1e-6,  # ns -> ms
        "us": 1e-3,  # µs -> ms
        "ms": 1.0,  # ms -> ms
        "s": 1e3,  # s -> ms
    }.get(unit, 1e-6)
    return value * scale


def _counter(entry: Dict[str, Any], key: str) -> float:
    counters = entry.get("counters", {})
    if key in counters:
        return float(counters[key])
    if key in entry:
        return float(entry[key])
    return math.nan


def summarize(benchmarks: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for b in benchmarks:
        name = b["name"]
        time_ms = _to_ms(b)
        iters = b.get("iterations", 1)
        gflops = _counter(b, "gflops")
        bytes_moved = _counter(b, "bytes")
        rows.append(
            {
                "name": name,
                "time_ms": time_ms,
                "iterations": iters,
                "gflops": gflops,
                "bytes": bytes_moved,
            }
        )
    return pd.DataFrame(rows)


def summarize_repetitions(benchmarks: List[Dict[str, Any]]) -> pd.DataFrame:
    groups: Dict[str, List[float]] = {}
    for b in benchmarks:
        name = b["name"]
        time_ms = _to_ms(b)
        groups.setdefault(name, []).append(time_ms)
    rows = []
    for name, samples in groups.items():
        stats = percentiles(samples, ps=(1, 50, 90, 95, 99))
        rows.append({"name": name, "mean": statistics.mean(samples), **stats})
    return pd.DataFrame(rows)


def aggregate_iteration_means(benchmarks: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for b in benchmarks:
        if b.get("run_type") != "iteration":
            continue
        name = b["name"]
        gflops = _counter(b, "gflops")
        bytes_moved = _counter(b, "bytes")
        rows.append({"name": name, "gflops": gflops, "bytes": bytes_moved})
    if not rows:
        return pd.DataFrame(columns=["name", "avg_gflops", "avg_bytes"])
    df = pd.DataFrame(rows)
    agg = df.groupby("name", as_index=False).mean(numeric_only=True)
    agg = agg.rename(columns={"gflops": "avg_gflops", "bytes": "avg_bytes"})
    return agg.sort_values("avg_gflops", ascending=True)


def plot_gflops(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No iteration entries found; skipping gflops plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(df["name"], df["avg_gflops"], marker="o", color="#4C72B0")
    plt.ylabel("Average GFLOPS")
    plt.xlabel("Benchmark name")
    plt.title("Average GFLOPS per benchmark (iteration runs)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def maybe_log_wandb(
    df: pd.DataFrame,
    df_rep: pd.DataFrame,
    df_avg: pd.DataFrame,
    out_csv: Path,
    out_json: Path,
    plot_path: Path,
    args: argparse.Namespace,
) -> None:
    if not args.wandb_project:
        return
    if wandb is None:
        raise SystemExit(
            "wandb not installed; pip install wandb or omit --wandb-project"
        )
    run = wandb.init(project=args.wandb_project, config={"run_tag": args.run_tag})
    wandb.log({"summary_table": wandb.Table(dataframe=df)})
    wandb.log({"percentiles": wandb.Table(dataframe=df_rep)})
    if not df_avg.empty:
        wandb.log({"averaged_iteration": wandb.Table(dataframe=df_avg)})
    if plot_path.exists():
        wandb.log({"gflops_plot": wandb.Image(str(plot_path))})

    artifact = wandb.Artifact("bench_results", type="benchmark")
    if out_csv.exists():
        artifact.add_file(str(out_csv))
    if out_json.exists():
        artifact.add_file(str(out_json))
    if plot_path.exists():
        artifact.add_file(str(plot_path))
    run.log_artifact(artifact)
    run.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True, type=Path, help="benchmark JSON output from gbench"
    )
    ap.add_argument("--out-csv", type=Path, default=Path("bench_summary.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("bench_summary.json"))
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--run-tag", type=str, default=None)
    args = ap.parse_args()

    benches = load_benchmark(args.input)
    df = summarize(benches)
    df_rep = summarize_repetitions(benches)
    df_avg = aggregate_iteration_means(benches)

    df.to_csv(args.out_csv, index=False)
    df_rep.to_json(args.out_json, orient="records", indent=2)
    plot_path = Path("gflops_benchmark.png")
    plot_gflops(df_avg, plot_path)

    print("=== Summary ===")
    print(df.to_markdown(index=False))
    print("\n=== Percentiles ===")
    print(df_rep.to_markdown(index=False))
    if not df_avg.empty:
        print("\n=== Averaged (iteration) ===")
        print(df_avg.to_markdown(index=False))

    maybe_log_wandb(
        df,
        df_rep,
        df_avg,
        args.out_csv,
        args.out_json,
        plot_path,
        args,
    )


if __name__ == "__main__":
    main()
