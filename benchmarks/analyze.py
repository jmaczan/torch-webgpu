#!/usr/bin/env python
import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List

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


def maybe_log_wandb(
    df: pd.DataFrame, df_rep: pd.DataFrame, args: argparse.Namespace
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

    df.to_csv(args.out_csv, index=False)
    df_rep.to_json(args.out_json, orient="records", indent=2)

    print("=== Summary ===")
    print(df.to_markdown(index=False))
    print("\n=== Percentiles ===")
    print(df_rep.to_markdown(index=False))

    maybe_log_wandb(df, df_rep, args)


if __name__ == "__main__":
    main()
