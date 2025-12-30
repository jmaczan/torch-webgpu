#!/usr/bin/env bash
python benchmarks/analyze.py --input bench.json \
  --out-csv bench_summary.csv \
  --out-json bench_percentiles.json \
  --wandb-project torch-webgpu \
  --run-tag "$(date +%Y%m%d-%H%M%S)"