# Reviewer-Requested Experiments

This directory contains all experiments requested by the TMLR reviewer.

## Quick Start

```bash
cd /home/jedrzej/dev/torch-webgpu/experiments

# Install dependencies (on each machine)
pip install -r requirements.txt

# Run all experiments and collect results
./run_all_experiments.sh
```

## Experiment Overview

| # | Experiment | Machine | Output File |
|---|------------|---------|-------------|
| 1 | Non-NVIDIA GPU WebGPU | MacBook M2 | `results/exp1_apple_m2_webgpu.json` |
| 2 | Device-side Argmax | Linux (RTX 5090) | `results/exp2_device_argmax.json` |
| 3 | Multi-dispatch Tiled Mega | Linux (RTX 5090) | `results/exp3_tiled_mega.json` |
| 4 | GPU Timeline Visualization | Linux (RTX 5090) | `results/exp4_timeline.json` |
| 5 | CUDA Fusion Comparison | Linux (RTX 5090) | `results/exp5_cuda_fusion.json` |

## Detailed Instructions

### Experiment 1: Apple M2 WebGPU (Run on MacBook)

```bash
# On MacBook Air M2
cd experiments
pip install wgpu glfw numpy

# Run the WebGPU benchmark
python exp1_cross_gpu_webgpu.py --output results/exp1_apple_m2_webgpu.json

# Copy results back to main machine
scp results/exp1_apple_m2_webgpu.json user@main-machine:~/torch-webgpu/experiments/results/
```

### Experiment 2-5: Run on Linux (RTX 5090)

```bash
# On main Linux machine
cd experiments
./run_all_experiments.sh
```

## After Running

Copy all results back and run:
```bash
python collect_results.py
```

This will output a summary you can paste back to Claude.
