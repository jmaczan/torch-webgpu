# Reviewer-Requested Experiments

This directory contains all experiments requested by the TMLR reviewer.

## Quick Start

### On your main Linux machine (RTX 5090):

```bash
cd /home/jedrzej/dev/torch-webgpu/experiments
./run_all_experiments.sh
```

### On MacBook Air M2:

```bash
# Copy experiments folder to Mac
scp -r experiments/ macbook:~/torch-webgpu-experiments/

# SSH to Mac and run
ssh macbook
cd ~/torch-webgpu-experiments
./run_on_mac.sh

# Copy results back
scp results/exp1_apple_m2_webgpu.json linux-machine:~/torch-webgpu/experiments/results/
```

### Collect all results:

```bash
python3 collect_results.py
```

Copy-paste the output back to Claude for paper incorporation.

---

## Experiment Overview

| # | Experiment | What it measures | Reviewer concern addressed |
|---|------------|------------------|---------------------------|
| 1 | Cross-GPU WebGPU | Per-dispatch overhead, fusion speedup, mega-kernel slowdown on non-NVIDIA GPU | "Validate on at least one non-NVIDIA GPU" |
| 2 | Device-Side Argmax | Full readback vs device-side argmax sync overhead | "Implement device-side argmax prototype" |
| 3 | Tiled Mega Strategy | 7-dispatch vs 3-dispatch vs 1-dispatch MLP | "Implement multi-dispatch tiled mega strategy" |
| 4 | Timeline Visualization | CPU/GPU overlap, per-operation breakdown | "Provide timeline visualization" |
| 5 | CUDA Comparison | CUDA vs WebGPU launch overhead, CUDA fusion benefit | "Isolate API/spec vs kernel maturity" |

---

## Detailed Instructions

### Experiment 1: Cross-GPU WebGPU Validation

**Purpose**: Validate findings on Apple M2 GPU (or other non-NVIDIA GPU)

**Run on MacBook M2**:
```bash
pip install wgpu numpy scipy
python exp1_cross_gpu_webgpu.py --output results/exp1_apple_m2_webgpu.json
```

**Expected output**:
- Per-dispatch overhead in µs (compare to ~95µs on RTX 5090)
- RMSNorm fusion speedup (compare to ~1.5x on RTX 5090)
- Mega-kernel slowdown (compare to ~16x on RTX 5090)

### Experiment 2: Device-Side Argmax

**Purpose**: Measure whether computing argmax on GPU reduces sync overhead

**Run on Linux**:
```bash
python exp2_device_argmax.py --output results/exp2_device_argmax.json
```

**Expected output**:
- Full logits readback time (current ~11ms approach)
- Device-side argmax time (should be lower due to smaller transfer)
- Buffer map/unmap microbenchmark (isolates mapping overhead)

### Experiment 3: Multi-Dispatch Tiled Mega Strategy

**Purpose**: Test middle-ground between fully unfused and single-dispatch mega-kernel

**Run on Linux**:
```bash
python exp3_tiled_mega.py --output results/exp3_tiled_mega.json
```

**Expected output**:
- Unfused MLP time (7 dispatches, many workgroups each)
- Tiled MLP time (3 dispatches, preserves parallelism)
- Mega-kernel time (1 dispatch, 256 threads only)

### Experiment 4: Timeline Visualization

**Purpose**: Show CPU/GPU overlap that explains why CPU overhead doesn't sum to wall-clock

**Run on Linux**:
```bash
python exp4_timeline.py --output results/exp4_timeline.json
```

**Expected output**:
- Per-dispatch CPU breakdown (encoder, bind group, submit)
- Total CPU time vs wall-clock time
- Overlap ratio showing pipelining
- Timeline figure (if matplotlib available)

### Experiment 5: CUDA Fusion Comparison

**Purpose**: Compare WebGPU overhead to CUDA and show CUDA fusion benefit is minimal

**Run on Linux**:
```bash
python exp5_cuda_fusion.py --output results/exp5_cuda_fusion.json
```

**Expected output**:
- CUDA kernel launch overhead (~5-10µs)
- WebGPU dispatch overhead (~95µs)
- Overhead ratio (WebGPU/CUDA)
- CUDA fusion speedup (should be minimal, ~1.0-1.2x)

---

## File Structure

```
experiments/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_all_experiments.sh       # Run all experiments on Linux
├── run_on_mac.sh               # Run experiment 1 on Mac
├── collect_results.py          # Generate summary for Claude
├── exp1_cross_gpu_webgpu.py    # Experiment 1: Cross-GPU
├── exp2_device_argmax.py       # Experiment 2: Device argmax
├── exp3_tiled_mega.py          # Experiment 3: Tiled mega
├── exp4_timeline.py            # Experiment 4: Timeline
├── exp5_cuda_fusion.py         # Experiment 5: CUDA comparison
└── results/                    # Output JSON files
    ├── exp1_webgpu.json
    ├── exp1_apple_m2_webgpu.json
    ├── exp2_device_argmax.json
    ├── exp3_tiled_mega.json
    ├── exp4_timeline.json
    ├── exp4_timeline_timeline.png
    └── exp5_cuda_fusion.json
```

---

## After Running All Experiments

1. Collect results:
   ```bash
   python collect_results.py > experiment_results.txt
   ```

2. Copy-paste `experiment_results.txt` content back to Claude

3. Claude will incorporate the results into the paper

---

## Troubleshooting

### "wgpu not found"
```bash
pip install wgpu
```

### "CUDA not available" in exp5
- This is OK on Mac - CUDA experiments will be skipped
- Run exp5 on the Linux machine with RTX 5090

### "Metal" backend on Mac
- This is expected - Apple Silicon uses Metal for WebGPU
- Results are still valid for cross-vendor validation

### Low performance numbers
- Ensure power is plugged in (laptops may throttle on battery)
- Close other GPU-intensive applications
- Run warmup iterations (scripts do this automatically)
