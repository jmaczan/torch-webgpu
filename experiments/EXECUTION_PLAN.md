# Experiment Execution Plan

## Overview

You need to run experiments on **two machines**:
1. **Linux workstation** (RTX 5090) - Experiments 1-5
2. **MacBook Air M2** - Experiment 1 only (cross-vendor validation)

## Step-by-Step Instructions

### STEP 1: Run experiments on Linux (RTX 5090)

```bash
cd /home/jedrzej/dev/torch-webgpu/experiments
./run_all_experiments.sh
```

**Expected time**: ~10-15 minutes

**What happens**:
- Installs wgpu, numpy, scipy, matplotlib if needed
- Runs all 5 experiments
- Saves results to `results/exp*.json`

### STEP 2: Copy experiments folder to MacBook

```bash
# From Linux machine
cd /home/jedrzej/dev/torch-webgpu
scp -r experiments/ YOUR_MAC:~/torch-webgpu-experiments/
```

### STEP 3: Run experiment on MacBook

```bash
# SSH to Mac
ssh YOUR_MAC

# Run the script
cd ~/torch-webgpu-experiments
./run_on_mac.sh
```

**Expected time**: ~5 minutes

### STEP 4: Copy Mac results back to Linux

```bash
# From Mac
scp results/exp1_apple_m2_webgpu.json YOUR_LINUX:~/dev/torch-webgpu/experiments/results/

# Or from Linux
scp YOUR_MAC:~/torch-webgpu-experiments/results/exp1_apple_m2_webgpu.json experiments/results/
```

### STEP 5: Collect all results

```bash
cd /home/jedrzej/dev/torch-webgpu/experiments
python3 collect_results.py
```

### STEP 6: Copy-paste the output

Copy the entire output from Step 5 and paste it back to Claude.

---

## What Each Experiment Produces

### Experiment 1 (Cross-GPU)
- Per-dispatch overhead on Apple M2 (compare to ~95µs on RTX 5090)
- Whether mega-kernel is also slow on Apple Silicon
- Whether fusion benefit is similar

### Experiment 2 (Device Argmax)
- How much sync overhead can be reduced by device-side argmax
- Buffer map/unmap overhead at different sizes

### Experiment 3 (Tiled Mega)
- Whether 3-dispatch tiled approach is faster than 7-dispatch unfused
- Comparison to single-dispatch mega-kernel

### Experiment 4 (Timeline)
- Visual proof of CPU/GPU overlap (pipelining)
- Breakdown of per-dispatch CPU overhead

### Experiment 5 (CUDA Comparison)
- CUDA launch overhead (~5-10µs) vs WebGPU (~95µs)
- Whether CUDA benefits from fusion (it shouldn't much)

---

## Troubleshooting

### "Permission denied" on scripts
```bash
chmod +x run_all_experiments.sh run_on_mac.sh
```

### "wgpu not found" on Mac
```bash
pip3 install wgpu
```

### CUDA experiments fail on Mac
This is expected - CUDA is not available on Mac. The script handles this gracefully.

### Results look wrong
- Make sure no other GPU-intensive apps are running
- Check that laptop is plugged in (not on battery)
- Try running with more iterations: `--iterations 200`

---

## Expected Results (approximate)

Based on our RTX 5090 findings, we expect on Apple M2:

| Metric | RTX 5090 (Dawn) | Apple M2 (Metal) - Expected |
|--------|-----------------|------------------------------|
| Per-dispatch overhead | ~95 µs | ~50-200 µs (unknown) |
| RMSNorm fusion speedup | ~1.5x | Similar (~1.3-2x) |
| Mega-kernel slowdown | ~16x | Similar (>5x) |

The key validation is whether the **qualitative findings** hold:
1. Does fusion help on Apple Silicon too?
2. Is mega-kernel also slow on Apple Silicon?
3. What is the per-dispatch overhead on Metal?

---

## After Pasting Results to Claude

Claude will:
1. Incorporate the cross-vendor results into the paper
2. Add the device-side argmax findings
3. Add the tiled mega-kernel comparison
4. Include the timeline visualization analysis
5. Update the CUDA comparison section

This should address all major reviewer concerns about:
- Single-vendor limitation
- Device-side argmax not implemented
- Multi-dispatch tiled strategy not explored
- No timeline visualization
- API vs maturity separation
