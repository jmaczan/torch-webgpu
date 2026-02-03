# Benchmark Protocol for Cross-Platform Testing

This document provides instructions for running benchmarks on different hardware platforms to address reviewer concerns about hardware generalizability.

## Overview

We provide portable benchmark scripts that work on any platform with pip-installable packages. No custom builds or compilation required.

**Benchmarks:**
- CPU baseline (PyTorch)
- ONNX Runtime (CPU, DirectML, CoreML)

## Hardware Targets

| Device | OS | CPU | GPU | Expected Provider |
|--------|-----|-----|-----|-------------------|
| Mac M2 Air | macOS | Apple M2 | Integrated | CPU, CoreML |
| Windows Laptop | Windows | Intel | RTX PRO 2000 | CPU, DirectML, CUDA |

---

## Mac M2 Air (macOS)

### Setup

```bash
# Create virtual environment
python3 -m venv bench_env
source bench_env/bin/activate

# Install dependencies
pip install torch transformers

# Optional: For ONNX benchmarks
pip install onnxruntime optimum[onnxruntime]

# Optional: For better system info
pip install py-cpuinfo psutil scipy
```

### Run Benchmarks

```bash
cd benchmarks/portable

# CPU benchmark (required)
python bench_portable_cpu.py --output results_mac_cpu.json --runs 30

# ONNX CPU benchmark (optional)
python bench_portable_onnx.py --provider cpu --output results_mac_onnx_cpu.json --runs 30
```

### Expected Output

The benchmark will output:
- Tokens/second with 95% confidence interval
- Time to first token
- Coefficient of variation (CV%)
- Full results in JSON format

---

## Windows Laptop with RTX PRO 2000

### Safety Notes for Corporate Laptop

These benchmarks are **read-only** and **safe** for corporate environments:
- No admin rights required
- No system modifications
- Only downloads ML model weights to user directory
- Can be fully removed by deleting the virtual environment

### Setup

```powershell
# Create virtual environment
python -m venv bench_env
.\bench_env\Scripts\Activate.ps1

# Install dependencies
pip install torch transformers

# For DirectML (WebGPU-like acceleration on Windows)
pip install onnxruntime-directml optimum[onnxruntime]

# For CUDA (if available and allowed)
# pip install onnxruntime-gpu

# Optional: For better system info
pip install py-cpuinfo psutil scipy
```

### Run Benchmarks

```powershell
cd benchmarks\portable

# CPU benchmark (required)
python bench_portable_cpu.py --output results_win_cpu.json --runs 30

# DirectML benchmark (GPU acceleration via DirectX)
python bench_portable_onnx.py --provider dml --output results_win_onnx_dml.json --runs 30

# CUDA benchmark (if onnxruntime-gpu installed)
# python bench_portable_onnx.py --provider cuda --output results_win_onnx_cuda.json --runs 30
```

---

## Alternative: Unified Runner

You can use the unified runner script:

```bash
# Run all benchmarks
python run_benchmarks.py --output-dir results/

# CPU only
python run_benchmarks.py --cpu-only --output-dir results/

# With specific ONNX provider
python run_benchmarks.py --onnx-provider dml --output-dir results/
```

---

## Expected Results Format

Each benchmark produces a JSON file with:

```json
{
  "tokens_per_second": 12.34,
  "tokens_per_second_std": 0.56,
  "tokens_per_second_ci95": [11.89, 12.79],
  "coefficient_of_variation": 4.5,
  "time_to_first_token_ms": 123.45,
  "time_to_first_token_ci95_ms": [120.1, 126.8],
  "runs": 30,
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "backend": "pytorch-cpu",
  "system_info": {
    "platform": "...",
    "cpu_brand": "...",
    "ram_gb": 16
  }
}
```

---

## Collecting Results

After running benchmarks on each device, please share:

1. **JSON result files** from each benchmark
2. **Any error messages** encountered during setup/execution
3. **Device specifications** (confirmed CPU/GPU model, RAM, OS version)

### Files to Collect

| Device | Expected Files |
|--------|----------------|
| Mac M2 Air | `results_mac_cpu.json`, optionally `results_mac_onnx_cpu.json` |
| Windows Laptop | `results_win_cpu.json`, `results_win_onnx_dml.json` |

---

## Troubleshooting

### "torch not found"
```bash
pip install torch
```

### "onnxruntime not found"
```bash
pip install onnxruntime  # CPU
pip install onnxruntime-directml  # Windows DirectML
```

### "optimum not found" (for ONNX export)
```bash
pip install optimum[onnxruntime]
```

### DirectML provider not available
Ensure you installed `onnxruntime-directml`, not `onnxruntime`:
```bash
pip uninstall onnxruntime
pip install onnxruntime-directml
```

### Out of memory
Reduce tokens or use smaller prompt:
```bash
python bench_portable_cpu.py --n-tokens 20 --output results.json
```

---

## Statistical Rigor

All benchmarks use:
- **30 runs** by default (configurable with `--runs`)
- **3 warmup runs** before timing
- **95% confidence intervals** using t-distribution
- **Coefficient of variation** to assess measurement stability

A CV below 10% indicates stable measurements. Higher CV may indicate:
- Background processes interfering
- Thermal throttling
- Memory pressure

Consider closing other applications and re-running if CV is high.
