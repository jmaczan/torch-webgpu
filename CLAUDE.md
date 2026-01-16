# torch-webgpu

WebGPU backend for PyTorch. Currently optimizing for paper submission.

## Current Mission

Optimize and benchmark Qwen2.5-0.5B-Instruct inference. This is the ONLY priority.

## Task 1: Squeeze Maximum Performance

Optimize torch-webgpu until Qwen2.5-0.5B-Instruct runs as fast as possible.

### Optimization Checklist

Work through systematically. Check off as completed.

#### Profiling (do first)
- [ ] Profile current inference end-to-end
- [ ] Identify top 5 slowest ops
- [ ] Measure time per op category (matmul, attention, normalization, etc.)
- [ ] Identify memory bottlenecks
- [ ] Log shader dispatch overhead

#### Matmul / Linear (usually biggest bottleneck)
- [ ] Optimize WGSL workgroup size (try 8x8, 16x16, 32x32)
- [ ] Implement tiled matmul if not done
- [ ] Test shared memory / workgroup memory usage
- [ ] Batch small matmuls where possible
- [ ] Profile different tile sizes for different matrix shapes

#### Attention
- [ ] Fuse Q, K, V projections if separate
- [ ] Optimize softmax (numerically stable + fast)
- [ ] Implement flash-attention-style tiling if memory bound
- [ ] Optimize attention score matmul
- [ ] KV-cache efficiency

#### Memory
- [ ] Minimize buffer allocations during inference
- [ ] Reuse buffers where possible
- [ ] Reduce GPU↔CPU data transfers
- [ ] Profile memory bandwidth utilization

#### Fusion
- [ ] Fuse elementwise ops chains (add + mul + activation)
- [ ] Fuse RMSNorm + subsequent op
- [ ] Fuse bias + activation
- [ ] Identify fusion opportunities from FX graph

#### Shader Optimizations
- [ ] Minimize workgroup barriers
- [ ] Optimize memory access patterns (coalesced)
- [ ] Use vec4 loads where applicable
- [ ] Reduce register pressure in hot shaders

#### Launch Overhead
- [ ] Batch shader dispatches where possible
- [ ] Minimize pipeline recreation
- [ ] Profile WebGPU command submission overhead

### Optimization Loop

```
while performance_improving:
    1. Profile
    2. Find biggest bottleneck
    3. Optimize it
    4. Measure improvement
    5. Record in OPTIMIZATION_LOG.md
    6. Repeat
```

### Performance Target

Measure: **tokens/second** on Qwen2.5-0.5B-Instruct

Record baseline before any optimization. Then track progress.

### Output: OPTIMIZATION_LOG.md

Create and maintain this file:

```markdown
# Optimization Log

## Baseline
- Date: YYYY-MM-DD
- Model: Qwen2.5-0.5B-Instruct
- Tokens/sec: X.XX
- Time to first token: X.XX ms
- Hardware: [GPU model]

## Optimization 1: [Name]
- Change: [what you did]
- Before: X.XX tokens/sec
- After: X.XX tokens/sec
- Improvement: +X%

## Optimization 2: [Name]
...
```

### Stop Condition for Task 1

Stop optimizing when:
- 3 consecutive optimization attempts yield <5% improvement each, OR
- You've exhausted all items in the checklist above

Then move to Task 2.

---

## Task 2: Benchmark torch-webgpu

After optimization complete, run comprehensive benchmarks.

### Benchmark Script

Create `benchmarks/bench_qwen.py`:

```python
import torch
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark_inference(model, tokenizer, prompt, n_tokens=50, warmup=3, runs=10):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=n_tokens)
    
    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=n_tokens)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append(end - start)
    
    total_tokens = n_tokens * runs
    total_time = sum(times)
    
    return {
        "tokens_per_second": total_tokens / total_time,
        "avg_time_per_run": total_time / runs,
        "std_time": torch.tensor(times).std().item(),
        "n_tokens": n_tokens,
        "runs": runs,
    }

# Run benchmark
# ... 
```

### Metrics to Collect

For each backend, measure:

| Metric | Unit | How |
|--------|------|-----|
| Tokens/second | tok/s | total_tokens / total_time |
| Time to first token | ms | time until first token generated |
| Peak memory | MB | monitor GPU memory |
| Total inference time | ms | end-to-end for N tokens |

### Test Configurations

Run benchmarks with:
- Prompt lengths: 32, 128, 512 tokens input
- Generation lengths: 32, 64, 128 tokens output
- Batch size: 1 (focus on single request latency)

### Output: benchmarks/results_webgpu.json

```json
{
  "backend": "torch-webgpu",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "hardware": {
    "gpu": "RTX 5090",
    "driver": "...",
    "browser": "N/A (native)"
  },
  "results": [
    {
      "input_tokens": 32,
      "output_tokens": 64,
      "tokens_per_second": 123.45,
      "time_to_first_token_ms": 45.2,
      "peak_memory_mb": 1024,
      "runs": 10,
      "std": 2.3
    },
    ...
  ]
}
```

---

## Task 3: Benchmark Baselines

Same benchmarks, different backends.

### 3A: ONNX Runtime Web

1. Export Qwen to ONNX
2. Run with ONNX Runtime Web (WebGPU execution provider)
3. Measure same metrics

Create `benchmarks/bench_onnx_web.py` or `benchmarks/bench_onnx_web.js`

Output: `benchmarks/results_onnx_web.json`

### 3B: CPU (PyTorch native)

1. Run Qwen with standard PyTorch on CPU
2. No torch.compile, just eager mode
3. Measure same metrics

Create `benchmarks/bench_cpu.py`

Output: `benchmarks/results_cpu.json`

### 3C: CUDA (PyTorch native)

1. Run Qwen with standard PyTorch on CUDA
2. Also test with torch.compile(backend="inductor")
3. Measure same metrics

Create `benchmarks/bench_cuda.py`

Output: `benchmarks/results_cuda.json`

---

## Task 4: Comparative Analysis

Create `benchmarks/ANALYSIS.md`:

```markdown
# Benchmark Analysis

## Hardware
- GPU: RTX 5090
- CPU: ...
- RAM: ...

## Results Summary

| Backend | Tokens/sec | vs CUDA | vs CPU |
|---------|------------|---------|--------|
| CUDA (inductor) | XXX | 1.00x | X.Xx |
| CUDA (eager) | XXX | X.Xx | X.Xx |
| torch-webgpu | XXX | X.Xx | X.Xx |
| ONNX Runtime Web | XXX | X.Xx | X.Xx |
| CPU | XXX | X.Xx | 1.00x |

## Analysis

### Where torch-webgpu wins
- ...

### Where torch-webgpu loses
- ...

### Bottleneck analysis
- ...

### Comparison to ONNX Runtime Web
- ...

## Conclusions
- ...
```

Also create charts (use matplotlib):
- Bar chart: tokens/sec by backend
- Line chart: scaling with input length
- Memory comparison

Output charts to `benchmarks/figures/`

---

## File Structure

```
torch-webgpu/
├── CLAUDE.md
├── OPTIMIZATION_LOG.md      # Create and maintain
├── benchmarks/
│   ├── bench_qwen.py        # Main benchmark script
│   ├── bench_cpu.py
│   ├── bench_cuda.py
│   ├── bench_onnx_web.py
│   ├── results_webgpu.json
│   ├── results_cpu.json
│   ├── results_cuda.json
│   ├── results_onnx_web.json
│   ├── ANALYSIS.md
│   └── figures/
│       ├── tokens_per_sec.png
│       └── memory_comparison.png
```

---

## Commands

```bash
# Profile torch-webgpu
python -m cProfile -o profile.prof benchmarks/bench_qwen.py
# or use py-spy, scalene, etc.

# Run benchmarks
python benchmarks/bench_qwen.py --backend webgpu --output results_webgpu.json
python benchmarks/bench_cpu.py --output results_cpu.json
python benchmarks/bench_cuda.py --output results_cuda.json

# Generate analysis
python benchmarks/analyze.py
```

---

## Priority Order

1. **Task 1**: Optimize torch-webgpu (MOST IMPORTANT)
2. **Task 2**: Benchmark torch-webgpu
3. **Task 3**: Benchmark baselines
4. **Task 4**: Comparative analysis

Do not start Task 2 until Task 1 stop condition is met.
Do not start Task 3 until Task 2 is complete.
Do not start Task 4 until Task 3 is complete.

---

## Current Status

- [ ] Task 1: Optimization complete
- [ ] Task 2: torch-webgpu benchmarked
- [ ] Task 3A: ONNX Runtime Web benchmarked
- [ ] Task 3B: CPU benchmarked
- [ ] Task 3C: CUDA benchmarked
- [ ] Task 4: Analysis complete

## Start Here

1. Run Qwen2.5-0.5B-Instruct inference once to establish baseline
2. Record baseline in OPTIMIZATION_LOG.md
3. Profile to find biggest bottleneck
4. Start optimization loop
