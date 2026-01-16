# Optimization Log

## Baseline
- Date: 2026-01-16
- Model: Qwen/Qwen2.5-0.5B-Instruct (0.49B parameters)
- Hardware: NVIDIA GeForce RTX 5090
- Driver: 570.195.03
- Backend: torch-webgpu

### Baseline Performance
| Metric | Value |
|--------|-------|
| Tokens/sec (avg) | 11.18 |
| Tokens/sec (stable, excl. first run) | ~13.4 |
| Tokens/sec std | 2.94 |
| Time to first token | 70.76 ms |
| Avg time per run (32 tokens) | 2.86 s |

### Initial Observations
1. First benchmark run is significantly slower (~4.5 tok/s vs ~13.4 tok/s for subsequent runs)
2. `torch._dynamo hit config.recompile_limit (8)` - dynamic shape causes recompilations
3. High variance suggests compilation overhead even after warmup
4. Stable throughput of ~13.4 tokens/second once compiled

### Bottleneck Analysis (Completed)
- [x] Profile end-to-end inference
- [x] Identify top 5 slowest ops
- [x] Measure time per op category

**Key Findings:**
1. **Softmax on LM head (vocab 151936)**: 45-52ms per call - MAJOR BOTTLENECK
2. **Matmul operations**: ~0.5-3ms each, 144 matmuls per forward pass
3. **Forward pass**: 99.9% of generation time
4. **Per-dispatch overhead**: ~0.4-0.5ms minimum per kernel

---

## Optimization 1: Parallel Softmax for Large Vocabularies
- **Date**: 2026-01-16
- **Change**: Implemented parallel reduction softmax kernel using workgroup shared memory
  - 256 threads per workgroup (one workgroup per batch row)
  - Parallel max reduction, parallel sum reduction
  - Threshold: use parallel kernel for dim_size > 1024
- **File**: `csrc/ops/softmax.cpp`

### Results (isolated op benchmark)
| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| softmax (1, 151936) | 45.46ms | 0.54ms | **84x** |
| softmax (10, 151936) | 51.65ms | 0.62ms | **83x** |

### Correctness
- All tests pass with max_diff < 1e-6
- Numerically stable (uses max subtraction)

### Impact on Full Model
- Softmax no longer a bottleneck (~0.6ms vs 45ms)
- Forward pass now dominated by matmul operations
- Next optimization target: matmul kernel

---

## Current Bottleneck Analysis (Post-Optimization 1)

With softmax optimized, the forward pass breakdown is:
- **Matmul operations**: ~100ms total (144 matmuls × ~0.7ms each)
- **Forward pass**: ~70-90ms per token generation step
- **GFLOPS**: Only 1-2% of theoretical RTX 5090 peak (~100 TFLOPS)

The matmul kernel is naive (no tiling, no shared memory), achieving only ~1700 GFLOPS on 1024×1024 matrices (vs theoretical 100+ TFLOPS).

---

## Optimization 2: [Next - Tiled Matmul with Shared Memory]
- Change: Implement tiled matrix multiplication with workgroup shared memory
- Target: 10-20x improvement on large matrices
- Status: Pending

