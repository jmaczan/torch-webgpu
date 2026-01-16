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

## Optimization 2: Tiled Matmul with Shared Memory
- **Date**: 2026-01-16
- **Change**: Implemented tiled matrix multiplication using workgroup shared memory
  - 16x16 tile size
  - Cooperatively load tiles of A and B into shared memory
  - Each thread computes one output element by iterating over tiles
  - Handles strided/transposed matrices correctly
- **File**: `csrc/shaders/mm.wgsl`

### Results (isolated op benchmark)
| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| mlp_down_proj_seq50 (10x4864 @ 4864x896) | 1.11ms, 394 GFLOPS | 0.46ms, 954 GFLOPS | **2.4x** |
| lm_head_seq10 (10x896 @ 896x151936) | 2.84ms, 958 GFLOPS | 1.07ms, 2556 GFLOPS | **2.7x** |
| mm_1024x1024 | 1.24ms, 1728 GFLOPS | 0.65ms, 3285 GFLOPS | **1.9x** |

Peak GFLOPS improved from ~1700 to ~3285 (3.3% of theoretical RTX 5090 peak).

### Correctness
- All standard tests pass
- Handles transposed matrices (like weight.t() in linear layers)
- Small numerical differences (~6e-4) for very large matrices are expected

### Impact on Full Model
- Isolated matmul ops show 2-3x speedup
- Full model forward pass shows minimal improvement (~75ms -> ~75ms)
- **Root cause**: High per-dispatch overhead (~0.4ms per kernel)
- With ~200+ kernel dispatches per forward pass, overhead dominates

---

## Current Bottleneck: Per-Dispatch Overhead

After softmax (84x) and matmul (2-3x) optimizations, the new bottleneck is:
- **~0.4ms per kernel dispatch** (WebGPU command submission overhead)
- ~200+ kernel dispatches per forward pass
- Total overhead: ~80ms just from dispatch

### Potential Solutions
1. **Kernel fusion**: Combine multiple operations into single dispatches
2. **Graph-level optimization**: Batch shader dispatches
3. **Reduce recompilations**: Fix dynamic shape issues (torch.dynamo)

---

## Optimization 3: Kernel Fusion (SILU+MUL, ADD+SILU, ADD+GELU)
- **Date**: 2026-01-16
- **Change**: Implemented fused binary operations to reduce kernel dispatch count
  - `fused_mul_silu`: gate * silu(up) - for GLU activation in MLP
  - `fused_add_silu`: silu(a + b) - add followed by SiLU
  - `fused_add_gelu`: gelu(a + b) - add followed by GELU
- **Files**: `csrc/ops/binary.cpp`, `csrc/ops/binary.h`, `csrc/ops/webgpu_ops.cpp`, compiler passes

### Results
| Operation | Correctness |
|-----------|-------------|
| fused_mul_silu | max_diff < 1e-7 |
| fused_add_silu | max_diff < 1e-6 |
| fused_add_gelu | max_diff < 1e-3 (approximation) |

### Impact on Full Model
- **Tokens/sec**: ~9.27 (no significant change from ~9.87 baseline after opt 2)
- Fusion reduces dispatch count marginally (~10-20 fewer dispatches)
- With ~200 total dispatches at ~0.4ms each, saving 10-20 dispatches = 4-8ms
- Overall dispatch overhead still dominates (~80ms)

### Analysis
Kernel fusion provides correct implementations but minimal end-to-end improvement because:
1. Per-dispatch overhead (~0.4ms) is a WebGPU/Dawn limitation
2. ~200 kernel dispatches per forward pass = ~80ms overhead minimum
3. Even eliminating 10-20% of dispatches only saves ~8-16ms

---

## Optimization Phase Conclusion

**Stop condition reached**: 3 consecutive optimization attempts yielded <5% end-to-end improvement.

### Final Performance
| Metric | Baseline | After All Optimizations |
|--------|----------|------------------------|
| Tokens/sec (avg) | 11.18 | 10.04 |
| Tokens/sec (stable, excl. first run) | ~13.4 | ~11.8 |
| TTFT | ~71ms | ~73ms |

### Root Cause Analysis
The fundamental bottleneck is **WebGPU command submission overhead**:
- Each kernel dispatch costs ~0.4ms
- Qwen2.5-0.5B requires ~200 kernel dispatches per forward pass
- Total dispatch overhead: ~80ms (dominates the ~100ms forward pass)
- Even 84x faster softmax and 2-3x faster matmul kernels can't overcome this

### What Would Help
1. **Dawn/WebGPU improvements**: Batch command submission, pipeline caching
2. **Graph compiler**: Fuse entire subgraphs into single kernels
3. **Vulkan backend**: Direct GPU API access to reduce overhead

---

## Summary

| Optimization | Isolated Speedup | End-to-End Impact |
|--------------|------------------|-------------------|
| Parallel Softmax | 84x | Softmax no longer bottleneck |
| Tiled Matmul | 2-3x | Limited by dispatch overhead |
| Kernel Fusion | N/A | <5% improvement |

**Conclusion**: torch-webgpu performance is fundamentally limited by WebGPU per-dispatch overhead (~0.4ms).
Further optimization requires architectural changes beyond kernel-level improvements.

---

## Optimization 4: Command Batching (Experimental)
- **Date**: 2026-01-16
- **Change**: Implemented command batching to reduce WebGPU submission overhead
  - Created `CommandBatcher` class to batch multiple dispatches into single command buffer
  - Batch size: 16 dispatches before auto-flush
  - Flush on GPU→CPU copy to ensure correctness
- **Files**: `csrc/core/command_batcher.h`, `csrc/core/command_batcher.cpp`, updated ops

### Architecture

**Before (per-op submission):**
```
op1: CreateEncoder → Dispatch → Submit
op2: CreateEncoder → Dispatch → Submit
... (200 times per forward)
```

**After (batched submission):**
```
CreateEncoder → Dispatch(op1) → Dispatch(op2) → ... → Dispatch(op16) → Submit
CreateEncoder → Dispatch(op17) → ... → Submit
```

### Results
| Metric | Before Batching | After Batching | Change |
|--------|-----------------|----------------|--------|
| Tokens/sec (stable) | ~11.8 | ~11.8 | ~0% |
| TTFT | ~73ms | ~74ms | ~0% |

### Analysis: Why No Improvement?

1. **Token generation is inherently sequential**
   - Each token requires the previous token's output
   - Forces GPU→CPU copy (argmax) after each forward pass
   - This triggers a flush, preventing cross-token batching

2. **Within-forward batching limited**
   - 200 ops / 16 batch size = ~12 submits (vs 200)
   - But even 12 submits at 0.4ms each = only 5ms saved
   - Other overhead (buffer creation, bind groups) still ~0.3ms per op

3. **The fundamental issue remains**
   - Dispatch overhead is only ~40% of total per-op overhead
   - Buffer creation, bind group creation, etc. are not batched
   - True solution requires graph-level compilation or persistent kernels

---

## Optimization 5: Buffer Pooling
- **Date**: 2026-01-16
- **Change**: Implemented buffer pooling to reduce uniform buffer allocation overhead
  - Created `BufferPool` class with size classes (64, 128, 256, 512, 1024, 2048, 4096 bytes)
  - Buffers are reused instead of created per-op
  - `acquireUniformBuffer()` helper function for easy integration
  - Buffers automatically released back to pool on batch flush
- **Files**: `csrc/core/buffer_pool.h`, `csrc/core/buffer_pool.cpp`, updated ops

### Architecture

**Before (per-op buffer creation):**
```
CreateBuffer(256 bytes) → WriteBuffer → Use → (buffer garbage collected)
CreateBuffer(256 bytes) → WriteBuffer → Use → (buffer garbage collected)
... (200 times per forward)
```

**After (buffer pooling):**
```
// First forward pass:
pool.acquire(256) → WriteBuffer → Use → pool.release() on flush
pool.acquire(256) → WriteBuffer → Use → pool.release() on flush
... (200 times, but buffers reused on subsequent passes)

// Subsequent passes:
pool.acquire(256) → [reuses existing buffer] → WriteBuffer → Use → release
```

### Results
| Metric | Before Pooling | After Pooling | Change |
|--------|----------------|---------------|--------|
| Tokens/sec (avg) | ~9.4 | ~9.4 | ~0% |
| Tokens/sec (stable, excl. first run) | ~10.3 | ~10.3 | ~0% |
| TTFT | ~74ms | ~77ms | ~0% |

### Analysis: Why Minimal Improvement?

1. **Buffer creation is a small fraction of overhead**
   - CreateBuffer: ~0.06ms per call
   - 200 ops × 0.06ms = ~12ms total
   - Even if we eliminate 100% of this, it's only ~12ms savings

2. **WriteBuffer still happens every time**
   - We still call `WriteBuffer()` on each op to update params
   - This is unavoidable since params change per-op

3. **Bind group creation remains dominant**
   - CreateBindGroup: ~0.15-0.2ms per call
   - 200 ops × 0.15ms = ~30ms total
   - This is not addressed by buffer pooling

---

## Optimization 6: Bind Group Caching
- **Date**: 2026-01-16
- **Change**: Implemented bind group caching to avoid recreation overhead
  - Created `BindGroupCache` class with hash-based lookup
  - Cache key: (pipeline pointer, buffer pointers, buffer sizes)
  - Max cache size: 1024 entries with simple eviction
  - Integrated into binary ops, unary ops, softmax, and matmul
- **Files**: `csrc/core/bind_group_cache.h`, `csrc/core/bind_group_cache.cpp`, updated ops

### Architecture

**Before (per-op bind group creation):**
```
CreateBindGroup({buf1, buf2, buf3, params}) → Use → (bind group GC'd)
CreateBindGroup({buf4, buf5, buf6, params}) → Use → (bind group GC'd)
... (200 times per forward)
```

**After (bind group caching):**
```
cache.get(key) → miss → CreateBindGroup() → cache.put() → Use
cache.get(key) → hit → Use (skip CreateBindGroup)
```

### Results
| Metric | Before Caching | After Caching | Change |
|--------|----------------|---------------|--------|
| Tokens/sec (avg) | ~9.4 | ~9.4 | ~0% |
| Tokens/sec (stable, excl. first run) | ~10.3 | ~10.3 | ~0% |
| TTFT | ~77ms | ~75ms | ~0% |

### Analysis: Why Minimal Improvement?

1. **Very low cache hit rate**
   - Input/output tensor buffers change per token
   - Even with buffer pooling, different ops use different data buffers
   - Cache key includes all buffer pointers, so different buffers = cache miss

2. **Token generation is sequential**
   - Each token uses different activations (hidden states)
   - Different activations = different buffers = cache miss
   - Only weight buffers are reused, but combined with different activation buffers

3. **Inference pattern doesn't enable caching**
   - Forward pass: input1, weight1 → hidden1 → ... → output1
   - Next token: input2 (from output1), weight1 → hidden2 → ... → output2
   - The input/output buffers change each token, breaking cache key match

### Conclusion

Bind group caching doesn't help because:
- In LLM inference, activation buffers change every token
- Only weight buffers remain constant, but they're combined with changing activation buffers
- The cache key (all buffer pointers) rarely matches between operations

---

## Optimization Summary (After 6 Attempts)

We've now tried:
1. **Parallel Softmax** - 84x isolated improvement, softmax no longer bottleneck
2. **Tiled Matmul** - 2-3x isolated improvement, limited by dispatch overhead
3. **Kernel Fusion** - <5% end-to-end improvement
4. **Command Batching** - ~0% improvement (sequential token generation forces flushes)
5. **Buffer Pooling** - ~0% improvement (buffer creation is small fraction of overhead)
6. **Bind Group Caching** - ~0% improvement (very low hit rate due to changing buffers)

### Root Cause Confirmed

The fundamental bottleneck is **per-operation overhead in WebGPU**, which includes:
- Command encoder/dispatch overhead: ~0.15ms
- Bind group creation: ~0.15ms
- Buffer operations (even with pooling): ~0.05ms
- Other overhead (pipeline state, etc.): ~0.05ms

Total per-op overhead: ~0.3-0.4ms × 200 ops = ~60-80ms per forward pass

### What Would Actually Help

1. **Graph compilation** - Compile entire forward pass into single command buffer
2. **Mega-kernel** - Single kernel that executes multiple operations
3. **Different API** - Direct Vulkan/CUDA access to reduce overhead
4. **Smaller models** - Fewer ops per forward = less overhead

