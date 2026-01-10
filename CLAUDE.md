# torch-webgpu

WebGPU backend for PyTorch. Goal: run `@torch.compile(backend="webgpu")` on real models.

## Current State

- Basic ops implemented
- WebGPU runtime working
- Basic IR, fx-to-IR conversion, lowerings in place

## Immediate Goal

Run Qwen/Qwen2.5-0.5B-Instruct fully on WebGPU via `@torch.compile(backend="webgpu")`.

## Architecture

```
PyTorch model
    ↓
torch.compile(backend="webgpu")
    ↓
FX Graph
    ↓
torch-webgpu IR
    ↓
Lowerings
    ↓
WGSL Shaders
    ↓
WebGPU Runtime
```

## What Qwen 0.5B Needs

Transformer architecture requires these ops:

### High Priority (blocking inference)
- [ ] Linear (matmul + bias)
- [ ] RMSNorm / LayerNorm
- [ ] RoPE (rotary position embeddings)
- [ ] Softmax
- [ ] Attention (scaled dot-product)
- [ ] SiLU / GELU activations
- [ ] Embedding lookup
- [ ] KV-cache operations

### Supporting Ops
- [ ] Transpose / permute
- [ ] Reshape / view
- [ ] Concatenation
- [ ] Slice / index
- [ ] Element-wise: add, mul, div
- [ ] Reduction: sum, mean, max
- [ ] Cast / type conversion

## Development Approach

1. **Op-by-op**: Pick one op, implement shader, add lowering, test
2. **Test against PyTorch**: Every op must match PyTorch output within tolerance
3. **Trace Qwen first**: Run FX trace on Qwen, get list of all ops needed
4. **Implement in dependency order**: Start from embedding, work through transformer layers

## Commands

### C++ unit tests

0. Remember to rebuild your code before testing - `./build.sh`
1. `chmod +x build-ctests.sh run-ctests.sh`
2. Update `build-ctests.sh` with your paths
3. `rm -rf build/ctests`
4. `./build-ctests.sh`
5. `./run-ctests.sh`

### C++ benchmarks

### Python unit tests

0. Remember to rebuild your code before testing - `./build.sh`
1. `pytest tests` to run all tests. `pytest tests/ops/test_cos.py` to run a chosen test file, like here we test cosinus

## Code Style

- Shaders in WGSL, one file per op when complex
- Lowerings map FX ops to IR ops
- IR ops map to shader dispatches
- Keep it simple, optimize later

## Testing Strategy

```python
def test_op_matches_pytorch():
    x = torch.randn(32, 64)
    expected = torch.some_op(x)
    
    compiled = torch.compile(torch.some_op, backend="webgpu")
    result = compiled(x)
    
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
```

## Key Files

```
torch_webgpu/
├── bench.json
├── benchmarks
│   ├── analyze.py
│   ├── bench_mm.cpp
│   └── CMakeLists.txt
├── bench_percentiles.json
├── bench_summary.csv
├── bench_summary.json
├── build/
├── build-benchmark.sh
├── build-ctests.sh
├── build.sh
├── CITATION.cff
├── CLAUDE.md
├── csrc
│   ├── bindings.cpp
│   ├── core
│   │   ├── webgpu_allocator.cpp
│   │   ├── webgpu_allocator.h
│   │   ├── webgpu_context.cpp
│   │   ├── webgpu_context.h
│   │   ├── webgpu_device_guard.cpp
│   │   └── webgpu_device_guard.h
│   ├── ops
│   │   ├── activation_functions.cpp
│   │   ├── arithmetic.cpp
│   │   ├── basic.cpp
│   │   ├── binary.cpp
│   │   ├── binary.h
│   │   ├── copy.cpp
│   │   ├── trig.cpp
│   │   ├── unary.cpp
│   │   ├── unary.h
│   │   └── webgpu_ops.cpp
│   ├── shaders
│   │   └── mm.wgsl
│   └── utils
│       ├── math.h
│       └── string.h
├── ctests
│   ├── CMakeLists.txt
│   ├── test_add.cpp
│   ├── test_as_strided.cpp
│   ├── test_copy.cpp
│   ├── test_copy_from_and_resize.cpp
│   ├── test_copy_from.cpp
│   ├── test_cos.cpp
│   ├── test_create_buffer.cpp
│   ├── test_empty_memory_format.cpp
│   ├── test_empty_strided.cpp
│   ├── test_fused_add_relu.cpp
│   ├── test_gelu.cpp
│   ├── test_math.cpp
│   ├── test_mm.cpp
│   ├── test_mul.cpp
│   ├── test_relu.cpp
│   ├── test_reshape.cpp
│   ├── test_resize.cpp
│   ├── test_silu.cpp
│   ├── test_to_device.cpp
│   ├── test_view.cpp
│   └── test_write_buffer.cpp
├── gflops_benchmark.png
├── pyproject.toml
├── python
│   ├── torch_webgpu
│   │   ├── _C.cpython-312-x86_64-linux-gnu.so
│   │   ├── compiler
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── webgpu
│   └── torch_webgpu.egg-info
│       ├── dependency_links.txt
│       ├── PKG-INFO
│       ├── requires.txt
│       ├── SOURCES.txt
│       └── top_level.txt
├── README.md
├── run-benchmark.sh
├── run-ctests.sh
├── run.sh
├── setup.py
├── temp.py
├── temp.txt
├── tests
│   ├── compile
│   │   └── test_ir.py
│   ├── ops
│   │   ├── test_add.py
│   │   ├── test_cos.py
│   │   ├── test_gelu.py
│   │   ├── test_mm.py
│   │   ├── test_mul.py
│   │   ├── test_reshape.py
│   │   └── test_silu.py
│   ├── playground.py
│   ├── test_compile_dev.py
│   ├── test_dev.py
│   └── test_nograd_reshape.py
├── test.sh
├── wandb/
```

## List of missing ops for running Qwen model:
 'G_model_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_',
 'G_model_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_',
 'G_model_modules_model_modules_norm_parameters_weight_', 'G_model_modules_model_modules_rotary_emb_buffers_inv_freq_',
 'L_input_ids_', '__and__', '__eq__', 'contiguous', 'cos', 'cumsum', 'expand', 'float', 'le', 'mean', 'new_ones',
 'operator.add', 'operator.getitem', 'operator.iadd', 'operator.matmul', 'operator.mul', 'operator.ne', 'operator.neg',
 'operator.sub', 'pow', 'reshape', 'sin', 'to', 'torch._C._log_api_usage_once', 'torch._C._nn.linear',
 'torch._C._nn.scaled_dot_product_attention', 'torch._C._set_grad_enabled',
 'torch._functorch.predispatch._add_batch_dim', 'torch._functorch.predispatch._remove_batch_dim',
 'torch._functorch.predispatch._vmap_decrement_nesting', 'torch._functorch.predispatch._vmap_increment_nesting',
 'torch._functorch.predispatch.lazy_load_decompositions', 'torch.amp.autocast_mode._enter_autocast',
 'torch.amp.autocast_mode._exit_autocast', 'torch.arange', 'torch.cat', 'torch.diff', 'torch.nn.functional.embedding',
 'torch.nn.functional.silu', 'torch.ops.aten.index', 'torch.rsqrt', 'transpose', 'unsqueeze', 'view'

## Next Steps
1. Implement ops one by one, starting with most common
2. Test each op in isolation
3. Run full model when all ops green

## Notes

- Numerical precision: fp32 first, fp16 later for performance
- Memory: WebGPU has different memory model, watch buffer sizes
- Async: WebGPU is async, batch operations where possible
- Write meaningful tests for all ops - both C++ and Python
- Use this approach: Make it work. Make it right. Make it fast.