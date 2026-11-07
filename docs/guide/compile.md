# Using torch.compile

`torch.compile` works on webgpu tensors with PyTorch's default compiler,
Inductor. torch-webgpu registers an Inductor backend for the `webgpu`
device that generates fused WGSL compute shaders:

```python
import torch
import torch_webgpu  # registers the device and the compiler backend

model = MyModel().to("webgpu")
compiled = torch.compile(model)          # no backend argument needed

with torch.no_grad():
    out = compiled(torch.randn(8, 512, device="webgpu"))
```

Training works too: forward, backward and optimizer steps are compiled
(`torch.compile(train_step)`).

## What it does

* **Fusion.** Chains of pointwise ops, reductions (sum, mean, max, argmax,
  var / Welford, softmax, norms) and their epilogues become single WGSL
  kernels: one dispatch where eager mode would launch one per op.
* **Every dtype eager supports.** float32 math for float32 / float16 /
  bfloat16 (rounded on store), exact int64 (emulated), int32, the narrow
  integer types and bool. float64 has no WGSL type; graph nodes using it run
  eagerly (through the CPU fallback), like in eager mode.
* **Matrix products** use the eager kernels (tiled GEMM, vectorized GEMV).
  Decode-size linear layers that read the same input (attention q / k / v,
  MLP gate / up) are merged into one multi-weight GEMV dispatch.
* **NaN / Inf semantics** match eager PyTorch inside fused kernels even
  though GPU drivers compile shaders with fast-math (special constants and
  divisions are kept opaque to the driver's optimizer).
* **Indexing** (gather, embedding, index_put with accumulate via atomics)
  reports out-of-bounds indices as `IndexError` at the next synchronization,
  as eager mode does.
* **Random numbers** (dropout, rand / randn) use a Philox generator inside
  the kernels.

Compiled graphs are cached by Inductor's FX graph cache; the cache key
includes the backend's code, so upgrades invalidate stale entries.

## Performance tips

* Small-batch LLM decoding is bound by reading the weights; compiled decode
  of Qwen2.5-0.5B runs in about 2 ms per token on an RTX 5090 (3x eager).
* Shapes that change every call (growing sequence lengths) recompile; use
  `dynamic=True` or a static KV cache.
* Find hot kernels with the GPU profiler:

```python
with torch.webgpu.profiler.profile() as prof:
    compiled(x)
print(prof.table())        # per-kernel GPU time from timestamp queries
```

## Inspecting generated code

```bash
TORCH_LOGS=output_code python my_script.py     # wrapper + WGSL sources
TORCH_WEBGPU_DUMP_SHADERS=/tmp/wgsl python my_script.py   # every compiled shader as a .wgsl file
```

Environment knobs:

| Variable | Effect |
|----------|--------|
| `TORCH_WEBGPU_NO_LINEAR_FUSION=1` | Keep parallel linear layers as separate GEMVs. |

## Legacy compiler

`torch_webgpu.compiler.webgpu_backend` is the original experimental
compiler (FX graph -> custom IR -> eager ops). It is kept for
compatibility; the Inductor path above is faster and covers all ops.
