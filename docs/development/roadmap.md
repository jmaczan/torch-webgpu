# Roadmap

## Goal

Make WebGPU a first-class PyTorch backend for portable AI/ML: complete and
correct like CUDA / MPS, as fast as the hardware allows, running anywhere
Dawn does (Vulkan, Metal, D3D12, OpenGL ES; spec-minimum limits), and
developed fearlessly behind extensive differential tests (everything is
checked against PyTorch's CPU results).

Phases: (1) completeness and correctness, (2) per-op performance,
(3) compiler (torch.compile / Inductor) for whole-model performance.

## Where things stand (Nov 2026)

- Eager: every OpInfo op passes forward and backward (float32; subset of
  other dtypes); CPU fallback only for the gaps below. HF LLMs generate
  token-exact vs CPU across ~15 families; ResNet / ViT / ConvNeXt match.
- Compiler: Inductor backend generating fused WGSL (`python/torch_webgpu/_inductor`).
  All OpInfo ops pass compiled, all dtypes. FX passes merge parallel linears
  and strip GQA KV repeats. Qwen2.5-0.5B static-cache generate ~300 tok/s
  (eager ~165) on an RTX 5090; GPT-2 ~480.
- Kernels: tiled GEMM (~33 TFLOPS f32), vectorized GEMVs (1.2-1.7 TB/s),
  implicit-GEMM convolutions with split-K, flash attention, split-key
  decode attention.
- Tests: OpInfo eager / gradients / compiled, compiler fuzzer, targeted op
  suites, runtime, e2e models, C++ gtests; GPU profiler; `benchmarks/suite.py`.
- CI (`.github/workflows/tests.yml`): Dawn builds and caches; the suites on
  lavapipe were down to two failures (asin, fixed); the OpenGL ES job needed
  libEGL (fixed). Confirm it goes green.

## Next steps

1. **CI green** on both matrix entries; then add macOS (Metal) and Windows
   (D3D12) jobs, and the nightly slow tier.
2. **Prefill attention**: the flash kernel runs at ~4-5 TFLOPS (scalar inner
   loops); register-tiled score / PV micro-tiles (more rows per workgroup
   alone did not help).
3. **Tensor cores**: Dawn's experimental subgroup-matrix feature for f16 /
   bf16 GEMM and attention (needs `allow_unsafe_apis`; shader-f16 on NVIDIA
   needs the `vulkan_enable_f16_on_nvidia` toggle).
4. **Decode fusion**: GEMV prologue / epilogue templates in Inductor (RMSNorm,
   SiLU * mul, residual add, RoPE) to cut the ~11 dispatches per layer;
   lower per-launch host cost (~2.2 us per compiled kernel) since generate is
   host-bound under its per-token sync.
5. **Eager completeness gaps** (currently CPU fallback): linalg
   decompositions, FFT (stft for audio models), complex dtypes, special
   functions pass 2 (Bessel, orthogonal polynomials, igamma), ctc_loss,
   segment_reduce, sparse.
6. **Robustness**: a rare segfault at process exit (mitigated by draining the
   queue at exit) and a one-off hang seen once in the compiler tests; recheck
   the Dawn lazy-clear race workaround when upgrading Dawn.
