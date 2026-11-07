# Running Tests

torch-webgpu is tested differentially: everything runs on webgpu and is
compared with PyTorch's CPU results for the same inputs. There are Python
(pytest) and C++ (gtest) suites.

```bash
./test.sh                      # everything: pytest + C++ (SKIP_CTESTS=1 for Python only)
pytest tests -m "not slow"     # the fast subset (a few minutes on a desktop GPU)
pytest tests/ops/test_sdpa.py  # one file
./build-ctests.sh && ./run-ctests.sh
```

## Suites

| Suite | What it checks |
|-------|----------------|
| `tests/opinfo/test_opinfo.py` | Every OpInfo in PyTorch's op database, eager on webgpu vs CPU: forward (float32 for all ops; float16, bfloat16, int64, int32, uint8, bool for a representative subset) and input gradients. |
| `tests/opinfo/test_opinfo_compiled.py` | The same op database through `torch.compile` (Inductor generating WGSL kernels, eager fallbacks for the rest). |
| `tests/compile/test_inductor.py` | Compiler unit tests: dtype classes, NaN / Inf semantics inside fused kernels, indexing and bounds errors, atomics, random numbers, dynamic shapes, aliasing, training steps. |
| `tests/compile/test_fuzz.py` | Differential fuzzing of the compiler with seeded random programs (fusions of pointwise ops, reductions, views, casts). |
| `tests/ops/` | Targeted op tests: edge cases OpInfo samples don't reach (NaN semantics, long sequences, large tensors, attention variants, sorting ties...). |
| `tests/runtime/` | Device API (memory stats, streams, RNG, errors, GPU profiler) and runs of the suites under the WebGPU spec's minimum limits. |
| `tests/e2e/` | Hugging Face models: greedy generation must match the CPU token for token without CPU fallbacks; training steps track the CPU. Models come from the local HF cache and are skipped when absent. |
| `ctests/` | C++ runtime core (allocator, launch grids, stream ordering, binding windows, WGSL encodings, device errors) and ops through the C++ API. |

## Shared infrastructure (`tests/conftest.py`)

* Every test ends with `torch.webgpu.synchronize()`, so asynchronous device
  errors (WebGPU validation errors, out-of-bounds indices, device loss) fail
  the test that caused them.
* `@pytest.mark.no_fallback` (or the `no_fallback` fixture) fails a test if
  any op went through the CPU fallback; the compile tests forbid fallbacks by
  default and opt out with `@pytest.mark.allow_fallback`.
* `@pytest.mark.slow` marks long tests (deselect with `-m "not slow"`).
* The pytest header shows the adapter and the `TORCH_WEBGPU_*` settings.

## Knobs

| Variable | Effect |
|----------|--------|
| `TORCH_WEBGPU_BACKEND=vulkan\|metal\|d3d12\|opengles\|...` | Pick the Dawn backend (OpenGL ES runs in WebGPU compatibility mode). |
| `TORCH_WEBGPU_LIMITS=default` | Request only the WebGPU spec's minimum limits (portability check). |
| `TORCH_WEBGPU_OPINFO_ALL_DTYPES=1` | OpInfo suites: every op for every dtype. |
| `TORCH_WEBGPU_OPINFO_MAX_SAMPLES`, `TORCH_WEBGPU_OPINFO_COMPILED_SAMPLES` | Samples per op (eager: 20, compiled: 3). |
| `TORCH_WEBGPU_FUZZ_PROGRAMS`, `TORCH_WEBGPU_FUZZ_SEED` | Number and seed offset of fuzzed programs (default 60 from 0). |

## Surveys

The OpInfo runner also works standalone and prints every failing op, which
is handy when working on coverage:

```bash
python tests/opinfo/opinfo_runner.py                    # eager forward, float32
python tests/opinfo/opinfo_runner.py --dtype bfloat16
python tests/opinfo/opinfo_runner.py --grad             # gradients
python tests/opinfo/opinfo_runner.py --compiled         # through torch.compile
python tests/opinfo/opinfo_runner.py --filter '^nn\.functional\.'
python scripts/op_coverage.py --missing                 # aten ops without a webgpu kernel
```

## Performance

`benchmarks/suite.py` measures kernels (GEMM, decode GEMVs, convolution,
attention, launch overhead; GPU time from timestamp queries) and models in
the local Hugging Face cache (decode, prefill and `generate`, eager and
compiled). Save a baseline before a change and compare after it:

```bash
python benchmarks/suite.py --out before.json
# ... change ...
python benchmarks/suite.py --baseline before.json   # exits 1 on >10% regressions
```

For a single workload, `torch.webgpu.profiler.profile()` gives per-kernel
GPU times (`prof.table()`).

## Continuous integration

`.github/workflows/tests.yml` builds Dawn at the pinned revision
(`scripts/build-dawn.sh`, cached) with the Vulkan and OpenGL ES backends and
runs the suites on GPU-less runners through Mesa's software drivers:
lavapipe (Vulkan) and llvmpipe (OpenGL ES, compatibility mode). Pull
requests run `-m "not slow"`; the nightly run includes the slow tests.

## Writing tests

* Compare against the CPU: `torch.testing.assert_close(out.cpu(), ref)`,
  with tolerances that follow the compute dtype (float32 math everywhere;
  half types round on store).
* Prefer extending OpInfo coverage (it gets forward, gradients, dtypes and
  compiled runs for free); add a targeted test in `tests/ops/` for edge
  cases samples miss (NaN / Inf, empty and huge tensors, ties, aliasing).
* Kernels run under drivers' fast-math: any WGSL that handles NaN / Inf
  needs a test with those values computed *inside* the kernel (see
  `tests/ops/test_nan_semantics.py`, `test_special_values` in
  `tests/compile/test_inductor.py`).
