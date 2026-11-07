# torch-webgpu

**PyTorch compiler and WebGPU runtime, capable of running LLMs like [LLama 3.2 3B](examples/llama-3.2-3b.py)**

![PyPI status](https://img.shields.io/pypi/status/torch-webgpu)
![License](https://img.shields.io/pypi/l/torch-webgpu)
![Python versions](https://img.shields.io/pypi/pyversions/torch-webgpu)
![PyPI version](https://img.shields.io/pypi/v/torch-webgpu)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

See also the WebGPU benchmark paper: [Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers](https://arxiv.org/abs/2604.02344)

## Example: Tensors and models on WebGPU

```python
import torch
import torch_webgpu  # registers the "webgpu" device

x = torch.tensor([1.0, 2.0, 3.0], device="webgpu")
print(x * 2)  # tensor([2., 4., 6.], device='webgpu:0')

# Any PyTorch model, eager mode, inference or training
model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4)).to("webgpu")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 16, device="webgpu")), torch.randint(0, 4, (8,), device="webgpu"))
loss.backward()
opt.step()
```

## Example: Run an LLM with Hugging Face transformers

```python
import torch, torch_webgpu
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).to("webgpu")
ids = tok("The universe is", return_tensors="pt").input_ids.to("webgpu")
print(tok.decode(model.generate(ids, max_new_tokens=20, do_sample=False)[0]))
```

Greedy generation matches the CPU token for token, without CPU fallbacks, for GPT-2, GPT-Neo, OPT,
Pythia, Bloom, Llama 3.2, SmolLM2, Qwen 1.5 / 2.5 / 3, Phi-1.5, StableLM 2, OLMo 2, Granite 3.3 and
Danube 3 (`scripts/e2e_models.py`).

## Example: Compile with the WebGPU backend

```python
compiled_model = torch.compile(model, backend="webgpu_backend")
```

The default Inductor backend (`torch.compile(model)`) does not generate WebGPU code yet.

## What is supported

- **Device**: `torch.device("webgpu")`, `.to("webgpu")`, `Tensor.webgpu()`, `Module.webgpu()`, and the
  `torch.webgpu` module (synchronize, memory stats / `empty_cache`, RNG state, streams and events,
  device properties), like `torch.cuda`.
- **Ops**: native WGSL kernels for elementwise math (incl. special functions), reductions and scans,
  matmul, convolutions, pooling, normalization, indexing / scatter / gather, sorting, random sampling,
  losses, upsampling, padding, grid sampling, embedding bags, histograms and more. Every op in PyTorch's
  OpInfo database is checked against the CPU, forward and backward.
- **Dtypes**: float32, float16, bfloat16, int64 (exact, emulated), int32, int16, int8, uint8, bool.
  Half types compute in float32 (like CUDA's "opmath").
- **Autograd, `torch.autocast(device_type="webgpu")`, `torch.save` / `load`, pickling**, weight tying.
- **CPU fallback** for anything without a native kernel (it warns once per op; see below).

### Differences from CPU / CUDA

- **No float64 / complex compute**: WebGPU hardware has no float64. Such tensors can live on the device,
  but ops on them run on the CPU (with a warning saying so). Use float32.
- **Float32 denormals flush to zero** (as on most GPUs).
- Results agree with the CPU to float32 precision, not bit for bit (different summation orders, etc.).

## Configuration

| Environment variable | Effect |
| --- | --- |
| `TORCH_WEBGPU_FALLBACK_WARN=0` | Don't warn when an op falls back to the CPU |
| `TORCH_WEBGPU_DISABLE_CPU_FALLBACK=1` | Raise instead of falling back to the CPU |
| `TORCH_WEBGPU_BACKEND=vulkan\|metal\|d3d12\|opengl\|opengles` | Choose the native GPU API |
| `TORCH_WEBGPU_COMPAT=1` | Use WebGPU compatibility mode (implied by the OpenGL backends) |
| `TORCH_WEBGPU_LIMITS=default` | Create the device with the WebGPU spec's default limits (portability testing) |
| `TORCH_WEBGPU_FORCE_FALLBACK_ADAPTER=1` | Use a software adapter, if available |
| `TORCH_WEBGPU_MAX_BATCH=N` | Dispatches recorded before a command buffer is submitted (default 128) |
| `TORCH_WEBGPU_DUMP_SHADERS=dir` | Write every generated WGSL shader to `dir` |
| `TORCH_WEBGPU_DEBUG=1` | Log adapter / device information |

## Portability

Tensors larger than one storage-buffer binding (128 MiB on the smallest devices, often 1-4 GiB on
desktop GPUs) work: kernels bind only the bytes they touch and split work that doesn't fit. A single
tensor can't exceed the device's maximum buffer size. The test suite also runs against the spec's
minimum limits (`TORCH_WEBGPU_LIMITS=default`) and on Dawn's OpenGL / OpenGL ES backends.

## Installation

```bash
pip install torch-webgpu
```

## Development

```bash
python -m pip install -e . --no-build-isolation   # build (Dawn in ./dawn-install)
./test.sh                                          # pytest (ops, OpInfo forward + gradients, runtime) + C++ gtests
HF_HUB_OFFLINE=1 python scripts/e2e_models.py gpt2 Qwen/Qwen2.5-0.5B   # end-to-end model checks
```

## Documentation

https://torch-webgpu.maczan.pl

## Supported platforms

- Linux (x86_64)
- macOS (Apple Silicon)
- Windows (x86_64)

## FAQ

### Why?

WebGPU promises to run everywhere - on almost every hardware - and becomes well supported in web browser. This project is a bridge between PyTorch world and WebGPU world

### There is "web" in "WebGPU", so does it mean that I can run PyTorch in a browser now?

This is a step towards running PyTorch in a browser. The next step is to run PyTorch inside a browser. I am [actively researching](https://github.com/brython-dev/brython/issues/2656) how to do it - if this topic excites you too, contact me on [Twitter](https://x.com/jedmaczan) or open an Issue in this GitHub repo

### How serious are you about this project? Is it a research or PoC in mind or are you going to make it production quality?

Once we hit version 1.0.0, `torch-webgpu` will be a production-ready PyTorch backend. WebGPU is an exciting, emerging technology. As of Nov 2025 [all major browsers support WebGPU](https://web.dev/blog/webgpu-supported-major-browsers). I think that it's highly important to build a bridge between PyTorch and WebGPU.

### Will you upstream WebGPU backend to PyTorch or keep it out-of-tree forever?

We'll see, ideally I'd see it as a part of PyTorch core, but we need to get a very high quality first to allow ourselves to ask PyTorch maintainers about it

### Contributor policy

I have a very little time and need to be picky about contributions, so please make sure you contribute code that is:
- well thought
- covered with unit tests
- you understand everything what you wrote
- as concise as possible - I can't handle too big PRs, sorry!

Use LLM at your discretion, but provide exhaustive explanation of what you built and why. Write it by yourself to show that you really understand

I can understand if that sounds too picky, but since I build this project after hours, I need to cut any additional noise. Sorry and thanks for understanding!

### I don't like X about this project

That's ok. The main goal here is to build a bridge (for community) and learn ML compilers in depth (for me). The project moves regularly, at its own pace. Things improve, cover more use cases, get more tests, get rethinked and rewrote. A journey, insights and learning over a raw development velocity. That's a tradeoff I choose

### I wish you moved faster

You can fund the project to give me more spare time to work on it. My email: `github@maczan.pl`

### Did AI built it?

The project started 26 Oct 2025. I have been coding it by hand and learning a lot about PyTorch internals and ML compilation in general. Once I made the project to the point where you could compile and run MLP on WebGPU, on 10 Jan 2026 I started to generate many missing ops using AI agents. In just 2 days, AI boosted the project from compiling and running MLPs to compiling and running LLMs ❤️

### Open a GitHub issue if you have more questions. Thanks and let's build this bridge!

## Ops support 
Many of important ops are implemented. If any is missing, feel free to open a PR or an issue. Thanks!

## Device / to

- [x] CPU <-> WebGPU
- [ ] CUDA <-> WebGPU
- [ ] MPS <-> WebGPU
- [ ] Intel Gaudi <-> WebGPU
- [ ] XLA <-> WebGPU

## TODOs

- performance wasn't a priority yet
- only float32 supported
- `wgpu::Queue.Submit()` handled synchronously
- some ops fallback to CPU
- add more compiler optimizations
- get high performance without platform specific (CUDA, MPS, ROCm) kernels. Five ingredients should be enough to get there - PyTorch, Python, C++, WGSL shaders and WebGPU runtime. Currently, `torch-webpgu` uses Google Dawn
- implement missing ops

## Resources

- Ascend's NPU backend for PyTorch https://github.com/ascend/pytorch
- Elie's WebGPU guide https://eliemichel.github.io/LearnWebGPU/index.html
- WGSL spec https://www.w3.org/TR/WGSL/
- PyTorch PrivateUse1 custom backend docs as a reference https://docs.pytorch.org/tutorials/advanced/privateuseone.html https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html https://docs.pytorch.org/tutorials/advanced/dispatcher
- https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel
- https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html

Note: This project is unrelated to [webgpu-torch](https://github.com/praeclarum/webgpu-torch), which is a neat PyTorch reimplementation in TypeScript targeting WebGPU

## Dev resources

### Build from source (only for development)

1. Clone this repo
2. Build Dawn: `./scripts/build-dawn.sh` (or set `DAWN_PREFIX` to your Dawn installation)
3. Build: `./build.sh`

### C++ unit tests

0. Remember to rebuild your code before testing - `./build.sh`
1. `chmod +x build-ctests.sh run-ctests.sh`
2. Update `build-ctests.sh` with your paths
3. `rm -rf build/ctests`
4. `./build-ctests.sh`
5. `./run-ctests.sh`

### C++ benchmarks

0. Remember to rebuild your code before testing - `./build.sh` and optionally log in to your wandb.ai account
1. `chmod +x build-benchmark.sh run-benchmark.sh`
2. Update `build-benchmark.sh` with your paths
3. `rm -rf build/benchmarks`
4. `./build-benchmark.sh`
5. `./run-benchmark.sh`

### Python unit tests

0. Remember to rebuild your code before testing - `./build.sh`
1. `pytest tests` to run all tests. `pytest tests/ops/test_cos.py` to run a chosen test file, like here we test cosinus

## Credits

[Jędrzej Maczan, 2025 - ∞](https://jedrzej.maczan.pl/)
