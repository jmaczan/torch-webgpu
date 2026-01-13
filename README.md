# torch-webgpu

PyTorch compiler and WebGPU runtime, capable of running LLMs like [LLama 3.2 3B](examples/llama-3.2-3b.py) 

## Example: Compile and run an LLM

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch_webgpu.compiler.webgpu_compiler import webgpu_backend

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model.eval()

compiled_model = torch.compile(model, backend=webgpu_backend)

with torch.no_grad():
    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    input_ids = inputs["input_ids"]
    generated_ids = input_ids.clone()
    outputs = compiled_model(input_ids)
    for _ in range(10):
        outputs = compiled_model(generated_ids)
        next_token = outputs.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```

## Example: Tensor on WebGPU

```python
import torch
import torch_webgpu

# Use WebGPU as a device
x = torch.tensor([1.0, 2.0, 3.0], device="webgpu")
y = x * 2
print(y)  # tensor([2., 4., 6.], device='webgpu')
```

## Use

In Python:

`from torch_webgpu import webgpu_backend`

And now you can use `@torch.compile(backend=webgpu_backend)`, `device="webgpu"`, `to="webgpu"` to run and compile PyTorch on a real WebGPU!


## Installation

```bash
pip install torch-webgpu
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

## Cite

If you use this software, please cite it as below.

```bibtex
@software{Maczan_torch-webgpu_2025,
author = {Maczan, Jędrzej Paweł},
month = oct,
title = {{torch-webgpu - PyTorch compiler and WebGPU runtime}},
url = {https://github.com/jmaczan/torch-webgpu},
version = {1.0.0},
year = {2025}
}
```

## Credits

[Jędrzej Maczan, 2025 - ∞](https://jedrzej.maczan.pl/)