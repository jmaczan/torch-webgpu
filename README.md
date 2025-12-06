# torch-webgpu
Experimental WebGPU backend for PyTorch

Not even 0.0.1 release yet! I make the repository public, so you give me support and I can get some dopamine out of it (building alone, in private, after day job, without a positive feedback - it is quite difficult, at least to me!)

**Goals**:
1. Run PyTorch on WebGPU `device="webgpu"`
2. Compile PyTorch code for WebGPU - `@torch.compile(m, backend=webgpu)`
3. High performance without platform specific (CUDA, MPS, ROCm) kernels. Five ingredients are enough to get there - PyTorch, Python, C++, WGSL shaders and WebGPU runtime. Currently, `torch-webpgu` uses Google Dawn

<p align="center">
<img src="webgpu.png" height="200" width="200">
</p>
<span>WebGPU logo by <a href="https://www.w3.org/"><abbr title="World Wide Web Consortium">W3C</abbr></a></span>

## Coolest thing you can do with torch-webgpu now
**Add tensors on WebGPU and move data between CPU and WebGPU!**

```py
a = torch.tensor([-1.5, 2.7, 1.0, 2.0], device="webgpu")
b = torch.tensor([-1.0, 0.9, 1.1, -2.1], device="webgpu")
result = a + b
expected = torch.tensor([-2.5, 3.6, 2.1, -0.1], device="cpu")
assert torch.allclose(result.to("cpu"), expected)
```

This is a TL;DR showcase of where we currently are with torch-webgpu. It will get regularly updated when new features land

## Installation
Only for developers and curious *very* early adopters

1. Clone this repo

2. Install `google/dawn`

Guide: https://github.com/google/dawn/blob/main/docs/quickstart-cmake.md. Set `DAWN_PREFIX=` to `dawn/install/Release` based on there is your `dawn` repo, like `DAWN_PREFIX=/home/user/dawn/install/Release`

3. In this repo, run `./build.sh`

## Use
In Python:

`import torch_webgpu`

And now you can use `device="webgpu"` and `to="webgpu"` to run pytorch on a real webgpu!

## Rough edges

This list helps me pick up what to work on next, aside of adding new ops

- only float32 supported
- `wgpu::Queue.Submit()` handled synchronously
- not enough unit tests ([a standarized testing out-of-tree backends is still in progress as of Dec 2025](https://dev-discuss.pytorch.org/t/testing-in-privateuse1-for-out-of-tree-pytorch-backends/3270), I hope to [involve torch-webgpu into this effort](https://dev-discuss.pytorch.org/t/testing-in-privateuse1-for-out-of-tree-pytorch-backends/3270/6))
- some ops might fallback to CPU

## Device / to

- [x] CPU <-> WebGPU
- [ ] CUDA <-> WebGPU
- [ ] MPS <-> WebGPU
- [ ] Intel Gaudi <-> WebGPU
- [ ] XLA <-> WebGPU

## FAQ

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

### Open a GitHub issue if you have more questions. Thanks and let's build this bridge!

## ATen Ops

### Core

- [x] empty.memory_format
- [x] empty_strided
- [x] as_strided
- [x] copy_
- [x] _copy_from
- [x] to.device
- [ ] empty_like
- [ ] zeros_like
- [ ] ones_like
- [ ] arange
- [ ] full
- [ ] rand
- [ ] randn
- [ ] clone
- [ ] to.dtype
- [ ] to
- [ ] quantize_per_tensor
- [ ] dequantize

### Arithmetic and activation functions

f32 only for now!

- [x] add.Tensor
- [x] gelu
- [x] silu
- [x] relu
- [x] masked_select
- [ ] add.Scalar
- [ ] add
- [ ] sub.Tensor
- [ ] sub
- [ ] mul.Tensor
- [ ] mul
- [ ] div.Tensor
- [ ] div
- [ ] neg
- [ ] pow.Tensor_Scalar
- [ ] pow
- [ ] sqrt
- [ ] rsqrt
- [ ] abs
- [ ] exp
- [ ] log
- [ ] tanh
- [ ] sigmoid
- [ ] clamp_min
- [ ] clamp
- [ ] round
- [ ] floor
- [ ] ceil
- [ ] minimum
- [ ] maximum
- [ ] where.self
- [ ] where
- [ ] masked_fill

### Comparisons

- [x] bitwise_and.Tensor
- [x] eq.Tensor
- [x] ne.Scalar
- [ ] ne.Tensor
- [ ] lt.Tensor
- [ ] le.Tensor
- [ ] gt.Tensor
- [ ] ge.Tensor

### Stats

- [ ] sum.dim_IntList
- [ ] sum
- [ ] mean.dim
- [ ] mean
- [ ] amax
- [ ] amin
- [ ] argmax
- [ ] argmin
- [ ] var_mean
- [ ] topk

### Tensor shapes, view, etc

- [x] view
- [x] resize
- [x] reshape
- [ ] flatten
- [ ] permute
- [ ] transpose.int
- [ ] transpose
- [ ] contiguous
- [ ] unsqueeze
- [ ] squeeze
- [ ] cat
- [ ] stack
- [ ] slice.Tensor
- [ ] slice
- [ ] select
- [ ] narrow
- [ ] expand
- [ ] broadcast_to
- [ ] index_select

### Linalg and attn

- [ ] addmm
- [ ] mm
- [ ] bmm
- [ ] matmul
- [ ] scaled_dot_product_attention
- [ ] _log_softmax
- [ ] softmax.int
- [ ] softmax
- [ ] layer_norm
- [ ] native_layer_norm
- [ ] rms_norm
- [ ] batch_norm
- [ ] group_norm
- [ ] embedding

### Convolutions

- [ ] conv2d
- [ ] conv2d_backward
- [ ] adaptive_avg_pool2d
- [ ] max_pool2d
- [ ] interpolate

## Resources

I mainly use Ascend's NPU backend for PyTorch https://github.com/ascend/pytorch, Elie's WebGPU guide https://eliemichel.github.io/LearnWebGPU/index.html, WGSL spec https://www.w3.org/TR/WGSL/ and PyTorch PrivateUse1 custom backend docs as a reference https://docs.pytorch.org/tutorials/advanced/privateuseone.html https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html https://docs.pytorch.org/tutorials/advanced/dispatcher

Note: This project is unrelated to [webgpu-torch](https://github.com/praeclarum/webgpu-torch), which is a neat PyTorch reimplementation in TypeScript targeting WebGPU

## Credits

[Jędrzej Maczan, 2025 - ∞](https://jedrzej.maczan.pl/)