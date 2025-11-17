# torch-webgpu
Experimental WebGPU backend for PyTorch

## Installation
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
- wgpu::Queue.Submit() handled synchronously
- many ops fallback to CPU
- very few unit tests

## Device / to

- [x] CPU <-> WebGPU
- [ ] CUDA <-> WebGPU
- [ ] MPS <-> WebGPU
- [ ] Intel Gaudi <-> WebGPU
- [ ] XLA <-> WebGPU

## ATen Ops

### Core

- [x] empty.memory_format
- [x] empty_strided
- [x] as_strided
- [ ] empty_like
- [ ] zeros_like
- [ ] ones_like
- [ ] arange
- [ ] full
- [ ] rand
- [ ] randn
- [x] copy_
- [x] _copy_from
- [ ] clone
- [x] to.device
- [ ] to.dtype
- [ ] to
- [ ] quantize_per_tensor
- [ ] dequantize

### Arithmetic and activation functions

- [x] add.Tensor (f32)
- [ ] add.Scalar
- [ ] add
- [ ] sub.Tensor
- [ ] sub
- [x] mul.Tensor
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
- [x] gelu
- [x] silu
- [x] relu
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
- [x] masked_select

### Comparisons

- [x] bitwise_and.Tensor
- [x] eq.Tensor
- [ ] ne.Tensor
- [x] ne.Scalar
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

Note: This project is unrelated to [webgpu-torch](https://github.com/praeclarum/webgpu-torch), which is a neat PyTorch reimplementation in TypeScript targeting WebGPU

I mainly use Ascend's NPU backend for PyTorch https://github.com/ascend/pytorch, Elie's WebGPU guide https://eliemichel.github.io/LearnWebGPU/index.html, WGSL spec https://www.w3.org/TR/WGSL/ and PyTorch PrivateUse1 custom backend docs as a reference https://docs.pytorch.org/tutorials/advanced/privateuseone.html https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html https://docs.pytorch.org/tutorials/advanced/dispatcher

Built by [Jędrzej Maczan, 2025](https://jedrzej.maczan.pl/)