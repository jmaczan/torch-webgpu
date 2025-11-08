# torch-webgpu
Experimental WebGPU backend for PyTorch

A very early stage of development, with limited functionality, lots of assumptions about data etc, like float32 only, contiguous data only. You probably don't want to use it yet, other than for playing around or contributing

Even though I code it after hours and in addition to contributing to PyTorch compiler, I'm serious about this project. I'd like to upstream the WebGPU backend to the PyTorch once we get good ops coverage, performance and a long tail of small things that have to work (different dtypes, etc). Estimated: late 2026

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

- contiguous data only
- (mostly) same dtypes of arguments expected; sporadic dtype conversions on CPU
- many ops fallback to CPU
- 1D-only tensor ops
- zero unit tests
- shaders not yet cached (see `add.Tensor`)
- one big beautiful™ `bindings.cpp` file
- probably at:: and c10:: mixed somewhere when using pytorch imports

## Device / to
- [x] WebGPU -> WebGPU
- [x] WebGPU -> CPU
- [x] CPU -> WebGPU
- [ ] CUDA <-> WebGPU
- [ ] MPS <-> WebGPU
- [ ] Intel Gaudi <-> WebGPU
- [ ] XLA <-> WebGPU

## ATen Ops
- [x] empty.memory_format
- [x] empty_strided
- [x] copy_
- [x] _copy_from
- [x] to.device
- [x] add.Tensor (f32, 1D)
- [x] ne.Scalar
- [x] bitwise_and.Tensor
- [x] eq.Tensor
- [x] abs
- [x] view
- [x] masked_select
- [ ] to.dtype
- [ ] add.Scalar
- [ ] sub.Tensor
- [ ] mul.Tensor
- [ ] div.Tensor
- [ ] neg
- [ ] pow.Tensor_Scalar
- [ ] ne.Tensor
- [ ] lt.Tensor
- [ ] le.Tensor
- [ ] gt.Tensor
- [ ] ge.Tensor
- [ ] relu
- [ ] clamp_min
- [ ] sigmoid
- [ ] tanh
- [ ] exp
- [ ] log
- [ ] sqrt
- [ ] rsqrt
- [ ] round
- [ ] floor
- [ ] ceil
- [ ] reshape
- [ ] transpose.int
- [ ] permute
- [ ] squeeze
- [ ] unsqueeze
- [ ] contiguous
- [ ] sum.dim_IntList
- [ ] mean.dim
- [ ] amax
- [ ] amin
- [ ] argmax
- [ ] argmin
- [ ] mm
- [ ] bmm
- [ ] matmul
- [ ] _log_softmax
- [ ] softmax.int
- [ ] layer_norm
- [ ] batch_norm
- [ ] conv2d
- [ ] conv2d_backward
- [ ] cat
- [ ] stack
- [ ] slice.Tensor
- [ ] index_select
- [ ] where.self
- [ ] clamp


Note: This project is unrelated to [webgpu-torch](https://github.com/praeclarum/webgpu-torch), which is a neat PyTorch reimplementation in TypeScript targeting WebGPU

I mainly use Ascend's NPU backend for PyTorch https://github.com/ascend/pytorch, Elie's WebGPU guide https://eliemichel.github.io/LearnWebGPU/index.html and PyTorch PrivateUse1 custom backend docs as a reference https://docs.pytorch.org/tutorials/advanced/privateuseone.html https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html https://docs.pytorch.org/tutorials/advanced/dispatcher