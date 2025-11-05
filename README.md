# torch-webgpu
WebGPU backend for PyTorch

## Install
1. Clone this repo

2. Install `google/dawn`
Guide: https://github.com/google/dawn/blob/main/docs/quickstart-cmake.md. Set `DAWN_PREFIX=` to `dawn/install/Release` based on there is your `dawn` repo, like `DAWN_PREFIX=/home/user/dawn/install/Release`

3. In this repo, run `./build.sh`

## Use
`import torch_webgpu`

And now you can use `device="webgpu"` and `to="webgpu"` to run computation on webpgu in pytorch!

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
- [x] to.device  
- [ ] to.dtype  
- [ ] add.Tensor  
- [ ] add.Scalar  
- [ ] sub.Tensor  
- [ ] mul.Tensor  
- [ ] div.Tensor  
- [ ] neg  
- [ ] pow.Tensor_Scalar  
- [ ] eq.Tensor  
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
- [ ] abs  
- [ ] round  
- [ ] floor  
- [ ] ceil  
- [ ] view  
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
