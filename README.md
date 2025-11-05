# torch-webgpu
WebGPU backend for PyTorch

## Install
1. Clone this repo

2. Install `google/dawn`
Guide: https://github.com/google/dawn/blob/main/docs/quickstart-cmake.md. Set `DAWN_PREFIX=` to `dawn/install/Release` based on there is your `dawn` repo, like `DAWN_PREFIX=/home/user/dawn/install/Release`

3. In this repo, run `./build.sh`

## Use
`import torch_webgpu`

And now you can use `device="webgpu"` in pytorch!


Note: This project is unrelated to [webgpu-torch](https://github.com/praeclarum/webgpu-torch), which is a neat PyTorch reimplementation in TypeScript targeting WebGPU

I mainly use Ascend's NPU backend for PyTorch https://github.com/ascend/pytorch, Elie's WebGPU guide https://eliemichel.github.io/LearnWebGPU/index.html and PyTorch PrivateUse1 custom backend docs as a reference https://docs.pytorch.org/tutorials/advanced/privateuseone.html https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html https://docs.pytorch.org/tutorials/advanced/dispatcher