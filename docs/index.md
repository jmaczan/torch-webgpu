# torch-webgpu

Compile and run PyTorch models on WebGPU

<p align="center">
<img src="https://raw.githubusercontent.com/jmaczan/torch-webgpu/main/webgpu.png" height="100" width="100">
</p>

## What is torch-webgpu?

torch-webgpu is an experimental WebGPU backend for PyTorch that allows you to:

- **Run PyTorch on WebGPU** using `device="webgpu"`
- **Compile PyTorch models** with `@torch.compile(backend="webgpu")`
- **Run LLMs on WebGPU** - Qwen 0.5B works today!

## Why WebGPU?

WebGPU is a modern graphics and compute API that:

- Runs everywhere - Windows, macOS, Linux
- Works in all major browsers (Chrome, Firefox, Safari, Edge)
- Provides a unified API across different GPU vendors
- Is the future of portable GPU computing

## Quick Example

```python
import torch
import torch_webgpu

# Use WebGPU as a device
x = torch.tensor([1.0, 2.0, 3.0], device="webgpu")
y = x * 2
print(y)  # tensor([2., 4., 6.], device='webgpu')
```

## Compile an LLM

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_webgpu.compiler.webgpu_compiler import webgpu_backend

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
compiled_model = torch.compile(model, backend=webgpu_backend)

# Run inference on WebGPU!
outputs = compiled_model(input_ids)
```

## Current Status

torch-webgpu is at version **0.0.1** - experimental but functional.

- ✅ Basic tensor operations
- ✅ torch.compile backend
- ✅ LLM inference (Qwen 0.5B)
- 🚧 Performance optimizations
- 🚧 More op coverage

## Get Started

Ready to try it? Head to the [Installation](getting-started/installation.md) guide.

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