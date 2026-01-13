# torch-webgpu

**PyTorch compiler and WebGPU runtime, capable of running LLM inference**

![PyPI status](https://img.shields.io/pypi/status/torch-webgpu)
![License](https://img.shields.io/pypi/l/torch-webgpu)
![Python versions](https://img.shields.io/pypi/pyversions/torch-webgpu)
![PyPI version](https://img.shields.io/pypi/v/torch-webgpu)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)


## Use

In Python:

`from torch_webgpu import webgpu_backend`

And now you can use `@torch.compile(backend=webgpu_backend)`, `device="webgpu"`, `to="webgpu"` to run and compile PyTorch on a real WebGPU!

## Installation

```bash
pip install torch-webgpu
```

## Why WebGPU?

WebGPU is a modern graphics and compute API that:

- Runs almost everywhere - Windows, macOS, Linux
- Works in all major browsers (Chrome, Firefox, Safari, Edge)
- Provides a unified API across different GPU vendors
- I believe is the future of portable GPU computing

## Example: Tensor on WebGPU

```python
import torch
import torch_webgpu

# Use WebGPU as a device
x = torch.tensor([1.0, 2.0, 3.0], device="webgpu")
y = x * 2
print(y)  # tensor([2., 4., 6.], device='webgpu')
```

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
