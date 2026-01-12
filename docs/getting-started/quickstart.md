# Quick Start

This guide will get you running PyTorch on WebGPU in 5 minutes.

## Basic Usage

### Import torch-webgpu

```python
import torch
import torch_webgpu  # This registers the WebGPU backend
```

### Create Tensors on WebGPU

```python
# Create directly on WebGPU
x = torch.tensor([1.0, 2.0, 3.0], device="webgpu")

# Or move existing tensors
y = torch.randn(3, 4)
y_webgpu = y.to("webgpu")
```

### Perform Operations

```python
# All standard operations work
a = torch.randn(3, 3, device="webgpu")
b = torch.randn(3, 3, device="webgpu")

c = a + b
d = torch.matmul(a, b)
e = torch.relu(c)
```

### Move Back to CPU

```python
result = e.to("cpu")
print(result.numpy())
```

## Using torch.compile

For best performance with models, use `torch.compile`:

```python
from torch_webgpu.compiler.webgpu_compiler import webgpu_backend

@torch.compile(backend=webgpu_backend)
def my_model(x):
    x = torch.relu(x @ weights + bias)
    return x

# The function is compiled for WebGPU
output = my_model(input_tensor)
```

## Full Example: Simple Neural Network

```python
import torch
import torch.nn as nn
import torch_webgpu
from torch_webgpu.compiler.webgpu_compiler import webgpu_backend

# Define a simple model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Create and compile
model = MLP()
model.eval()
compiled_model = torch.compile(model, backend=webgpu_backend)

# Run inference
with torch.no_grad():
    input_data = torch.randn(1, 784)
    output = compiled_model(input_data)
    print(output.shape)  # torch.Size([1, 10])
```

## Next Steps

- [Run Qwen LLM](../examples/qwen.md) - Compile and run a real LLM
- [Supported Ops](../guide/ops.md) - See what operations are available
- [torch.compile Guide](../guide/compile.md) - Learn more about the compiler
