# Using torch.compile

The `torch.compile` backend is the recommended way to run models on WebGPU. It traces your model and optimizes it for WebGPU execution.

## Basic Usage

```python
import torch
from torch_webgpu.compiler.webgpu_compiler import webgpu_backend

# Compile a function
@torch.compile(backend=webgpu_backend)
def my_function(x, y):
    return torch.relu(x @ y)

# Or compile a model
model = MyModel()
compiled_model = torch.compile(model, backend=webgpu_backend)
```

## How It Works

1. **Tracing** - PyTorch traces your function/model to create an FX graph
2. **IR Conversion** - The graph is converted to torch-webgpu's IR
3. **Lowering** - IR is lowered to WebGPU operations
4. **Execution** - Operations run on WebGPU

```
PyTorch Model → FX Graph → High IR → Low IR → WebGPU Ops
```

## Compiling Models

```python
import torch.nn as nn
from torch_webgpu.compiler.webgpu_compiler import webgpu_backend

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 256)

    def forward(self, x):
        return torch.relu(self.linear(x))

model = MyModel()
model.eval()  # Important: set to eval mode

# Compile
compiled = torch.compile(model, backend=webgpu_backend)

# Run
with torch.no_grad():
    output = compiled(torch.randn(1, 512))
```

## Compiler Options

```python
# Dynamic shapes (for variable input sizes)
compiled = torch.compile(model, backend=webgpu_backend, dynamic=True)

# Disable for debugging
compiled = torch.compile(model, backend=webgpu_backend, disable=True)
```

## What Gets Compiled

The compiler handles:

- ✅ Linear layers
- ✅ Activations (ReLU, SiLU, GELU, etc.)
- ✅ Normalization (LayerNorm, RMSNorm)
- ✅ Attention (scaled_dot_product_attention)
- ✅ Embeddings
- ✅ Element-wise ops (add, mul, div, etc.)
- ✅ Reductions (sum, mean, max, etc.)
- ✅ Shape ops (view, reshape, transpose, etc.)

## Debugging

Enable debug output to see what's happening:

```python
# In lowering.py, set:
DEBUG_LOWERING = True
```

This prints shape information for each operation.

## Common Issues

### Graph Break

If you see "graph break", some operation isn't supported:

```python
# This may cause a graph break
def forward(self, x):
    if x.sum() > 0:  # Dynamic condition - graph break!
        return x * 2
    return x
```

Solution: Keep control flow static.

### Unsupported Op

```
Unsupported FX op: some_op
```

The operation isn't implemented yet. Check [Supported Ops](ops.md) or open an issue.

## Performance Tips

1. **Use eval mode** - `model.eval()` before compiling
2. **Use no_grad** - `with torch.no_grad():` for inference
3. **Warm up** - First run compiles; subsequent runs are faster
4. **Batch inputs** - Larger batches are more efficient
