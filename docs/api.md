# API Reference

## torch_webgpu Module

### Importing

```python
import torch_webgpu
```

Importing `torch_webgpu` registers the WebGPU backend with PyTorch. After importing, you can use `device="webgpu"`.

## Compiler

### webgpu_backend

```python
from torch_webgpu.compiler.webgpu_compiler import webgpu_backend
```

The `torch.compile` backend for WebGPU.

**Usage:**

```python
compiled_fn = torch.compile(fn, backend=webgpu_backend)
compiled_model = torch.compile(model, backend=webgpu_backend)
```

**Parameters for torch.compile:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | callable | Function or nn.Module to compile |
| `backend` | str or callable | Use `webgpu_backend` |
| `dynamic` | bool | Enable dynamic shapes (default: False) |
| `fullgraph` | bool | Require full graph capture (default: False) |
| `disable` | bool | Disable compilation (default: False) |

## Device Operations

### Creating Tensors

```python
# Direct creation
x = torch.tensor([1, 2, 3], device="webgpu")
x = torch.randn(3, 4, device="webgpu")
x = torch.zeros(10, device="webgpu")
x = torch.ones(5, 5, device="webgpu")
```

### Moving Tensors

```python
# To WebGPU
x_webgpu = x.to("webgpu")

# To CPU
x_cpu = x_webgpu.to("cpu")
```

## Internal APIs

!!! warning
    Internal APIs may change without notice.

### High-Level IR

```python
from torch_webgpu.compiler.high_ir import HighIROp, HighIRNode
```

Operations in the high-level intermediate representation.

### Low-Level IR

```python
from torch_webgpu.compiler.low_ir import LowIROp, LowIRNode
```

Operations in the low-level intermediate representation.

### Lowering

```python
from torch_webgpu.compiler.lowering import lowering, SHADER_TO_FUNC
```

Maps IR operations to actual PyTorch/WebGPU operations.

## C++ Extension

The C++ extension provides the core WebGPU operations:

```python
import torch_webgpu._C as _C
```

### torch.ops.webgpu

Registered operations accessible via `torch.ops.webgpu`:

| Operation | Description |
|-----------|-------------|
| `create_buffer` | Create a WebGPU buffer |
| `write_buffer` | Write data to a buffer |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DAWN_PREFIX` | Path to Dawn installation (for building) |

## Supported Devices

| Device String | Description |
|---------------|-------------|
| `"webgpu"` | WebGPU device |
| `"cpu"` | CPU device (standard PyTorch) |
