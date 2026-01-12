# Using device="webgpu"

torch-webgpu registers a custom PyTorch device called `webgpu`. You can use it just like `cuda` or `mps`.

## Creating Tensors

```python
import torch
import torch_webgpu

# Create directly on WebGPU
x = torch.tensor([1.0, 2.0, 3.0], device="webgpu")
y = torch.randn(3, 4, device="webgpu")
z = torch.zeros(10, device="webgpu")
```

## Moving Tensors

```python
# CPU to WebGPU
cpu_tensor = torch.randn(3, 3)
webgpu_tensor = cpu_tensor.to("webgpu")

# WebGPU to CPU
back_to_cpu = webgpu_tensor.to("cpu")

# Alternative syntax
webgpu_tensor = cpu_tensor.webgpu()  # Not yet supported
```

## Checking Device

```python
x = torch.tensor([1.0], device="webgpu")
print(x.device)  # webgpu
print(x.is_webgpu)  # Not yet supported, use str(x.device) == "webgpu"
```

## Supported Data Types

Currently, torch-webgpu supports:

- ✅ `torch.float32` (default, recommended)
- 🚧 `torch.float16` (coming soon)
- 🚧 `torch.int32`
- 🚧 `torch.int64`

```python
# Use float32 for now
x = torch.tensor([1.0, 2.0], dtype=torch.float32, device="webgpu")
```

## Device Transfers

Data transfer between CPU and WebGPU:

```python
# CPU → WebGPU
x_cpu = torch.randn(1000, 1000)
x_webgpu = x_cpu.to("webgpu")  # Copies data to GPU memory

# WebGPU → CPU
x_back = x_webgpu.to("cpu")  # Copies data back

# In-place operations stay on device
x_webgpu.mul_(2)  # Still on WebGPU
```

## Limitations

Current limitations of the device backend:

1. **Only float32** - Other dtypes coming soon
2. **Synchronous execution** - Operations block until complete
3. **No direct CUDA interop** - Must go through CPU

## Best Practices

1. **Minimize transfers** - Keep data on WebGPU as long as possible
2. **Use torch.compile** - For models, the compiler is more optimized
3. **Batch operations** - Larger operations are more efficient
