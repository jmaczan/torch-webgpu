# Installation

## From PyPI (Recommended)

```bash
pip install torch-webgpu
```

Supported platforms:

- Linux (x86_64)
- macOS (arm64 - Apple Silicon)
- Windows (x86_64)

## Requirements

- Python 3.10 or higher
- PyTorch 2.0 or higher

## Verify Installation

```python
import torch_webgpu
import torch

# Create a tensor on WebGPU
x = torch.tensor([1.0, 2.0, 3.0], device="webgpu")
print(x)
# tensor([1., 2., 3.], device='webgpu')
```

## From Source

For development or if you need the latest changes:

```bash
# Clone the repository
git clone https://github.com/jmaczan/torch-webgpu.git
cd torch-webgpu

# Build Dawn (WebGPU runtime)
./scripts/build-dawn.sh

# Install in development mode
pip install -e .
```

See [Building from Source](../development/building.md) for detailed instructions.

## Troubleshooting

### Import Error

If you get an import error, make sure PyTorch is installed:

```bash
pip install torch
```

### GPU Not Found

torch-webgpu uses Dawn (Google's WebGPU implementation) which supports:

- Vulkan (Linux, Windows)
- Metal (macOS)
- D3D12 (Windows)

Make sure you have appropriate GPU drivers installed.
