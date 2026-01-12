# Building from Source

This guide explains how to build torch-webgpu from source.

## Prerequisites

- Python 3.10+
- CMake 3.16+
- C++17 compatible compiler
- Git

### Platform-Specific

**Linux:**
```bash
sudo apt-get install cmake ninja-build libx11-dev libx11-xcb-dev libxcb1-dev
```

**macOS:**
```bash
brew install cmake ninja
```

**Windows:**
- Visual Studio 2019+ with C++ support
- CMake (from cmake.org or Visual Studio)

## Building Dawn

torch-webgpu uses Google Dawn as its WebGPU runtime.

### Automatic Build

```bash
./scripts/build-dawn.sh
```

This will:

1. Clone Dawn repository
2. Configure with CMake
3. Build `libwebgpu_dawn`
4. Install to `dawn-install/`

Build takes 20-40 minutes depending on your machine.

### Manual Build

If the script doesn't work, build Dawn manually:

```bash
# Clone Dawn
git clone https://dawn.googlesource.com/dawn
cd dawn

# Configure
cmake -S . -B out/Release \
    -DCMAKE_BUILD_TYPE=Release \
    -DDAWN_FETCH_DEPENDENCIES=ON \
    -DDAWN_ENABLE_D3D11=OFF \
    -DDAWN_ENABLE_D3D12=OFF \
    -DDAWN_BUILD_SAMPLES=OFF

# Build
cmake --build out/Release --target webgpu_dawn -j$(nproc)

# Set environment variable
export DAWN_PREFIX=/path/to/dawn/out/Release
```

## Building torch-webgpu

Once Dawn is built:

```bash
# Set Dawn location (if not using default)
export DAWN_PREFIX=/path/to/dawn-install

# Build and install
pip install -e .
```

Or use the build script:

```bash
./build.sh
```

## Verifying the Build

```python
import torch_webgpu
import torch

x = torch.tensor([1.0, 2.0, 3.0], device="webgpu")
print(x * 2)  # Should print: tensor([2., 4., 6.], device='webgpu')
```

## Build Options

### Debug Build

For debugging with symbols:

```bash
# Edit setup.py to use debug flags
extra_compile_args = ["-g", "-O0"]  # Instead of "-O2"
```

### Custom Dawn Location

```bash
export DAWN_PREFIX=/custom/path/to/dawn
pip install -e .
```

## Troubleshooting

### CMake not found

Install CMake 3.16 or higher.

### Dawn build fails

- Check you have enough disk space (~10GB needed)
- Check you have enough RAM (~8GB recommended)
- Try with fewer parallel jobs: `NUM_JOBS=2 ./scripts/build-dawn.sh`

### Missing headers

Install development packages for your platform (see Prerequisites).

### Linker errors

Make sure `DAWN_PREFIX` points to a directory containing:
- `lib/libwebgpu_dawn.so` (Linux)
- `lib/libwebgpu_dawn.dylib` (macOS)
- `lib/webgpu_dawn.dll` (Windows)
