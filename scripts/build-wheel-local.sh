#!/bin/bash
# Build wheel locally for testing
# Usage: ./scripts/build-wheel-local.sh

set -e

echo "=== Building torch-webgpu wheel locally ==="

# Check DAWN_PREFIX
if [ -z "$DAWN_PREFIX" ]; then
    # Try common locations
    if [ -d "dawn-install" ]; then
        export DAWN_PREFIX="$PWD/dawn-install"
    elif [ -d "$HOME/dawn/install/Release" ]; then
        export DAWN_PREFIX="$HOME/dawn/install/Release"
    else
        echo "ERROR: DAWN_PREFIX not set and Dawn not found"
        echo "Either set DAWN_PREFIX or run: ./scripts/build-dawn.sh"
        exit 1
    fi
fi

echo "Using DAWN_PREFIX=$DAWN_PREFIX"

# Verify Dawn library exists
if [ "$(uname)" = "Darwin" ]; then
    LIB_NAME="libwebgpu_dawn.dylib"
elif [ "$(uname)" = "Linux" ]; then
    LIB_NAME="libwebgpu_dawn.so"
else
    LIB_NAME="webgpu_dawn.dll"
fi

if [ ! -f "$DAWN_PREFIX/lib/$LIB_NAME" ]; then
    echo "ERROR: Dawn library not found at $DAWN_PREFIX/lib/$LIB_NAME"
    echo "Contents of $DAWN_PREFIX/lib/:"
    ls -la "$DAWN_PREFIX/lib/" 2>/dev/null || echo "(directory doesn't exist)"
    exit 1
fi

echo "Dawn library found: $DAWN_PREFIX/lib/$LIB_NAME"

# Clean old builds
rm -rf dist/ build/ *.egg-info python/*.egg-info

# Build wheel
echo "Building wheel..."
pip wheel . --no-deps -w dist/

echo ""
echo "=== Build complete ==="
ls -la dist/

echo ""
echo "To install: pip install dist/*.whl --force-reinstall"
echo "To test: python -c \"import torch_webgpu; import torch; print(torch.tensor([1.0], device='webgpu'))\""
