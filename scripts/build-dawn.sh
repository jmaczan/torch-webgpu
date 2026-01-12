#!/bin/bash
# Build Google Dawn WebGPU runtime for torch-webgpu
# This script downloads and builds Dawn from source using CMake
# Usage: ./scripts/build-dawn.sh [install_prefix]

set -e

INSTALL_PREFIX="${1:-$PWD/dawn-install}"
DAWN_DIR="${DAWN_DIR:-$PWD/dawn}"
NUM_JOBS="${NUM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

echo "Building Dawn..."
echo "  Install prefix: $INSTALL_PREFIX"
echo "  Dawn source dir: $DAWN_DIR"
echo "  Parallel jobs: $NUM_JOBS"

# Clone Dawn if not exists
if [ ! -d "$DAWN_DIR" ]; then
    echo "Cloning Dawn repository..."
    git clone --depth 1 https://dawn.googlesource.com/dawn "$DAWN_DIR"
fi

cd "$DAWN_DIR"

# Create build directory
BUILD_DIR="$DAWN_DIR/out/Release"
mkdir -p "$BUILD_DIR"

# Generate build files with automatic dependency fetching
echo "Configuring CMake (dependencies will be fetched automatically)..."
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DDAWN_FETCH_DEPENDENCIES=ON \
    -DDAWN_ENABLE_D3D11=OFF \
    -DDAWN_ENABLE_D3D12=OFF \
    -DDAWN_ENABLE_NULL=OFF \
    -DDAWN_ENABLE_DESKTOP_GL=OFF \
    -DDAWN_ENABLE_OPENGLES=OFF \
    -DDAWN_BUILD_SAMPLES=OFF \
    -DTINT_BUILD_TESTS=OFF \
    -DTINT_BUILD_CMD_TOOLS=OFF

# Build
echo "Building Dawn (this may take 20-40 minutes)..."
cmake --build "$BUILD_DIR" --target webgpu_dawn -j "$NUM_JOBS"

# Install
echo "Installing Dawn to $INSTALL_PREFIX..."
mkdir -p "$INSTALL_PREFIX/lib" "$INSTALL_PREFIX/include"

# Copy library (handle different platforms)
if [ -f "$BUILD_DIR/src/dawn/native/libwebgpu_dawn.so" ]; then
    cp "$BUILD_DIR/src/dawn/native/libwebgpu_dawn.so" "$INSTALL_PREFIX/lib/"
elif [ -f "$BUILD_DIR/src/dawn/native/libwebgpu_dawn.dylib" ]; then
    cp "$BUILD_DIR/src/dawn/native/libwebgpu_dawn.dylib" "$INSTALL_PREFIX/lib/"
elif [ -f "$BUILD_DIR/src/dawn/native/Release/webgpu_dawn.dll" ]; then
    cp "$BUILD_DIR/src/dawn/native/Release/webgpu_dawn.dll" "$INSTALL_PREFIX/lib/"
    cp "$BUILD_DIR/src/dawn/native/Release/webgpu_dawn.lib" "$INSTALL_PREFIX/lib/" 2>/dev/null || true
fi

# Copy headers
cp -r "$DAWN_DIR/include/webgpu" "$INSTALL_PREFIX/include/" 2>/dev/null || true
cp -r "$DAWN_DIR/include/dawn" "$INSTALL_PREFIX/include/" 2>/dev/null || true
cp -r "$BUILD_DIR/gen/include/dawn" "$INSTALL_PREFIX/include/" 2>/dev/null || true

echo ""
echo "Dawn built successfully!"
echo "Library installed to: $INSTALL_PREFIX/lib/"
echo "Headers installed to: $INSTALL_PREFIX/include/"
ls -la "$INSTALL_PREFIX/lib/"
