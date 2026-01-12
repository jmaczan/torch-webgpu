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
    -DBUILD_SHARED_LIBS=OFF \
    -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED \
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

# Find and copy library (handle different platforms and build configurations)
echo "Searching for Dawn library..."
FOUND_LIB=""

# Search for the library in common locations
for lib_path in \
    "$BUILD_DIR/src/dawn/native/libwebgpu_dawn.so" \
    "$BUILD_DIR/src/dawn/native/libwebgpu_dawn.dylib" \
    "$BUILD_DIR/src/dawn/native/Release/webgpu_dawn.dll" \
    "$BUILD_DIR/libwebgpu_dawn.so" \
    "$BUILD_DIR/libwebgpu_dawn.dylib" \
    ; do
    if [ -f "$lib_path" ]; then
        echo "Found: $lib_path"
        FOUND_LIB="$lib_path"
        cp "$lib_path" "$INSTALL_PREFIX/lib/"
        break
    fi
done

# Also try find command as fallback
if [ -z "$FOUND_LIB" ]; then
    echo "Searching with find..."
    FOUND_LIB=$(find "$BUILD_DIR" -name "libwebgpu_dawn.*" -o -name "webgpu_dawn.dll" 2>/dev/null | head -1)
    if [ -n "$FOUND_LIB" ]; then
        echo "Found via find: $FOUND_LIB"
        cp "$FOUND_LIB" "$INSTALL_PREFIX/lib/"
    fi
fi

# Windows: also copy .lib file if present
if [ -f "$BUILD_DIR/src/dawn/native/Release/webgpu_dawn.lib" ]; then
    cp "$BUILD_DIR/src/dawn/native/Release/webgpu_dawn.lib" "$INSTALL_PREFIX/lib/"
fi

if [ -z "$FOUND_LIB" ]; then
    echo "ERROR: Could not find Dawn library!"
    echo "Contents of BUILD_DIR:"
    find "$BUILD_DIR" -name "*webgpu*" -o -name "*dawn*" 2>/dev/null | head -20
    exit 1
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
