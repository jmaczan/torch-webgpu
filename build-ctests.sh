#!/usr/bin/env bash
# Build the C++ (gtest) test-suite against the active Python's libtorch.
set -euo pipefail
cd "$(dirname "$0")"

DAWN_PREFIX=${DAWN_PREFIX:-$PWD/dawn-install}
PYTHON=${PYTHON:-python}
TORCH_CMAKE_PREFIX_PATH=${TORCH_CMAKE_PREFIX_PATH:-$($PYTHON -c "import torch, sys; sys.stdout.write(torch.utils.cmake_prefix_path)")}

cmake -S ctests -B build/ctests -G Ninja \
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-RelWithDebInfo} \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX_PATH" \
  -DPython3_EXECUTABLE="$(command -v $PYTHON)" \
  -DDAWN_PREFIX="$DAWN_PREFIX"

cmake --build build/ctests -j
