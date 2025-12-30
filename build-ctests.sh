#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
export DAWN_PREFIX=/home/jedrzej/dev/dawn/install/Release
TORCH_CMAKE_PREFIX_PATH=${TORCH_CMAKE_PREFIX_PATH:-$(python -c "import torch, sys; sys.stdout.write(torch.utils.cmake_prefix_path)")}

cmake -S ctests -B build/ctests \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX_PATH" \
  -DDAWN_PREFIX="$DAWN_PREFIX" \
  -DCMAKE_CUDA_ARCHITECTURES=120

cmake --build build/ctests -j