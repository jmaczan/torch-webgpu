#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
export DAWN_PREFIX=/home/jedrzej/dev/dawn/install/Release
TORCH_CMAKE_PREFIX_PATH=${TORCH_CMAKE_PREFIX_PATH:-$(python -c "import torch, sys; sys.stdout.write(torch.utils.cmake_prefix_path)")}

cmake -S benchmarks -B build/benchmarks \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX_PATH" \
  -DDAWN_PREFIX="$DAWN_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120

cmake --build build/benchmarks -j

./build/benchmarks/bench_webgpu \
  --benchmark_repetitions=30 \
  --benchmark_min_time=0.2s \
  --benchmark_format=json \
  --benchmark_out=bench.json \
  --benchmark_out_format=json