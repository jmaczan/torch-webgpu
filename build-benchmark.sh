#!/usr/bin/env bash
set -euo pipefail

DAWN_PREFIX=${DAWN_PREFIX:-$PWD/dawn-install}
TORCH_CMAKE_PREFIX_PATH=${TORCH_CMAKE_PREFIX_PATH:-$(python -c "import torch, sys; sys.stdout.write(torch.utils.cmake_prefix_path)")}

cmake -S benchmarks -B build/benchmarks \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX_PATH" \
  -DDAWN_PREFIX="$DAWN_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build/benchmarks -j

./build/benchmarks/bench_webgpu \
  --benchmark_repetitions=30 \
  --benchmark_min_time=0.2s \
  --benchmark_format=json \
  --benchmark_out=bench.json \
  --benchmark_out_format=json