#!/bin/bash
# Run experiments on MacBook Air M2
# Copy this directory to your Mac and run: ./run_on_mac.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Running Experiments on Apple Silicon"
echo "========================================"
echo ""

# Create results directory
mkdir -p results

# Install dependencies
echo "Installing dependencies..."
pip3 install wgpu numpy scipy --quiet

echo ""
echo "Running Experiment 1: Cross-GPU WebGPU..."
python3 exp1_cross_gpu_webgpu.py --output results/exp1_apple_m2_webgpu.json --iterations 100

echo ""
echo "========================================"
echo "Complete! Results saved to:"
echo "  results/exp1_apple_m2_webgpu.json"
echo ""
echo "Copy this file back to your main machine:"
echo "  scp results/exp1_apple_m2_webgpu.json user@main-machine:~/torch-webgpu/experiments/results/"
echo "========================================"
