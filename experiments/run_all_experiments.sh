#!/bin/bash
# Run all reviewer-requested experiments
# Usage: ./run_all_experiments.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Running All Reviewer-Requested Experiments"
echo "========================================"
echo ""

# Create results directory
mkdir -p results

# Check dependencies
echo "Checking dependencies..."
python3 -c "import wgpu" 2>/dev/null || {
    echo "Installing wgpu..."
    pip install wgpu
}
python3 -c "import numpy" 2>/dev/null || pip install numpy
python3 -c "import scipy" 2>/dev/null || pip install scipy
python3 -c "import matplotlib" 2>/dev/null || pip install matplotlib

echo ""
echo "========================================"
echo "Experiment 1: Cross-GPU WebGPU Validation"
echo "========================================"
echo "NOTE: Run this on MacBook M2 for Apple Silicon results"
echo ""
python3 exp1_cross_gpu_webgpu.py --output results/exp1_webgpu.json --iterations 100

echo ""
echo "========================================"
echo "Experiment 2: Device-Side Argmax"
echo "========================================"
python3 exp2_device_argmax.py --output results/exp2_device_argmax.json --iterations 100

echo ""
echo "========================================"
echo "Experiment 3: Multi-Dispatch Tiled Mega"
echo "========================================"
python3 exp3_tiled_mega.py --output results/exp3_tiled_mega.json --iterations 100

echo ""
echo "========================================"
echo "Experiment 4: GPU Timeline Visualization"
echo "========================================"
python3 exp4_timeline.py --output results/exp4_timeline.json --dispatches 100

echo ""
echo "========================================"
echo "Experiment 5: CUDA Fusion Comparison"
echo "========================================"
python3 exp5_cuda_fusion.py --output results/exp5_cuda_fusion.json --iterations 1000

echo ""
echo "========================================"
echo "All Experiments Complete!"
echo "========================================"
echo ""
echo "Results saved to: experiments/results/"
echo ""
echo "Run 'python3 collect_results.py' to generate summary"
