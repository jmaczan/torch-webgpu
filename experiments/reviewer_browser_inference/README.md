# Browser LLM Inference Benchmarks

This directory contains fully operational browser-based LLM inference benchmarks to address reviewer concerns about incomplete browser testing.

## Quick Start

### Option 1: Using Python HTTP Server (Recommended)

```bash
cd experiments/reviewer_browser_inference
python3 -m http.server 8080
```

Then open in your browser:
- **WebLLM Benchmark** (recommended): http://localhost:8080/webllm_benchmark.html
- **Transformers.js Benchmark**: http://localhost:8080/index.html

### Option 2: Direct File Opening

Some browsers allow opening HTML files directly, but CORS may block model loading. Use the HTTP server method for best results.

## Benchmark Files

### 1. `webllm_benchmark.html` (Recommended)

Uses [WebLLM](https://github.com/mlc-ai/web-llm) for full WebGPU-accelerated LLM inference.

**Supported Models:**
- Qwen2.5-0.5B (q4f16, ~300MB) - matches paper's test model
- Qwen2.5-1.5B (q4f16, ~900MB) - larger model for generalization
- Llama-3.2-1B (q4f16, ~700MB)
- Llama-3.2-3B (q4f16, ~1.8GB)
- TinyLlama-1.1B (q4f16, ~600MB)
- Phi-2 (q4f16, ~1.5GB)
- Gemma-2B (q4f16, ~1.2GB)

**Features:**
- Full end-to-end inference (not micro-benchmarks)
- Reports tokens/second with 95% confidence intervals
- Separates prefill and decode speeds
- Downloads results as JSON

### 2. `index.html`

Uses [Transformers.js](https://huggingface.co/docs/transformers.js) with ONNX Runtime WebGPU.

**Note:** Transformers.js WebGPU support is still maturing. WebLLM typically provides better performance.

## Where to Run

### Chrome (Recommended)
- Version 113+ required for WebGPU
- Best WebGPU performance on Windows/Linux with NVIDIA/AMD
- Run on: Windows (NVIDIA/AMD), Linux (NVIDIA/AMD), macOS (Apple Silicon)

### Safari
- Version 17+ required for WebGPU
- Good performance on Apple Silicon
- Run on: macOS (Apple M1/M2/M3)

### Edge
- Version 113+ required (same as Chrome)
- Run on: Windows

### Firefox
- WebGPU available but throttled (as documented in paper)
- Expected: ~1040 µs dispatch overhead, <1 tok/s
- Run for comparison to validate paper's Firefox findings

## Expected Results

Based on the paper's findings, you should expect:

| Browser | Platform | Expected Performance |
|---------|----------|---------------------|
| Chrome | Linux/NVIDIA | 5-15 tok/s (Qwen2.5-0.5B) |
| Chrome | Windows/NVIDIA | 5-15 tok/s |
| Chrome | macOS/Apple Silicon | 3-10 tok/s |
| Safari | macOS/Apple Silicon | 3-10 tok/s |
| Firefox | Any | <1 tok/s (throttled) |

**Note:** Browser inference will be slower than native Dawn due to:
- JavaScript/WASM overhead
- Model quantization (q4f16 vs fp16)
- Browser security sandboxing

## Collecting Results for Paper

1. Run each benchmark with:
   - 10+ runs for statistical validity
   - 3+ warmup runs
   - 50 tokens generated

2. Download JSON results using the "Download JSON" button

3. Run on multiple configurations:
   - Chrome on Linux with NVIDIA GPU
   - Chrome on Windows with NVIDIA GPU
   - Safari on macOS with Apple Silicon
   - Firefox on any platform (for throttling validation)

4. Collect results in `results/` directory

## Troubleshooting

### "WebGPU not supported"
- Update browser to latest version
- Chrome 113+, Safari 17+, Edge 113+ required
- Firefox: enable `dom.webgpu.enabled` in about:config

### Model loading fails
- Check network connection
- Models download from Hugging Face Hub
- First load may take several minutes

### Low performance
- Close other GPU-intensive applications
- Ensure GPU drivers are up to date
- Check if discrete GPU is being used (not integrated)

### CORS errors
- Use the Python HTTP server method, not direct file opening
