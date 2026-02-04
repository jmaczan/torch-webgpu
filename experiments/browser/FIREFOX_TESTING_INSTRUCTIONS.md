# Firefox WebGPU Testing Instructions (Windows 11 + RTX 2000)

## Prerequisites

1. **Firefox 121+** (stable) or Firefox Nightly
2. **Windows 11** with NVIDIA RTX 2000 laptop GPU
3. **WebGPU enabled** in Firefox

## Step 1: Enable WebGPU in Firefox

1. Open Firefox
2. Type `about:config` in the address bar
3. Accept the risk warning
4. Search for `dom.webgpu.enabled`
5. Set it to `true`
6. Search for `gfx.webgpu.force-enabled`
7. Set it to `true` (if available)
8. **Restart Firefox**

## Step 2: Run the Benchmark

### Option A: Local Server (Recommended)

1. Open a terminal/PowerShell in the `experiments/browser/` directory
2. Start a local HTTP server:
   ```powershell
   # Using Python
   python -m http.server 8765

   # Or using Node.js
   npx http-server -p 8765
   ```
3. Open Firefox and navigate to: `http://localhost:8765/benchmark.html`
4. Verify "WebGPU ready!" appears in the status
5. Click "Run All Benchmarks"
6. Wait for all tests to complete
7. Click "Download Results JSON" to save the results

### Option B: Direct File (May not work due to CORS)

1. Open Firefox
2. Press `Ctrl+O` to open file
3. Navigate to `experiments/browser/benchmark.html`
4. If WebGPU doesn't work, use Option A instead

## Step 3: Save Results

1. Click "Download Results JSON"
2. Save the file as `exp1_browser_firefox_win11_rtx2000.json`
3. Copy the file to `experiments/results/`

Alternatively, copy the JSON from the results display:
1. Select all text in the "Results" section
2. Save to `experiments/results/exp1_browser_firefox_win11_rtx2000.json`

## Expected Results

Based on our other measurements, we expect:

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| Sequential dispatch overhead | 25-50 µs | Similar to Chrome/Vulkan |
| Single-op overhead | 200-2000 µs | Includes sync overhead |
| RMSNorm fusion speedup | 1.0-1.7x | Depends on Firefox's backend |

## Troubleshooting

### "WebGPU not supported"
- Ensure Firefox 121+ is installed
- Verify `dom.webgpu.enabled = true` in about:config
- Try Firefox Nightly if stable doesn't work
- Check that you're using the NVIDIA GPU (not integrated Intel)

### Software Renderer Detected
If adapter info shows "llvmpipe" or "SwiftShader":
- Firefox is using software rendering
- Ensure NVIDIA drivers are up to date
- Check Windows Display Settings to confirm RTX 2000 is available

### Benchmark Hangs
- Some WebGPU operations may timeout
- Wait up to 60 seconds for each test
- If still hanging, reload the page and try again

## Data We Need

Please provide:
1. The downloaded JSON results file
2. The adapter info shown on the page (vendor, device, description)
3. Firefox version (Help > About Firefox)
4. NVIDIA driver version (Device Manager > Display Adapters > RTX 2000 > Properties > Driver)

## File Location

Results should be saved to:
```
experiments/results/exp1_browser_firefox_win11_rtx2000.json
```
