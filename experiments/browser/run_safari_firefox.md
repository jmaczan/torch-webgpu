# Safari and Firefox WebGPU Testing on macOS

## Prerequisites

1. **macOS with M2** (or later Apple Silicon)
2. **Safari 17+** (WebGPU enabled by default)
3. **Firefox 121+** (WebGPU needs to be enabled)

## Setup

### Install Playwright (supports Safari/Firefox unlike Puppeteer)
```bash
cd experiments/browser
npm install playwright
npx playwright install webkit firefox
```

### Enable Firefox WebGPU
1. Open Firefox
2. Go to `about:config`
3. Search for `dom.webgpu.enabled`
4. Set to `true`
5. Restart Firefox

## Manual Testing (Recommended)

Since automated browser testing with hardware GPU can be unreliable, manual testing is recommended:

### Step 1: Start the benchmark server
```bash
cd experiments/browser
node -e "
const http = require('http');
const fs = require('fs');
const path = require('path');
const server = http.createServer((req, res) => {
    const filePath = path.join(__dirname, 'benchmark.html');
    fs.readFile(filePath, (err, data) => {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(data);
    });
});
server.listen(8765, () => console.log('Server at http://localhost:8765'));
"
```

### Step 2: Open in Safari
1. Open Safari
2. Navigate to `http://localhost:8765`
3. Check WebGPU status (should show "WebGPU ready!")
4. Click "Run Benchmarks"
5. Copy the JSON results

### Step 3: Open in Firefox
1. Open Firefox (with WebGPU enabled)
2. Navigate to `http://localhost:8765`
3. Check WebGPU status
4. Click "Run Benchmarks"
5. Copy the JSON results

### Step 4: Save results
Save results to:
- `experiments/results/exp1_browser_safari_m2.json`
- `experiments/results/exp1_browser_firefox_m2.json`

## Expected Results

Based on our wgpu-native Metal results (71 µs dispatch overhead), we expect:
- Safari: ~70-100 µs per dispatch (Metal backend + browser overhead)
- Firefox: ~70-100 µs per dispatch (Metal backend + browser overhead)

If WebGPU is unavailable, you'll see "WebGPU not supported" - this means the browser version doesn't support WebGPU or it's not enabled.

## Troubleshooting

### Safari shows "WebGPU not supported"
- Ensure Safari 17+
- Try Safari Technology Preview
- Check System Preferences > Safari > Advanced > Show Develop menu, then Develop > Experimental Features > WebGPU

### Firefox shows "WebGPU not supported"
- Ensure Firefox 121+
- Verify `dom.webgpu.enabled = true` in about:config
- Try Firefox Nightly if stable doesn't work

### Results show software renderer
- Check adapter info for "SwiftShader" or "llvmpipe"
- These indicate no hardware GPU access
- May need to run with display attached (not headless)
