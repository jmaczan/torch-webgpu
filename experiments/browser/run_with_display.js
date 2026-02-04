#!/usr/bin/env node
/**
 * Browser WebGPU benchmark with hardware GPU access via display
 */

const puppeteer = require('puppeteer');
const http = require('http');
const fs = require('fs');
const path = require('path');

async function startServer(port = 8765) {
    return new Promise((resolve) => {
        const server = http.createServer((req, res) => {
            const filePath = path.join(__dirname, 'benchmark.html');
            fs.readFile(filePath, (err, data) => {
                if (err) {
                    res.writeHead(500);
                    res.end('Error loading benchmark.html');
                    return;
                }
                res.writeHead(200, { 'Content-Type': 'text/html' });
                res.end(data);
            });
        });
        server.listen(port, () => {
            console.log(`Server running at http://localhost:${port}`);
            resolve(server);
        });
    });
}

async function runBenchmark() {
    console.log('Starting WebGPU browser benchmark with hardware GPU...');
    const startTime = Date.now();

    const server = await startServer(8765);

    try {
        // Launch Chrome with GPU enabled - NOT headless for GPU access
        const browser = await puppeteer.launch({
            headless: false,  // Required for hardware GPU
            args: [
                '--enable-unsafe-webgpu',
                '--enable-features=Vulkan,UseSkiaRenderer',
                '--use-vulkan',
                '--enable-gpu',
                '--disable-software-rasterizer',
                '--ignore-gpu-blocklist',
                '--no-sandbox',
                '--disable-gpu-sandbox',
                '--disable-dev-shm-usage',
            ],
            env: {
                ...process.env,
                DISPLAY: process.env.DISPLAY || ':0',
            }
        });

        const page = await browser.newPage();

        page.on('console', msg => {
            const type = msg.type();
            if (type === 'error') {
                console.log('Browser error:', msg.text());
            }
        });

        console.log('Navigating to benchmark page...');
        await page.goto('http://localhost:8765', { waitUntil: 'networkidle0', timeout: 30000 });

        await new Promise(r => setTimeout(r, 3000));

        // Check WebGPU status
        const status = await page.evaluate(() => document.getElementById('status')?.textContent || 'unknown');
        console.log('WebGPU Status:', status);

        const adapterInfo = await page.evaluate(() => document.getElementById('adapter-info')?.textContent || 'none');
        console.log('Adapter Info:', adapterInfo.trim().substring(0, 200));

        if (status.includes('not supported') || status.includes('error')) {
            // Get more diagnostic info
            const gpuCheck = await page.evaluate(async () => {
                if (!navigator.gpu) return 'navigator.gpu is undefined';
                try {
                    const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
                    if (!adapter) return 'requestAdapter returned null';
                    return 'Adapter available: ' + JSON.stringify(adapter.info || {});
                } catch (e) {
                    return 'Error: ' + e.message;
                }
            });
            console.log('GPU Check:', gpuCheck);

            if (gpuCheck.includes('undefined') || gpuCheck.includes('null')) {
                console.error('WebGPU not available with hardware GPU!');
                await browser.close();
                server.close();
                process.exit(1);
            }
        }

        // Check if run button is enabled
        const btnDisabled = await page.evaluate(() => document.getElementById('runBtn')?.disabled);
        if (btnDisabled) {
            console.error('Run button is disabled');
            await browser.close();
            server.close();
            process.exit(1);
        }

        console.log('Starting benchmarks...');
        await page.click('#runBtn');

        // Wait for benchmarks to complete
        let attempts = 0;
        while (attempts < 150) { // 5 minutes max
            await new Promise(r => setTimeout(r, 2000));
            attempts++;

            const resultsText = await page.evaluate(() => document.getElementById('results')?.textContent || '');

            if (resultsText.includes('Running:')) {
                console.log('Progress:', resultsText.substring(0, 50));
            }

            if (resultsText.startsWith('{') && resultsText.includes('"experiments"')) {
                console.log('Benchmarks complete!');
                break;
            }
        }

        // Get results
        const results = await page.evaluate(() => {
            const text = document.getElementById('results')?.textContent || '';
            try {
                return JSON.parse(text);
            } catch (e) {
                return { error: text };
            }
        });

        // Save results
        const outputPath = path.join(__dirname, '..', 'results', 'exp1_browser_chrome_hardware.json');
        fs.mkdirSync(path.dirname(outputPath), { recursive: true });
        fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
        console.log(`\nResults saved to: ${outputPath}`);

        // Print summary
        if (results.experiments) {
            console.log('\n' + '='.repeat(60));
            console.log('BROWSER BENCHMARK SUMMARY (HARDWARE GPU)');
            console.log('='.repeat(60));
            console.log(`Adapter: ${results.adapter_info?.device || 'unknown'}`);
            console.log(`Vendor: ${results.adapter_info?.vendor || 'unknown'}`);

            if (results.experiments.sequential_dispatches) {
                console.log(`TRUE per-dispatch overhead: ${results.experiments.sequential_dispatches.per_dispatch_us.toFixed(1)} µs`);
            }
            if (results.experiments.rmsnorm_fusion_speedup) {
                console.log(`RMSNorm fusion speedup: ${results.experiments.rmsnorm_fusion_speedup.toFixed(2)}x`);
            }
            console.log('='.repeat(60));
        } else if (results.error) {
            console.log('Error:', results.error.substring(0, 200));
        }

        await browser.close();

    } finally {
        server.close();
    }
}

runBenchmark().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
