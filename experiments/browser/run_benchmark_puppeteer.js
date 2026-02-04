#!/usr/bin/env node
/**
 * Automated browser WebGPU benchmark using Puppeteer
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
    console.log('Starting WebGPU browser benchmark...');
    const startTime = Date.now();

    // Start local server
    const server = await startServer(8765);

    try {
        // Launch Chrome with WebGPU enabled
        // Use 'new' headless mode which has better GPU support
        const browser = await puppeteer.launch({
            headless: 'new',
            args: [
                '--enable-unsafe-webgpu',
                '--enable-features=Vulkan,UseSkiaRenderer',
                '--use-vulkan',
                '--enable-gpu',
                '--disable-software-rasterizer',
                '--ignore-gpu-blocklist',
                '--disable-gpu-sandbox',
                '--no-sandbox',
                '--disable-setuid-sandbox',
            ]
        });

        const page = await browser.newPage();

        // Enable console logging
        page.on('console', msg => {
            if (msg.type() === 'error') {
                console.log('Browser error:', msg.text());
            }
        });

        console.log('Navigating to benchmark page...');
        await page.goto('http://localhost:8765', { waitUntil: 'networkidle0', timeout: 30000 });

        // Wait for initialization
        await new Promise(r => setTimeout(r, 2000));

        // Check WebGPU status
        const status = await page.evaluate(() => document.getElementById('status').textContent);
        console.log('WebGPU Status:', status);

        const adapterInfo = await page.evaluate(() => document.getElementById('adapter-info').textContent);
        console.log('Adapter Info:', adapterInfo.trim());

        if (status.includes('not supported') || status.includes('error')) {
            console.error('WebGPU not available!');

            // Try to get more info about why
            const gpuInfo = await page.evaluate(() => {
                return navigator.gpu ? 'navigator.gpu exists' : 'navigator.gpu is undefined';
            });
            console.log('GPU API:', gpuInfo);

            await browser.close();
            server.close();
            process.exit(1);
        }

        // Check if run button is enabled
        const btnDisabled = await page.evaluate(() => document.getElementById('runBtn').disabled);
        if (btnDisabled) {
            console.error('Run button is disabled - WebGPU initialization may have failed');
            await browser.close();
            server.close();
            process.exit(1);
        }

        // Click run button
        console.log('Starting benchmarks...');
        await page.click('#runBtn');

        // Wait for benchmarks to complete
        let lastProgress = '';
        while (true) {
            await new Promise(r => setTimeout(r, 2000));

            const resultsText = await page.evaluate(() => document.getElementById('results').textContent);

            // Show progress
            if (resultsText.includes('Running:') && resultsText !== lastProgress) {
                console.log('Progress:', resultsText);
                lastProgress = resultsText;
            }

            // Check if results contain JSON (benchmarks complete)
            if (resultsText.startsWith('{') && resultsText.includes('"experiments"')) {
                console.log('Benchmarks complete!');
                break;
            }

            // Timeout after 5 minutes
            if (Date.now() - startTime > 300000) {
                console.error('Benchmark timeout after 5 minutes!');
                break;
            }
        }

        // Get final results
        const results = await page.evaluate(() => {
            const text = document.getElementById('results').textContent;
            try {
                return JSON.parse(text);
            } catch (e) {
                return { error: text };
            }
        });

        // Get comparison text
        const comparison = await page.evaluate(() => document.getElementById('comparison').textContent);

        // Save results
        const outputPath = path.join(__dirname, '..', 'results', 'exp1_browser_chrome.json');
        fs.mkdirSync(path.dirname(outputPath), { recursive: true });
        fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
        console.log(`\nResults saved to: ${outputPath}`);

        // Print summary
        console.log('\n' + '='.repeat(60));
        console.log('SUMMARY');
        console.log('='.repeat(60));

        if (results.experiments) {
            console.log(`Adapter: ${results.adapter_info?.device || 'unknown'}`);

            if (results.experiments.dispatch_overhead_single_op) {
                console.log(`Single-op dispatch overhead: ${results.experiments.dispatch_overhead_single_op.mean_us.toFixed(1)} us`);
            }
            if (results.experiments.sequential_dispatches) {
                console.log(`TRUE per-dispatch overhead: ${results.experiments.sequential_dispatches.per_dispatch_us.toFixed(1)} us`);
            }
            if (results.experiments.rmsnorm_unfused) {
                console.log(`RMSNorm unfused: ${results.experiments.rmsnorm_unfused.mean_ms.toFixed(3)} ms`);
            }
            if (results.experiments.rmsnorm_fused) {
                console.log(`RMSNorm fused: ${results.experiments.rmsnorm_fused.mean_ms.toFixed(3)} ms`);
            }
            if (results.experiments.rmsnorm_fusion_speedup) {
                console.log(`Fusion speedup: ${results.experiments.rmsnorm_fusion_speedup.toFixed(2)}x`);
            }
        } else if (results.error) {
            console.log('Error:', results.error);
        }

        console.log('='.repeat(60));

        if (comparison) {
            console.log('\nComparison with native:');
            console.log(comparison);
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
