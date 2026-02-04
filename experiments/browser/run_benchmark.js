#!/usr/bin/env node
/**
 * Automated browser WebGPU benchmark using Puppeteer
 *
 * Usage:
 *   npm install puppeteer
 *   node run_benchmark.js [--browser chrome|firefox] [--output results.json]
 */

const puppeteer = require('puppeteer');
const http = require('http');
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const browserArg = args.find(a => a.startsWith('--browser='))?.split('=')[1] || 'chrome';
const outputArg = args.find(a => a.startsWith('--output='))?.split('=')[1] || `results_browser_${browserArg}.json`;

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
    console.log(`Running WebGPU benchmark in ${browserArg}...`);

    // Start local server
    const server = await startServer(8765);

    try {
        // Launch browser with WebGPU enabled
        const launchOptions = {
            headless: false, // WebGPU often requires non-headless mode
            args: [
                '--enable-unsafe-webgpu',
                '--enable-features=Vulkan',
                '--use-vulkan',
                '--disable-dawn-features=disallow_unsafe_apis',
                '--disable-gpu-sandbox',
            ]
        };

        if (browserArg === 'firefox') {
            launchOptions.product = 'firefox';
            launchOptions.args = []; // Firefox has different flags
        }

        const browser = await puppeteer.launch(launchOptions);
        const page = await browser.newPage();

        // Enable console logging
        page.on('console', msg => console.log('Browser:', msg.text()));

        // Navigate to benchmark page
        await page.goto('http://localhost:8765', { waitUntil: 'networkidle0' });

        // Wait for WebGPU to initialize
        await page.waitForFunction(() => {
            return document.getElementById('runBtn') && !document.getElementById('runBtn').disabled;
        }, { timeout: 10000 }).catch(() => {
            console.log('WebGPU may not be available, checking status...');
        });

        // Check if WebGPU is available
        const status = await page.evaluate(() => document.getElementById('status').textContent);
        console.log('Status:', status);

        if (status.includes('not supported') || status.includes('error')) {
            console.error('WebGPU not available in this browser!');
            await browser.close();
            server.close();
            process.exit(1);
        }

        // Click run button
        console.log('Starting benchmarks...');
        await page.click('#runBtn');

        // Wait for benchmarks to complete (monitor results element)
        let lastText = '';
        while (true) {
            await new Promise(r => setTimeout(r, 1000));
            const resultsText = await page.evaluate(() => document.getElementById('results').textContent);

            if (resultsText !== lastText) {
                console.log('Progress:', resultsText.substring(0, 60) + '...');
                lastText = resultsText;
            }

            // Check if results contain JSON (benchmarks complete)
            if (resultsText.startsWith('{') && resultsText.includes('"experiments"')) {
                console.log('Benchmarks complete!');
                break;
            }

            // Timeout after 5 minutes
            if (Date.now() - startTime > 300000) {
                console.error('Benchmark timeout!');
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

        // Save results
        const outputPath = path.join(__dirname, '..', 'results', outputArg);
        fs.mkdirSync(path.dirname(outputPath), { recursive: true });
        fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
        console.log(`Results saved to: ${outputPath}`);

        // Print summary
        if (results.experiments) {
            console.log('\n=== SUMMARY ===');
            console.log(`Browser: ${browserArg}`);
            console.log(`Adapter: ${results.adapter_info?.device || 'unknown'}`);
            if (results.experiments.sequential_dispatches) {
                console.log(`Per-dispatch overhead: ${results.experiments.sequential_dispatches.per_dispatch_us.toFixed(1)} us`);
            }
            if (results.experiments.rmsnorm_fusion_speedup) {
                console.log(`RMSNorm fusion speedup: ${results.experiments.rmsnorm_fusion_speedup.toFixed(2)}x`);
            }
        }

        await browser.close();

    } finally {
        server.close();
    }
}

const startTime = Date.now();
runBenchmark().catch(console.error);
