# Metal Profiling Guide for WebGPU Analysis

## Goal
Understand why Metal backend shows different optimization behavior than Vulkan:
- Fusion provides no benefit on Metal (0.95x) vs 1.4-1.7x on Vulkan
- Mega-kernels help Metal (1.5x) but not Vulkan
- Higher per-dispatch overhead (71 µs vs 24-36 µs)

## Tools Required

1. **Xcode** (includes Instruments)
2. **Metal System Trace** template in Instruments
3. **GPU Frame Debugger** (optional, for shader analysis)

## Profiling Steps

### 1. Profile wgpu-native dispatch overhead

```bash
# Run the benchmark with Instruments attached
cd /path/to/torch-webgpu
xcrun xctrace record --template 'Metal System Trace' --launch -- python experiments/exp1_cross_gpu_webgpu.py
```

Or attach to running process:
1. Start benchmark: `python experiments/exp1_cross_gpu_webgpu.py`
2. Open Instruments
3. Choose "Metal System Trace"
4. Attach to Python process
5. Record during benchmark execution

### 2. What to Look For

In the Metal System Trace, examine:

#### Command Buffer Submission
- Time between `commit` calls
- Queue wait time
- Encoder overhead

#### GPU Timeline
- Compute shader execution time
- Gaps between dispatches
- Memory transfer overhead

#### Specific Metrics
- `MTLCommandBuffer commit` duration
- `MTLComputeCommandEncoder dispatch` duration
- Buffer allocation/deallocation time

### 3. Key Questions to Answer

1. **Why is dispatch overhead higher?**
   - Is it command encoding time?
   - Is it queue submission?
   - Is it synchronization?

2. **Why doesn't fusion help?**
   - Does fused kernel take longer to execute?
   - Is there memory bandwidth difference?
   - Is there shader compilation overhead?

3. **Why do mega-kernels help?**
   - Is it purely dispatch count reduction?
   - Is there better memory locality?
   - Is there reduced synchronization?

### 4. Expected Findings (Hypotheses to Test)

Based on our data, we hypothesize:
- Metal's command buffer model has higher per-commit overhead
- Unified memory reduces data movement benefit from fusion
- Metal may have different shader compilation caching

### 5. Recording Results

Save profiling results to:
- `experiments/results/metal_trace_dispatch.trace` (Instruments file)
- `experiments/results/metal_analysis.md` (written analysis)

Include:
- Screenshots of GPU timeline
- Measured durations for key operations
- Comparison with expected Vulkan behavior

## Alternative: GPU Frame Capture

For deeper shader analysis:

1. In Xcode, enable GPU Frame Capture
2. Run benchmark
3. Capture a frame during compute execution
4. Analyze shader performance, memory access patterns

## Minimal Reproduction

If full profiling isn't feasible, at least run this minimal test:

```python
# Run on M2 and record timing breakdown
python -c "
import wgpu
import time

adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
device = adapter.request_device_sync()
queue = device.queue

# Measure individual components
times = {'encode': [], 'submit': [], 'wait': []}

for _ in range(100):
    encoder = device.create_command_encoder()

    t0 = time.perf_counter()
    # ... encode commands ...
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    queue.submit([encoder.finish()])
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    queue.on_submitted_work_done_sync()
    t5 = time.perf_counter()

    times['encode'].append(t1-t0)
    times['submit'].append(t3-t2)
    times['wait'].append(t5-t4)

import numpy as np
for k, v in times.items():
    print(f'{k}: {np.mean(v)*1e6:.1f} µs')
"
```

This will show where the overhead is concentrated.
