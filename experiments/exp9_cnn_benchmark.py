#!/usr/bin/env python3
"""
Experiment 9: CNN Dispatch Overhead Benchmark

Benchmarks ResNet-50 to validate dispatch overhead findings on CNN architecture.
CNNs have many small operations (conv-bn-relu per layer) which may exhibit
more pronounced dispatch overhead than LLMs.

Usage:
    pip install wgpu numpy scipy
    python exp9_cnn_benchmark.py --output results/exp9_cnn.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import wgpu
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    print("Warning: wgpu not available")


def get_system_info():
    """Collect system information."""
    import platform
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    if WGPU_AVAILABLE:
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            if adapter:
                adapter_info = adapter.info
                info["gpu_vendor"] = adapter_info.get("vendor", "unknown")
                info["gpu_device"] = adapter_info.get("device", "unknown")
                info["wgpu_backend"] = adapter_info.get("backend_type", "unknown")
        except Exception as e:
            info["gpu_error"] = str(e)

    return info


def benchmark_conv2d_sequence(device, queue, n_layers=50, channels=64, spatial=56, n_iterations=50):
    """
    Benchmark a sequence of conv2d operations (simulating ResNet-like architecture).
    Each layer = conv + relu (2 dispatches unfused, 1 dispatch fused).
    """

    # Simplified convolution shader (3x3 conv)
    conv_shader = f"""
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
        let idx = gid.x;
        let out_size = {channels}u * {spatial}u * {spatial}u;

        if (idx < out_size) {{
            // Simplified: just do element-wise operation to simulate workload
            var sum: f32 = 0.0;
            for (var i = 0u; i < 9u; i++) {{  // 3x3 kernel
                sum += input[idx] * weight[i];
            }}
            output[idx] = sum;
        }}
    }}
    """

    relu_shader = f"""
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
        let idx = gid.x;
        let size = {channels}u * {spatial}u * {spatial}u;

        if (idx < size) {{
            output[idx] = max(input[idx], 0.0);
        }}
    }}
    """

    # Fused conv+relu shader
    fused_shader = f"""
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
        let idx = gid.x;
        let out_size = {channels}u * {spatial}u * {spatial}u;

        if (idx < out_size) {{
            var sum: f32 = 0.0;
            for (var i = 0u; i < 9u; i++) {{
                sum += input[idx] * weight[i];
            }}
            output[idx] = max(sum, 0.0);  // Fused ReLU
        }}
    }}
    """

    # Create pipelines
    conv_module = device.create_shader_module(code=conv_shader)
    relu_module = device.create_shader_module(code=relu_shader)
    fused_module = device.create_shader_module(code=fused_shader)

    conv_pipeline = device.create_compute_pipeline(layout="auto", compute={"module": conv_module, "entry_point": "main"})
    relu_pipeline = device.create_compute_pipeline(layout="auto", compute={"module": relu_module, "entry_point": "main"})
    fused_pipeline = device.create_compute_pipeline(layout="auto", compute={"module": fused_module, "entry_point": "main"})

    # Buffer size for feature maps
    buffer_size = channels * spatial * spatial * 4
    weight_size = 9 * 4  # 3x3 kernel

    # Create buffers (alternating for ping-pong)
    buf_a = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    buf_b = device.create_buffer(size=buffer_size, usage=wgpu.BufferUsage.STORAGE)
    weight_buf = device.create_buffer(size=weight_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)

    n_workgroups = (channels * spatial * spatial + 63) // 64

    # Benchmark unfused (2 dispatches per layer)
    def run_unfused():
        for layer in range(n_layers):
            # Conv
            bg_conv = device.create_bind_group(
                layout=conv_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": buf_a}},
                    {"binding": 1, "resource": {"buffer": weight_buf}},
                    {"binding": 2, "resource": {"buffer": buf_b}}
                ]
            )
            encoder = device.create_command_encoder()
            pass1 = encoder.begin_compute_pass()
            pass1.set_pipeline(conv_pipeline)
            pass1.set_bind_group(0, bg_conv)
            pass1.dispatch_workgroups(n_workgroups)
            pass1.end()
            queue.submit([encoder.finish()])

            # ReLU
            bg_relu = device.create_bind_group(
                layout=relu_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": buf_b}},
                    {"binding": 1, "resource": {"buffer": buf_a}}
                ]
            )
            encoder = device.create_command_encoder()
            pass2 = encoder.begin_compute_pass()
            pass2.set_pipeline(relu_pipeline)
            pass2.set_bind_group(0, bg_relu)
            pass2.dispatch_workgroups(n_workgroups)
            pass2.end()
            queue.submit([encoder.finish()])

        queue.on_submitted_work_done_sync()

    # Benchmark fused (1 dispatch per layer)
    def run_fused():
        # Use list to allow modification in closure
        bufs = [buf_a, buf_b]
        for layer in range(n_layers):
            bg_fused = device.create_bind_group(
                layout=fused_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": bufs[0]}},
                    {"binding": 1, "resource": {"buffer": weight_buf}},
                    {"binding": 2, "resource": {"buffer": bufs[1]}}
                ]
            )
            encoder = device.create_command_encoder()
            pass1 = encoder.begin_compute_pass()
            pass1.set_pipeline(fused_pipeline)
            pass1.set_bind_group(0, bg_fused)
            pass1.dispatch_workgroups(n_workgroups)
            pass1.end()
            queue.submit([encoder.finish()])

            # Swap buffers
            bufs[0], bufs[1] = bufs[1], bufs[0]

        queue.on_submitted_work_done_sync()

    # Warmup
    for _ in range(3):
        run_unfused()
        run_fused()

    # Benchmark unfused
    unfused_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        run_unfused()
        end = time.perf_counter()
        unfused_times.append((end - start) * 1000)

    # Benchmark fused
    fused_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        run_fused()
        end = time.perf_counter()
        fused_times.append((end - start) * 1000)

    unfused_mean = np.mean(unfused_times)
    fused_mean = np.mean(fused_times)
    speedup = unfused_mean / fused_mean

    return {
        "n_layers": n_layers,
        "channels": channels,
        "spatial_size": spatial,
        "unfused_dispatches": n_layers * 2,
        "fused_dispatches": n_layers,
        "unfused_mean_ms": unfused_mean,
        "unfused_std_ms": np.std(unfused_times),
        "fused_mean_ms": fused_mean,
        "fused_std_ms": np.std(fused_times),
        "fusion_speedup": speedup,
        "per_dispatch_overhead_us": ((unfused_mean - fused_mean) / n_layers) * 1000,  # Time saved per layer
        "n_iterations": n_iterations
    }


def main():
    parser = argparse.ArgumentParser(description="CNN dispatch overhead benchmark")
    parser.add_argument("--output", type=str, default="results/exp9_cnn.json")
    parser.add_argument("--power-preference", type=str, default="high-performance",
                        choices=["high-performance", "low-power"])
    args = parser.parse_args()

    if not WGPU_AVAILABLE:
        print("ERROR: wgpu is required. Install with: pip install wgpu")
        return

    print("=" * 60)
    print("Experiment 9: CNN Dispatch Overhead Benchmark")
    print("=" * 60)

    # Setup wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference=args.power_preference)
    adapter_info = adapter.info
    print(f"\nGPU: {adapter_info.get('device', 'unknown')}")
    print(f"Backend: {adapter_info.get('backend_type', 'unknown')}")

    device = adapter.request_device_sync()
    queue = device.queue

    system_info = get_system_info()

    results = {
        "system_info": system_info,
        "experiments": {}
    }

    # Test different CNN configurations
    configs = [
        {"n_layers": 50, "channels": 64, "spatial": 56, "name": "ResNet50-early (64ch, 56x56)"},
        {"n_layers": 50, "channels": 256, "spatial": 14, "name": "ResNet50-mid (256ch, 14x14)"},
        {"n_layers": 50, "channels": 512, "spatial": 7, "name": "ResNet50-late (512ch, 7x7)"},
    ]

    for config in configs:
        print(f"\n{config['name']}...")
        result = benchmark_conv2d_sequence(
            device, queue,
            n_layers=config["n_layers"],
            channels=config["channels"],
            spatial=config["spatial"]
        )
        results["experiments"][config["name"]] = result

        print(f"  Unfused ({result['unfused_dispatches']} dispatches): {result['unfused_mean_ms']:.2f} ms")
        print(f"  Fused ({result['fused_dispatches']} dispatches): {result['fused_mean_ms']:.2f} ms")
        print(f"  Fusion speedup: {result['fusion_speedup']:.2f}x")
        print(f"  Implied per-dispatch overhead: {result['per_dispatch_overhead_us']:.1f} us")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, exp in results["experiments"].items():
        print(f"{name}: {exp['fusion_speedup']:.2f}x speedup, {exp['per_dispatch_overhead_us']:.1f} us/dispatch")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
