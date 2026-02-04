#!/usr/bin/env python3
"""
Experiment 8: Power Measurement During Inference

Measures GPU power consumption during WebGPU and CUDA inference
using nvidia-smi.

Usage:
    python exp8_power_measurement.py --output results/exp8_power.json
"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

# Check for torch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available")

# Check for wgpu availability
try:
    import wgpu
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    print("Warning: wgpu not available")


def get_gpu_power():
    """Get current GPU power draw in watts using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting GPU power: {e}")
    return None


def get_gpu_info():
    """Get GPU information."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,power.limit,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'name': parts[0],
                'power_limit_w': float(parts[1].replace(' W', '')),
                'memory_mb': int(parts[2].replace(' MiB', ''))
            }
    except Exception:
        pass
    return {}


class PowerMonitor:
    """Background power monitoring during inference."""

    def __init__(self, sample_interval_ms=50):
        self.sample_interval = sample_interval_ms / 1000
        self.samples = []
        self.running = False
        self.thread = None

    def _monitor_loop(self):
        while self.running:
            power = get_gpu_power()
            if power is not None:
                self.samples.append((time.perf_counter(), power))
            time.sleep(self.sample_interval)

    def start(self):
        self.samples = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def get_stats(self):
        if not self.samples:
            return None

        powers = [p for _, p in self.samples]
        times = [t for t, _ in self.samples]

        return {
            'n_samples': len(powers),
            'duration_s': times[-1] - times[0] if len(times) > 1 else 0,
            'mean_power_w': np.mean(powers),
            'max_power_w': np.max(powers),
            'min_power_w': np.min(powers),
            'std_power_w': np.std(powers),
        }


def measure_cuda_inference_power(model, tokenizer, prompt, n_tokens=50, n_runs=10):
    """Measure power during CUDA inference."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()

    monitor = PowerMonitor(sample_interval_ms=20)
    total_tokens = 0
    total_time = 0
    energy_joules = []

    for run in range(n_runs):
        torch.cuda.synchronize()

        monitor.start()
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        monitor.stop()

        run_time = end_time - start_time
        total_time += run_time
        total_tokens += n_tokens

        # Calculate energy for this run (power * time)
        stats = monitor.get_stats()
        if stats:
            run_energy = stats['mean_power_w'] * run_time
            energy_joules.append(run_energy)

        print(f"  Run {run+1}/{n_runs}: {run_time:.2f}s, {stats['mean_power_w']:.1f}W mean power")

    tokens_per_second = total_tokens / total_time
    joules_per_token = np.mean(energy_joules) / n_tokens if energy_joules else None

    return {
        'backend': 'CUDA',
        'tokens_per_second': tokens_per_second,
        'total_tokens': total_tokens,
        'total_time_s': total_time,
        'mean_power_w': np.mean([e / (total_time / n_runs) for e in energy_joules]) if energy_joules else None,
        'joules_per_token': joules_per_token,
        'n_runs': n_runs
    }


def measure_idle_power(duration_s=5):
    """Measure idle GPU power."""
    monitor = PowerMonitor(sample_interval_ms=50)
    monitor.start()
    time.sleep(duration_s)
    monitor.stop()
    return monitor.get_stats()


def measure_webgpu_power(n_dispatches=200, n_iterations=50, n_runs=10):
    """
    Measure power during WebGPU compute dispatches.

    This simulates the dispatch pattern of LLM inference
    (many small dispatches per forward pass).
    """
    if not WGPU_AVAILABLE:
        return None

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if not adapter:
        print("  No WebGPU adapter found")
        return None

    device = adapter.request_device_sync()
    queue = device.queue

    # Create a simple matmul-like compute shader
    shader = """
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> c: array<f32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if (idx < 4096u) {
            var sum: f32 = 0.0;
            for (var i = 0u; i < 256u; i++) {
                sum += a[idx] * b[i];
            }
            c[idx] = sum;
        }
    }
    """

    module = device.create_shader_module(code=shader)
    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": module, "entry_point": "main"}
    )

    # Create buffers (simulate LLM hidden states)
    size = 4096 * 4  # 4096 elements * 4 bytes
    buf_a = device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    buf_b = device.create_buffer(size=256 * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    buf_c = device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE)

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": buf_a}},
            {"binding": 1, "resource": {"buffer": buf_b}},
            {"binding": 2, "resource": {"buffer": buf_c}},
        ]
    )

    def run_dispatches():
        """Run n_dispatches (simulating one forward pass)."""
        for _ in range(n_dispatches):
            encoder = device.create_command_encoder()
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(16)  # 4096/256 = 16 workgroups
            compute_pass.end()
            queue.submit([encoder.finish()])
        queue.on_submitted_work_done_sync()

    # Warmup
    for _ in range(3):
        run_dispatches()

    monitor = PowerMonitor(sample_interval_ms=20)
    total_time = 0
    energy_joules = []

    for run in range(n_runs):
        monitor.start()
        start_time = time.perf_counter()

        for _ in range(n_iterations):
            run_dispatches()

        end_time = time.perf_counter()
        monitor.stop()

        run_time = end_time - start_time
        total_time += run_time

        stats = monitor.get_stats()
        if stats:
            run_energy = stats['mean_power_w'] * run_time
            energy_joules.append(run_energy)

        print(f"  Run {run+1}/{n_runs}: {run_time:.2f}s, {stats['mean_power_w']:.1f}W mean power")

    # Calculate metrics (treating each iteration as one "forward pass" / token)
    total_forward_passes = n_iterations * n_runs
    forward_passes_per_second = total_forward_passes / total_time
    joules_per_forward = np.mean(energy_joules) / n_iterations if energy_joules else None

    return {
        'backend': 'WebGPU',
        'dispatches_per_forward': n_dispatches,
        'forward_passes_per_second': forward_passes_per_second,
        'total_forward_passes': total_forward_passes,
        'total_time_s': total_time,
        'mean_power_w': np.mean([e / (total_time / n_runs) for e in energy_joules]) if energy_joules else None,
        'joules_per_forward': joules_per_forward,
        'n_runs': n_runs,
        'n_iterations': n_iterations
    }


def main():
    parser = argparse.ArgumentParser(description="Power measurement during inference")
    parser.add_argument("--output", type=str, default="results/exp8_power.json")
    parser.add_argument("--n-tokens", type=int, default=50)
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 8: Power Measurement During Inference")
    print("=" * 60)

    results = {
        'gpu_info': get_gpu_info(),
        'experiments': {}
    }

    print(f"\nGPU: {results['gpu_info'].get('name', 'unknown')}")
    print(f"Power limit: {results['gpu_info'].get('power_limit_w', 'unknown')} W")

    # Measure idle power
    print("\nMeasuring idle power...")
    idle_stats = measure_idle_power(5)
    results['experiments']['idle'] = idle_stats
    print(f"  Idle power: {idle_stats['mean_power_w']:.1f} W")

    # CUDA inference
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("\nLoading model for CUDA inference...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

        prompt = "The capital of France is"

        print(f"\nMeasuring CUDA inference power ({args.n_runs} runs, {args.n_tokens} tokens each)...")
        cuda_results = measure_cuda_inference_power(model, tokenizer, prompt, args.n_tokens, args.n_runs)
        results['experiments']['cuda'] = cuda_results

        if cuda_results:
            print(f"\n  CUDA Results:")
            print(f"    Tokens/sec: {cuda_results['tokens_per_second']:.1f}")
            print(f"    Mean power: {cuda_results['mean_power_w']:.1f} W")
            print(f"    Energy: {cuda_results['joules_per_token']:.2f} J/token")

            # Calculate efficiency vs idle
            inference_power = cuda_results['mean_power_w']
            delta_power = inference_power - idle_stats['mean_power_w']
            print(f"    Delta power (vs idle): {delta_power:.1f} W")

    # WebGPU power measurement
    if WGPU_AVAILABLE:
        print(f"\nMeasuring WebGPU power ({args.n_runs} runs, 50 iterations each)...")
        webgpu_results = measure_webgpu_power(n_dispatches=200, n_iterations=50, n_runs=args.n_runs)
        results['experiments']['webgpu'] = webgpu_results

        if webgpu_results:
            print(f"\n  WebGPU Results:")
            print(f"    Forward passes/sec: {webgpu_results['forward_passes_per_second']:.1f}")
            print(f"    Mean power: {webgpu_results['mean_power_w']:.1f} W")
            print(f"    Energy: {webgpu_results['joules_per_forward']:.4f} J/forward")

            # Calculate efficiency vs idle
            inference_power = webgpu_results['mean_power_w']
            delta_power = inference_power - idle_stats['mean_power_w']
            print(f"    Delta power (vs idle): {delta_power:.1f} W")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
