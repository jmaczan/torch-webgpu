#!/usr/bin/env python3
"""
Experiment 4: GPU Timeline Visualization

This experiment captures detailed timing information to create a timeline
showing the overlap between CPU dispatch preparation and GPU execution.

Usage:
    python exp4_timeline.py --output results/exp4_timeline.json

Produces:
1. Timeline data showing CPU and GPU overlap
2. Visualization figure (if matplotlib available)
3. Analysis reconciling per-dispatch CPU overhead with wall-clock TTFT
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

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def get_system_info():
    import platform
    info = {"platform": platform.platform()}
    if WGPU_AVAILABLE:
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            if adapter:
                info["gpu"] = adapter.info.get("description", "unknown")
                info["backend"] = adapter.info.get("backend_type", "unknown")
        except:
            pass
    return info


def capture_timeline(device, queue, n_dispatches=100):
    """
    Capture detailed timeline of CPU dispatch preparation.

    Returns timeline events showing:
    - encoder_create_start/end
    - pass_begin
    - set_pipeline
    - set_bind_group
    - dispatch
    - pass_end
    - encoder_finish
    - queue_submit
    """

    # Create a simple shader
    shader_code = """
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        // Do some work to have measurable GPU time
        var val = data[gid.x];
        for (var i = 0u; i < 100u; i++) {
            val = val * 1.0001 + 0.0001;
        }
        data[gid.x] = val;
    }
    """

    module = device.create_shader_module(code=shader_code)
    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": module, "entry_point": "main"}
    )

    buffer = device.create_buffer(
        size=4096 * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )
    queue.write_buffer(buffer, 0, np.random.randn(4096).astype(np.float32).tobytes())

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": buffer}}]
    )

    # Warmup
    for _ in range(10):
        encoder = device.create_command_encoder()
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, bind_group)
        p.dispatch_workgroups(64)
        p.end()
        queue.submit([encoder.finish()])
    queue.on_submitted_work_done_sync()

    # Capture timeline
    timeline_events = []
    wall_start = time.perf_counter()

    for dispatch_id in range(n_dispatches):
        dispatch_events = {"id": dispatch_id}

        # Encoder creation
        t0 = time.perf_counter()
        encoder = device.create_command_encoder()
        t1 = time.perf_counter()
        dispatch_events["encoder_create_us"] = (t1 - t0) * 1e6

        # Begin compute pass
        t0 = time.perf_counter()
        compute_pass = encoder.begin_compute_pass()
        t1 = time.perf_counter()
        dispatch_events["pass_begin_us"] = (t1 - t0) * 1e6

        # Set pipeline
        t0 = time.perf_counter()
        compute_pass.set_pipeline(pipeline)
        t1 = time.perf_counter()
        dispatch_events["set_pipeline_us"] = (t1 - t0) * 1e6

        # Set bind group
        t0 = time.perf_counter()
        compute_pass.set_bind_group(0, bind_group)
        t1 = time.perf_counter()
        dispatch_events["set_bind_group_us"] = (t1 - t0) * 1e6

        # Dispatch
        t0 = time.perf_counter()
        compute_pass.dispatch_workgroups(64)
        t1 = time.perf_counter()
        dispatch_events["dispatch_us"] = (t1 - t0) * 1e6

        # End pass
        t0 = time.perf_counter()
        compute_pass.end()
        t1 = time.perf_counter()
        dispatch_events["pass_end_us"] = (t1 - t0) * 1e6

        # Finish encoder
        t0 = time.perf_counter()
        command_buffer = encoder.finish()
        t1 = time.perf_counter()
        dispatch_events["encoder_finish_us"] = (t1 - t0) * 1e6

        # Submit
        t0 = time.perf_counter()
        queue.submit([command_buffer])
        t1 = time.perf_counter()
        dispatch_events["submit_us"] = (t1 - t0) * 1e6

        dispatch_events["relative_time_us"] = (time.perf_counter() - wall_start) * 1e6

        timeline_events.append(dispatch_events)

    # Wait for all GPU work to complete
    t0 = time.perf_counter()
    queue.on_submitted_work_done_sync()
    gpu_sync_time = (time.perf_counter() - t0) * 1e6

    wall_end = time.perf_counter()
    wall_clock_us = (wall_end - wall_start) * 1e6

    return {
        "n_dispatches": n_dispatches,
        "wall_clock_us": wall_clock_us,
        "gpu_sync_time_us": gpu_sync_time,
        "events": timeline_events
    }


def analyze_timeline(timeline_data):
    """Analyze the timeline to show CPU/GPU overlap."""
    events = timeline_data["events"]

    # Sum all CPU-side times
    total_encoder_create = sum(e["encoder_create_us"] for e in events)
    total_pass_begin = sum(e["pass_begin_us"] for e in events)
    total_set_pipeline = sum(e["set_pipeline_us"] for e in events)
    total_set_bind_group = sum(e["set_bind_group_us"] for e in events)
    total_dispatch = sum(e["dispatch_us"] for e in events)
    total_pass_end = sum(e["pass_end_us"] for e in events)
    total_encoder_finish = sum(e["encoder_finish_us"] for e in events)
    total_submit = sum(e["submit_us"] for e in events)

    total_cpu_time = (total_encoder_create + total_pass_begin + total_set_pipeline +
                      total_set_bind_group + total_dispatch + total_pass_end +
                      total_encoder_finish + total_submit)

    wall_clock = timeline_data["wall_clock_us"]
    gpu_sync = timeline_data["gpu_sync_time_us"]

    # Per-dispatch averages
    n = len(events)
    per_dispatch_cpu = total_cpu_time / n

    return {
        "breakdown": {
            "encoder_create_us": total_encoder_create,
            "pass_begin_us": total_pass_begin,
            "set_pipeline_us": total_set_pipeline,
            "set_bind_group_us": total_set_bind_group,
            "dispatch_us": total_dispatch,
            "pass_end_us": total_pass_end,
            "encoder_finish_us": total_encoder_finish,
            "submit_us": total_submit
        },
        "totals": {
            "total_cpu_time_us": total_cpu_time,
            "wall_clock_us": wall_clock,
            "gpu_sync_us": gpu_sync
        },
        "per_dispatch": {
            "encoder_create_us": total_encoder_create / n,
            "pass_begin_us": total_pass_begin / n,
            "set_pipeline_us": total_set_pipeline / n,
            "set_bind_group_us": total_set_bind_group / n,
            "dispatch_us": total_dispatch / n,
            "pass_end_us": total_pass_end / n,
            "encoder_finish_us": total_encoder_finish / n,
            "submit_us": total_submit / n,
            "total_us": per_dispatch_cpu
        },
        "overlap_analysis": {
            "cpu_time_exceeds_wall_clock": total_cpu_time > wall_clock,
            "overlap_ratio": total_cpu_time / wall_clock if wall_clock > 0 else 0,
            "explanation": "If overlap_ratio > 1, GPU execution overlaps with CPU dispatch preparation (pipelining)"
        }
    }


def create_timeline_figure(timeline_data, analysis, output_path):
    """Create visualization of the timeline."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping figure generation")
        return

    events = timeline_data["events"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Per-dispatch breakdown
    ax1 = axes[0]
    categories = ["encoder_create", "pass_begin", "set_pipeline", "set_bind_group",
                  "dispatch", "pass_end", "encoder_finish", "submit"]

    per_dispatch = analysis["per_dispatch"]
    values = [per_dispatch[f"{cat}_us"] for cat in categories]

    bars = ax1.barh(categories, values, color='steelblue')
    ax1.set_xlabel("Time (µs)")
    ax1.set_title("Per-Dispatch CPU Overhead Breakdown")
    ax1.bar_label(bars, fmt='%.1f')

    # Plot 2: Cumulative timeline showing overlap
    ax2 = axes[1]

    # Extract cumulative CPU time at each dispatch
    cpu_times = []
    wall_times = []
    cumulative_cpu = 0
    for e in events:
        cpu_time = (e["encoder_create_us"] + e["pass_begin_us"] + e["set_pipeline_us"] +
                    e["set_bind_group_us"] + e["dispatch_us"] + e["pass_end_us"] +
                    e["encoder_finish_us"] + e["submit_us"])
        cumulative_cpu += cpu_time
        cpu_times.append(cumulative_cpu)
        wall_times.append(e["relative_time_us"])

    dispatch_ids = range(len(events))

    ax2.plot(dispatch_ids, np.array(cpu_times) / 1000, label="Cumulative CPU time", linewidth=2)
    ax2.plot(dispatch_ids, np.array(wall_times) / 1000, label="Wall clock time", linewidth=2)
    ax2.set_xlabel("Dispatch #")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("CPU Time vs Wall Clock (Overlap = Pipelining)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add annotation
    if analysis["overlap_analysis"]["cpu_time_exceeds_wall_clock"]:
        ax2.annotate(
            f"Overlap ratio: {analysis['overlap_analysis']['overlap_ratio']:.2f}x\n(GPU executes while CPU prepares next dispatch)",
            xy=(len(events)*0.6, max(wall_times)/1000*0.8),
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

    plt.tight_layout()

    fig_path = output_path.replace(".json", "_timeline.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Timeline figure saved to: {fig_path}")
    plt.close()

    return fig_path


def main():
    parser = argparse.ArgumentParser(description="GPU timeline visualization experiment")
    parser.add_argument("--output", type=str, default="results/exp4_timeline.json")
    parser.add_argument("--dispatches", type=int, default=100)
    args = parser.parse_args()

    if not WGPU_AVAILABLE:
        print("ERROR: wgpu required")
        return

    print("=" * 60)
    print("Experiment 4: GPU Timeline Visualization")
    print("=" * 60)

    system_info = get_system_info()
    print(f"GPU: {system_info.get('gpu', 'unknown')}")

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    queue = device.queue

    print(f"\nCapturing timeline for {args.dispatches} dispatches...")
    timeline_data = capture_timeline(device, queue, n_dispatches=args.dispatches)

    print("Analyzing timeline...")
    analysis = analyze_timeline(timeline_data)

    results = {
        "system_info": system_info,
        "timeline": timeline_data,
        "analysis": analysis
    }

    # Print summary
    print("\n" + "=" * 60)
    print("TIMELINE ANALYSIS")
    print("=" * 60)

    print("\nPer-dispatch CPU overhead:")
    for key, val in analysis["per_dispatch"].items():
        print(f"  {key}: {val:.1f} µs")

    print(f"\nTotals:")
    print(f"  Total CPU time:     {analysis['totals']['total_cpu_time_us']/1000:.2f} ms")
    print(f"  Wall clock time:    {analysis['totals']['wall_clock_us']/1000:.2f} ms")
    print(f"  GPU sync time:      {analysis['totals']['gpu_sync_us']/1000:.2f} ms")

    print(f"\nOverlap analysis:")
    print(f"  CPU time exceeds wall clock: {analysis['overlap_analysis']['cpu_time_exceeds_wall_clock']}")
    print(f"  Overlap ratio: {analysis['overlap_analysis']['overlap_ratio']:.2f}x")
    print(f"  {analysis['overlap_analysis']['explanation']}")

    print("=" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate figure
    fig_path = create_timeline_figure(timeline_data, analysis, str(output_path))
    if fig_path:
        results["figure_path"] = fig_path


if __name__ == "__main__":
    main()
