#!/usr/bin/env python3
"""
Collect and summarize all experiment results.

Usage:
    python collect_results.py

This generates a formatted summary that can be pasted back to Claude
for incorporation into the paper.
"""

import json
from pathlib import Path


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON in {path}"}


def format_summary():
    results_dir = Path("results")

    print("=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    print("\nCopy-paste this entire output back to Claude for paper incorporation.\n")

    # Experiment 1: Cross-GPU WebGPU
    print("-" * 70)
    print("EXPERIMENT 1: Cross-GPU WebGPU Validation")
    print("-" * 70)

    exp1 = load_json(results_dir / "exp1_webgpu.json")
    if exp1 is None:
        exp1 = load_json(results_dir / "exp1_apple_m2_webgpu.json")

    if exp1:
        sys_info = exp1.get("system_info", {})
        print(f"Platform: {sys_info.get('platform', 'unknown')}")
        print(f"GPU: {sys_info.get('gpu_description', sys_info.get('gpu', 'unknown'))}")
        print(f"Backend: {sys_info.get('wgpu_backend', sys_info.get('backend', 'unknown'))}")

        exps = exp1.get("experiments", {})

        if "dispatch_overhead" in exps:
            do = exps["dispatch_overhead"]
            print(f"\nPer-dispatch overhead: {do['mean_dispatch_us']:.1f} ± {do['std_dispatch_us']:.1f} µs")

        if "rmsnorm_unfused" in exps and "rmsnorm_fused" in exps:
            unfused = exps["rmsnorm_unfused"]
            fused = exps["rmsnorm_fused"]
            print(f"\nRMSNorm unfused (5 dispatches): {unfused['mean_ms']:.3f} ± {unfused['std_ms']:.3f} ms")
            print(f"RMSNorm fused (1 dispatch):     {fused['mean_ms']:.3f} ± {fused['std_ms']:.3f} ms")
            print(f"Fusion speedup: {exps.get('rmsnorm_fusion_speedup', unfused['mean_ms']/fused['mean_ms']):.2f}x")

        if "mega_kernel_single_wg" in exps and "multi_workgroup" in exps:
            mega = exps["mega_kernel_single_wg"]
            multi = exps["multi_workgroup"]
            print(f"\nMega-kernel (1 dispatch, 256 threads): {mega['mean_ms']:.3f} ms")
            print(f"Multi-workgroup (2 dispatches):         {multi['mean_ms']:.3f} ms")
            print(f"Mega-kernel slowdown: {exps.get('mega_kernel_slowdown', mega['mean_ms']/multi['mean_ms']):.1f}x")
    else:
        print("NO RESULTS - Run exp1_cross_gpu_webgpu.py")

    # Experiment 2: Device-Side Argmax
    print("\n" + "-" * 70)
    print("EXPERIMENT 2: Device-Side Argmax")
    print("-" * 70)

    exp2 = load_json(results_dir / "exp2_device_argmax.json")
    if exp2:
        sys_info = exp2.get("system_info", {})
        print(f"GPU: {sys_info.get('gpu', 'unknown')}")

        exps = exp2.get("experiments", {})
        if "full_readback" in exps:
            fr = exps["full_readback"]
            print(f"\nFull logits readback ({fr['buffer_size_kb']:.0f} KB): {fr['mean_ms']:.2f} ± {fr['std_ms']:.2f} ms")

        if "device_argmax" in exps:
            da = exps["device_argmax"]
            print(f"Device-side argmax (4 bytes):             {da['mean_ms']:.2f} ± {da['std_ms']:.2f} ms")
            print(f"Correctness: {'PASS' if da.get('correctness', False) else 'FAIL'}")

        if "improvement_ms" in exps:
            print(f"\nImprovement: {exps['improvement_ms']:.2f} ms ({exps['improvement_pct']:.1f}%)")

        if "buffer_map_unmap" in exps:
            print("\nBuffer map/unmap microbenchmark:")
            for r in exps["buffer_map_unmap"]:
                print(f"  {r['size_kb']:>8.1f} KB: {r['mean_ms']:.2f} ± {r['std_ms']:.2f} ms")
    else:
        print("NO RESULTS - Run exp2_device_argmax.py")

    # Experiment 3: Tiled Mega Strategy
    print("\n" + "-" * 70)
    print("EXPERIMENT 3: Multi-Dispatch Tiled Mega Strategy")
    print("-" * 70)

    exp3 = load_json(results_dir / "exp3_tiled_mega.json")
    if exp3:
        print(f"Hidden dim: {exp3.get('hidden_dim', 896)}, Intermediate dim: {exp3.get('intermediate_dim', 4864)}")

        exps = exp3.get("experiments", {})
        if "unfused" in exps:
            uf = exps["unfused"]
            print(f"\nUnfused MLP (7 dispatches): {uf['mean_ms']:.3f} ± {uf['std_ms']:.3f} ms")

        if "tiled" in exps:
            ti = exps["tiled"]
            print(f"Tiled MLP (3 dispatches):   {ti['mean_ms']:.3f} ± {ti['std_ms']:.3f} ms")

        if "mega_kernel" in exps:
            mk = exps["mega_kernel"]
            print(f"Mega-kernel (1 dispatch):   {mk['mean_ms']:.3f} ± {mk['std_ms']:.3f} ms")
            print(f"  Note: {mk.get('note', 'reduced intermediate_dim')}")

        analysis = exp3.get("analysis", {})
        if "tiled_speedup_vs_unfused" in analysis:
            print(f"\nTiled speedup vs unfused: {analysis['tiled_speedup_vs_unfused']:.2f}x")
        if "mega_slowdown_vs_tiled" in analysis:
            print(f"Mega-kernel slowdown vs tiled: {analysis['mega_slowdown_vs_tiled']:.1f}x")
    else:
        print("NO RESULTS - Run exp3_tiled_mega.py")

    # Experiment 4: Timeline
    print("\n" + "-" * 70)
    print("EXPERIMENT 4: GPU Timeline Visualization")
    print("-" * 70)

    exp4 = load_json(results_dir / "exp4_timeline.json")
    if exp4:
        analysis = exp4.get("analysis", {})
        per_dispatch = analysis.get("per_dispatch", {})
        totals = analysis.get("totals", {})
        overlap = analysis.get("overlap_analysis", {})

        print(f"\nPer-dispatch breakdown (µs):")
        for key in ["encoder_create_us", "pass_begin_us", "set_pipeline_us",
                    "set_bind_group_us", "dispatch_us", "pass_end_us",
                    "encoder_finish_us", "submit_us"]:
            if key in per_dispatch:
                print(f"  {key.replace('_us', ''):20s}: {per_dispatch[key]:.1f}")
        if "total_us" in per_dispatch:
            print(f"  {'total':20s}: {per_dispatch['total_us']:.1f}")

        print(f"\nTotals:")
        if "total_cpu_time_us" in totals:
            print(f"  Total CPU time:  {totals['total_cpu_time_us']/1000:.2f} ms")
        if "wall_clock_us" in totals:
            print(f"  Wall clock time: {totals['wall_clock_us']/1000:.2f} ms")
        if "gpu_sync_us" in totals:
            print(f"  GPU sync time:   {totals['gpu_sync_us']/1000:.2f} ms")

        print(f"\nOverlap analysis:")
        if "overlap_ratio" in overlap:
            print(f"  Overlap ratio: {overlap['overlap_ratio']:.2f}x")
            print(f"  {overlap.get('explanation', '')}")
    else:
        print("NO RESULTS - Run exp4_timeline.py")

    # Experiment 5: CUDA Comparison
    print("\n" + "-" * 70)
    print("EXPERIMENT 5: CUDA Fusion Comparison")
    print("-" * 70)

    exp5 = load_json(results_dir / "exp5_cuda_fusion.json")
    if exp5:
        sys_info = exp5.get("system_info", {})
        print(f"CUDA device: {sys_info.get('cuda_device', 'unknown')}")
        print(f"WebGPU device: {sys_info.get('webgpu_device', 'unknown')}")

        exps = exp5.get("experiments", {})
        if "cuda_launch" in exps and "error" not in exps["cuda_launch"]:
            cl = exps["cuda_launch"]
            print(f"\nCUDA kernel launch overhead: {cl['mean_us']:.1f} ± {cl['std_us']:.1f} µs")

        if "webgpu_dispatch" in exps and "error" not in exps["webgpu_dispatch"]:
            wd = exps["webgpu_dispatch"]
            print(f"WebGPU dispatch overhead:    {wd['mean_us']:.1f} ± {wd['std_us']:.1f} µs")

        if "cuda_unfused_rmsnorm" in exps and "error" not in exps["cuda_unfused_rmsnorm"]:
            cuf = exps["cuda_unfused_rmsnorm"]
            print(f"\nCUDA RMSNorm unfused: {cuf['mean_us']:.1f} ± {cuf['std_us']:.1f} µs")

        if "cuda_fused_rmsnorm" in exps and "error" not in exps["cuda_fused_rmsnorm"]:
            cf = exps["cuda_fused_rmsnorm"]
            print(f"CUDA RMSNorm fused:   {cf['mean_us']:.1f} ± {cf['std_us']:.1f} µs")

        if "cuda_compiled_rmsnorm" in exps and "error" not in exps["cuda_compiled_rmsnorm"]:
            cc = exps["cuda_compiled_rmsnorm"]
            print(f"CUDA RMSNorm compiled:{cc['mean_us']:.1f} ± {cc['std_us']:.1f} µs")

        analysis = exp5.get("analysis", {})
        if "webgpu_vs_cuda_overhead_ratio" in analysis:
            print(f"\nWebGPU/CUDA overhead ratio: {analysis['webgpu_vs_cuda_overhead_ratio']:.1f}x")
        if "cuda_fusion_speedup" in analysis:
            print(f"CUDA fusion speedup: {analysis['cuda_fusion_speedup']:.2f}x")
    else:
        print("NO RESULTS - Run exp5_cuda_fusion.py")

    # Final summary
    print("\n" + "=" * 70)
    print("END OF RESULTS")
    print("=" * 70)


if __name__ == "__main__":
    format_summary()
