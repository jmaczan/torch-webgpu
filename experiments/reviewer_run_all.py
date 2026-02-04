#!/usr/bin/env python3
"""
Master Script: Run All Experiments for TMLR Reviewer Response

This script runs all experiments needed to address the reviewer's concerns
and generates a summary report.

Usage:
    pip install torch transformers wgpu numpy scipy
    python reviewer_run_all.py --platform [m2|amd|nvidia]

The script will:
1. Run end-to-end LLM inference benchmark (addresses Major #1)
2. Collect dispatch overhead data (validates cross-platform claims)
3. Collect raw data for tables needing CIs (addresses Minor #6, #8)
4. Generate summary report for paper revision
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_script(script_path, args_list, description):
    """Run a Python script and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*70}\n")

    cmd = [sys.executable, str(script_path)] + args_list
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Script failed with return code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"ERROR: Script not found: {script_path}")
        return False


def generate_summary_report(output_dir, platform):
    """Generate summary report from all results."""
    report_path = output_dir / "REVIEWER_RESPONSE_SUMMARY.md"

    report = f"""# TMLR Reviewer Response: Experimental Results

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Platform: {platform}

## Overview

This document summarizes the experimental results collected to address the TMLR reviewer's concerns.

## Major Concerns Addressed

### 1. End-to-End Inference on Additional GPUs (Major #1)

"""

    # Load e2e results if available
    e2e_path = output_dir / f"reviewer_e2e_{platform}.json"
    if e2e_path.exists():
        with open(e2e_path) as f:
            e2e = json.load(f)
        report += f"""**Results on {platform.upper()}:**
- Device: {e2e.get('device', 'unknown')}
- Tokens/second: {e2e['tokens_per_second']:.2f} +/- {e2e['tokens_per_second_std']:.2f}
- 95% CI: [{e2e['tokens_per_second_ci95'][0]:.2f}, {e2e['tokens_per_second_ci95'][1]:.2f}]
- CV: {e2e['coefficient_of_variation']:.1f}%
- TTFT: {e2e['time_to_first_token_ms']:.2f} ms

"""
        if "note" in e2e:
            report += f"**Note:** {e2e['note']}\n\n"
    else:
        report += f"*Results not yet collected for {platform}*\n\n"

    report += """### 2. Dispatch Overhead Validation

"""

    # Load dispatch overhead results
    dispatch_path = output_dir / f"reviewer_dispatch_{platform}.json"
    if dispatch_path.exists():
        with open(dispatch_path) as f:
            dispatch = json.load(f)
        si = dispatch.get("system_info", {})
        exp = dispatch.get("experiments", {})

        report += f"""**Platform:** {si.get('gpu_description', 'unknown')}
**Backend:** {si.get('wgpu_backend', 'unknown')}

| Measurement | Value |
|------------|-------|
| Single-op dispatch overhead | {exp.get('dispatch_overhead', {}).get('mean_dispatch_us', 'N/A'):.1f} µs |
| Sequential dispatch overhead | {exp.get('sequential_dispatches', {}).get('per_dispatch_us', 'N/A'):.1f} µs |
| RMSNorm fusion speedup | {exp.get('rmsnorm_fusion_speedup', 'N/A'):.2f}x |

"""
    else:
        report += f"*Dispatch overhead results not yet collected*\n\n"

    report += """## Minor Concerns Addressed

### 6. Confidence Intervals for Tables 7, 8, 14, 15

"""

    table_path = output_dir / f"reviewer_table_data_{platform}.json"
    if table_path.exists():
        with open(table_path) as f:
            tables = json.load(f)

        # Table 14
        t14 = tables.get("table14_mega_kernel", {})
        if t14:
            mega = t14.get("mega_kernel", {})
            multi = t14.get("multi_workgroup", {})
            sig = t14.get("mega_vs_multi_significance", {})

            report += f"""**Table 14: Mega-kernel vs Multi-workgroup (256x256)**

| Approach | Time (ms) | 95% CI | Std |
|----------|-----------|--------|-----|
| Mega-kernel | {mega.get('mean', 0):.4f} | [{mega.get('ci95_lower', 0):.4f}, {mega.get('ci95_upper', 0):.4f}] | {mega.get('std', 0):.4f} |
| Multi-workgroup | {multi.get('mean', 0):.4f} | [{multi.get('ci95_lower', 0):.4f}, {multi.get('ci95_upper', 0):.4f}] | {multi.get('std', 0):.4f} |

**Statistical Significance:**
- Speedup: {t14.get('mega_vs_multi_speedup', 0):.2f}x
- p-value: {sig.get('p_value', 'N/A')}
- Cohen's d: {sig.get('cohens_d', 'N/A')}
- Significant at p<0.05: {sig.get('significant_p05', 'N/A')}

"""

        # Table 15
        t15 = tables.get("table15_device_argmax", {})
        if t15:
            full = t15.get("full_readback", {})
            argmax = t15.get("device_argmax", {})
            sig = t15.get("argmax_significance", {})

            report += f"""**Table 15: Device-side Argmax**

| Approach | Time (ms) | 95% CI | Std |
|----------|-----------|--------|-----|
| Full readback | {full.get('mean', 0):.3f} | [{full.get('ci95_lower', 0):.3f}, {full.get('ci95_upper', 0):.3f}] | {full.get('std', 0):.3f} |
| Device argmax | {argmax.get('mean', 0):.3f} | [{argmax.get('ci95_lower', 0):.3f}, {argmax.get('ci95_upper', 0):.3f}] | {argmax.get('std', 0):.3f} |

**Statistical Significance:**
- Improvement: {t15.get('improvement_percent', 0):.1f}%
- p-value: {sig.get('p_value', 'N/A')}
- Significant at p<0.05: {sig.get('significant_p05', 'N/A')}

"""
    else:
        report += "*Table data not yet collected*\n\n"

    report += """## Files Generated

"""

    for f in output_dir.glob("reviewer_*.json"):
        report += f"- `{f.name}`\n"

    report += """
## Next Steps

1. Copy relevant CIs and p-values into paper tables
2. Update Table 14 with significance test results
3. Update Table 15 with significance test results
4. Add cross-platform inference results to paper (if WebGPU inference available)

## Notes for Rebuttal

- If using MPS instead of WebGPU: Note that MPS results provide GPU baseline but are not directly comparable to WebGPU dispatch overhead measurements
- Dispatch overhead measurements via wgpu validate the cross-platform findings even without full inference
"""

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{'='*70}")
    print(f"Summary report saved to: {report_path}")
    print(f"{'='*70}")

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Run all reviewer response experiments")
    parser.add_argument("--platform", type=str, required=True,
                       choices=["m2", "amd", "nvidia", "intel"],
                       help="Platform identifier for output files")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--runs", type=int, default=30,
                       help="Number of benchmark runs")
    parser.add_argument("--skip-e2e", action="store_true",
                       help="Skip end-to-end inference benchmark")
    parser.add_argument("--skip-dispatch", action="store_true",
                       help="Skip dispatch overhead benchmark")
    parser.add_argument("--skip-tables", action="store_true",
                       help="Skip table data collection")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments_dir = Path(__file__).parent

    print("=" * 70)
    print("TMLR REVIEWER RESPONSE: Running All Experiments")
    print("=" * 70)
    print(f"Platform: {args.platform}")
    print(f"Output directory: {output_dir}")
    print(f"Runs per benchmark: {args.runs}")

    success = True

    # 1. End-to-end inference
    if not args.skip_e2e:
        script = experiments_dir / "reviewer_exp1_e2e_inference.py"
        output = output_dir / f"reviewer_e2e_{args.platform}.json"
        success &= run_script(
            script,
            ["--output", str(output), "--runs", str(args.runs)],
            "End-to-end LLM inference benchmark"
        )

    # 2. Dispatch overhead (using existing exp1)
    if not args.skip_dispatch:
        script = experiments_dir / "exp1_cross_gpu_webgpu.py"
        output = output_dir / f"reviewer_dispatch_{args.platform}.json"
        success &= run_script(
            script,
            ["--output", str(output), "--iterations", str(args.runs)],
            "Dispatch overhead benchmark (wgpu)"
        )

    # 3. Table data collection
    if not args.skip_tables:
        script = experiments_dir / "reviewer_exp2_collect_table_data.py"
        output = output_dir / f"reviewer_table_data_{args.platform}.json"
        success &= run_script(
            script,
            ["--output", str(output), "--runs", str(args.runs)],
            "Table data collection (CIs and p-values)"
        )

    # Generate summary report
    generate_summary_report(output_dir, args.platform)

    print("\n" + "=" * 70)
    if success:
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    else:
        print("SOME EXPERIMENTS FAILED - Check output above")
    print("=" * 70)

    print(f"""
NEXT STEPS:
1. Review {output_dir}/REVIEWER_RESPONSE_SUMMARY.md
2. Copy CI values into paper tables
3. Update statistical significance sections
4. Run on other platforms if needed:
   - Apple M2: python {__file__} --platform m2
   - AMD: python {__file__} --platform amd
""")


if __name__ == "__main__":
    main()
