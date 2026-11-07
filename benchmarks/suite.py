#!/usr/bin/env python
"""torch-webgpu performance suite.

Kernel benchmarks report GPU time from timestamp queries
(torch.webgpu.profiler) where available; end-to-end benchmarks report wall
time. Results go to JSON; --baseline compares with an earlier run and flags
regressions.

    python benchmarks/suite.py                         # kernels + models in the HF cache
    python benchmarks/suite.py --only kernels --out now.json
    python benchmarks/suite.py --baseline before.json  # exit 1 on >10% regressions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Callable

import torch
import torch.nn.functional as F
import torch_webgpu  # noqa: F401

D = "webgpu"
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
PROFILER = torch.webgpu.profiler.is_supported()


def gpu_us(fn: Callable[[], object], reps: int = 20) -> float:
    """Mean GPU microseconds per call (wall time when no timestamps)."""
    for _ in range(3):
        fn()
    torch.webgpu.synchronize()
    if PROFILER:
        with torch.webgpu.profiler.profile() as p:
            for _ in range(reps):
                fn()
        return p.total_us / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.webgpu.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6


def wall_ms(fn: Callable[[], object], reps: int = 10) -> float:
    for _ in range(3):
        fn()
    torch.webgpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.webgpu.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def kernels() -> dict[str, dict]:
    r: dict[str, dict] = {}
    for n in (1024, 4096):
        a = torch.randn(n, n, device=D)
        t = gpu_us(lambda: a @ a, reps=10)
        r[f"gemm_f32_{n}"] = {"us": t, "tflops": 2 * n**3 / t / 1e6}
    # Decode GEMVs over many weight copies, so weights stream from DRAM.
    for name, (N, K, conv1d) in {"gemv_4864x896": (4864, 896, False), "gemv_896x4864": (896, 4864, False),
                                 "gemv_conv1d_768x3072": (3072, 768, True)}.items():
        copies = max(4, int(1.2e9 // (N * K * 4)))
        ws = [torch.randn(K, N, device=D) if conv1d else torch.randn(N, K, device=D) for _ in range(copies)]
        x = torch.randn(1, K, device=D)
        it = iter(range(10**9))

        def call():
            w = ws[next(it) % copies]
            return x @ w if conv1d else F.linear(x, w)

        t = gpu_us(call, reps=copies)
        r[name] = {"us": t, "gbps": N * K * 4 / t / 1e3}
        del ws
    x = torch.randn(8, 512, 7, 7, device=D)
    w = torch.randn(512, 512, 3, 3, device=D)
    t = gpu_us(lambda: F.conv2d(x, w, padding=1), reps=10)
    r["conv3x3_512_7x7_b8"] = {"us": t, "tflops": 2 * 8 * 49 * 512 * 512 * 9 / t / 1e6}
    q = torch.randn(1, 14, 1, 64, device=D)
    k = torch.randn(1, 2, 2048, 64, device=D)
    r["attention_decode_2k"] = {"us": gpu_us(lambda: F.scaled_dot_product_attention(q, k, k, enable_gqa=True))}
    q = torch.randn(1, 14, 1024, 64, device=D)
    k = torch.randn(1, 2, 1024, 64, device=D)
    t = gpu_us(lambda: F.scaled_dot_product_attention(q, k, k, is_causal=True, enable_gqa=True), reps=5)
    r["attention_prefill_1k_causal"] = {"us": t}
    s = torch.zeros(16, device=D)
    n = 2000
    for _ in range(100):  # pipeline compile + allocator warm-up
        s.add_(1)
    torch.webgpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        s.add_(1)
    torch.webgpu.synchronize()
    r["eager_op_overhead"] = {"us": (time.perf_counter() - t0) / n * 1e6}
    return r


def models() -> dict[str, dict]:
    try:
        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoModelForCausalLM
    except ImportError:
        return {}
    r: dict[str, dict] = {}
    for mid in ("Qwen/Qwen2.5-0.5B", "openai-community/gpt2"):
        if not isinstance(try_to_load_from_cache(mid, "config.json"), str):
            continue
        name = mid.split("/")[-1]
        m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32, attn_implementation="sdpa").eval().to(D)
        ids1 = torch.randint(0, 1000, (1, 1), device=D)
        ids512 = torch.randint(0, 1000, (1, 512), device=D)
        with torch.no_grad():
            r[f"{name}_decode_eager"] = {"ms": wall_ms(lambda: m(ids1))}
            r[f"{name}_prefill512_eager"] = {"ms": wall_ms(lambda: m(ids512), reps=5)}
            cm = torch.compile(m)
            r[f"{name}_decode_compiled"] = {"ms": wall_ms(lambda: cm(ids1))}
            r[f"{name}_prefill512_compiled"] = {"ms": wall_ms(lambda: cm(ids512), reps=5)}
            torch._dynamo.reset()
            prompt = torch.randint(0, 1000, (1, 16), device=D)
            kw = dict(max_new_tokens=32, do_sample=False)
            for mode in ("eager", "compiled_static"):
                if mode == "compiled_static":
                    m.forward = torch.compile(m.forward)
                    kw["cache_implementation"] = "static"
                m.generate(prompt, **kw)
                torch.webgpu.synchronize()
                t0 = time.perf_counter()
                m.generate(prompt, **kw)
                torch.webgpu.synchronize()
                r[f"{name}_generate_{mode}"] = {"tok_s": 32 / (time.perf_counter() - t0)}
        torch._dynamo.reset()
        del m
    return r


# Metric direction: higher is better for these keys, lower for the others.
HIGHER = {"tflops", "gbps", "tok_s"}


def compare(now: dict, base: dict, tol: float) -> list[str]:
    bad = []
    for name, metrics in now.items():
        for key, v in metrics.items():
            b = base.get(name, {}).get(key)
            if b is None or b == 0:
                continue
            change = (v - b) / b if key in HIGHER else (b - v) / b
            mark = "REGRESSION" if change < -tol else ""
            print(f"  {name:34s} {key:6s} {b:10.2f} -> {v:10.2f}  {change * 100:+6.1f}% {mark}")
            if mark:
                bad.append(f"{name}.{key}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["kernels", "models"])
    ap.add_argument("--out", default=None, help="write results as JSON")
    ap.add_argument("--baseline", default=None, help="compare with an earlier --out file")
    ap.add_argument("--tolerance", type=float, default=0.10)
    args = ap.parse_args()
    props = torch.webgpu.get_device_properties()
    print(f"device: {props['device']} ({props['backend']}), timestamps: {PROFILER}")
    results: dict[str, dict] = {}
    if args.only in (None, "kernels"):
        results.update(kernels())
    if args.only in (None, "models"):
        results.update(models())
    for name, metrics in results.items():
        print(f"{name:34s} " + "  ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"device": props, "torch": torch.__version__, "results": results}, f, indent=1)
    if args.baseline:
        with open(args.baseline) as f:
            base = json.load(f)["results"]
        print(f"vs {args.baseline}:")
        bad = compare(results, base, args.tolerance)
        if bad:
            print("regressions:", ", ".join(bad))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
