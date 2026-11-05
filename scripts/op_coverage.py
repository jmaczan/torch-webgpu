"""Report which ATen ops need a backend kernel and whether torch_webgpu has one.

An op "needs a backend kernel" when native_functions.yaml gives it a CPU
dispatch entry (i.e. it is not Composite*). For those ops, a missing
PrivateUse1 kernel means we go through the boxed CPU fallback.

Usage: python scripts/op_coverage.py [--missing] [--filter REGEX]
"""

import argparse
import os
import re

import torch
import torchgen
import yaml


def backend_ops():
    path = os.path.join(
        os.path.dirname(torchgen.__file__), "packaged/ATen/native/native_functions.yaml"
    )
    with open(path) as f:
        entries = yaml.load(f, Loader=yaml.CSafeLoader)
    ops = []
    for e in entries:
        name = e["func"].split("(")[0]
        dispatch = e.get("dispatch") or {}
        keys = set()
        for k in dispatch:
            keys.update(x.strip() for x in k.split(","))
        if "CPU" in keys:
            ops.append(name)
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", action="store_true")
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()

    import torch_webgpu  # noqa: F401 - registers kernels

    ops = backend_ops()
    have, missing = [], []
    for name in ops:
        if args.filter and not re.search(args.filter, name):
            continue
        qual = "aten::" + name.split(".")[0]
        overload = name.split(".")[1] if "." in name else ""
        try:
            has = torch._C._dispatch_has_kernel_for_dispatch_key(
                qual + ("." + overload if overload else ""), "PrivateUse1"
            )
        except RuntimeError:
            has = False
        (have if has else missing).append(name)
    total = len(have) + len(missing)
    print(f"native PrivateUse1 kernels: {len(have)}/{total}")
    if args.missing:
        for n in missing:
            print("  missing:", n)


if __name__ == "__main__":
    main()
