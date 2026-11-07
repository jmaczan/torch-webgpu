"""Runs PyTorch OpInfo samples on webgpu and compares against CPU.

Used by test_opinfo.py and usable standalone for a coverage survey:

    python tests/opinfo/opinfo_runner.py [--dtype float32] [--filter REGEX]
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
import warnings

import torch
import torch_webgpu  # noqa: F401

from torch.testing._internal.common_methods_invocations import op_db

DEVICE = "webgpu"

# dtype -> (atol, rtol). GPU math is f32; low precision dtypes round on store.
TOLERANCES = {
    torch.float32: (1e-4, 1.3e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (5e-2, 2e-2),
}


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        y = x.detach().to(device)
        return y.requires_grad_(x.requires_grad) if x.is_floating_point() else y
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu(v) for v in x)
    return x


def compare(actual, expected, atol, rtol, path="out"):
    if isinstance(expected, torch.Tensor):
        # Tolerance follows the result dtype (e.g. exp(int) -> float32).
        if expected.dtype in TOLERANCES:
            atol, rtol = max(atol, TOLERANCES[expected.dtype][0]), max(rtol, TOLERANCES[expected.dtype][1])
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{path}: expected tensor, got {type(actual)}")
        actual = actual.cpu()
        if actual.dtype != expected.dtype:
            raise AssertionError(f"{path}: dtype {actual.dtype} != {expected.dtype}")
        if actual.shape != expected.shape:
            raise AssertionError(f"{path}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=True,
                                   check_stride=False, check_device=False, msg=lambda m: f"{path}: {m}")
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: length {len(actual)} != {len(expected)}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, atol, rtol, f"{path}[{i}]")
    elif isinstance(expected, (int, float, bool, complex)):
        if isinstance(actual, torch.Tensor):
            actual = actual.item()
        if isinstance(expected, float) and abs(actual - expected) > atol + rtol * abs(expected):
            raise AssertionError(f"{path}: {actual} != {expected}")


def nondeterministic(op):
    # Ops whose results legitimately differ between backends.
    return op.name in {
        "rand_like", "randn_like", "randint_like", "bernoulli", "normal", "multinomial",
        "nn.functional.dropout", "nn.functional.dropout2d", "nn.functional.dropout3d",
        "nn.functional.alpha_dropout", "nn.functional.feature_alpha_dropout", "empty",
        "empty_like", "empty_strided", "new_empty", "new_empty_strided", "empty_permuted",
        "uniform", "exponential", "geometric", "cauchy", "log_normal", "randint", "randn", "rand",
        "nn.functional.rrelu", "torch._refs.empty", "torch._refs.empty_like",
        "nn.functional.fractional_max_pool2d", "nn.functional.fractional_max_pool3d",
        "svd_lowrank", "pca_lowrank",
    }


def random_sample(op, sample):
    # Samples that draw random numbers (dropout) can't match the CPU.
    kw = sample.kwargs
    if kw.get("dropout_p", 0) or kw.get("p", 0) and "dropout" in op.name:
        return True
    return op.name == "nn.functional.multi_head_attention_forward" and kw.get("training", False)


def harness_limited(op):
    # Samples that can't be reproduced by moving inputs to another device:
    # as_strided on partial views depends on the original storage layout;
    # tensor_split takes CPU index tensors; sparse layouts aren't supported.
    return (op.name, op.variant_test_name) in {("as_strided", "partial_views"), ("tensor_split", "")} or \
        op.name.startswith("sparse.") or op.name == "to_sparse"


# Ops returning indices into `input` along a dim, where the order among equal
# values is implementation-defined (CPU's unstable sort and partial-sort topk
# differ from ours, and from CUDA's). op name -> (index of dim among args,
# default dim, index of the indices output or None if the result *is* indices).
TIE_ORDER_OPS = {
    "sort": (0, -1, 1), "argsort": (0, -1, None), "topk": (1, -1, 1), "kthvalue": (1, -1, 1),
    "median": (0, None, 1), "nanmedian": (0, None, 1), "mode": (0, -1, 1),
}


def tie_equivalent(op, sample, actual, expected):
    """True if actual's indices select the same values as expected's."""
    spec = TIE_ORDER_OPS.get(op.name)
    if spec is None:
        return False
    arg_i, default_dim, out_i = spec
    x = sample.input
    dim = sample.args[arg_i] if len(sample.args) > arg_i else sample.kwargs.get("dim", default_dim)
    if dim is None or x.dim() == 0:
        return False
    if out_i is None:
        ia, ie = actual, expected
    else:
        if not isinstance(expected, (tuple, list)):
            return False
        try:
            compare(actual[0], expected[0], 0, 0)
        except AssertionError:
            return False
        ia, ie = actual[out_i], expected[out_i]
    ia = ia.cpu()
    if ia.dtype != ie.dtype or ia.shape != ie.shape:
        return False
    if ia.dim() < x.dim():  # keepdim=False
        ia, ie = ia.unsqueeze(dim), ie.unsqueeze(dim)
    if ia.numel() and (ia.min() < 0 or ia.max() >= x.size(dim)):
        return False
    va, ve = x.gather(dim, ia), x.gather(dim, ie)
    return torch.equal(va, ve) or (va.is_floating_point() and torch.equal(va.isnan(), ve.isnan())
                                   and torch.equal(va.nan_to_num(0), ve.nan_to_num(0)))


def run_op(op, dtype, max_samples=None):
    """Returns (status, message). status in {pass, fail, error, skip}."""
    if nondeterministic(op):
        return "skip", "nondeterministic"
    if harness_limited(op):
        return "skip", "not reproducible on another device"
    if dtype not in op.supported_dtypes("cpu"):
        return "skip", "dtype unsupported on cpu"
    atol, rtol = TOLERANCES.get(dtype, (0, 0))
    try:
        samples = list(op.sample_inputs("cpu", dtype, requires_grad=False))
    except Exception as e:  # noqa: BLE001
        return "skip", f"sample generation failed: {e}"
    if max_samples:
        samples = samples[:max_samples]
    for k, s in enumerate(samples):
        if random_sample(op, s):
            continue
        try:
            expected = op(s.input, *s.args, **s.kwargs)
        except Exception:  # noqa: BLE001
            continue  # CPU can't run this sample either
        di = to_device(s.input, DEVICE)
        da = to_device(s.args, DEVICE)
        dk = to_device(s.kwargs, DEVICE)
        try:
            actual = op(di, *da, **dk)
            compare(actual, expected, atol, rtol)
        except AssertionError as e:
            if tie_equivalent(op, s, actual, expected):
                continue
            return "fail", f"sample {k}: {str(e)[:300]}"
        except Exception as e:  # noqa: BLE001
            return "error", f"sample {k}: {type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
    return "pass", f"{len(samples)} samples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--show", default="fail,error")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    warnings.filterwarnings("ignore")
    counts = {}
    show = set(args.show.split(","))
    verbose = bool(int(__import__("os").environ.get("OPINFO_VERBOSE", "0")))
    for op in op_db:
        if args.filter and not re.search(args.filter, op.name):
            continue
        if verbose:
            print("RUN", op.name, op.variant_test_name, flush=True)
        status, msg = run_op(op, dtype, args.max_samples)
        counts[status] = counts.get(status, 0) + 1
        if status in show:
            print(f"{status.upper():5s} {op.name}{'.' + op.variant_test_name if op.variant_test_name else ''}: {msg}")
            sys.stdout.flush()
    print(counts)
    print("fallback ops:", sorted(torch_webgpu._C._fallback_ops()))


if __name__ == "__main__":
    main()
