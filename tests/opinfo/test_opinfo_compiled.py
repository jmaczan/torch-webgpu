"""OpInfo samples through torch.compile: every op in PyTorch's op database
is compiled with Inductor (generated WGSL kernels, eager fallbacks for the
rest) on webgpu and compared with the CPU eager result.

float32 covers every OpInfo; float16 / bfloat16 / int64 / int32 / bool run
on test_opinfo.SUBSET (all of them with TORCH_WEBGPU_OPINFO_ALL_DTYPES=1).
Each sample compiles its own graph; Inductor's FX graph cache makes reruns
fast. TORCH_WEBGPU_OPINFO_COMPILED_SAMPLES sets samples per op (default 3).
"""

import os
import warnings

import pytest
import torch

from opinfo_runner import op_db, run_op_compiled
from test_opinfo import ALL_DTYPES, EXPECTED_FAILURES as EAGER_EXPECTED_FAILURES, SUBSET

MAX_SAMPLES = int(os.environ.get("TORCH_WEBGPU_OPINFO_COMPILED_SAMPLES", "3"))

DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.bool]

# Known deviations of compiled code from CPU eager that are not webgpu bugs.
EXPECTED_FAILURES = {
    # Inductor folds reductions over zero elements to the reduction's
    # initial value: the mean of an empty tensor becomes 0 instead of NaN
    # (any device).
    ("var_mean.unbiased", torch.float32),
    ("std_mean.unbiased", torch.float32),
    # Inductor's batch norm decomposition returns per-channel saved
    # statistics in eval mode where eager returns empty tensors.
    ("native_batch_norm", torch.float32),
    ("_native_batch_norm_legit", torch.float32),
} | EAGER_EXPECTED_FAILURES


def _params():
    for op in op_db:
        name = op.name + ("." + op.variant_test_name if op.variant_test_name else "")
        for dtype in DTYPES:
            if dtype is not torch.float32 and not ALL_DTYPES and op.name not in SUBSET:
                continue
            marks = [pytest.mark.xfail(reason="known deviation, see EXPECTED_FAILURES", strict=False)] \
                if (name, dtype) in EXPECTED_FAILURES else []
            yield pytest.param(op, dtype, id=f"{name}-{str(dtype).replace('torch.', '')}", marks=marks)


@pytest.mark.parametrize("op,dtype", list(_params()))
def test_opinfo_compiled(op, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        status, msg = run_op_compiled(op, dtype, MAX_SAMPLES)
    if status == "skip":
        pytest.skip(msg)
    assert status == "pass", msg
