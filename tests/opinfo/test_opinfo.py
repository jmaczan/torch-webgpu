"""OpInfo-driven correctness tests: every op in PyTorch's op database is run
on webgpu and compared with the CPU result for the same sample inputs.

float32 covers every OpInfo; other dtypes run on a representative subset to
keep the suite fast. test_opinfo_grad compares input gradients (float32) of
every differentiable op. Set TORCH_WEBGPU_OPINFO_ALL_DTYPES=1 to run every op for
every dtype, TORCH_WEBGPU_OPINFO_MAX_SAMPLES to limit samples per op.
"""

import os
import warnings

import pytest
import torch

from opinfo_runner import op_db, run_op, run_op_grad

MAX_SAMPLES = int(os.environ.get("TORCH_WEBGPU_OPINFO_MAX_SAMPLES", "20"))
ALL_DTYPES = os.environ.get("TORCH_WEBGPU_OPINFO_ALL_DTYPES") == "1"

# Ops checked for reduced-precision / integer dtypes unless ALL_DTYPES.
SUBSET = {
    "add", "sub", "mul", "div", "neg", "abs", "exp", "log", "sqrt", "rsqrt", "sin", "cos", "tanh",
    "sigmoid", "nn.functional.gelu", "nn.functional.silu", "nn.functional.relu", "softmax",
    "log_softmax", "sum", "mean", "amax", "amin", "argmax", "max", "min", "cumsum", "matmul", "mm",
    "bmm", "addmm", "nn.functional.linear", "nn.functional.layer_norm", "nn.functional.embedding",
    "where", "masked_fill", "clamp", "eq", "lt", "ge", "cat", "stack", "index_select", "gather",
    "scatter_add", "transpose", "permute", "reshape", "contiguous", "clone", "fill", "zeros_like",
    "full_like", "arange", "triu", "tril", "pow", "remainder", "floor_divide", "bitwise_and",
    "logical_not", "any", "all", "nonzero", "topk", "sort", "nn.functional.conv2d",
    "nn.functional.max_pool2d", "nn.functional.scaled_dot_product_attention",
}

# Known, intentional deviations from the CPU reference.
EXPECTED_FAILURES = {
    # CPU rounds every bf16 intermediate of div_floor/fmod to bf16; like CUDA
    # we compute in float32 ("opmath"), which differs at rounding boundaries.
    ("div.trunc_rounding", torch.bfloat16),
    ("div.floor_rounding", torch.bfloat16),
    ("remainder", torch.bfloat16),
    ("floor_divide", torch.bfloat16),
}

DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.uint8, torch.bool]


def _params():
    for op in op_db:
        for dtype in DTYPES:
            if dtype is not torch.float32 and not ALL_DTYPES and op.name not in SUBSET:
                continue
            name = op.name + ("." + op.variant_test_name if op.variant_test_name else "")
            # Not strict: whether the deviation shows depends on the driver's rounding.
            marks = [pytest.mark.xfail(reason="intentional deviation from CPU, see EXPECTED_FAILURES", strict=False)] \
                if (name, dtype) in EXPECTED_FAILURES else []
            yield pytest.param(op, dtype, id=f"{name}-{str(dtype).replace('torch.', '')}", marks=marks)


@pytest.mark.parametrize("op,dtype", list(_params()))
def test_opinfo(op, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        status, msg = run_op(op, dtype, MAX_SAMPLES)
    if status == "skip":
        pytest.skip(msg)
    assert status == "pass", msg


def _grad_params():
    for op in op_db:
        name = op.name + ("." + op.variant_test_name if op.variant_test_name else "")
        yield pytest.param(op, id=name)


@pytest.mark.parametrize("op", list(_grad_params()))
def test_opinfo_grad(op):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        status, msg = run_op_grad(op, torch.float32, min(MAX_SAMPLES, 10))
    if status == "skip":
        pytest.skip(msg)
    assert status == "pass", msg
