"""IEEE special-value semantics. GPU shader compilers may assume NaN-free
math (e.g. compile float `!=` as an ordered comparison), so every op whose
result depends on NaN/Inf/-0 handling is checked against the CPU here."""

import re

import pytest
import torch
import torch_webgpu  # noqa: F401

D = "webgpu"
NAN, INF = float("nan"), float("inf")
A = torch.tensor([NAN, NAN, 1.0, NAN, 0.0, INF, -0.0, -INF, 2.0])
B = torch.tensor([NAN, 1.0, NAN, 0.0, NAN, NAN, 0.0, -INF, 2.0])


def same(actual, expected):
    actual = actual.cpu()
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


BINARY = ["eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or", "logical_xor",
          "maximum", "minimum", "fmax", "fmin", "isclose", "heaviside"]
UNARY = {
    "bool": lambda t: t.bool(),
    "logical_not": torch.logical_not,
    "nonzero": lambda t: t.nonzero(),
    "count_nonzero": torch.count_nonzero,
    "ne_zero": lambda t: t != 0,
    "ne_nan": lambda t: t != NAN,
    "eq_nan": lambda t: t == NAN,
    "any": lambda t: t.any(),
    "all": lambda t: t.all(),
    "where": lambda t: torch.where(t.bool(), 1.0, 2.0),
    "sign": torch.sign,
    "signbit": torch.signbit,
    "clamp": lambda t: t.clamp(0, 1),
    "isnan": torch.isnan,
    "isinf": torch.isinf,
    "isfinite": torch.isfinite,
    "nan_to_num": torch.nan_to_num,
    "norm0": lambda t: torch.linalg.vector_norm(t, 0),
    "amax": lambda t: t.amax(),
    "amin": lambda t: t.amin(),
    "argmax": lambda t: t.argmax(),
    "argmin": lambda t: t.argmin(),
    "masked_select": lambda t: t.masked_select(t.bool()),
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize("name", BINARY)
def test_binary(name, dtype):
    f = getattr(torch, name)
    a, b = A.to(dtype), B.to(dtype)
    same(f(a.to(D), b.to(D)), f(a, b))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize("name", list(UNARY))
def test_unary(name, dtype):
    f = UNARY[name]
    a = A.to(dtype)
    same(f(a.to(D)), f(a))


COMPUTED = {
    "sub_self": lambda t: t - t,  # inf - inf -> NaN produced inside the kernel
    "abs": torch.abs,
    "neg": torch.neg,
    "mul0": lambda t: t * 0,
    "add_inf": lambda t: t + INF,
    "exp": torch.exp,
    "sqrt": torch.sqrt,
    "isnan_sub": lambda t: torch.isnan(t - t),
    "sum": lambda t: t.sum(dim=0, keepdim=True),
    "amax": lambda t: t.amax(dim=0, keepdim=True),
}


# NaN/Inf created by arithmetic (not just loaded) must survive the store to
# half types. Denormals are excluded: GPUs flush f32 denormals to zero.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize("name", list(COMPUTED))
def test_computed_special_values(name, dtype):
    a = torch.tensor([NAN, INF, -INF, 1.0, -NAN, 6e4, -0.0, 2.0]).to(dtype)
    f = COMPUTED[name]
    # Special values must match exactly; finite results of transcendental
    # functions may differ by an ulp between drivers.
    actual, expected = f(a.to(D)).cpu(), f(a)
    torch.testing.assert_close(actual, expected, atol=0, rtol=2e-7 if dtype == torch.float32 else 0, equal_nan=True)


def test_float_to_int_and_bool_casts():
    for dt in [torch.bool, torch.uint8, torch.int32, torch.int64]:
        x = torch.tensor([NAN, -0.0, 0.0, 0.5, -1.5]) if dt == torch.bool else torch.tensor([-0.0, 0.0, 0.5, -1.5, 3.99])
        same(x.to(D).to(dt), x.to(dt))


def test_printing_matches_cpu_without_fallbacks():
    before = set(torch_webgpu._C._fallback_ops())
    x = torch.tensor([[NAN, 1e-8, 3.0], [INF, -2.5, 0.0]])
    for t in [x, x.half(), torch.arange(5), torch.tensor([True, False]), torch.randn(500, 500), torch.tensor(3.5)]:
        assert re.sub(r",\s*device='webgpu:0'", "", repr(t.to(D))) == repr(t)
    assert set(torch_webgpu._C._fallback_ops()) == before
