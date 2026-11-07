"""Special functions and less common pointwise ops vs CPU, over wide input
ranges including poles, special values and the branch points of the
piecewise approximations."""

import os

import pytest
import torch
import torch_webgpu  # noqa: F401

D = "webgpu"
NAN, INF = float("nan"), float("inf")

SPECIAL_VALUES = torch.tensor([
    NAN, INF, -INF, 0.0, -0.0, 1.0, 2.0, -1.0, -2.5, 0.5, 1e-30, 1e30, -1e30, 1e-6, 7.9999, 8.0, 8.0001,
    10.0, 50.0, 51.0, -6.0, -27.0, 0.9999, 1e8])


def wide():
    g = torch.Generator().manual_seed(0)
    return torch.cat([SPECIAL_VALUES, torch.randn(2000, generator=g) * 3,
                      torch.rand(500, generator=g) * 100 - 50, torch.rand(500, generator=g)])


def check(f, *args, atol=1e-5, rtol=2e-5):
    expected = f(*args)
    actual = f(*[a.to(D) if isinstance(a, torch.Tensor) else a for a in args])
    torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=rtol, equal_nan=True)


@pytest.fixture(autouse=True)
def no_fallback(monkeypatch):
    before = set(torch_webgpu._C._fallback_ops())
    yield
    new = set(torch_webgpu._C._fallback_ops()) - before
    assert not new, f"unexpected CPU fallbacks: {new}"


UNARY = {
    "lgamma": (torch.lgamma, 1e-5),
    "digamma": (torch.digamma, 1e-4),
    "i0": (torch.i0, 1e-5),
    "i0e": (torch.special.i0e, 1e-5),
    "i1": (torch.special.i1, 1e-5),
    "i1e": (torch.special.i1e, 1e-5),
    "log_ndtr": (torch.special.log_ndtr, 1e-5),
    "erfcx": (torch.special.erfcx, 1e-5),
    "entr": (torch.special.entr, 1e-5),
    "angle": (torch.angle, 1e-5),
}


@pytest.mark.parametrize("name", list(UNARY))
def test_unary(name):
    f, tol = UNARY[name]
    check(f, wide(), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32], ids=str)
@pytest.mark.parametrize("name", ["lgamma", "digamma", "i0", "i1e", "angle"])
def test_unary_other_dtypes(name, dtype):
    f, _ = UNARY[name]
    x = (torch.randn(500) * 4).to(dtype) if dtype.is_floating_point else torch.randint(-20, 20, (500,), dtype=dtype)
    check(f, x, atol=1e-2, rtol=1e-2)


def test_ndtri():
    p = torch.cat([torch.tensor([0.0, 1.0, 0.5, 1e-10, 1 - 1e-7, NAN, -0.1, 1.1, 0.13533, 0.8646]), torch.rand(2000)])
    check(torch.special.ndtri, p)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5])
def test_polygamma(n):
    x = torch.cat([wide().clamp(min=0.1), torch.tensor([-3.0, -2.0, -1.0, 0.0, -0.0, -2.5, NAN, INF, -INF, 1e-3, 30.0, 1e4])])
    check(lambda t: torch.polygamma(n, t), x, atol=1e-4, rtol=1e-4)


def test_zeta():
    x = wide()
    # (x huge: CPU's float series yields inf*0 = NaN; we return the limit q^-x -> 0.)
    x = x[x.abs() < 1e6]
    check(torch.special.zeta, x.abs() + 1, x.flip(0).abs() + 0.1, atol=1e-4, rtol=1e-4)
    check(torch.special.zeta, torch.tensor([1.0, 0.5, 2.0, 2.0, 3.0, 40.0, 60.0, 2.0, NAN]),
          torch.tensor([1.0, 1.0, -2.0, -2.5, 0.0, 1e6, 1.5, NAN, 1.0]), atol=1e-4, rtol=1e-4)


def test_xlog1py():
    x = wide()
    check(torch.special.xlog1py, x, x.flip(0))
    check(torch.special.xlog1py, torch.tensor([2.0, -2.0, 0.0, 1.0, NAN]), torch.tensor([-1.0, -1.0, -1.0, -3.0, 0.0]))


def test_clamp_tensor_bounds():
    x = wide()
    b = x.flip(0)
    check(lambda a, lo: torch.clamp(a, lo, lo + 1), x, b)
    check(lambda a, lo: torch.clamp(a, min=lo), x, b)
    check(lambda a, hi: torch.clamp(a, max=hi), x, b)
    i = torch.randint(-10, 10, (100,))
    check(lambda a, lo: torch.clamp(a, lo, lo + 3), i, i.flip(0))
    check(lambda a, lo: torch.clamp(a, lo, lo + 3), i.long(), i.flip(0).long())


@pytest.mark.parametrize("decimals", [0, 2, -1, -3])
def test_round_decimals(decimals):
    check(lambda t: torch.round(t, decimals=decimals), wide() * 100)


def test_nextafter_and_frexp_exact():
    x = wide()
    check(torch.nextafter, x, x.flip(0), atol=0, rtol=0)
    denormals = torch.tensor([1.4013e-45, 2.8026e-45, 1.7796e-43, 1e-40, -3e-39, 1.17e-38, -0.0, 0.0, INF, -INF, NAN])
    for t in [x, denormals]:
        check(lambda v: torch.frexp(v)[0], t, atol=0, rtol=0)
        check(lambda v: torch.frexp(v)[1], t, atol=0, rtol=0)


def test_rsub():
    x = wide()
    check(lambda a, b: torch.rsub(a, b, alpha=2), x, x.flip(0))
    i = torch.randint(-10, 10, (50,))
    check(lambda a, b: torch.rsub(a, b), i, i.flip(0))
