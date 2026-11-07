"""Sorting and searching: properties the OpInfo samples don't cover
(stability, NaN ordering, long rows, all dtypes), compared exactly to CPU."""

import pytest
import torch
import torch_webgpu  # noqa: F401

D = "webgpu"


def assert_same(actual, expected):
    if isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            assert_same(a, e)
        return
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0, equal_nan=True)


def make(shape, dtype, low=0, high=50):
    if dtype.is_floating_point:
        return (torch.randn(shape) * 10).to(dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape).bool()
    return torch.randint(low, high, shape).to(dtype)


DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.int16, torch.uint8, torch.bool]
SHAPES = [(1,), (7,), (5, 33), (3, 1000), (2, 4, 9), (1, 151936)]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("descending", [False, True])
def test_stable_sort_matches_cpu(dtype, shape, descending):
    torch.manual_seed(0)
    x = make(shape, dtype, high=8)  # many ties: exercises stability
    for dim in range(-len(shape), 0):
        assert_same(x.to(D).sort(dim=dim, descending=descending, stable=True),
                    x.sort(dim=dim, descending=descending, stable=True))
        assert_same(x.to(D).argsort(dim=dim, descending=descending, stable=True),
                    x.argsort(dim=dim, descending=descending, stable=True))


@pytest.mark.parametrize("descending", [False, True])
def test_sort_nan_and_inf(descending):
    x = torch.tensor([[3.0, float("nan"), -float("inf"), 0.0, -0.0, float("inf"), float("nan"), 1.0]])
    assert_same(x.to(D).sort(dim=-1, descending=descending, stable=True),
                x.sort(dim=-1, descending=descending, stable=True))


def test_sort_noncontiguous_and_scalar():
    x = torch.randn(6, 10).t()
    assert_same(x.to(D).sort(dim=0, stable=True), x.sort(dim=0, stable=True))
    s = torch.tensor(2.5)
    assert_same(s.to(D).sort(), s.sort())
    e = torch.empty(0, 3)
    assert_same(e.to(D).sort(dim=0), e.sort(dim=0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.int32, torch.float16], ids=str)
@pytest.mark.parametrize("largest", [True, False])
def test_topk_values(dtype, largest):
    torch.manual_seed(1)
    x = make((4, 5000), dtype, high=100000)
    for k in [1, 5, 64]:
        gv, gi = x.to(D).topk(k, dim=-1, largest=largest)
        cv, _ = x.topk(k, dim=-1, largest=largest)
        assert_same(gv, cv)
        assert_same(x.gather(-1, gi.cpu()), cv)


def test_median_kthvalue():
    torch.manual_seed(2)
    x = torch.randn(5, 17)
    x[1, 3] = float("nan")
    assert_same(x.to(D).median(dim=-1)[0], x.median(dim=-1)[0])
    assert_same(x.to(D).nanmedian(dim=-1), x.nanmedian(dim=-1))
    assert_same(x.to(D).nanmedian(), x.nanmedian())
    assert_same(x.to(D).median(), x.median())
    for k in [1, 9, 17]:
        assert_same(x[2:].to(D).kthvalue(k, dim=-1), x[2:].kthvalue(k, dim=-1))
    allnan = torch.full((2, 4), float("nan"))
    assert_same(allnan.to(D).nanmedian(dim=1)[0], allnan.nanmedian(dim=1)[0])


@pytest.mark.parametrize("right", [False, True])
def test_searchsorted_bucketize(right):
    torch.manual_seed(3)
    s = torch.randint(0, 20, (4, 30)).sort(dim=-1)[0].float()
    v = torch.randint(-2, 22, (4, 9)).float()
    assert_same(torch.searchsorted(s.to(D), v.to(D), right=right), torch.searchsorted(s, v, right=right))
    assert_same(torch.searchsorted(s.to(D), v.to(D), right=right, out_int32=True),
                torch.searchsorted(s, v, right=right, out_int32=True))
    assert_same(torch.bucketize(v.to(D), s[0].to(D), right=right), torch.bucketize(v, s[0], right=right))
    assert_same(torch.searchsorted(s[0].to(D), 7.0, right=right), torch.searchsorted(s[0], 7.0, right=right))


def test_isin_unique_misc():
    torch.manual_seed(4)
    a = torch.randint(0, 20, (300,))
    b = torch.randint(0, 20, (7,))
    assert_same(torch.isin(a.to(D), b.to(D)), torch.isin(a, b))
    assert_same(torch.isin(a.to(D), b.to(D), invert=True), torch.isin(a, b, invert=True))
    assert_same(torch.unique(a.to(D), return_inverse=True, return_counts=True),
                torch.unique(a, return_inverse=True, return_counts=True))
    v = torch.randn(4, 7)
    assert_same(torch.aminmax(v.to(D), dim=1), torch.aminmax(v, dim=1))
    assert_same(torch.count_nonzero(a.to(D) - 5, dim=0), torch.count_nonzero(a - 5, dim=0))
    assert torch.equal(a.to(D), a.to(D))
    assert not torch.equal(a.to(D), (a + 1).to(D))
    assert not torch.equal(a.to(D), a[:-1].to(D))


def test_no_cpu_fallback():
    before = set(torch_webgpu._C._fallback_ops())
    x = torch.randn(3, 50, device=D)
    x.sort(dim=-1)
    x.topk(4)
    x.median(dim=0)
    torch.searchsorted(x.sort()[0], x)
    torch.isin(x, x[0])
    new = set(torch_webgpu._C._fallback_ops()) - before
    assert not new, f"unexpected CPU fallbacks: {new}"
