"""Scans (cummax/cummin/logcumsumexp), mode, unique variants and histograms
vs CPU, including NaN, ties, multi-chunk rows and values on bin edges."""

import pytest
import torch
import torch_webgpu  # noqa: F401

D = "webgpu"
NAN, INF = float("nan"), float("inf")


@pytest.fixture(autouse=True)
def no_fallback():
    before = set(torch_webgpu._C._fallback_ops())
    yield
    new = set(torch_webgpu._C._fallback_ops()) - before
    assert not new, f"unexpected CPU fallbacks: {new}"


def same(a, e):
    torch.testing.assert_close(a.cpu(), e, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64, torch.bool, torch.uint8], ids=str)
@pytest.mark.parametrize("shape,dim", [((7,), 0), ((5, 300), 1), ((300, 4), 0), ((2, 3, 1000), -1), ((1, 70000), 1)])
def test_cummax_cummin(dtype, shape, dim):
    g = torch.Generator().manual_seed(0)
    if dtype.is_floating_point:
        x = (torch.randn(shape, generator=g) * 3).round().to(dtype)  # many ties
        if x.numel() > 10:
            x.view(-1)[3] = NAN
            x.view(-1)[9] = NAN
    else:
        x = torch.randint(0, 5, shape, generator=g).to(dtype)
    for f in (torch.cummax, torch.cummin):
        c, d = f(x, dim), f(x.to(D), dim)
        same(d.values, c.values)
        same(d.indices, c.indices)


@pytest.mark.parametrize("shape,dim", [((7,), 0), ((5, 300), 1), ((2, 3, 1000), -1), ((1, 70000), 1)])
def test_logcumsumexp(shape, dim):
    x = torch.randn(shape) * 5
    torch.testing.assert_close(torch.logcumsumexp(x.to(D), dim).cpu(), torch.logcumsumexp(x, dim), atol=1e-4, rtol=1e-5)
    s = torch.tensor([-INF, -INF, 1.0, INF, 2.0, -INF, NAN, 3.0])
    same(torch.logcumsumexp(s.to(D), 0), torch.logcumsumexp(s, 0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.int32], ids=str)
@pytest.mark.parametrize("shape,dim", [((50,), 0), ((6, 40), 1), ((40, 7), 0), ((3, 5, 30), 2)])
def test_mode(dtype, shape, dim):
    x = torch.randint(0, 4, shape).to(dtype)
    for keepdim in (False, True):
        c, d = torch.mode(x, dim, keepdim), torch.mode(x.to(D), dim, keepdim)
        same(d.values, c.values)
        # The index among equal values is implementation-defined (unstable
        # sort on CPU); it must point at the mode value.
        idx = d.indices.cpu() if keepdim else d.indices.cpu().unsqueeze(dim)
        vals = c.values if keepdim else c.values.unsqueeze(dim)
        same(x.gather(dim, idx), vals)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bool], ids=str)
def test_unique_variants(dtype):
    for shape in [(50,), (6, 40), (40, 7), (3, 5, 4)]:
        x = torch.randint(0, 3, shape).to(dtype)
        for d in range(len(shape)):
            for f in (lambda t: torch.unique(t, dim=d, return_inverse=True, return_counts=True),
                      lambda t: torch.unique_consecutive(t, dim=d, return_inverse=True, return_counts=True)):
                for a, e in zip(f(x.to(D)), f(x)):
                    same(a, e)
        for a, e in zip(torch.unique_consecutive(x.to(D), return_inverse=True, return_counts=True),
                        torch.unique_consecutive(x, return_inverse=True, return_counts=True)):
            same(a, e)
    empty = torch.empty(5, 0, 5, dtype=dtype)
    assert torch.unique(empty.to(D), dim=1).shape == (5, 0, 5)
    with pytest.raises(RuntimeError, match="0 sized dimensions"):
        torch.unique(empty.to(D), dim=0)


def test_histc():
    g = torch.Generator().manual_seed(0)
    for x in [torch.randn(1000, generator=g), torch.randint(0, 10, (500,), generator=g).float(),
              torch.tensor([1.0, 2.0, 1.0, NAN, INF]), torch.tensor([3.0, 3.0])]:
        for bins, lo, hi in [(10, 0, 0), (7, -1, 1), (10, 0, 10), (3, 2, 2)]:
            try:
                expected = torch.histc(x, bins, lo, hi)
            except RuntimeError as e:  # e.g. data range [nan, nan]: same error on webgpu
                with pytest.raises(RuntimeError, match="not finite"):
                    torch.histc(x.to(D), bins, lo, hi)
                assert "not finite" in str(e)
                continue
            same(torch.histc(x.to(D), bins, lo, hi), expected)


@pytest.mark.parametrize("density", [False, True])
def test_histogram(density):
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 10, (400,), generator=g).float()  # exactly on edges
    w = torch.rand(400, generator=g)
    for kw in [{"bins": 10}, {"bins": 7, "range": (2.0, 8.0)}, {"bins": torch.tensor([0.0, 1.0, 2.5, 9.0, 9.5])}]:
        for weight in [None, w]:
            kwd = dict(kw)
            if isinstance(kwd["bins"], torch.Tensor):
                kwd_dev = {**kwd, "bins": kwd["bins"].to(D)}
            else:
                kwd_dev = kwd
            c = torch.histogram(x, weight=weight, density=density, **kwd)
            d = torch.histogram(x.to(D), weight=None if weight is None else weight.to(D), density=density, **kwd_dev)
            torch.testing.assert_close(d.hist.cpu(), c.hist, atol=1e-5, rtol=1e-5)
            # (edges may differ by an ulp where a driver fuses start + step * i)
            torch.testing.assert_close(d.bin_edges.cpu(), c.bin_edges, atol=0, rtol=2e-7)


def test_histogramdd():
    x = torch.randint(0, 5, (300, 3)).float()
    c = torch.histogramdd(x, bins=[4, 5, 3])
    d = torch.histogramdd(x.to(D), bins=[4, 5, 3])
    same(d.hist, c.hist)
    for a, e in zip(d.bin_edges, c.bin_edges):
        same(a, e)
    edges = [torch.tensor([0.0, 2.0, 5.0]), torch.tensor([0.0, 1.0, 3.0, 4.0]), torch.tensor([1.0, 4.0])]
    c = torch.histogramdd(x, bins=edges, density=True)
    d = torch.histogramdd(x.to(D), bins=[e.to(D) for e in edges], density=True)
    torch.testing.assert_close(d.hist.cpu(), c.hist)
