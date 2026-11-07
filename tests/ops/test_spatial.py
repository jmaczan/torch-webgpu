"""Spatial ops vs CPU, forward and backward: grid_sample (every mode x
padding x align_corners, 2-D and 3-D), 3-D / adaptive pooling, unpooling,
3-D and anti-aliased upsampling, padding modes."""

import itertools

import pytest
import torch
import torch.nn.functional as F
import torch_webgpu  # noqa: F401

D = "webgpu"


@pytest.fixture(autouse=True)
def no_fallback():
    before = set(torch_webgpu._C._fallback_ops())
    yield
    new = set(torch_webgpu._C._fallback_ops()) - before
    assert not new, f"unexpected CPU fallbacks: {new}"


def check_fwd_bwd(f, *inputs, atol=1e-4, rtol=1e-4, seed=0):
    """Compares f(*inputs) and the gradients of a random weighting of it."""
    cpu = [x.detach().clone().requires_grad_(x.is_floating_point()) for x in inputs]
    dev = [x.detach().to(D).requires_grad_(x.is_floating_point()) for x in inputs]
    out_c, out_d = f(*cpu), f(*dev)
    outs_c = out_c if isinstance(out_c, tuple) else (out_c,)
    outs_d = out_d if isinstance(out_d, tuple) else (out_d,)
    g = torch.Generator().manual_seed(seed)
    loss_c, loss_d = 0, 0
    for oc, od in zip(outs_c, outs_d):
        torch.testing.assert_close(od.detach().cpu(), oc.detach(), atol=atol, rtol=rtol)
        if oc.is_floating_point():
            w = torch.randn(oc.shape, generator=g)
            loss_c = loss_c + (oc * w).sum()
            loss_d = loss_d + (od * w.to(D)).sum()
    if not isinstance(loss_c, torch.Tensor):
        return
    loss_c.backward()
    loss_d.backward()
    for xc, xd in zip(cpu, dev):
        if xc.requires_grad:
            torch.testing.assert_close(xd.grad.cpu(), xc.grad, atol=atol * 10, rtol=rtol * 10)


@pytest.mark.parametrize("nd", [2, 3])
@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [False, True])
def test_grid_sample(nd, mode, padding_mode, align_corners):
    g = torch.Generator().manual_seed(1)
    spatial = (5, 6) if nd == 2 else (4, 5, 3)
    out_spatial = (7, 3) if nd == 2 else (3, 4, 2)
    x = torch.randn((2, 3) + spatial, generator=g)
    grid = torch.rand((2,) + out_spatial + (nd,), generator=g) * 3 - 1.5  # includes out-of-range points
    flat = grid.view(-1)
    flat[:6] = torch.tensor([-1.0, 1.0, 0.0, 1.5, -1.5, 0.25])  # exact boundaries
    # Nearest: avoid exact .5 ties where rounding of float noise decides.
    check_fwd_bwd(lambda a, b: F.grid_sample(a, b, mode=mode, padding_mode=padding_mode, align_corners=align_corners),
                  x, grid)


@pytest.mark.parametrize("ceil_mode", [False, True])
def test_pool3d(ceil_mode):
    x = torch.randn(2, 3, 7, 8, 6)
    check_fwd_bwd(lambda t: F.max_pool3d(t, 3, stride=2, padding=1, ceil_mode=ceil_mode, return_indices=True), x)
    check_fwd_bwd(lambda t: F.max_pool3d(t, (2, 3, 2), dilation=(1, 2, 1), ceil_mode=ceil_mode), x)
    for cip, div in itertools.product([False, True], [None, 5]):
        check_fwd_bwd(lambda t: F.avg_pool3d(t, 3, stride=2, padding=1, ceil_mode=ceil_mode,
                                             count_include_pad=cip, divisor_override=div), x)


@pytest.mark.parametrize("out", [(1, 1), (3, 4), (5, 7), (2, 9)])
def test_adaptive_max_pool2d(out):
    x = torch.randn(2, 3, 5, 9)
    check_fwd_bwd(lambda t: F.adaptive_max_pool2d(t, out, return_indices=True), x)
    check_fwd_bwd(lambda t: F.adaptive_max_pool2d(t, out), x[0])


def test_adaptive_max_pool3d_and_unpool():
    x = torch.randn(2, 3, 5, 6, 7)
    check_fwd_bwd(lambda t: F.adaptive_max_pool3d(t, (2, 4, 3), return_indices=True), x)
    y, idx = F.max_pool2d(torch.randn(2, 3, 8, 8), 2, return_indices=True)
    check_fwd_bwd(lambda t: F.max_unpool2d(t, idx.to(t.device), 2), y)
    y3, idx3 = F.max_pool3d(torch.randn(1, 2, 4, 4, 6), 2, return_indices=True)
    check_fwd_bwd(lambda t: F.max_unpool3d(t, idx3.to(t.device), 2), y3)


@pytest.mark.parametrize("mode", ["nearest", "nearest-exact", "trilinear"])
@pytest.mark.parametrize("size", [(3, 9, 4), (8, 2, 5)])
def test_upsample3d(mode, size):
    x = torch.randn(2, 3, 4, 5, 6)
    kw = {"align_corners": False} if mode == "trilinear" else {}
    check_fwd_bwd(lambda t: F.interpolate(t, size=size, mode=mode, **kw), x)
    if mode == "trilinear":
        check_fwd_bwd(lambda t: F.interpolate(t, size=size, mode=mode, align_corners=True), x)


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
@pytest.mark.parametrize("size", [(3, 4), (12, 5), (31, 2)])
@pytest.mark.parametrize("align_corners", [False, True])
def test_antialias(mode, size, align_corners):
    x = torch.randn(2, 3, 13, 9)
    check_fwd_bwd(lambda t: F.interpolate(t, size=size, mode=mode, antialias=True, align_corners=align_corners), x)


@pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
@pytest.mark.parametrize("nd", [1, 2, 3])
def test_pad_modes(mode, nd):
    x = torch.randn((2, 3) + (5, 6, 4)[:nd])
    pad = (2, 1, 3, 0, 1, 2)[: 2 * nd]
    check_fwd_bwd(lambda t: F.pad(t, pad, mode=mode), x)
    if mode == "replicate":
        check_fwd_bwd(lambda t: F.pad(t, tuple(-1 if i % 2 else p for i, p in enumerate(pad)), mode=mode), x)
