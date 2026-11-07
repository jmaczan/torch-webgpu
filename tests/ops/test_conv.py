"""2-d convolutions (implicit GEMM) vs CPU, forward and gradients, over the
shapes that select different paths: shallow / wide layers, deep layers with
few output pixels and long reductions (split-K), batch folded into the
GEMM, groups, strides, dilation, padding, half types."""

import pytest
import torch
import torch.nn.functional as F
import torch_webgpu  # noqa: F401

D = "webgpu"
pytestmark = pytest.mark.no_fallback

CASES = [  # (N, C, H, W, O, k, stride, padding, dilation, groups)
    (2, 3, 32, 32, 16, 3, 1, 1, 1, 1),
    (5, 64, 14, 14, 64, 3, 1, 1, 1, 1),
    (8, 512, 7, 7, 512, 3, 1, 1, 1, 1),      # deep: split-K
    (3, 1024, 7, 7, 256, 1, 1, 0, 1, 1),     # 1x1, long K
    (4, 32, 9, 11, 48, 3, 2, 1, 1, 1),       # stride
    (2, 16, 12, 12, 16, 3, 1, 2, 2, 1),      # dilation
    (2, 32, 10, 10, 64, 3, 1, 1, 1, 4),      # groups
    (3, 64, 8, 8, 64, 3, 1, 1, 1, 64),       # depthwise
    (1, 256, 4, 4, 128, 3, 1, 1, 1, 1),
    (6, 8, 5, 5, 4, 5, 1, 0, 1, 1),          # output 1x1
]


@pytest.mark.parametrize("case", CASES, ids=str)
@pytest.mark.parametrize("bias", [False, True])
def test_conv2d_forward_backward(case, bias):
    N, C, H, W, O, k, s, p, d, g = case
    x = torch.randn(N, C, H, W, requires_grad=True)
    w = (torch.randn(O, C // g, k, k) / (C // g * k * k) ** 0.5).requires_grad_()
    b = torch.randn(O, requires_grad=True) if bias else None
    out = F.conv2d(x, w, b, s, p, d, g)
    xd, wd = x.detach().to(D).requires_grad_(), w.detach().to(D).requires_grad_()
    bd = b.detach().to(D).requires_grad_() if bias else None
    outd = F.conv2d(xd, wd, bd, s, p, d, g)
    torch.testing.assert_close(outd.detach().cpu(), out.detach(), atol=2e-4, rtol=2e-4)
    go = torch.randn_like(out)
    out.backward(go)
    outd.backward(go.to(D))
    torch.testing.assert_close(xd.grad.cpu(), x.grad, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(wd.grad.cpu(), w.grad, atol=2e-3, rtol=2e-3)
    if bias:
        torch.testing.assert_close(bd.grad.cpu(), b.grad, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
def test_conv2d_half_deep(dtype):
    x, w = torch.randn(4, 256, 7, 7).to(dtype), (torch.randn(256, 256, 3, 3) / 48).to(dtype)
    out = F.conv2d(x.to(D), w.to(D), padding=1)
    assert out.dtype == dtype
    torch.testing.assert_close(out.float().cpu(), F.conv2d(x.float(), w.float(), padding=1), atol=5e-2, rtol=5e-2)


def test_conv2d_noncontiguous_input():
    x = torch.randn(2, 8, 16, 16).permute(0, 1, 3, 2)
    w = torch.randn(8, 8, 3, 3)
    torch.testing.assert_close(F.conv2d(x.to(D), w.to(D), padding=1).cpu(), F.conv2d(x, w, padding=1), atol=2e-4, rtol=2e-4)
