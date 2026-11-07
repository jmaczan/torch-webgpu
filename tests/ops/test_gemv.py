"""Decode-size matrix products (GEMV paths) and torch.ops.webgpu.multi_linear
vs CPU: row counts 1..16, biases, batches, unaligned / offset / strided
operands (which must take the generic path and still be right)."""

import pytest
import torch
import torch.nn.functional as F
import torch_webgpu  # noqa: F401

D = "webgpu"
pytestmark = pytest.mark.no_fallback


def close(a, e, tol=2e-4):
    torch.testing.assert_close(a.cpu(), e, atol=tol, rtol=tol)


@pytest.mark.parametrize("M", [1, 2, 3, 8, 16, 17])
@pytest.mark.parametrize("N,K", [(1, 4), (3, 8), (128, 896), (896, 896), (37, 1028), (1000, 4), (5, 4096), (7, 6)])
@pytest.mark.parametrize("bias", [False, True])
def test_linear(M, N, K, bias):
    torch.manual_seed(M * N + K)
    x, w = torch.randn(M, K), torch.randn(N, K)
    b = torch.randn(N) if bias else None
    out = F.linear(x.to(D), w.to(D), None if b is None else b.to(D))
    close(out, F.linear(x, w, b), 1e-3)


@pytest.mark.parametrize("B", [1, 3])
def test_batched_and_strided(B):
    torch.manual_seed(B)
    x, w = torch.randn(B, 2, 64), torch.randn(48, 64)
    close(F.linear(x.to(D), w.to(D)), F.linear(x, w))
    # x @ W (weight contiguous along N), transposed activations
    wt = torch.randn(64, 48)
    close(x.to(D) @ wt.to(D), x @ wt)
    xt = torch.randn(B, 64, 2).transpose(1, 2)
    close(xt.to(D) @ wt.to(D), xt @ wt)


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_unaligned_views(offset):
    """Weights / inputs starting at element offsets that break 16-byte
    alignment fall back to the scalar kernel."""
    w = torch.randn(64, 140)
    x = torch.randn(3, 140)
    wd, xd = w.to(D), x.to(D)
    close(xd[:, offset:offset + 128] @ wd[:, offset:offset + 128].t(), x[:, offset:offset + 128] @ w[:, offset:offset + 128].t())


def test_alpha_beta():
    x, w, b = torch.randn(2, 32), torch.randn(16, 32), torch.randn(16)
    close(torch.addmm(b.to(D), x.to(D), w.t().to(D), beta=0.5, alpha=-2.0), torch.addmm(b, x, w.t(), beta=0.5, alpha=-2.0))


def test_half_inputs():
    x, w = torch.randn(1, 256).half(), torch.randn(64, 256).half()
    close(F.linear(x.to(D), w.to(D)).float(), F.linear(x.float(), w.float()), 2e-2)


MULTI_CASES = [
    ([896, 128, 128], 896, [True, True, True]),
    ([4864, 4864], 896, [False, False]),
    ([3, 5], 4, [True, False]),
    ([7], 12, [False]),
    ([16, 16, 16, 16], 32, [False] * 4),
    ([10, 20, 30, 40], 64, [True] * 4),   # 2 + 4 + 4 bindings > 8 on minimum-limit devices
    ([6, 6], 10, [False, False]),          # K not a multiple of 4: generic path
]


@pytest.mark.parametrize("M", [1, 4, 16, 20])
@pytest.mark.parametrize("case", MULTI_CASES, ids=str)
def test_multi_linear(M, case):
    Ns, K, has_bias = case
    torch.manual_seed(K + M)
    x = torch.randn(M, K)
    ws = [torch.randn(n, K) for n in Ns]
    bs = [torch.randn(n) if hb else None for n, hb in zip(Ns, has_bias)]
    ref = torch.cat([F.linear(x, w, b) for w, b in zip(ws, bs)], -1)
    out = torch.ops.webgpu.multi_linear(x.to(D), [w.to(D) for w in ws], [None if b is None else b.to(D) for b in bs])
    assert out.shape == ref.shape
    close(out, ref, 1e-3)


def test_multi_linear_aliased_and_views():
    """Weights that are views of one buffer (and an input sharing it)."""
    big = torch.randn(64, 32)
    d = big.to(D)
    x = torch.randn(2, 32)
    ref = torch.cat([x @ big[:16].t(), x @ big[16:40].t()], -1)
    close(torch.ops.webgpu.multi_linear(x.to(D), [d[:16], d[16:40]], [None, None]), ref)
    # The input is a row of the same buffer.
    ref = torch.cat([big[60:62] @ big[:16].t(), big[60:62] @ big[16:40].t()], -1)
    close(torch.ops.webgpu.multi_linear(d[60:62], [d[:16], d[16:40]], [None, None]), ref)


def test_multi_linear_one_dispatch():
    if not torch.webgpu.profiler.is_supported():
        pytest.skip("no timestamp queries")
    x = torch.randn(1, 256, device=D)
    ws = [torch.randn(n, 256, device=D) for n in (256, 64, 64)]
    with torch.webgpu.profiler.profile() as p:
        torch.ops.webgpu.multi_linear(x, ws, [None, None, None])
    assert [r.name for r in p.records] == ["multi_gemv"]
