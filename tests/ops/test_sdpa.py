"""scaled_dot_product_attention on webgpu (fused kernel and math path) vs
CPU: shapes, causal masking, GQA, boolean / float / broadcast masks, fully
masked rows, half types, and gradients."""

import pytest
import torch
import torch.nn.functional as F
import torch_webgpu  # noqa: F401

D = "webgpu"

CASES = [  # (B, H, Hk, Lq, Lk, head dim)
    (2, 4, 4, 37, 37, 64),
    (1, 8, 2, 100, 100, 64),    # GQA
    (2, 2, 2, 1, 50, 32),       # decode
    (1, 4, 4, 16, 80, 128),
    (1, 3, 3, 70, 20, 8),       # Lq > Lk
    (2, 2, 1, 33, 33, 80),
    (1, 2, 2, 5, 5, 160),       # head dim beyond the fused kernel -> math path
]


@pytest.fixture(autouse=True)
def no_fallback():
    before = set(torch_webgpu._C._fallback_ops())
    yield
    assert not set(torch_webgpu._C._fallback_ops()) - before


def close(a, e, tol):
    torch.testing.assert_close(a.detach().cpu().float(), e.detach().float(), atol=tol, rtol=tol, equal_nan=True)


@pytest.mark.parametrize("case", CASES, ids=str)
@pytest.mark.parametrize("variant", ["plain", "causal", "bool_mask", "float_mask", "float_mask_bcast"])
def test_sdpa_forward_backward(case, variant):
    torch.manual_seed(0)
    B, H, Hk, Lq, Lk, Dh = case
    q, k, v = torch.randn(B, H, Lq, Dh), torch.randn(B, Hk, Lk, Dh), torch.randn(B, Hk, Lk, Dh)
    mask = {"bool_mask": lambda: torch.rand(B, 1, Lq, Lk) > 0.3, "float_mask": lambda: torch.randn(B, H, Lq, Lk),
            "float_mask_bcast": lambda: torch.randn(Lq, Lk)}.get(variant, lambda: None)()
    causal = variant == "causal"
    rep = H // Hk
    qc, kc, vc = [t.clone().requires_grad_() for t in (q, k, v)]
    oc = F.scaled_dot_product_attention(qc, kc.repeat_interleave(rep, 1), vc.repeat_interleave(rep, 1),
                                        attn_mask=mask, is_causal=causal)
    qd, kd, vd = [t.to(D).requires_grad_() for t in (q, k, v)]
    od = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=None if mask is None else mask.to(D),
                                        is_causal=causal, enable_gqa=rep > 1)
    g = torch.randn_like(oc)
    oc.backward(g)
    od.backward(g.to(D))
    close(od, oc, 2e-4)
    for a, e in [(qd.grad, qc.grad), (kd.grad, kc.grad), (vd.grad, vc.grad)]:
        close(a, e, 1e-3)


def test_fully_masked_rows_are_zero():
    q, k, v = torch.randn(1, 2, 4, 16), torch.randn(1, 2, 6, 16), torch.randn(1, 2, 6, 16)
    mask = torch.ones(1, 1, 4, 6, dtype=torch.bool)
    mask[..., 1, :] = False
    out = F.scaled_dot_product_attention(q.to(D), k.to(D), v.to(D), attn_mask=mask.to(D))
    close(out, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), 1e-4)
    assert torch.equal(out[0, :, 1].cpu(), torch.zeros(2, 16))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
def test_half_types(dtype):
    q, k, v = [torch.randn(2, 4, 50, 64).to(dtype) for _ in range(3)]
    out = F.scaled_dot_product_attention(q.to(D), k.to(D), v.to(D), is_causal=True)
    assert out.dtype == dtype
    close(out, F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True), 2e-2)


def test_long_context_memory():
    """The fused kernel never materializes the L x L scores."""
    L = 8192
    q = torch.randn(1, 4, L, 64, device=D)
    k = torch.randn(1, 1, L, 64, device=D)
    v = torch.randn(1, 1, L, 64, device=D)
    torch.webgpu.synchronize()
    torch.webgpu.reset_peak_memory_stats()
    base = torch.webgpu.memory_allocated()
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    torch.webgpu.synchronize()
    assert torch.webgpu.max_memory_allocated() - base < 64 * 2**20  # scores alone would be 1 GiB
    assert torch.isfinite(out).all().item()


DECODE_CASES = [  # (B, H, Hk, Lq, Lk, head dim, value head dim)
    (1, 14, 2, 1, 1, 64, 64),
    (1, 14, 2, 1, 48, 64, 64),
    (2, 4, 4, 1, 255, 32, 32),
    (1, 4, 2, 1, 256, 64, 64),
    (1, 4, 2, 1, 257, 64, 64),
    (1, 2, 1, 3, 1000, 128, 128),
    (1, 2, 2, 4, 2000, 80, 80),
    (1, 2, 2, 2, 33, 160, 96),
    (1, 2, 2, 1, 6000, 64, 64),    # may exceed the shared score buffer: other path
    (1, 3, 3, 1, 70, 8, 200),
]


@pytest.mark.parametrize("case", DECODE_CASES, ids=str)
@pytest.mark.parametrize("variant", ["plain", "causal", "bool_mask", "hf_additive_mask", "fully_masked_row"])
def test_sdpa_decode(case, variant):
    """Few queries against a long key sequence (LLM decoding with a KV
    cache), incl. Hugging Face's additive masks (finfo.min, finite)."""
    B, H, Hk, Lq, Lk, Dh, Dv = case
    q, k, v = torch.randn(B, H, Lq, Dh), torch.randn(B, Hk, Lk, Dh), torch.randn(B, Hk, Lk, Dv)
    mask = None
    if variant == "bool_mask":
        mask = torch.rand(B, 1, Lq, Lk) > 0.3
    elif variant == "hf_additive_mask":
        visible = torch.rand(B, 1, Lq, Lk) > 0.5
        visible[..., 0] = True
        mask = torch.where(visible, 0.0, torch.finfo(torch.float32).min)
    elif variant == "fully_masked_row":
        mask = torch.rand(B, 1, Lq, Lk) > 0.3
        mask[..., 0, :] = False
    rep = H // Hk
    ref = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1),
                                         attn_mask=mask, is_causal=variant == "causal")
    out = F.scaled_dot_product_attention(q.to(D), k.to(D), v.to(D), attn_mask=None if mask is None else mask.to(D),
                                         is_causal=variant == "causal", enable_gqa=rep > 1)
    if variant == "fully_masked_row":
        ref = ref.nan_to_num(0.0)  # CPU math gives NaN rows; ours (like the fused CPU kernels) gives 0
    close(out, ref, 2e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
def test_sdpa_decode_half(dtype):
    q, k, v = torch.randn(1, 8, 1, 64).to(dtype), torch.randn(1, 2, 300, 64).to(dtype), torch.randn(1, 2, 300, 64).to(dtype)
    out = F.scaled_dot_product_attention(q.to(D), k.to(D), v.to(D), enable_gqa=True)
    assert out.dtype == dtype
    ref = F.scaled_dot_product_attention(q.float(), k.float().repeat_interleave(4, 1), v.float().repeat_interleave(4, 1))
    close(out, ref, 2e-2)


def test_sdpa_decode_grad():
    q, k, v = torch.randn(1, 4, 1, 32), torch.randn(1, 4, 100, 32), torch.randn(1, 4, 100, 32)
    qc, kc, vc = [t.clone().requires_grad_() for t in (q, k, v)]
    qd, kd, vd = [t.to(D).requires_grad_() for t in (q, k, v)]
    g = torch.randn(1, 4, 1, 32)
    F.scaled_dot_product_attention(qc, kc, vc).backward(g)
    F.scaled_dot_product_attention(qd, kd, vd).backward(g.to(D))
    for a, e in [(qd.grad, qc.grad), (kd.grad, kc.grad), (vd.grad, vc.grad)]:
        close(a, e, 1e-3)


def test_sdpa_decode_uses_one_dispatch():
    if not torch.webgpu.profiler.is_supported():
        pytest.skip("no timestamp queries")
    q, k, v = torch.randn(1, 14, 1, 64, device=D), torch.randn(1, 2, 500, 64, device=D), torch.randn(1, 2, 500, 64, device=D)
    F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    with torch.webgpu.profiler.profile() as p:
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    assert [r.name for r in p.records] == ["attention_decode"]
