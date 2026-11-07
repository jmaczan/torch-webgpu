"""PyTorch's inference fast paths for nn.MultiheadAttention,
nn.TransformerEncoderLayer and nn.TransformerEncoder (eval mode, no grad):
_native_multi_head_attention and _transformer_encoder_layer_fwd run on the
device, and padded batches (which the encoder would turn into nested
tensors) work. Compared with the same modules on the CPU."""

import copy

import pytest
import torch
import torch.nn as nn
import torch_webgpu  # noqa: F401

D = "webgpu"
pytestmark = pytest.mark.no_fallback


def close(a, e, tol=1e-4):
    torch.testing.assert_close(a.cpu(), e, atol=tol, rtol=tol)


def padding_mask(B, L, kind):
    m = torch.zeros(B, L, dtype=torch.bool)
    m[1:, L - 3:] = True  # right padding in all but the first sequence
    return m if kind == "bool" else torch.zeros(B, L).masked_fill(m, float("-inf"))


@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("mask", [None, "bool", "float"])
def test_encoder_layer(norm_first, activation, mask):
    layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True, norm_first=norm_first,
                                       activation=activation).eval()
    x = torch.randn(3, 10, 64)
    pm = padding_mask(3, 10, mask) if mask else None
    with torch.no_grad():
        ref = layer(x, src_key_padding_mask=pm)
        out = copy.deepcopy(layer).to(D)(x.to(D), src_key_padding_mask=None if pm is None else pm.to(D))
    close(out, ref)


@pytest.mark.parametrize("nested", [True, False])
def test_encoder_with_padding(nested):
    enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, batch_first=True), 2,
                                enable_nested_tensor=nested).eval()
    x = torch.randn(3, 12, 64)
    pm = padding_mask(3, 12, "bool")
    with torch.no_grad():
        ref = enc(x, src_key_padding_mask=pm)
        out = copy.deepcopy(enc).to(D)(x.to(D), src_key_padding_mask=pm.to(D))
    # The CPU's nested path leaves padded positions zero; compare the rest.
    keep = ~pm
    close(out.cpu()[keep], ref[keep])


def test_encoder_causal_mask():
    enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(32, 2, 64, batch_first=True), 2).eval()
    x = torch.randn(2, 9, 32)
    causal = nn.Transformer.generate_square_subsequent_mask(9)
    with torch.no_grad():
        ref = enc(x, mask=causal, is_causal=True)
        out = copy.deepcopy(enc).to(D)(x.to(D), mask=causal.to(D), is_causal=True)
    close(out, ref)


@pytest.mark.parametrize("need_weights", [False, True])
@pytest.mark.parametrize("mask", [None, "bool", "float"])
def test_multihead_attention_self(need_weights, mask):
    mha = nn.MultiheadAttention(64, 4, batch_first=True).eval()
    x = torch.randn(3, 10, 64)
    pm = padding_mask(3, 10, mask) if mask else None
    with torch.no_grad():
        ref, ref_w = mha(x, x, x, key_padding_mask=pm, need_weights=need_weights)
        xd = x.to(D)
        out, w = copy.deepcopy(mha).to(D)(xd, xd, xd, key_padding_mask=None if pm is None else pm.to(D),
                                           need_weights=need_weights)
    close(out, ref)
    if need_weights:
        close(w, ref_w)


def test_multihead_attention_cross():
    mha = nn.MultiheadAttention(32, 2, batch_first=True).eval()
    q, kv = torch.randn(2, 5, 32), torch.randn(2, 9, 32)
    with torch.no_grad():
        ref = mha(q, kv, kv, need_weights=False)[0]
        md = copy.deepcopy(mha).to(D)
        kvd = kv.to(D)
        out = md(q.to(D), kvd, kvd, need_weights=False)[0]
    close(out, ref)
