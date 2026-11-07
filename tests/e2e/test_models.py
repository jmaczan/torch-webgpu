"""End-to-end Hugging Face models on webgpu vs CPU: greedy generation must
match token for token without CPU fallbacks, and a few training steps must
track the CPU. Models are taken from the local HF cache (skipped if absent).
"""

import copy
import os
import sys
import warnings

import pytest
import torch
import torch_webgpu  # noqa: F401

os.environ.setdefault("HF_HUB_OFFLINE", "1")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from e2e_models import check_model  # noqa: E402

MODELS = [
    "openai-community/gpt2",              # learned positions, GELU, Conv1D
    "HuggingFaceTB/SmolLM2-135M",         # Llama: RoPE, RMSNorm, SwiGLU, GQA
    "EleutherAI/pythia-70m-deduped",      # GPT-NeoX: partial rotary, parallel residual
    "facebook/opt-125m",                  # ReLU, learned positions with offset
    "Qwen/Qwen2.5-0.5B",                  # QKV bias, tied embeddings
]


def cached(model_id):
    try:
        from huggingface_hub import try_to_load_from_cache
        return try_to_load_from_cache(model_id, "config.json") not in (None,) and \
            isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.parametrize("model_id", MODELS)
def test_generation_matches_cpu(model_id):
    if not cached(model_id):
        pytest.skip(f"{model_id} not in the local HF cache")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = check_model(model_id, new_tokens=12)
    assert r["fallbacks"] == [], r
    assert r["top1_match"], r
    assert r["generation_match"], r
    assert r["logits_max_abs_diff"] < 1e-4 * max(1.0, r["logits_scale"]) * 10, r


def test_gpt2_training_tracks_cpu():
    model_id = "openai-community/gpt2"
    if not cached(model_id):
        pytest.skip(f"{model_id} not in the local HF cache")
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    m_cpu = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    m_cpu.config.n_layer = 2
    m_cpu.transformer.h = m_cpu.transformer.h[:2]  # keep it quick
    m_cpu.train()
    for mod in m_cpu.modules():  # deterministic: no dropout
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    m_dev = copy.deepcopy(m_cpu).to("webgpu")
    opt_c = torch.optim.AdamW(m_cpu.parameters(), lr=1e-4)
    opt_d = torch.optim.AdamW(m_dev.parameters(), lr=1e-4)
    ids = torch.randint(0, 50257, (2, 32))
    losses = []
    for _ in range(3):
        step = []
        for m, opt, x in [(m_cpu, opt_c, ids), (m_dev, opt_d, ids.to("webgpu"))]:
            opt.zero_grad()
            loss = m(x, labels=x).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            step.append(loss.item())
        losses.append(step)
    for lc, ld in losses:
        assert abs(lc - ld) < 1e-3 * max(1.0, abs(lc)), losses
    # Adam moves every weight by ~lr per step whatever the gradient's size, so
    # near-zero gradients whose sign differs in the last bits can put a weight
    # up to 2 * lr per step apart: bound by steps * 2 * lr.
    atol = 3 * 2 * 1e-4
    for (n, pc), pd in zip(m_cpu.named_parameters(), m_dev.parameters()):
        torch.testing.assert_close(pd.detach().cpu(), pc.detach(), atol=atol, rtol=1e-3, msg=lambda s: f"{n}: {s}")
