"""End-to-end check of Hugging Face causal LMs on webgpu vs CPU (float32).

For each model: next-token logits of a prompt are compared with the CPU,
greedy generation must produce the same tokens, and CPU fallbacks and
throughput are reported.

    HF_HUB_OFFLINE=1 python scripts/e2e_models.py gpt2 Qwen/Qwen2.5-0.5B ...
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings

import torch
import torch_webgpu

PROMPT = "The quick brown fox jumps over the lazy dog. In the beginning, the universe was"


def check_model(model_id: str, new_tokens: int = 20, dtype=torch.float32) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa").eval()
    except ValueError:  # architecture without an SDPA path
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="eager").eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    res = {"model": model_id, "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}

    with torch.no_grad():
        ref_logits = model(ids).logits[0, -1].float()
        ref_gen = model.generate(ids, max_new_tokens=new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)

        fallbacks_before = set(torch_webgpu._C._fallback_ops())
        model = model.to("webgpu")
        dids = ids.to("webgpu")
        logits = model(dids).logits[0, -1].float().cpu()
        diff = (logits - ref_logits).abs().max().item()
        res["logits_max_abs_diff"] = diff
        res["logits_scale"] = ref_logits.abs().max().item()
        res["top1_match"] = logits.argmax().item() == ref_logits.argmax().item()

        model.generate(dids, max_new_tokens=2, do_sample=False, pad_token_id=tok.eos_token_id)  # warm up
        torch.webgpu.synchronize()
        t0 = time.perf_counter()
        gen = model.generate(dids, max_new_tokens=new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.webgpu.synchronize()
        dt = time.perf_counter() - t0
        gen = gen.cpu()
        res["generation_match"] = torch.equal(gen, ref_gen)
        if not res["generation_match"]:
            n = min(gen.shape[1], ref_gen.shape[1])
            first = (gen[0, :n] != ref_gen[0, :n]).nonzero()
            res["first_divergence"] = int(first[0]) - ids.shape[1] if len(first) else n
        res["tokens_per_s"] = round(new_tokens / dt, 2)
        res["fallbacks"] = sorted(set(torch_webgpu._C._fallback_ops()) - fallbacks_before)
        res["sample"] = tok.decode(gen[0, ids.shape[1]:])[:80]
    del model
    gc.collect()
    torch.webgpu.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    ok = True
    for m in args.models:
        try:
            r = check_model(m, args.tokens)
        except Exception as e:  # noqa: BLE001
            r = {"model": m, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"}
        good = "error" not in r and r["generation_match"] and not r["fallbacks"]
        ok &= good
        if args.json:
            print(json.dumps(r), flush=True)
        else:
            status = "OK  " if good else "FAIL"
            if "error" in r:
                print(f"{status} {m}: {r['error']}", flush=True)
            else:
                print(f"{status} {m} ({r['params_m']}M): logits diff {r['logits_max_abs_diff']:.2e} "
                      f"(scale {r['logits_scale']:.1f}), gen match {r['generation_match']}"
                      f"{'' if r['generation_match'] else ' (diverges at token %d)' % r['first_divergence']}, "
                      f"{r['tokens_per_s']} tok/s, fallbacks {r['fallbacks']}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
