#!/usr/bin/env python3
"""
test_srfv_mmvp.py — Test SRF-V on MMVP.

SRF-V = SRF (attention logit boost, single pass) + value-projection scaling
of salient visual tokens. No contrastive second pass.

Value scaling: v_proj output for salient image tokens is multiplied by
(1 + alpha_v * salience) at prefill. Scaled V vectors are cached → every
subsequent generation step reads amplified values automatically.

Compares on MMVP (150 pairs, 300 images):
  baseline    — vanilla model, no intervention
  srf         — attention logit boost only (single pass)
  srfv_<α>    — SRF attention + V scaling at alpha_v ∈ {0.1, 0.3, 0.5, 0.8, 1.0}

Run:
    source activate mllm
    cd /volumes2/mllm/lmms-eval
    python srf/test_srfv_mmvp.py 2>&1 | tee /tmp/srfv_mmvp.log
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

_SRF_DIR      = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR / "saliency"))
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import os
os.environ.setdefault("HF_HOME", "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
from collections import defaultdict
from datasets import load_dataset as hf_load
from qwen_vl_utils import process_vision_info

import config as CFG
import qwen_attn_patch as patch
import srf
import srf_v
import eval as eval_mod


# ---------------------------------------------------------------------------
# Single-pass generation (SRF or SRF-V) — manual KV-cache loop
# ---------------------------------------------------------------------------

def generate_single_pass(model, inp: dict, method: str = "srf",
                         max_new_tokens: int = 16) -> list[int]:
    """
    Single forward-pass generation with method="srf" (SRF attention active)
    or method="baseline". V hooks fire automatically if srf_v is active.
    """
    device    = next(model.parameters()).device
    input_len = inp["input_ids"].shape[1]

    eos_ids: set[int] = set()
    eid = getattr(model.config, "eos_token_id", None)
    if isinstance(eid, int):
        eos_ids.add(eid)
    elif isinstance(eid, (list, tuple)):
        eos_ids.update(eid)

    generated: list[int] = []
    past_kv = None

    for step in range(max_new_tokens):
        seq_len = input_len + step
        if step == 0:
            kw = dict(inp, use_cache=True)
        else:
            new_tok   = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
            attn_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)
            kw = dict(input_ids=new_tok, attention_mask=attn_mask,
                      past_key_values=past_kv, use_cache=True)

        patch._STATE["method"] = method
        with torch.inference_mode():
            out = model(**kw)
        logits  = out.logits[:, -1, :].float()
        past_kv = out.past_key_values

        next_token = int(logits.argmax(dim=-1).item())
        generated.append(next_token)
        if next_token in eos_ids:
            break

    patch._STATE["method"] = "srf"
    return generated


# ---------------------------------------------------------------------------
# MMVP eval — baseline / SRF / SRF-V sweep
# ---------------------------------------------------------------------------

def run_mmvp(model, processor, device, args) -> None:
    import pandas as pd

    v_alphas = [0.1, 0.3, 0.5, 0.8, 1.0]
    configs  = ["baseline", "srf"] + [f"srfv_{a}" for a in v_alphas]

    print("\n" + "=" * 60)
    print("DATASET: MMVP (150 pairs, 300 images)")
    print(f"Configs: {configs}")
    print("=" * 60)

    srf.reset_for_dataset(dataset="mmvp", **{})

    df         = pd.read_csv(CFG.MMVP_CSV)
    img_ds     = hf_load("MMVP/MMVP", split="train")
    lex_sorted = sorted(range(1, 301), key=str)
    csv_to_hf  = {c: h for h, c in enumerate(lex_sorted)}

    img_token_id = processor.tokenizer.convert_tokens_to_ids(CFG.IMAGE_TOKEN)

    pair_results = {c: defaultdict(dict) for c in configs}
    n_img_ok     = {c: 0 for c in configs}
    _printed_shape = False

    for csv_1idx in range(1, 301):
        row_idx = csv_1idx - 1
        row     = df.iloc[row_idx]
        opts    = re.findall(r'\(([ab])\)\s*([^(]+)', str(row["Options"]), re.IGNORECASE)
        if not opts:
            continue
        opt_text = "\n".join(f"{m[0].upper()}. {m[1].strip()}" for m in opts)
        prompt   = (f"{row['Question']}\n{opt_text}\n"
                    "Answer with the option's letter from the given choices directly.")
        gt_raw   = str(row["Correct Answer"]).strip().strip("()").upper()
        gt       = CFG.MMVP_GT_CORRECTIONS.get(row_idx, gt_raw)
        image    = img_ds[csv_to_hf[csv_1idx]]["image"].convert("RGB")
        pair_id  = row_idx // 2
        img_key  = "a" if row_idx % 2 == 0 else "b"

        msgs   = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  prompt},
        ]}]
        text   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = process_vision_info(msgs)
        inp    = processor(text=[text], images=vis, return_tensors="pt",
                           padding=True).to(device)

        s, e = eval_mod.get_img_range(inp["input_ids"], img_token_id)

        if not _printed_shape and "pixel_values" in inp:
            print(f"  [sanity] pixel_values.shape = {tuple(inp['pixel_values'].shape)}")
            _printed_shape = True

        srf.prepare_sample(inp, s, e, image, row["Question"], model, processor)

        # ── baseline (no attention hook, no V hook) ────────────────────────
        srf_v.teardown_v()
        tok_ids = generate_single_pass(model, inp, method="baseline")
        raw     = processor.decode(tok_ids, skip_special_tokens=True)
        pred    = eval_mod._decode_ab(raw)
        ok      = (pred == gt)
        if ok:
            n_img_ok["baseline"] += 1
        pair_results["baseline"][pair_id][img_key] = ok

        # ── SRF (attention boost only, no V hook) ─────────────────────────
        srf_v.teardown_v()
        tok_ids = generate_single_pass(model, inp, method="srf")
        raw     = processor.decode(tok_ids, skip_special_tokens=True)
        pred    = eval_mod._decode_ab(raw)
        ok      = (pred == gt)
        if ok:
            n_img_ok["srf"] += 1
        pair_results["srf"][pair_id][img_key] = ok

        # ── SRF-V sweep (attention + V scaling, single pass) ──────────────
        for alpha in v_alphas:
            cfg = f"srfv_{alpha}"
            srf_v.setup_v(model, alpha_v=alpha)
            tok_ids = generate_single_pass(model, inp, method="srf")
            raw     = processor.decode(tok_ids, skip_special_tokens=True)
            pred    = eval_mod._decode_ab(raw)
            ok      = (pred == gt)
            if ok:
                n_img_ok[cfg] += 1
            pair_results[cfg][pair_id][img_key] = ok

        srf_v.teardown_v()
        srf.cleanup()

        if csv_1idx % 60 == 0:
            print(f"\n  [{csv_1idx:3d}/300]")
            for c in configs:
                pa = sum(1 for r in pair_results[c].values()
                         if "a" in r and "b" in r and r["a"] and r["b"])
                n  = len([r for r in pair_results[c].values() if "a" in r and "b" in r])
                print(f"    {c:<18}: img={n_img_ok[c]/csv_1idx:.3f}  "
                      f"pair={pa/max(n,1):.3f}")

    # ── Final results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL — MMVP")
    print("=" * 60)
    print(f"  {'Config':<18}  {'pair_acc':>8}  {'img_acc':>8}  {'Δ pair':>8}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*8}")

    base_pair = None
    for c in configs:
        pairs    = [(pid, r) for pid, r in pair_results[c].items()
                    if "a" in r and "b" in r]
        pair_ok  = sum(1 for _, r in pairs if r["a"] and r["b"])
        pair_acc = pair_ok / max(1, len(pairs))
        img_acc  = n_img_ok[c] / 300
        if base_pair is None:
            base_pair = pair_acc
        delta     = pair_acc - base_pair
        delta_str = f"{delta:+.4f}" if c != "baseline" else "    ref"
        print(f"  {c:<18}  {pair_acc:>8.4f}  {img_acc:>8.4f}  {delta_str:>8}  "
              f"({pair_ok}/{len(pairs)} pairs)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Test SRF-V (attention + value scaling) on MMVP")
    p.add_argument("--model", default=CFG.DEFAULT_MODEL)
    return p.parse_args()


def main():
    args = parse_args()

    model, processor = eval_mod.load_model(args.model)
    device = next(model.parameters()).device

    print(f"\n[SRF-V TEST]  model={args.model}")
    srf.setup(model, processor, calib_dataset="mmvp")

    run_mmvp(model, processor, device, args)


if __name__ == "__main__":
    main()
