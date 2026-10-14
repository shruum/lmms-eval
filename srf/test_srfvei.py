#!/usr/bin/env python3
"""
test_srfvei.py — Test SRF + VEI on MMVP and POPE adversarial.

SRF-VEI = SRF (attention logit boost) + Visual Evidence Injection (VEI):
  at each active layer's FFN input, inject a salience-weighted mean of
  image token hidden states into the answer token position.

SRF addresses the routing bottleneck (attention weights).
VEI addresses the signal bottleneck (answer token FFN input).

Tests both datasets in one model load:
  • MMVP (150 pairs, 300 images, ~10 min)
  • POPE adversarial (300 samples, ~3 min)

Configs compared:
  baseline      — no intervention
  srf           — SRF attention boost only
  srfvei_<γ>    — SRF + VEI at gamma ∈ {0.1, 0.3, 0.5, 1.0}

Layer range for VEI: fusion → last (8 to n_layers-1 = 35) by default.

Run:
    source activate mllm
    cd /volumes2/mllm/lmms-eval
    python srf/test_srfvei.py 2>&1 | tee /tmp/srfvei.log
"""
from __future__ import annotations

import argparse
import pathlib
import random
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
import srf_vei
import eval as eval_mod


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_single_pass(model, inp: dict, method: str = "srf",
                         max_new_tokens: int = 16) -> list[int]:
    """Manual KV-cache generation loop. VEI hooks fire automatically at prefill."""
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


def get_logits_single(model, inp: dict, method: str = "srf") -> torch.Tensor:
    """Single forward pass — returns last-position logits (for Yes/No POPE eval)."""
    patch._STATE["method"] = method
    with torch.inference_mode():
        out = model(**inp)
    patch._STATE["method"] = "srf"
    return out.logits[:, -1, :].float()


# ---------------------------------------------------------------------------
# MMVP evaluation
# ---------------------------------------------------------------------------

def run_mmvp(model, processor, device, configs, gammas,
             layer_start=None, layer_end=None) -> None:
    import pandas as pd

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
        s, e   = eval_mod.get_img_range(inp["input_ids"], img_token_id)

        srf.prepare_sample(inp, s, e, image, row["Question"], model, processor)

        # ── baseline ──────────────────────────────────────────────────────
        srf_vei.teardown_vei()
        tok_ids = generate_single_pass(model, inp, method="baseline")
        raw     = processor.decode(tok_ids, skip_special_tokens=True)
        pred    = eval_mod._decode_ab(raw)
        ok      = (pred == gt)
        if ok: n_img_ok["baseline"] += 1
        pair_results["baseline"][pair_id][img_key] = ok

        # ── SRF only ──────────────────────────────────────────────────────
        srf_vei.teardown_vei()
        tok_ids = generate_single_pass(model, inp, method="srf")
        raw     = processor.decode(tok_ids, skip_special_tokens=True)
        pred    = eval_mod._decode_ab(raw)
        ok      = (pred == gt)
        if ok: n_img_ok["srf"] += 1
        pair_results["srf"][pair_id][img_key] = ok

        # ── SRF + VEI sweep ───────────────────────────────────────────────
        for gamma in gammas:
            cfg = f"srfvei_{gamma}"
            srf_vei.setup_vei(model, gamma=gamma,
                              layer_start=layer_start, layer_end=layer_end)
            tok_ids = generate_single_pass(model, inp, method="srf")
            raw     = processor.decode(tok_ids, skip_special_tokens=True)
            pred    = eval_mod._decode_ab(raw)
            ok      = (pred == gt)
            if ok: n_img_ok[cfg] += 1
            pair_results[cfg][pair_id][img_key] = ok

        srf_vei.teardown_vei()
        srf.cleanup()

        if csv_1idx % 60 == 0:
            print(f"\n  [{csv_1idx:3d}/300]")
            for c in configs:
                pa = sum(1 for r in pair_results[c].values()
                         if "a" in r and "b" in r and r["a"] and r["b"])
                n  = len([r for r in pair_results[c].values() if "a" in r and "b" in r])
                print(f"    {c:<20}: img={n_img_ok[c]/csv_1idx:.3f}  pair={pa/max(n,1):.3f}")

    # ── Final ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL — MMVP")
    print("=" * 60)
    print(f"  {'Config':<20}  {'pair_acc':>8}  {'img_acc':>8}  {'Δ pair':>8}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}")
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
        print(f"  {c:<20}  {pair_acc:>8.4f}  {img_acc:>8.4f}  {delta_str:>8}  "
              f"({pair_ok}/{len(pairs)} pairs)")


# ---------------------------------------------------------------------------
# POPE adversarial evaluation
# ---------------------------------------------------------------------------

def _decode_yn(logits: torch.Tensor, processor) -> str:
    """Decode argmax token and return 'yes' or 'no'. Matches eval.py behaviour."""
    tok_id  = int(logits.argmax(-1).item())
    decoded = processor.decode([tok_id], skip_special_tokens=True).strip().lower()
    return "yes" if decoded.startswith("yes") else "no"


def run_pope_adv(model, processor, device, configs, gammas, n_samples: int = 300,
                 layer_start=None, layer_end=None, seed: int = 42) -> None:
    print("\n" + "=" * 60)
    print(f"DATASET: POPE adversarial ({n_samples} samples)")
    print(f"Configs: {configs}")
    print("=" * 60)

    srf.reset_for_dataset(dataset="pope")

    img_token_id = processor.tokenizer.convert_tokens_to_ids(CFG.IMAGE_TOKEN)

    ds   = hf_load("lmms-lab/POPE", split="test")
    rows = [r for r in ds
            if str(r.get("category", r.get("type", ""))).lower() == "adversarial"]
    rng  = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n_samples]
    print(f"  Loaded {len(rows)} POPE adversarial samples")

    n_correct = {c: 0 for c in configs}

    for i, r in enumerate(rows):
        image = r["image"].convert("RGB")
        q     = str(r["question"]).strip() + "\nAnswer with Yes or No only."
        gt    = "yes" if str(r.get("answer", "")).strip().lower() == "yes" else "no"

        msgs   = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  q},
        ]}]
        text   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = process_vision_info(msgs)
        inp    = processor(text=[text], images=vis, return_tensors="pt",
                           padding=True).to(device)
        s, e   = eval_mod.get_img_range(inp["input_ids"], img_token_id)

        srf.prepare_sample(inp, s, e, image, q, model, processor)

        # ── baseline ──────────────────────────────────────────────────────
        srf_vei.teardown_vei()
        logits = get_logits_single(model, inp, method="baseline")
        pred   = _decode_yn(logits[0], processor)
        if pred == gt: n_correct["baseline"] += 1

        # ── SRF only ──────────────────────────────────────────────────────
        srf_vei.teardown_vei()
        logits = get_logits_single(model, inp, method="srf")
        pred   = _decode_yn(logits[0], processor)
        if pred == gt: n_correct["srf"] += 1

        # ── SRF + VEI sweep ───────────────────────────────────────────────
        for gamma in gammas:
            cfg = f"srfvei_{gamma}"
            srf_vei.setup_vei(model, gamma=gamma,
                              layer_start=layer_start, layer_end=layer_end)
            logits = get_logits_single(model, inp, method="srf")
            pred   = _decode_yn(logits[0], processor)
            if pred == gt: n_correct[cfg] += 1

        srf_vei.teardown_vei()
        srf.cleanup()

        if (i + 1) % 100 == 0:
            print(f"  [{i+1:3d}/{len(rows)}]  " +
                  "  ".join(f"{c}:{n_correct[c]/(i+1):.3f}" for c in configs))

    # ── Final ──────────────────────────────────────────────────────────────
    n = len(rows)
    print("\n" + "=" * 60)
    print(f"FINAL — POPE adversarial ({n} samples)")
    print("=" * 60)
    print(f"  {'Config':<20}  {'acc':>8}  {'Δ acc':>8}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}")
    base_acc = None
    for c in configs:
        acc = n_correct[c] / n
        if base_acc is None:
            base_acc = acc
        delta     = acc - base_acc
        delta_str = f"{delta:+.4f}" if c != "baseline" else "    ref"
        print(f"  {c:<20}  {acc:>8.4f}  {delta_str:>8}  ({n_correct[c]}/{n})")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Test SRF+VEI on MMVP and POPE adversarial")
    p.add_argument("--model",    default=CFG.DEFAULT_MODEL)
    p.add_argument("--n_pope",   type=int, default=300,
                   help="POPE adversarial samples (default 300 for quick sweep)")
    p.add_argument("--gammas",   type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0],
                   help="VEI injection strengths to sweep")
    p.add_argument("--layer_start", type=int, default=None,
                   help="First layer for VEI (default: vaf_layer_start=8)")
    p.add_argument("--layer_end",   type=int, default=None,
                   help="Last layer for VEI (default: n_layers-1=35)")
    p.add_argument("--skip_mmvp",  action="store_true")
    p.add_argument("--skip_pope",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    gammas  = args.gammas
    configs = ["baseline", "srf"] + [f"srfvei_{g}" for g in gammas]

    model, processor = eval_mod.load_model(args.model)
    device = next(model.parameters()).device

    print(f"\n[SRF-VEI TEST]  model={args.model}")
    print(f"  gamma sweep: {gammas}")
    print(f"  layer_range: [{args.layer_start or 'vaf_start'}, {args.layer_end or 'n_layers-1'}]")
    srf.setup(model, processor, calib_dataset="mmvp")

    if not args.skip_mmvp:
        run_mmvp(model, processor, device, configs, gammas,
                 layer_start=args.layer_start, layer_end=args.layer_end)

    if not args.skip_pope:
        run_pope_adv(model, processor, device, configs, gammas,
                     n_samples=args.n_pope,
                     layer_start=args.layer_start, layer_end=args.layer_end)


if __name__ == "__main__":
    main()
