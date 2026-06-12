#!/usr/bin/env python3
"""
Quick POPE accuracy evaluation on the fixed 60-sample validation set.

Runs Qwen2.5-VL-3B-Instruct with or without SRF and reports yes/no accuracy.
Use this instead of the full 3000-sample POPE eval to rapidly compare changes.

Usage (from /volumes2/mllm/lmms-eval/):
  # Baseline (no SRF)
  python srf/eval_pope_val.py

  # With SRF (current config)
  python srf/eval_pope_val.py --srf

  # Compare saved results
  python srf/eval_pope_val.py --compare results/pope_val_baseline.json results/pope_val_srf.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))         # srf/
_REPO = os.path.abspath(os.path.join(_HERE, ".."))         # lmms-eval/
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "srf"))
sys.path.insert(0, os.path.join(_REPO, "srf", "saliency"))
sys.path.insert(0, os.path.join(_REPO, "my_analysis"))

os.environ.setdefault("HF_HOME",        "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ── Constants ─────────────────────────────────────────────────────────────────
SPLITS  = ["adversarial", "popular", "random"]
ANSWERS = ["Yes", "No"]
MODEL_ID       = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_PIXELS     = 512 * 28 * 28
SYSTEM_PROMPT  = "You are a helpful assistant."
POST_PROMPT    = "\nAnswer with Yes or No only."


# ── POPE loader (same as eval_presence.py — fixed seed=42) ───────────────────
def load_balanced_pope(n_per_cell: int = 10, seed: int = 42) -> List[Dict]:
    from datasets import load_dataset

    print("Loading POPE dataset…")
    ds  = load_dataset("lmms-lab/POPE", split="test")
    rng = random.Random(seed)

    buckets: Dict[tuple, list] = {
        (sp, ans): [] for sp in SPLITS for ans in ANSWERS
    }
    for r in ds:
        sp  = str(r.get("category", "")).strip().lower()
        ans = str(r.get("answer",   "")).strip().capitalize()
        if sp in SPLITS and ans in ANSWERS:
            buckets[(sp, ans)].append(r)

    samples: List[Dict] = []
    for (sp, ans), rows in sorted(buckets.items()):
        rng.shuffle(rows)
        for r in rows[:n_per_cell]:
            samples.append({
                "image":    r["image"].convert("RGB"),
                "question": str(r["question"]).strip(),
                "gt":       ans,
                "split":    sp,
            })

    rng.shuffle(samples)
    counts = {
        (sp, ans): sum(1 for s in samples if s["split"] == sp and s["gt"] == ans)
        for sp in SPLITS for ans in ANSWERS
    }
    print(f"  Loaded {len(samples)} samples:")
    for (sp, ans), c in sorted(counts.items()):
        print(f"    {sp:12s} GT={ans}: {c}")
    return samples


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(use_srf: bool = False):
    import torch
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    # Import the right model class
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        ModelClass = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration
        ModelClass = Qwen2VLForConditionalGeneration

    # SRF patches torch.nn.functional.softmax — requires eager attention.
    # SDPA and flash_attention_2 bypass the softmax call entirely.
    attn_impl = "eager" if use_srf else "sdpa"
    print(f"Loading {MODEL_ID}  (attn={attn_impl})…")
    model = ModelClass.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation=attn_impl,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, max_pixels=MAX_PIXELS, min_pixels=256 * 28 * 28
    )
    print("  Model loaded.")
    return model, processor, process_vision_info


# ── Single-sample inference ───────────────────────────────────────────────────
def run_one(
    model,
    processor,
    process_vision_info,
    sample: Dict,
    use_srf: bool = False,
) -> str:
    import torch

    image    = sample["image"]
    question = sample["question"] + POST_PROMPT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": question},
        ]},
    ]

    text        = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inp, _ = process_vision_info(messages)
    inputs       = processor(
        text=[text], images=image_inp, padding=True, return_tensors="pt"
    ).to("cuda:0")

    if use_srf:
        import srf as srf_module
        import qwen_attn_patch as patch
        img_start, img_end = patch.get_image_token_range(inputs, processor=processor)
        srf_module.prepare_sample(
            inputs, img_start, img_end, image, question, model, processor
        )

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

    if use_srf:
        srf_module.cleanup()

    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def run_one_b4(
    model,
    processor,
    process_vision_info,
    sample: Dict,
    retry_threshold: float = 0.25,
    retry_multiplier: float = 2.0,
) -> str:
    """B4: confidence-gated two-pass correction.

    Run normal SRF generation. If first token is 'No' but CLIP is high-confidence
    (full_img_sim >= retry_threshold), re-run with retry_multiplier * alpha.
    Only costs 2× forward passes on ~5-10% of samples.
    """
    import torch
    import srf as srf_module
    import qwen_attn_patch as patch

    image    = sample["image"]
    question = sample["question"] + POST_PROMPT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": question},
        ]},
    ]
    text        = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inp, _ = process_vision_info(messages)
    inputs       = processor(text=[text], images=image_inp, padding=True, return_tensors="pt").to("cuda:0")

    img_start, img_end = patch.get_image_token_range(inputs, processor=processor)
    srf_module.prepare_sample(inputs, img_start, img_end, image, question, model, processor)

    # ── First pass: generate 1 token ─────────────────────────────────────────
    with torch.inference_mode():
        first_out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    first_tok = processor.batch_decode(
        first_out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip().lower()

    # ── Check retry condition ─────────────────────────────────────────────────
    clip_info  = srf_module.last_clip_result
    should_retry = (
        first_tok.startswith("no")
        and clip_info.get("object_present", False)
        and clip_info.get("full_img_sim", 0.0) >= retry_threshold
    )

    if should_retry:
        # Re-prepare with amplified alpha
        orig_value = patch._STATE["value"]
        srf_module.prepare_sample(inputs, img_start, img_end, image, question, model, processor)
        patch._STATE["value"] = patch._STATE["value"] * retry_multiplier
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        srf_module.cleanup()
    else:
        # Continue normal generation (re-prepare to reset state after 1-token pass)
        srf_module.prepare_sample(inputs, img_start, img_end, image, question, model, processor)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        srf_module.cleanup()

    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def parse_answer(raw: str) -> str:
    """Normalise model output to 'Yes' or 'No'."""
    r = raw.strip().lower()
    if r.startswith("yes"):
        return "Yes"
    if r.startswith("no"):
        return "No"
    # Fallback: look for yes/no anywhere
    if "yes" in r:
        return "Yes"
    if "no" in r:
        return "No"
    return raw.strip()


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(records: List[Dict]) -> Dict:
    total   = len(records)
    correct = sum(1 for r in records if r["correct"])
    acc     = correct / total

    yes_rec = [r for r in records if r["gt"] == "Yes"]
    no_rec  = [r for r in records if r["gt"] == "No"]
    tp = sum(1 for r in yes_rec if r["pred"] == "Yes")
    tn = sum(1 for r in no_rec  if r["pred"] == "No")
    fp = sum(1 for r in no_rec  if r["pred"] == "Yes")
    fn = sum(1 for r in yes_rec if r["pred"] == "No")

    tpr  = tp / (tp + fn + 1e-9)
    fpr  = fp / (fp + tn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    f1   = 2 * prec * tpr / (prec + tpr + 1e-9)
    yes_ratio = sum(1 for r in records if r["pred"] == "Yes") / total

    per_split = {}
    for sp in SPLITS:
        sp_recs = [r for r in records if r["split"] == sp]
        if sp_recs:
            per_split[sp] = {
                "acc": sum(1 for r in sp_recs if r["correct"]) / len(sp_recs),
                "n":   len(sp_recs),
            }

    return {
        "n": total, "correct": correct,
        "accuracy": acc, "tpr": tpr, "fpr": fpr,
        "precision": prec, "f1": f1,
        "yes_ratio": yes_ratio,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "per_split": per_split,
    }


def print_metrics(m: Dict, label: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {label}   (n={m['n']})")
    print(f"{'='*55}")
    print(f"  accuracy   : {m['accuracy']:.3f}  ({m['correct']}/{m['n']})")
    print(f"  TPR        : {m['tpr']:.3f}  (TP={m['tp']} FN={m['fn']})")
    print(f"  FPR        : {m['fpr']:.3f}  (FP={m['fp']} TN={m['tn']})")
    print(f"  F1         : {m['f1']:.3f}")
    print(f"  yes_ratio  : {m['yes_ratio']:.3f}  (model bias check)")
    print()
    print("  Per-split:")
    for sp, v in m["per_split"].items():
        print(f"    {sp:12s}  acc={v['acc']:.3f}  (n={v['n']})")
    print()


def print_comparison(path_a: str, path_b: str) -> None:
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    ma, mb = a["metrics"], b["metrics"]
    la, lb = a["label"],   b["label"]

    print(f"\n{'='*62}")
    print(f"  {la}  vs  {lb}")
    print(f"{'='*62}")
    print(f"  {'Metric':<14}  {la:>18}  {lb:>18}  {'Δ':>6}")
    print(f"  {'-'*60}")
    for key in ("accuracy", "tpr", "fpr", "f1", "yes_ratio"):
        delta = mb[key] - ma[key]
        arrow = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "─")
        print(f"  {key:<14}  {ma[key]:>18.3f}  {mb[key]:>18.3f}  {arrow}{delta:>+5.3f}")
    print()
    print("  Per-split accuracy:")
    for sp in SPLITS:
        va = ma["per_split"].get(sp, {}).get("acc", 0)
        vb = mb["per_split"].get(sp, {}).get("acc", 0)
        delta = vb - va
        arrow = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "─")
        print(f"    {sp:12s}  {va:.3f}  →  {vb:.3f}  {arrow}{delta:>+.3f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Quick POPE accuracy eval on 60-sample val set")
    parser.add_argument("--srf",        action="store_true", help="Enable SRF")
    parser.add_argument("--srf_mode",   default=None,
                        help="Override saliency mode (e.g. clip_full_gate, clip_full_gate_v2)")
    parser.add_argument("--alpha",            type=float, default=None,
                        help="Override boost alpha (e.g. 8.0)")
    parser.add_argument("--neg_absent_alpha", type=float, default=None,
                        help="Suppression alpha for absent objects (e.g. 1.0, 2.0, 4.0)")
    parser.add_argument("--bias_mode", default=None,
                        help="Override srf_bias_mode (e.g. budget_shift, prob_interp, additive_logit)")
    parser.add_argument("--interp_lambda", type=float, default=None,
                        help="Mixing weight for prob_interp mode (0=no-op, 1=full redistrib)")
    parser.add_argument("--vr_target", type=float, default=None,
                        help="B1: target image attention fraction (e.g. 0.15, 0.20)")
    parser.add_argument("--vr_k", type=float, default=None,
                        help="B1: deficit amplification factor (default 3.0)")
    parser.add_argument("--b4", action="store_true",
                        help="B4: confidence-gated two-pass retry")
    parser.add_argument("--b4_threshold", type=float, default=0.25,
                        help="B4: CLIP confidence threshold to trigger retry (default 0.25)")
    parser.add_argument("--b4_multiplier", type=float, default=2.0,
                        help="B4: alpha multiplier on retry (default 2.0)")
    parser.add_argument("--n_per_cell", type=int, default=10,
                        help="Samples per cell — keep at 10 for comparability (default 10)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out",        default=None,
                        help="Output JSON path (default: results/pope_val_{label}.json)")
    parser.add_argument("--compare",    nargs=2, metavar=("A_JSON", "B_JSON"),
                        help="Compare two saved result JSONs and exit")
    args = parser.parse_args()

    if args.compare:
        print_comparison(args.compare[0], args.compare[1])
        return

    label = "srf" if args.srf else "baseline"
    if args.srf and args.srf_mode:
        label = f"srf_{args.srf_mode}"
    if args.srf and args.alpha:
        label += f"_a{args.alpha}"
    if args.srf and args.neg_absent_alpha:
        label += f"_neg{args.neg_absent_alpha}"
    if args.srf and args.bias_mode:
        label += f"_{args.bias_mode}"
    if args.srf and args.vr_target:
        label += f"_vr{args.vr_target}"
    if args.b4:
        label += f"_B4t{args.b4_threshold}x{args.b4_multiplier}"

    out_path = args.out or os.path.join(
        _REPO, "results", f"pope_val_{label}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    samples = load_balanced_pope(args.n_per_cell, args.seed)

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor, process_vision_info = load_model(use_srf=args.srf or args.b4)

    # ── Setup SRF if requested ────────────────────────────────────────────────
    if args.srf or args.b4:
        import srf as srf_module
        print("Setting up SRF (calibrating heads)…")
        srf_module.setup(model, processor, calib_dataset="pope")
        overrides = {}
        if args.srf_mode:
            overrides["saliency_mode"] = args.srf_mode
        if args.alpha:
            overrides["alpha"] = args.alpha
        if args.neg_absent_alpha is not None:
            overrides["neg_absent_alpha"] = args.neg_absent_alpha
        if args.bias_mode is not None:
            overrides["bias_mode"] = args.bias_mode
        if args.interp_lambda is not None:
            overrides["interp_lambda"] = args.interp_lambda
        if args.vr_target is not None:
            overrides["vr_target"] = args.vr_target
        if args.vr_k is not None:
            overrides["vr_k"] = args.vr_k
        if overrides:
            srf_module.reset_for_dataset("pope", **overrides)
        print("  SRF ready.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\nRunning inference ({'SRF on' if args.srf else 'baseline'})…\n")
    records: List[Dict] = []

    for i, sample in enumerate(samples):
        if args.b4 and args.srf:
            raw = run_one_b4(model, processor, process_vision_info, sample,
                             retry_threshold=args.b4_threshold,
                             retry_multiplier=args.b4_multiplier)
        else:
            raw  = run_one(model, processor, process_vision_info, sample, use_srf=args.srf)
        pred = parse_answer(raw)
        gt   = sample["gt"]
        correct = pred == gt
        records.append({
            "idx":     i,
            "split":   sample["split"],
            "gt":      gt,
            "pred":    pred,
            "raw":     raw,
            "correct": correct,
            "question": sample["question"],
        })
        sym = "✓" if correct else "✗"
        print(f"  [{i+1:02d}/{len(samples)}] {sym} {sample['split'][:3]} GT={gt:<3}  pred={pred:<3}  raw='{raw}'")

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(records)
    print_metrics(metrics, label)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_data = {
        "label":      label,
        "srf":        args.srf,
        "srf_mode":   args.srf_mode,
        "n_per_cell": args.n_per_cell,
        "seed":       args.seed,
        "metrics":    metrics,
        "records":    records,
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
