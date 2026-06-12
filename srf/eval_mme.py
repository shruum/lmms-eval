#!/usr/bin/env python3
"""
MME evaluation — baseline vs SRF.

Runs Qwen2.5-VL-3B-Instruct on the full lmms-lab/MME test set (2374 samples).
Reports per-category accuracy and the standard MME total score
(sum of per-category accuracy * 200, capped at 200 per category).

Usage (from /volumes2/mllm/lmms-eval/):
  # Baseline (no SRF)
  python srf/eval_mme.py

  # With SRF
  python srf/eval_mme.py --srf

  # Compare saved results
  python srf/eval_mme.py --compare results/mme_baseline.json results/mme_srf.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "srf"))
sys.path.insert(0, os.path.join(_REPO, "srf", "saliency"))
sys.path.insert(0, os.path.join(_REPO, "my_analysis"))

os.environ.setdefault("HF_HOME",        "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

MODEL_ID      = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_PIXELS    = 512 * 28 * 28
SYSTEM_PROMPT = "You are a helpful assistant."
POST_PROMPT   = "\nAnswer with Yes or No only."

# Standard MME perception categories (used for the 'perception score')
PERCEPTION_CATS = {
    "existence", "count", "position", "color",
    "posters", "celebrity", "scene", "landmark", "artwork", "OCR",
}
# Cognition categories
COGNITION_CATS = {
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning",
}


# ── Data loader ───────────────────────────────────────────────────────────────
def load_mme() -> List[Dict]:
    from datasets import load_dataset
    print("Loading MME dataset…")
    ds = load_dataset("lmms-lab/MME", split="test")
    samples = []
    for r in ds:
        ans = str(r.get("answer", "")).strip().capitalize()
        if ans not in ("Yes", "No"):
            continue
        samples.append({
            "image":    r["image"].convert("RGB"),
            "question": str(r["question"]).strip(),
            "gt":       ans,
            "category": str(r.get("category", "")).strip(),
        })
    print(f"  Loaded {len(samples)} samples across {len({s['category'] for s in samples})} categories.")
    return samples


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(use_srf: bool = False):
    import torch
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        ModelClass = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration
        ModelClass = Qwen2VLForConditionalGeneration

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
def run_one(model, processor, process_vision_info, sample: Dict, use_srf: bool = False) -> str:
    import torch

    question = sample["question"] + POST_PROMPT
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text",  "text": question},
        ]},
    ]
    text       = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inp, _ = process_vision_info(messages)
    inputs     = processor(text=[text], images=image_inp, padding=True, return_tensors="pt").to("cuda:0")

    if use_srf:
        import srf as srf_module
        import qwen_attn_patch as patch
        img_start, img_end = patch.get_image_token_range(inputs, processor=processor)
        srf_module.prepare_sample(inputs, img_start, img_end, sample["image"], question, model, processor)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    if use_srf:
        srf_module.cleanup()

    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def parse_answer(raw: str) -> str:
    r = raw.strip().lower()
    if r.startswith("yes"): return "Yes"
    if r.startswith("no"):  return "No"
    if "yes" in r: return "Yes"
    if "no"  in r: return "No"
    return raw.strip()


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(records: List[Dict]) -> Dict:
    total   = len(records)
    correct = sum(1 for r in records if r["correct"])

    per_cat: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        per_cat[r["category"]]["total"]   += 1
        per_cat[r["category"]]["correct"] += int(r["correct"])

    cat_scores = {}
    for cat, v in per_cat.items():
        acc = v["correct"] / v["total"] if v["total"] else 0.0
        cat_scores[cat] = {"acc": acc, "correct": v["correct"], "total": v["total"],
                           "mme_score": acc * 200}

    perception_score = sum(
        cat_scores[c]["mme_score"] for c in PERCEPTION_CATS if c in cat_scores
    )
    cognition_score  = sum(
        cat_scores[c]["mme_score"] for c in COGNITION_CATS  if c in cat_scores
    )

    return {
        "n": total, "correct": correct,
        "accuracy": correct / total,
        "perception_score": perception_score,
        "cognition_score":  cognition_score,
        "total_score":      perception_score + cognition_score,
        "per_category":     dict(cat_scores),
    }


def print_metrics(m: Dict, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}   (n={m['n']})")
    print(f"{'='*60}")
    print(f"  Overall accuracy  : {m['accuracy']:.3f}  ({m['correct']}/{m['n']})")
    print(f"  Perception score  : {m['perception_score']:.1f}")
    print(f"  Cognition  score  : {m['cognition_score']:.1f}")
    print(f"  Total MME  score  : {m['total_score']:.1f}")
    print()
    print("  Per-category:")
    for cat in sorted(m["per_category"]):
        v = m["per_category"][cat]
        tag = "(P)" if cat in PERCEPTION_CATS else "(C)"
        print(f"    {cat:25s} {tag}  acc={v['acc']:.3f}  score={v['mme_score']:.1f}  (n={v['total']})")
    print()


def print_comparison(path_a: str, path_b: str) -> None:
    with open(path_a) as f: a = json.load(f)
    with open(path_b) as f: b = json.load(f)
    ma, mb = a["metrics"], b["metrics"]
    la, lb = a["label"],   b["label"]

    print(f"\n{'='*65}")
    print(f"  {la}  vs  {lb}")
    print(f"{'='*65}")
    print(f"  {'Metric':<22}  {la:>16}  {lb:>16}  {'Δ':>7}")
    print(f"  {'-'*63}")
    for key, fmt in [("accuracy",":.3f"), ("perception_score",":.1f"),
                     ("cognition_score",":.1f"), ("total_score",":.1f")]:
        va, vb = ma[key], mb[key]
        delta  = vb - va
        arrow  = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        print(f"  {key:<22}  {va:>16{fmt[1:]}}  {vb:>16{fmt[1:]}}  {arrow}{delta:>+7.2f}")
    print()
    print("  Per-category accuracy:")
    all_cats = sorted(set(ma["per_category"]) | set(mb["per_category"]))
    for cat in all_cats:
        va = ma["per_category"].get(cat, {}).get("acc", 0)
        vb = mb["per_category"].get(cat, {}).get("acc", 0)
        delta = vb - va
        arrow = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "─")
        print(f"    {cat:25s}  {va:.3f}  →  {vb:.3f}  {arrow}{delta:>+.3f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="MME evaluation — baseline vs SRF")
    parser.add_argument("--srf",      action="store_true", help="Enable SRF")
    parser.add_argument("--srf_mode", default=None,        help="Override saliency mode")
    parser.add_argument("--alpha",    type=float, default=None)
    parser.add_argument("--out",      default=None)
    parser.add_argument("--compare",  nargs=2, metavar=("A_JSON", "B_JSON"))
    args = parser.parse_args()

    if args.compare:
        print_comparison(args.compare[0], args.compare[1])
        return

    label = "srf" if args.srf else "baseline"
    if args.srf and args.srf_mode:
        label = f"srf_{args.srf_mode}"

    out_path = args.out or os.path.join(_REPO, "results", f"mme_{label}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    samples = load_mme()
    model, processor, process_vision_info = load_model(use_srf=args.srf)

    if args.srf:
        import srf as srf_module
        print("Setting up SRF…")
        srf_module.setup(model, processor, calib_dataset="mme")
        overrides = {}
        if args.srf_mode: overrides["saliency_mode"] = args.srf_mode
        if args.alpha:    overrides["alpha"]          = args.alpha
        if overrides:
            srf_module.reset_for_dataset("mme", **overrides)
        else:
            srf_module.reset_for_dataset("mme")
        print("  SRF ready.")

    print(f"\nRunning inference ({'SRF on' if args.srf else 'baseline'})…\n")
    records: List[Dict] = []

    for i, sample in enumerate(samples):
        raw     = run_one(model, processor, process_vision_info, sample, use_srf=args.srf)
        pred    = parse_answer(raw)
        correct = pred == sample["gt"]
        records.append({
            "idx": i, "category": sample["category"],
            "gt": sample["gt"], "pred": pred, "raw": raw, "correct": correct,
            "question": sample["question"],
        })
        sym = "✓" if correct else "✗"
        print(f"  [{i+1:04d}/{len(samples)}] {sym} {sample['category'][:15]:15s} "
              f"GT={sample['gt']:<3}  pred={pred:<3}  raw='{raw}'")

    metrics = compute_metrics(records)
    print_metrics(metrics, label)

    out_data = {"label": label, "srf": args.srf, "srf_mode": args.srf_mode,
                "metrics": metrics, "records": records}
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
