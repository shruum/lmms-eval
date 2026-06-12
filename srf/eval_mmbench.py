#!/usr/bin/env python3
"""
MMBench evaluation — baseline vs SRF.

Runs Qwen2.5-VL-3B-Instruct on HuggingFaceM4/MMBench validation split (4329 samples).
Multiple-choice (A/B/C/D) format. Reports per-category accuracy.

Usage (from /volumes2/mllm/lmms-eval/):
  python srf/eval_mmbench.py
  python srf/eval_mmbench.py --srf
  python srf/eval_mmbench.py --compare results/mmbench_baseline.json results/mmbench_srf.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List

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


def load_mmbench() -> List[Dict]:
    from datasets import load_dataset
    print("Loading MMBench dataset…")
    ds = load_dataset("HuggingFaceM4/MMBench", split="validation")
    samples = []
    for r in ds:
        hint = str(r.get("hint", "")).strip()
        q    = str(r["question"]).strip()
        opts = {}
        for letter in ("A", "B", "C", "D"):
            v = str(r.get(letter, "")).strip()
            if v and v != "None":
                opts[letter] = v
        if len(opts) < 2:
            continue  # skip malformed
        opts_str = "\n".join(f"{l}. {v}" for l, v in opts.items())
        prompt   = (f"{hint}\n{q}" if hint and hint != "None" else q)
        prompt   = f"{prompt}\n{opts_str}\nAnswer with the option letter only."
        samples.append({
            "image":    r["image"].convert("RGB"),
            "prompt":   prompt,
            "question": q,
            "gt":       str(r["answer"]).strip().upper(),
            "category": str(r.get("category", "")).strip(),
            "l2cat":    str(r.get("l2-category", "")).strip(),
            "opts":     opts,
        })
    print(f"  Loaded {len(samples)} samples, {len({s['category'] for s in samples})} categories.")
    return samples


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
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation=attn_impl,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, max_pixels=MAX_PIXELS, min_pixels=256 * 28 * 28
    )
    print("  Model loaded.")
    return model, processor, process_vision_info


def run_one(model, processor, process_vision_info, sample: Dict, use_srf: bool = False) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text",  "text": sample["prompt"]},
        ]},
    ]
    text         = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inp, _ = process_vision_info(messages)
    inputs       = processor(text=[text], images=image_inp, padding=True, return_tensors="pt").to("cuda:0")

    if use_srf:
        import srf as srf_module
        import qwen_attn_patch as patch
        img_start, img_end = patch.get_image_token_range(inputs, processor=processor)
        srf_module.prepare_sample(
            inputs, img_start, img_end, sample["image"], sample["prompt"], model, processor
        )

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    if use_srf:
        srf_module.cleanup()

    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def parse_answer(raw: str, valid_opts: set) -> str:
    """Extract A/B/C/D from model output."""
    # First character if it's a valid letter
    r = raw.strip()
    if r and r[0].upper() in valid_opts:
        return r[0].upper()
    # Look for standalone letter: "A." or "(A)" or "Answer: A"
    m = re.search(r'\b([A-D])[.\):]?\s*$', r.upper())
    if m and m.group(1) in valid_opts:
        return m.group(1)
    m = re.search(r'\b([A-D])\b', r.upper())
    if m and m.group(1) in valid_opts:
        return m.group(1)
    return raw.strip()[:1].upper() if raw.strip() else "?"


def compute_metrics(records: List[Dict]) -> Dict:
    total   = len(records)
    correct = sum(1 for r in records if r["correct"])

    per_cat: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        per_cat[r["category"]]["total"]   += 1
        per_cat[r["category"]]["correct"] += int(r["correct"])

    cat_scores = {
        cat: {"acc": v["correct"] / v["total"], "correct": v["correct"], "total": v["total"]}
        for cat, v in per_cat.items()
    }
    return {
        "n": total, "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "per_category": dict(cat_scores),
    }


def print_metrics(m: Dict, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}   (n={m['n']})")
    print(f"{'='*60}")
    print(f"  Overall accuracy : {m['accuracy']:.3f}  ({m['correct']}/{m['n']})")
    print()
    print("  Per-category (top 10 by sample count):")
    cats_sorted = sorted(m["per_category"].items(), key=lambda x: -x[1]["total"])
    for cat, v in cats_sorted[:10]:
        print(f"    {cat:38s}  acc={v['acc']:.3f}  (n={v['total']})")
    print()


def print_comparison(path_a: str, path_b: str) -> None:
    with open(path_a) as f: a = json.load(f)
    with open(path_b) as f: b = json.load(f)
    ma, mb = a["metrics"], b["metrics"]
    la, lb = a["label"],   b["label"]

    print(f"\n{'='*65}")
    print(f"  {la}  vs  {lb}")
    print(f"{'='*65}")
    delta = mb["accuracy"] - ma["accuracy"]
    arrow = "▲" if delta > 0.002 else ("▼" if delta < -0.002 else "─")
    print(f"  accuracy  {ma['accuracy']:.3f}  →  {mb['accuracy']:.3f}  {arrow}{delta:>+.3f}")
    print()
    print("  Per-category accuracy (sorted by Δ):")
    all_cats = sorted(
        set(ma["per_category"]) | set(mb["per_category"]),
        key=lambda c: mb["per_category"].get(c, {}).get("acc", 0) -
                      ma["per_category"].get(c, {}).get("acc", 0),
        reverse=True,
    )
    for cat in all_cats:
        va = ma["per_category"].get(cat, {}).get("acc", 0)
        vb = mb["per_category"].get(cat, {}).get("acc", 0)
        n  = mb["per_category"].get(cat, {}).get("total", 0)
        d  = vb - va
        arrow = "▲" if d > 0.01 else ("▼" if d < -0.01 else "─")
        print(f"    {cat:38s}  {va:.3f} → {vb:.3f}  {arrow}{d:>+.3f}  (n={n})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="MMBench evaluation — baseline vs SRF")
    parser.add_argument("--srf",      action="store_true")
    parser.add_argument("--srf_mode", default=None)
    parser.add_argument("--alpha",    type=float, default=None)
    parser.add_argument("--out",      default=None)
    parser.add_argument("--compare",  nargs=2, metavar=("A_JSON", "B_JSON"))
    args = parser.parse_args()

    if args.compare:
        print_comparison(args.compare[0], args.compare[1])
        return

    label    = "srf" if args.srf else "baseline"
    out_path = args.out or os.path.join(_REPO, "results", f"mmbench_{label}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    samples = load_mmbench()
    model, processor, process_vision_info = load_model(use_srf=args.srf)

    if args.srf:
        import srf as srf_module
        print("Setting up SRF…")
        srf_module.setup(model, processor, calib_dataset="mmbench")
        overrides = {}
        if args.srf_mode: overrides["saliency_mode"] = args.srf_mode
        if args.alpha:    overrides["alpha"]          = args.alpha
        srf_module.reset_for_dataset("mmbench", **overrides)
        print("  SRF ready.")

    print(f"\nRunning inference ({'SRF on' if args.srf else 'baseline'})…\n")
    records: List[Dict] = []

    for i, sample in enumerate(samples):
        raw     = run_one(model, processor, process_vision_info, sample, use_srf=args.srf)
        pred    = parse_answer(raw, set(sample["opts"].keys()))
        correct = pred == sample["gt"]
        records.append({
            "idx": i, "category": sample["category"], "l2cat": sample["l2cat"],
            "gt": sample["gt"], "pred": pred, "raw": raw, "correct": correct,
        })
        sym = "✓" if correct else "✗"
        print(f"  [{i+1:04d}/{len(samples)}] {sym} {sample['category'][:20]:20s} "
              f"GT={sample['gt']}  pred={pred}  raw='{raw[:20]}'")

    metrics = compute_metrics(records)
    print_metrics(metrics, label)

    out_data = {"label": label, "srf": args.srf, "srf_mode": args.srf_mode,
                "metrics": metrics, "records": records}
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
