#!/usr/bin/env python3
"""
HallusionBench evaluation — baseline vs SRF.

Runs Qwen2.5-VL-3B-Instruct on lmms-lab/HallusionBench (image split, 951 samples).
Reports the three standard HallusionBench metrics:
  aAcc  — all accuracy  (each question correct independently)
  fAcc  — figure accuracy (all questions on same figure correct)
  qAcc  — question-pair accuracy (both questions in a pair correct)

Usage (from /volumes2/mllm/lmms-eval/):
  # Baseline
  python srf/eval_hallusionbench.py

  # With SRF
  python srf/eval_hallusionbench.py --srf

  # Compare
  python srf/eval_hallusionbench.py --compare results/hallusionbench_baseline.json results/hallusionbench_srf.json
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


# ── Data loader ───────────────────────────────────────────────────────────────
def load_hallusionbench() -> List[Dict]:
    from datasets import load_dataset
    print("Loading HallusionBench dataset…")
    ds = load_dataset("lmms-lab/HallusionBench", split="image")

    samples = []
    for r in ds:
        gt_raw = str(r.get("gt_answer", "")).strip()
        # gt_answer: "1" = Yes, "0" = No
        if gt_raw == "1":
            gt = "Yes"
        elif gt_raw == "0":
            gt = "No"
        else:
            continue  # skip samples with missing/unclear GT

        img = r.get("image")
        if img is None:
            continue

        samples.append({
            "image":      img.convert("RGB"),
            "question":   str(r["question"]).strip(),
            "gt":         gt,
            "gt_raw":     gt_raw,
            "category":   str(r.get("category", "")).strip(),
            "subcategory": str(r.get("subcategory", "")).strip(),
            "set_id":     str(r.get("set_id", "")),
            "figure_id":  str(r.get("figure_id", "")),
            "question_id": str(r.get("question_id", "")),
        })

    print(f"  Loaded {len(samples)} samples.")
    cats = {}
    for s in samples:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    for cat, n in sorted(cats.items()):
        print(f"    {cat}: {n}")
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
    text         = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inp, _ = process_vision_info(messages)
    inputs       = processor(text=[text], images=image_inp, padding=True, return_tensors="pt").to("cuda:0")

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
    """
    Standard HallusionBench metrics:
      aAcc — per-question accuracy (simple)
      fAcc — figure accuracy: all questions on the same (set_id, figure_id) are correct
      qAcc — question-pair accuracy: questions with same (set_id, question_id) across figures
    """
    total   = len(records)
    correct = sum(1 for r in records if r["correct"])
    aAcc    = correct / total if total else 0.0

    # fAcc: group by (set_id, figure_id)
    figures: Dict[tuple, List[bool]] = defaultdict(list)
    for r in records:
        key = (r["set_id"], r["figure_id"])
        figures[key].append(r["correct"])
    fig_correct = sum(1 for v in figures.values() if all(v))
    fAcc        = fig_correct / len(figures) if figures else 0.0

    # qAcc: group by (set_id, question_id) — same question asked across multiple figures
    qpairs: Dict[tuple, List[bool]] = defaultdict(list)
    for r in records:
        key = (r["set_id"], r["question_id"])
        qpairs[key].append(r["correct"])
    qp_correct = sum(1 for v in qpairs.values() if all(v))
    qAcc       = qp_correct / len(qpairs) if qpairs else 0.0

    # Per-category
    per_cat: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        per_cat[r["category"]]["total"]   += 1
        per_cat[r["category"]]["correct"] += int(r["correct"])
    cat_scores = {
        cat: {"acc": v["correct"] / v["total"], "correct": v["correct"], "total": v["total"]}
        for cat, v in per_cat.items()
    }

    yes_ratio = sum(1 for r in records if r["pred"] == "Yes") / total

    return {
        "n": total, "correct": correct,
        "aAcc": aAcc, "fAcc": fAcc, "qAcc": qAcc,
        "yes_ratio": yes_ratio,
        "n_figures": len(figures), "figures_correct": fig_correct,
        "n_qpairs":  len(qpairs),  "qpairs_correct":  qp_correct,
        "per_category": dict(cat_scores),
    }


def print_metrics(m: Dict, label: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {label}   (n={m['n']})")
    print(f"{'='*55}")
    print(f"  aAcc (question)  : {m['aAcc']:.3f}  ({m['correct']}/{m['n']})")
    print(f"  fAcc (figure)    : {m['fAcc']:.3f}  ({m['figures_correct']}/{m['n_figures']})")
    print(f"  qAcc (pair)      : {m['qAcc']:.3f}  ({m['qpairs_correct']}/{m['n_qpairs']})")
    print(f"  yes_ratio        : {m['yes_ratio']:.3f}")
    print()
    print("  Per-category:")
    for cat, v in sorted(m["per_category"].items()):
        print(f"    {cat:6s}  acc={v['acc']:.3f}  (n={v['total']})")
    print()


def print_comparison(path_a: str, path_b: str) -> None:
    with open(path_a) as f: a = json.load(f)
    with open(path_b) as f: b = json.load(f)
    ma, mb = a["metrics"], b["metrics"]
    la, lb = a["label"],   b["label"]

    print(f"\n{'='*62}")
    print(f"  {la}  vs  {lb}")
    print(f"{'='*62}")
    print(f"  {'Metric':<14}  {la:>18}  {lb:>18}  {'Δ':>6}")
    print(f"  {'-'*60}")
    for key in ("aAcc", "fAcc", "qAcc", "yes_ratio"):
        va, vb = ma[key], mb[key]
        delta  = vb - va
        arrow  = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "─")
        print(f"  {key:<14}  {va:>18.3f}  {vb:>18.3f}  {arrow}{delta:>+5.3f}")
    print()
    print("  Per-category aAcc:")
    all_cats = sorted(set(ma["per_category"]) | set(mb["per_category"]))
    for cat in all_cats:
        va = ma["per_category"].get(cat, {}).get("acc", 0)
        vb = mb["per_category"].get(cat, {}).get("acc", 0)
        delta = vb - va
        arrow = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "─")
        print(f"    {cat:6s}  {va:.3f}  →  {vb:.3f}  {arrow}{delta:>+.3f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="HallusionBench evaluation — baseline vs SRF")
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

    out_path = args.out or os.path.join(_REPO, "results", f"hallusionbench_{label}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    samples = load_hallusionbench()
    model, processor, process_vision_info = load_model(use_srf=args.srf)

    if args.srf:
        import srf as srf_module
        print("Setting up SRF…")
        # HallusionBench uses POPE-style Yes/No questions → calibrate on pope
        srf_module.setup(model, processor, calib_dataset="pope")
        overrides = {}
        if args.srf_mode: overrides["saliency_mode"] = args.srf_mode
        if args.alpha:    overrides["alpha"]          = args.alpha
        # Use mme params (phase=both, alpha=4.0) — closest to HallusionBench
        srf_module.reset_for_dataset("mme", **overrides)
        print("  SRF ready.")

    print(f"\nRunning inference ({'SRF on' if args.srf else 'baseline'})…\n")
    records: List[Dict] = []

    for i, sample in enumerate(samples):
        raw     = run_one(model, processor, process_vision_info, sample, use_srf=args.srf)
        pred    = parse_answer(raw)
        correct = pred == sample["gt"]
        records.append({
            "idx": i, "category": sample["category"],
            "subcategory": sample["subcategory"],
            "set_id": sample["set_id"], "figure_id": sample["figure_id"],
            "question_id": sample["question_id"],
            "gt": sample["gt"], "pred": pred, "raw": raw, "correct": correct,
            "question": sample["question"],
        })
        sym = "✓" if correct else "✗"
        print(f"  [{i+1:04d}/{len(samples)}] {sym} {sample['category']:2s}/{sample['subcategory'][:12]:12s} "
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
