#!/usr/bin/env python3
"""
Saliency presence-detection evaluation on a fixed POPE validation set.

Usage (from /volumes2/mllm/lmms-eval/):
  # Establish baseline
  python srf/saliency/eval_presence.py --mode clip_full_gate --n_per_cell 10 --save_baseline

  # Test a different mode
  python srf/saliency/eval_presence.py --mode clip_improved --n_per_cell 10

  # Compare two saved result files
  python srf/saliency/eval_presence.py \\
      --compare results/saliency_val_baseline.json results/saliency_val_clip_improved/saliency_val_clip_improved.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional

# ── Path setup (mirrors pope_saliency_testset.py) ────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))   # srf/saliency/
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))  # lmms-eval/
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "srf", "saliency"))
sys.path.insert(0, os.path.join(_REPO, "srf"))

os.environ.setdefault("HF_HOME", "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from clip_salience import (
    ClipSalienceResult,
    compute_clip_salience,
    compute_clip_salience_multiscale,
    compute_clip_salience_multiscale_full_gate,
    compute_clip_salience_full_gate_v2,
    compute_clip_salience_full_gate_v3,
    extract_query_noun,
)

# ── Constants ─────────────────────────────────────────────────────────────────
SPLITS = ["adversarial", "popular", "random"]
ANSWERS = ["Yes", "No"]
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_GRID = 7


# ── POPE loader (self-contained; mirrors load_balanced_pope in pope_saliency_testset.py) ──
def load_balanced_pope(
    n_per_cell: int,
    seed: int,
    splits: Optional[List[str]] = None,
) -> List[Dict]:
    from datasets import load_dataset

    active_splits = splits or SPLITS
    print("Loading POPE dataset…")
    ds = load_dataset("lmms-lab/POPE", split="test")
    rng = random.Random(seed)

    buckets: Dict[tuple, list] = {
        (sp, ans): [] for sp in active_splits for ans in ANSWERS
    }
    for r in ds:
        sp  = str(r.get("category", "")).strip().lower()
        ans = str(r.get("answer",   "")).strip().capitalize()
        if sp in active_splits and ans in ANSWERS:
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


# ── Saliency dispatch ─────────────────────────────────────────────────────────
def run_saliency(
    sample: Dict,
    mode: str,
    clip_model: str,
    grid_n: int,
) -> ClipSalienceResult:
    image    = sample["image"]
    question = sample["question"]

    if mode == "clip":
        return compute_clip_salience(
            image, question, grid_n, grid_n,
            top_k_pct=0.30, clip_model_name=clip_model,
        )
    elif mode == "clip_improved":
        return compute_clip_salience_multiscale(
            image, question, grid_n, grid_n,
            top_k_pct=0.30, coarse_scales=(3, 5, 7),
            clip_model_name=clip_model,
        )
    elif mode == "clip_full_gate":
        return compute_clip_salience_multiscale_full_gate(
            image, question, grid_n, grid_n,
            top_k_pct=0.30, coarse_scales=(3, 5, 7),
            clip_model_name=clip_model,
        )
    elif mode == "clip_full_gate_v2":
        return compute_clip_salience_full_gate_v2(
            image, question, grid_n, grid_n,
            top_k_pct=0.30, coarse_scales=(3, 5, 7),
            clip_model_name=clip_model,
        )
    elif mode in ("clip_full_gate_v3", "clip_full_gate_v3_iou",
                   "clip_full_gate_v3_blur", "clip_full_gate_v3_entropy",
                   "clip_full_gate_v3_vitl"):
        backup = {"clip_full_gate_v3_iou":     "cross_scale",
                  "clip_full_gate_v3_blur":    "blur_delta",
                  "clip_full_gate_v3_entropy": "raw_entropy"}.get(mode, "none")
        vitl_model = "openai/clip-vit-large-patch14"
        model_name = vitl_model if mode == "clip_full_gate_v3_vitl" else clip_model
        # ViT-L/14 full_img_sim is higher than ViT-B/32; start at 0.26 then sweep.
        thresh = 0.26 if mode == "clip_full_gate_v3_vitl" else None
        return compute_clip_salience_full_gate_v3(
            image, question, grid_n, grid_n,
            top_k_pct=0.30, coarse_scales=(3, 5, 7),
            clip_model_name=model_name,
            full_img_thresh=thresh,
            backup=backup,
        )
    else:
        raise ValueError(
            f"Unknown saliency mode: {mode!r}. "
            "Choose: clip, clip_improved, clip_full_gate, clip_full_gate_v2, "
            "clip_full_gate_v3, clip_full_gate_v3_iou, clip_full_gate_v3_blur, "
            "clip_full_gate_v3_entropy, clip_full_gate_v3_vitl"
        )


# ── Spatial quality metrics ───────────────────────────────────────────────────
def spatial_metrics(sal: torch.Tensor) -> Dict[str, float]:
    sal_f = sal.float()
    n = sal_f.numel()

    # Entropy of sharpened softmax distribution (0 = spike, 1 = uniform)
    p       = torch.softmax(sal_f * 20.0, dim=0)
    entropy = float(-(p * (p + 1e-9).log()).sum() / math.log(n))

    # Peak-to-mean ratio
    peak_to_mean = float(sal_f.max()) / (float(sal_f.mean()) + 1e-9)

    # Contrast: top-30% mean / all-patches mean
    top_k    = max(1, round(n * 0.30))
    topk_mean = float(sal_f.topk(top_k).values.mean())
    all_mean  = float(sal_f.mean()) + 1e-9
    contrast  = topk_mean / all_mean

    return {
        "entropy":      entropy,
        "peak_to_mean": peak_to_mean,
        "contrast":     contrast,
    }


# ── Visualization helpers ─────────────────────────────────────────────────────
def saliency_overlay(
    sal: np.ndarray,
    grid_h: int,
    grid_w: int,
    image: Image.Image,
    alpha: float = 0.55,
) -> np.ndarray:
    sal_2d  = sal.reshape(grid_h, grid_w)
    sal_2d  = (sal_2d - sal_2d.min()) / (sal_2d.max() - sal_2d.min() + 1e-8)
    sal_img = Image.fromarray((sal_2d * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR
    )
    sal_arr = np.array(sal_img) / 255.0
    heat    = plt.colormaps["jet"](sal_arr)[..., :3]
    img_arr = np.array(image) / 255.0
    overlay = (1 - alpha) * img_arr + alpha * heat
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def save_sample_figure(
    sample: Dict,
    result: ClipSalienceResult,
    sm: Dict[str, float],
    grid_n: int,
    out_path: str,
) -> None:
    image    = sample["image"]
    gt       = sample["gt"]
    split    = sample["split"]
    noun     = extract_query_noun(sample["question"])
    correct  = result.object_present == (gt == "Yes")
    verdict  = "CORRECT ✓" if correct else "WRONG ✗"
    pred_str = "PRESENT" if result.object_present else "ABSENT"

    overlay = saliency_overlay(result.saliency.numpy(), grid_n, grid_n, image)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                              gridspec_kw={"width_ratios": [3, 3, 2]})
    fig.patch.set_facecolor("white")

    # Panel 0: original image
    axes[0].imshow(image)
    axes[0].axis("off")
    gt_color = "#1e8449" if gt == "Yes" else "#922b21"
    q_short  = (sample["question"][:60] + "…") if len(sample["question"]) > 60 else sample["question"]
    axes[0].set_title(f"[{split}]  GT: {gt}\n{q_short}", fontsize=8.5, color=gt_color, pad=4)

    # Panel 1: saliency heatmap
    axes[1].imshow(overlay)
    axes[1].axis("off")
    verdict_color = "#1a5276" if correct else "#922b21"
    axes[1].set_title(f"noun: '{noun}'\npred = {pred_str}  [{verdict}]",
                      fontsize=9, color=verdict_color, pad=4)

    # Panel 2: metrics text panel
    axes[2].axis("off")

    def tick(v: bool) -> str:
        return "✓" if v else "✗"

    has_full = bool(result.full_img_sim)
    has_v2   = bool(result.contrastive_gap) or result.patch_entropy > 0
    has_v3   = result.blur_delta != 0.0 or result.cross_scale_iou != 0.0
    lines = [
        ("── Presence signals ──────", "#333333", True),
        (f"  full_img   {result.full_img_sim:.3f}  {tick(result.gate_full)}", "#1a5276" if result.gate_full else "#922b21", has_full),
        (f"  patch_max  {result.max_sim:.3f}  {tick(result.gate_patch)}", "#555555", True),
        (f"  contrast   {result.patch_contrast:.2f}x  {tick(result.gate_contrast)}", "#1a5276" if result.gate_contrast else "#922b21", has_full),
        (f"  ctrstv_gap {result.contrastive_gap:+.4f}  {tick(result.gate_contrastive)}", "#1a5276" if result.gate_contrastive else "#922b21", has_v2),
        (f"  patch_ent  {result.patch_entropy:.3f}  {tick(result.gate_entropy)}", "#1a5276" if result.gate_entropy else "#922b21", has_v2),
        ("── v3 signals ────────────", "#444444", has_v3),
        (f"  raw_ent    {result.raw_entropy:.3f}  {tick(result.gate_raw_entropy)}", "#1a5276" if result.gate_raw_entropy else "#922b21", has_v3),
        (f"  xscale_iou {result.cross_scale_iou:.3f}  {tick(result.gate_cross_scale)}", "#1a5276" if result.gate_cross_scale else "#922b21", has_v3),
        (f"  blur_delta {result.blur_delta:+.4f}  {tick(result.gate_blur_delta)}", "#1a5276" if result.gate_blur_delta else "#922b21", has_v3),
        ("", "#333333", True),
        ("── Spatial quality ───────", "#333333", True),
        (f"  entropy    {sm['entropy']:.3f}  (↓=conc.)", "#555555", True),
        (f"  peak/mean  {sm['peak_to_mean']:.3f}", "#555555", True),
        (f"  contrast   {sm['contrast']:.3f}", "#555555", True),
        ("", "#333333", True),
        ("── Decision ──────────────", "#333333", True),
        (f"  GT         {gt}", gt_color, True),
        (f"  pred       {pred_str}", verdict_color, True),
        (f"  {verdict}", verdict_color, True),
    ]

    y = 0.97
    for text, color, show in lines:
        if not show:
            continue
        axes[2].text(0.05, y, text, transform=axes[2].transAxes,
                     fontsize=8, color=color, verticalalignment="top",
                     fontfamily="monospace")
        y -= 0.072

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Aggregate metrics ─────────────────────────────────────────────────────────
def compute_metrics(records: List[Dict]) -> Dict[str, Any]:
    yes_rec = [r for r in records if r["gt"] == "Yes"]
    no_rec  = [r for r in records if r["gt"] == "No"]

    tp = sum(1 for r in yes_rec if     r["object_present"])
    fn = sum(1 for r in yes_rec if not r["object_present"])
    fp = sum(1 for r in no_rec  if     r["object_present"])
    tn = sum(1 for r in no_rec  if not r["object_present"])

    tpr  = tp / (tp + fn + 1e-9)
    fpr  = fp / (fp + tn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    f1   = 2 * prec * tpr / (prec + tpr + 1e-9)
    acc  = (tp + tn) / len(records)

    # Per-split
    per_split: Dict[str, Dict] = {}
    for sp in SPLITS:
        sp_recs = [r for r in records if r["split"] == sp]
        if not sp_recs:
            continue
        sp_yes = [r for r in sp_recs if r["gt"] == "Yes"]
        sp_no  = [r for r in sp_recs if r["gt"] == "No"]
        sp_tp  = sum(1 for r in sp_yes if     r["object_present"])
        sp_tn  = sum(1 for r in sp_no  if not r["object_present"])
        sp_fp  = sum(1 for r in sp_no  if     r["object_present"])
        per_split[sp] = {
            "acc": (sp_tp + sp_tn) / len(sp_recs),
            "tpr": sp_tp / (len(sp_yes) + 1e-9),
            "fpr": sp_fp / (len(sp_no)  + 1e-9),
            "n":   len(sp_recs),
        }

    # Per-gate individual accuracy (full_gate mode)
    gate_acc: Dict[str, float] = {}
    for g in ("gate_full", "gate_patch", "gate_contrast", "gate_contrastive", "gate_entropy",
              "gate_raw_entropy", "gate_cross_scale", "gate_blur_delta"):
        vals = [r for r in records if g in r and r[g] is not None]
        if vals:
            correct_g = sum(1 for r in vals if bool(r[g]) == (r["gt"] == "Yes"))
            gate_acc[g] = correct_g / len(vals)

    # Spatial metrics averaged per GT group
    def mean_m(recs: List[Dict], key: str) -> float:
        vals = [r[key] for r in recs if key in r]
        return float(np.mean(vals)) if vals else 0.0

    spatial: Dict[str, Any] = {}
    for key in ("entropy", "peak_to_mean", "contrast",
                "raw_entropy", "cross_scale_iou", "blur_delta", "full_img_sim", "max_sim"):
        spatial[key] = {
            "yes": mean_m(yes_rec, key),
            "no":  mean_m(no_rec,  key),
        }

    return {
        "n": len(records),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy":  acc,
        "tpr":       tpr,
        "fpr":       fpr,
        "precision": prec,
        "f1":        f1,
        "per_split":    per_split,
        "gate_accuracy": gate_acc,
        "spatial":       spatial,
    }


def print_metrics(m: Dict, mode: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Mode: {mode}   (n={m['n']})")
    print(f"{'='*60}")
    print(f"  presence_acc : {m['accuracy']:.3f}  "
          f"({m['tp']+m['tn']}/{m['n']}  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']})")
    print(f"  TPR          : {m['tpr']:.3f}")
    print(f"  FPR          : {m['fpr']:.3f}")
    print(f"  precision    : {m['precision']:.3f}")
    print(f"  F1           : {m['f1']:.3f}")
    print()
    print("  Per-split:")
    for sp, v in m["per_split"].items():
        print(f"    {sp:12s}  acc={v['acc']:.3f}  TPR={v['tpr']:.3f}  FPR={v['fpr']:.3f}  (n={v['n']})")
    if m["gate_accuracy"]:
        print()
        print("  Per-gate accuracy (signal alone vs GT):")
        for g, a in m["gate_accuracy"].items():
            print(f"    {g:16s}  {a:.3f}")
    print()
    print("  Spatial metrics by GT group (↑ = more concentrated):")
    for key, v in m["spatial"].items():
        bar_yes = "█" * int(v["yes"] * 20)
        bar_no  = "█" * int(v["no"]  * 20)
        print(f"    {key:15s}  GT-Yes={v['yes']:.3f} {bar_yes}")
        print(f"    {'':15s}  GT-No ={v['no']:.3f} {bar_no}  Δ={v['yes']-v['no']:+.3f}")
    print()


def print_comparison(base_path: str, new_path: str) -> None:
    with open(base_path) as f:
        base = json.load(f)
    with open(new_path) as f:
        new = json.load(f)

    bm, nm = base["metrics"], new["metrics"]
    bn, nn = base["mode"],    new["mode"]

    print(f"\n{'='*68}")
    print(f"  Comparison: {bn}  vs  {nn}")
    print(f"{'='*68}")
    print(f"  {'Metric':<18}  {bn:>20}  {nn:>20}  {'Δ':>6}")
    print(f"  {'-'*66}")
    for key in ("accuracy", "tpr", "fpr", "precision", "f1"):
        delta = nm[key] - bm[key]
        arrow = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "─")
        print(f"  {key:<18}  {bm[key]:>20.3f}  {nm[key]:>20.3f}  {arrow}{delta:>+5.3f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Saliency presence-detection eval on POPE val set")
    parser.add_argument("--mode", default="clip_full_gate",
                        choices=["clip", "clip_improved", "clip_full_gate",
                                 "clip_full_gate_v2", "clip_full_gate_v3",
                                 "clip_full_gate_v3_iou", "clip_full_gate_v3_blur",
                                 "clip_full_gate_v3_entropy", "clip_full_gate_v3_vitl"],
                        help="Saliency mode to evaluate (default: clip_full_gate)")
    parser.add_argument("--n_per_cell", type=int, default=10,
                        help="Samples per (split × answer) cell → 6 cells total (default 10 → 60 samples)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for balanced sampling (default 42)")
    parser.add_argument("--clip_model", default=DEFAULT_CLIP_MODEL,
                        help="CLIP model name (default: openai/clip-vit-base-patch32)")
    parser.add_argument("--grid_n", type=int, default=DEFAULT_GRID,
                        help="Patch grid size (default 7)")
    parser.add_argument("--n_vis", type=int, default=12,
                        help="Number of sample figures to save (default 12, 0 to skip)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: results/saliency_val_{mode}/)")
    parser.add_argument("--save_baseline", action="store_true",
                        help="Also copy results to results/saliency_val_baseline.json")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE_JSON", "NEW_JSON"),
                        help="Compare two saved result JSONs and exit")
    args = parser.parse_args()

    # ── Compare mode (no data loading needed) ──────────────────────────────────
    if args.compare:
        print_comparison(args.compare[0], args.compare[1])
        return

    # ── Output directory ───────────────────────────────────────────────────────
    out_dir = args.out_dir or os.path.join(
        _REPO, "results", f"saliency_val_{args.mode}"
    )
    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # ── Load balanced POPE set ─────────────────────────────────────────────────
    samples = load_balanced_pope(args.n_per_cell, args.seed)
    print(f"\nRunning saliency mode: {args.mode}  ({args.clip_model})\n")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    records:        List[Dict]              = []
    result_cache:   List[ClipSalienceResult] = []

    for i, sample in enumerate(samples):
        result = run_saliency(sample, args.mode, args.clip_model, args.grid_n)
        sm     = spatial_metrics(result.saliency)
        correct = result.object_present == (sample["gt"] == "Yes")

        records.append({
            "idx":             i,
            "split":           sample["split"],
            "gt":              sample["gt"],
            "question":        sample["question"],
            "object_present":  result.object_present,
            "max_sim":         result.max_sim,
            "full_img_sim":    result.full_img_sim,
            "patch_contrast":  result.patch_contrast,
            "gate_full":       result.gate_full,
            "gate_patch":      result.gate_patch,
            "gate_contrast":   result.gate_contrast,
            "contrastive_gap": result.contrastive_gap,
            "patch_entropy":   result.patch_entropy,
            "gate_contrastive": result.gate_contrastive,
            "gate_entropy":    result.gate_entropy,
            # v3 signals
            "raw_entropy":     result.raw_entropy,
            "cross_scale_iou": result.cross_scale_iou,
            "blur_delta":      result.blur_delta,
            "gate_raw_entropy": result.gate_raw_entropy,
            "gate_cross_scale": result.gate_cross_scale,
            "gate_blur_delta":  result.gate_blur_delta,
            **sm,
        })
        result_cache.append(result)

        sym  = "✓" if correct else "✗"
        noun = extract_query_noun(sample["question"])
        print(
            f"  [{i+1:02d}/{len(samples)}] {sym} {sample['split'][:3]} GT={sample['gt']:<3}  "
            f"pred={'P' if result.object_present else 'A'}  "
            f"full={result.full_img_sim:.3f} patch={result.max_sim:.3f}  {noun}"
        )

    # ── Compute + print metrics ────────────────────────────────────────────────
    metrics = compute_metrics(records)
    print_metrics(metrics, args.mode)

    # ── Save visualizations ────────────────────────────────────────────────────
    if args.n_vis > 0:
        # Balance TP / TN / FP / FN
        tp_recs = [r for r in records if     r["object_present"] and r["gt"] == "Yes"]
        tn_recs = [r for r in records if not r["object_present"] and r["gt"] == "No"]
        fp_recs = [r for r in records if     r["object_present"] and r["gt"] == "No"]
        fn_recs = [r for r in records if not r["object_present"] and r["gt"] == "Yes"]

        per_group = max(1, args.n_vis // 4)
        vis_recs  = (
            tp_recs[:per_group] + tn_recs[:per_group] +
            fp_recs[:per_group] + fn_recs[:per_group]
        )[:args.n_vis]

        print(f"Saving {len(vis_recs)} visualizations → {vis_dir}/")
        print(f"  (TP={len(tp_recs[:per_group])} TN={len(tn_recs[:per_group])} "
              f"FP={len(fp_recs[:per_group])} FN={len(fn_recs[:per_group])})")

        for rec in vis_recs:
            idx    = rec["idx"]
            sample = samples[idx]
            result = result_cache[idx]
            sm_vis = {k: rec[k] for k in ("entropy", "peak_to_mean", "contrast")}
            noun   = extract_query_noun(sample["question"])
            pred   = "P" if result.object_present else "A"
            fname  = (
                f"{idx:03d}_{sample['split'][:3]}_GT{sample['gt']}_"
                f"pred{pred}_{noun.replace(' ', '_')}.png"
            )
            save_sample_figure(
                sample, result, sm_vis, args.grid_n,
                os.path.join(vis_dir, fname),
            )
        print(f"  Done.")

    # ── Save results JSON ──────────────────────────────────────────────────────
    result_data = {
        "mode":       args.mode,
        "clip_model": args.clip_model,
        "n_per_cell": args.n_per_cell,
        "seed":       args.seed,
        "metrics":    metrics,
        "records":    records,
    }
    out_json = os.path.join(out_dir, f"saliency_val_{args.mode}.json")
    with open(out_json, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Results → {out_json}")

    if args.save_baseline:
        baseline_path = os.path.join(_REPO, "results", "saliency_val_baseline.json")
        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"Baseline → {baseline_path}")


if __name__ == "__main__":
    main()
