#!/usr/bin/env python3
"""
Visualize False Negative (FN) cases from the full POPE SRF run.

FN = GT=Yes (object IS present) but model predicted No.
Loads images by index from HF POPE dataset, runs clip_full_gate_v3 saliency,
and saves per-sample figures using eval_presence.save_sample_figure.

Usage:
  python srf/visualize_pope_fn.py                         # 20 FNs from adversarial
  python srf/visualize_pope_fn.py --split popular --n 20
  python srf/visualize_pope_fn.py --split all --n 20
  python srf/visualize_pope_fn.py --type fp               # false positives instead
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "srf"))
sys.path.insert(0, os.path.join(_REPO, "srf", "saliency"))

os.environ.setdefault("HF_HOME", "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import matplotlib
matplotlib.use("Agg")

import clip_salience as clip_sal
from eval_presence import saliency_overlay

import matplotlib.pyplot as plt
import numpy as np

_CLIP_MODEL  = "openai/clip-vit-base-patch32"
_GRID_N      = 7
_SPATIAL     = 2
_FULL_POPE   = "/volumes2/mllm/lmms-eval/results/pope_full_srf_v3.json"


def _save_fig(image, question, gt, split, noun, result, sm, grid_h, grid_w, out_path):
    import matplotlib.pyplot as plt
    from eval_presence import saliency_overlay
    correct   = result.object_present == (gt == "Yes")
    pred_str  = "PRESENT" if result.object_present else "ABSENT"
    verdict   = "CORRECT ✓" if correct else "WRONG ✗"
    gt_color  = "#1e8449" if gt == "Yes" else "#922b21"
    v_color   = "#1a5276" if correct else "#922b21"

    overlay = saliency_overlay(result.saliency.numpy(), grid_h, grid_w, image)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                             gridspec_kw={"width_ratios": [3, 3, 2]})
    fig.patch.set_facecolor("white")

    axes[0].imshow(image); axes[0].axis("off")
    q_short = question[:60] + ("…" if len(question) > 60 else "")
    axes[0].set_title(f"[{split}]  GT: {gt}\n{q_short}", fontsize=8.5, color=gt_color, pad=4)

    axes[1].imshow(overlay); axes[1].axis("off")
    axes[1].set_title(f"noun: '{noun}'\npred = {pred_str}  [{verdict}]",
                      fontsize=9, color=v_color, pad=4)

    axes[2].axis("off")
    def tick(v): return "✓" if v else "✗"
    lines = [
        f"── Presence signals ──────",
        f"  full_img   {result.full_img_sim:.3f}  {tick(result.gate_full)}",
        f"  patch_max  {result.max_sim:.3f}  {tick(result.gate_patch)}",
        f"  contrast   {result.patch_contrast:.2f}x  {tick(result.gate_contrast)}",
        f"  ctrstv_gap {result.contrastive_gap:+.4f}  {tick(result.gate_contrastive)}",
        f"  patch_ent  {result.patch_entropy:.3f}  {tick(result.gate_entropy)}",
        f"── v3 signals ────────────",
        f"  raw_ent    {result.raw_entropy:.3f}  {tick(result.gate_raw_entropy)}",
        f"  xscale_iou {result.cross_scale_iou:.3f}  {tick(result.gate_cross_scale)}",
        f"  blur_delta {result.blur_delta:+.4f}  {tick(result.gate_blur_delta)}",
        f"",
        f"── Spatial ───────────────",
        f"  entropy    {sm['entropy']:.3f}",
        f"  peak/mean  {sm['peak_to_mean']:.3f}",
        f"  contrast   {sm['contrast']:.3f}",
        f"",
        f"── Decision ──────────────",
        f"  GT         {gt}",
        f"  pred       {pred_str}",
        f"  {verdict}",
    ]
    axes[2].text(0.05, 0.97, "\n".join(lines), transform=axes[2].transAxes,
                 fontsize=8, va="top", ha="left", fontfamily="monospace",
                 bbox=dict(facecolor="#f5f5f5", alpha=0.9, pad=4, boxstyle="round"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {os.path.basename(out_path)}")


def compute_saliency(image, noun):
    import torch
    try:
        w, h   = image.size
        grid_h = max(int(round((h / 28) / _SPATIAL)), 1)
        grid_w = max(int(round((w / 28) / _SPATIAL)), 1)
        return clip_sal.compute_clip_salience_full_gate_v3(
            image, noun, grid_h, grid_w,
            top_k_pct=0.30, clip_model_name=_CLIP_MODEL, backup="none",
        )
    except Exception as e:
        print(f"    [WARN] CLIP error for '{noun}': {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="adversarial",
                        choices=["adversarial", "popular", "random", "all"])
    parser.add_argument("--type",  default="fn", choices=["fn", "fp"],
                        help="fn=missed present, fp=false alarm on absent")
    parser.add_argument("--n",     type=int, default=20)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Load full POPE results to find error cases ────────────────────────────
    print(f"Loading {_FULL_POPE}…")
    with open(_FULL_POPE) as f:
        full = json.load(f)
    records = full["records"]

    if args.type == "fn":
        # False Negative: GT=Yes (present) but model said No
        errors = [r for r in records if r["gt"] == "Yes" and not r["correct"]]
        label  = "FN (present object — model said No)"
    else:
        # False Positive: GT=No (absent) but model said Yes
        errors = [r for r in records if r["gt"] == "No" and not r["correct"]]
        label  = "FP (absent object — model said Yes)"

    if args.split != "all":
        errors = [r for r in errors if r["split"] == args.split]

    rng.shuffle(errors)
    errors = errors[:args.n]
    print(f"Found {len(errors)} {args.type.upper()} cases from split='{args.split}'")

    # ── Load HF POPE dataset ──────────────────────────────────────────────────
    from datasets import load_dataset
    print("Loading POPE HF dataset…")
    ds = load_dataset("lmms-lab/POPE", split="test")
    # Build idx → row map
    idx_to_row = {i: ds[i] for i in range(len(ds))}

    # ── Output dir ───────────────────────────────────────────────────────────
    tag      = f"{args.split}_{args.type}"
    out_root = args.out_dir or os.path.join(
        _REPO, "results", "saliency_vis_pope", tag
    )
    os.makedirs(out_root, exist_ok=True)

    # ── Process each error case ───────────────────────────────────────────────
    from noun_extract import extract_clip_noun
    import numpy as np

    print(f"\nGenerating {len(errors)} visualizations → {out_root}/\n")
    for i, rec in enumerate(errors):
        idx   = rec["idx"]
        row   = idx_to_row[idx]
        image = row["image"].convert("RGB")
        q     = rec["question"]
        noun  = extract_clip_noun(q, mode="pope")
        split = rec["split"]
        gt    = rec["gt"]

        print(f"  [{i+1:02d}/{len(errors)}] idx={idx:04d} [{split}] GT={gt}  "
              f"noun='{noun}'  q='{q}'")

        result = compute_saliency(image, noun)
        if result is None:
            print(f"    → CLIP failed, skipping")
            continue

        # Spatial metrics (entropy, peak_to_mean, contrast)
        sal  = result.saliency.numpy()
        n    = len(sal)
        g    = int(round(n ** 0.5))
        # saliency may be non-square (grid_h × grid_w) — just use flat for stats
        flat = sal
        flat_norm = flat / (flat.sum() + 1e-8)
        entropy    = float(-np.sum(flat_norm * np.log(flat_norm + 1e-8)) / np.log(len(flat_norm)))
        peak2mean  = float(flat.max() / (flat.mean() + 1e-8))
        top_k      = int(0.3 * len(flat))
        contrast   = float(np.sort(flat)[::-1][:top_k].mean() / (flat.mean() + 1e-8))
        sm = {"entropy": entropy, "peak_to_mean": peak2mean, "contrast": contrast}

        # Infer actual grid dims from saliency length
        sal_np = result.saliency.numpy()
        n_tok  = len(sal_np)
        w_img, h_img = image.size
        gh = max(int(round((h_img / 28) / _SPATIAL)), 1)
        gw = max(int(round((w_img / 28) / _SPATIAL)), 1)
        if gh * gw != n_tok:  # fallback to square
            g  = int(round(n_tok ** 0.5))
            gh = gw = g if g * g == n_tok else _GRID_N

        sample = {
            "image":    image,
            "question": q,
            "gt":       gt,
            "split":    split,
        }

        fname = (f"{i:02d}_{idx:04d}_{split[:3]}_GT{gt}_pred"
                 f"{'P' if result.object_present else 'A'}_{noun.replace(' ','_')}.png")
        _save_fig(image, q, gt, split, noun, result, sm, gh, gw,
                  os.path.join(out_root, fname))

    print(f"\nDone. {len(errors)} figures saved to {out_root}/")
    print(f"\nLabel: {label}")


if __name__ == "__main__":
    main()
