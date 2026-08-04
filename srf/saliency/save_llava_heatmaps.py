#!/usr/bin/env python3
"""
Save CLIP v3 saliency heatmaps for LLaVA POPE samples.

Loads N adversarial POPE samples from HuggingFace, runs clip_full_gate_v3,
and saves side-by-side PNG: original image | saliency overlay.

Diagnostic text per sample:
  noun, full_img_sim, gate_full, gate_patch, patch_max_sim, object_present

LLaVA-1.5 image grid: 336px → 24×24 CLIP patch grid → 6×6 coarse CLIP grid.

Usage:
  python srf/saliency/save_llava_heatmaps.py \
      --n 8 \
      --repope_dir data/repope \
      --output results/saliency_heatmaps \
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

# Ensure srf package is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_LMMS = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _LMMS)
sys.path.insert(0, _HERE)

from srf.saliency import clip_salience as cs

# LLaVA-1.5 image token grid: 336px / 14px patch = 24 CLIP patch tokens,
# but clip_salience uses a coarser 6×6 grid (from get_grid_dims model_type="llava").
LLAVA_GRID_H = 6
LLAVA_GRID_W = 6


def saliency_overlay(
    sal: torch.Tensor,
    grid_h: int,
    grid_w: int,
    image: Image.Image,
    alpha: float = 0.55,
) -> np.ndarray:
    sal_np = sal.float().numpy().reshape(grid_h, grid_w)
    sal_np = (sal_np - sal_np.min()) / (sal_np.max() - sal_np.min() + 1e-8)
    sal_img = Image.fromarray((sal_np * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR)
    sal_arr = np.array(sal_img) / 255.0
    heat    = cm.get_cmap("jet")(sal_arr)[..., :3]
    img_arr = np.array(image.convert("RGB")) / 255.0
    overlay = (1 - alpha) * img_arr + alpha * heat
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def load_repope_labels(repope_dir: str) -> dict:
    labels = {}
    for split in ("adversarial", "popular", "random"):
        fpath = os.path.join(repope_dir, f"coco_repope_{split}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for line in f:
                e = json.loads(line)
                labels[(split, str(e["question_id"]))] = e["label"].lower()
    return labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8, help="Number of samples to visualize")
    p.add_argument("--split", default="adversarial",
                   choices=["adversarial", "popular", "random"])
    p.add_argument("--repope_dir", default=None,
                   help="Path to RepoPOPE dir (coco_repope_*.json). "
                        "If given, uses corrected labels.")
    p.add_argument("--output", default="results/saliency_heatmaps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full_img_thresh", type=float, default=0.20,
                   help="v3 full-image presence threshold (default 0.20)")
    p.add_argument("--patch_thresh", type=float, default=0.27,
                   help="v3 patch presence backup threshold (default 0.27)")
    p.add_argument("--backup", default="none",
                   choices=["none", "cross_scale", "blur_delta", "raw_entropy"],
                   help="v3 backup gate signal (default: none = gate_full OR gate_patch)")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load POPE from HuggingFace
    from datasets import load_dataset
    print("Loading lmms-lab/POPE from HuggingFace cache…")
    ds = load_dataset("lmms-lab/POPE", split="test")

    repope_labels = {}
    if args.repope_dir:
        repope_labels = load_repope_labels(args.repope_dir)
        print(f"  Loaded {len(repope_labels)} RePOPE corrected labels")

    rows = [r for r in ds if str(r.get("category", r.get("type", ""))).lower() == args.split]

    if repope_labels:
        rows = [r for r in rows
                if (args.split, str(r["question_id"])) in repope_labels]

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    samples = rows[:args.n]
    print(f"  {len(samples)} {args.split} samples selected")

    print("Loading CLIP ViT-B/32…")
    cs._load_clip()  # warm up CLIP once

    for idx, r in enumerate(samples):
        image   = r["image"].convert("RGB")
        q       = str(r["question"]).strip()
        qid     = str(r["question_id"])

        if repope_labels:
            gt = repope_labels.get((args.split, qid), "?")
        else:
            gt = ("yes" if str(r.get("answer", "")).strip().lower() == "yes" else "no")

        print(f"\n[{idx+1:02d}/{len(samples)}] qid={qid}  gt={gt}")
        print(f"  Q: {q}")

        # Run clip_v3 saliency
        result = cs.compute_clip_salience_full_gate_v3(
            image=image,
            text=q + "\nAnswer with Yes or No only.",
            grid_h=LLAVA_GRID_H,
            grid_w=LLAVA_GRID_W,
            top_k_pct=0.3,
            coarse_scales=(3, 5, 7),
            backup=args.backup,
            full_img_thresh=args.full_img_thresh,
            patch_thresh=args.patch_thresh,
        )

        print(f"  noun='{result.query_noun}'  full_img_sim={result.full_img_sim:.3f}"
              f"  patch_max={result.max_sim:.3f}  present={result.object_present}"
              f"  gate_full={result.gate_full}  gate_patch={result.gate_patch}")

        # Build figure: original | saliency overlay
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image)
        axes[0].set_title(
            f"Q: {q}\nGT={gt}  qid={qid}", fontsize=7, wrap=True)
        axes[0].axis("off")

        overlay = saliency_overlay(result.saliency, LLAVA_GRID_H, LLAVA_GRID_W, image)
        axes[1].imshow(overlay)
        gate_str = (f"gate_full={'Y' if result.gate_full else 'N'} "
                    f"gate_patch={'Y' if result.gate_patch else 'N'}")
        axes[1].set_title(
            f"CLIP v3 | noun='{result.query_noun}'\n"
            f"full_sim={result.full_img_sim:.3f}  patch_max={result.max_sim:.3f}\n"
            f"{gate_str}  present={'YES' if result.object_present else 'NO'}",
            fontsize=8)
        axes[1].axis("off")

        fig.suptitle(
            f"LLaVA POPE {args.split} | sample {idx+1}  "
            f"(full_thresh={args.full_img_thresh}  patch_thresh={args.patch_thresh}  "
            f"backup={args.backup})",
            fontsize=8)
        fig.tight_layout()

        fname = f"{args.split}_{idx+1:02d}_qid{qid}_gt{gt}.png"
        out_path = os.path.join(args.output, fname)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_path}")

    print(f"\nDone. {len(samples)} heatmaps saved to {args.output}/")


if __name__ == "__main__":
    main()
