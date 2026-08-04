#!/usr/bin/env python3
"""Save CLIP v3 saliency heatmaps for MMHal-Bench samples.

Loads N samples from the local MMHal-Bench JSON, runs clip_full_gate_v3
using image_content as the saliency noun, and saves side-by-side PNGs:
  original image | saliency overlay

Diagnostic text per sample:
  question_type, image_content, noun, full_img_sim, gate_full, gate_patch, object_present

Usage:
  python srf/saliency/save_mmhal_heatmaps.py \
      --n 8 \
      --output results/saliency_heatmaps/mmhal \
      --full_img_thresh 0.25 \
      --patch_thresh 0.27
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_LMMS = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _LMMS)
sys.path.insert(0, _HERE)

from srf.saliency import clip_salience as cs

# LLaVA-1.5: 336px images → 6×6 coarse CLIP grid (same as POPE heatmaps)
LLAVA_GRID_H = 6
LLAVA_GRID_W = 6

_MMHAL_JSON   = "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json"
_MMHAL_IMAGES = "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/images"


def saliency_overlay(sal: torch.Tensor, grid_h: int, grid_w: int,
                     image: Image.Image, alpha: float = 0.55) -> np.ndarray:
    sal_np = sal.float().numpy().reshape(grid_h, grid_w)
    sal_np = (sal_np - sal_np.min()) / (sal_np.max() - sal_np.min() + 1e-8)
    sal_img = Image.fromarray((sal_np * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR)
    sal_arr = np.array(sal_img) / 255.0
    heat    = cm.get_cmap("jet")(sal_arr)[..., :3]
    img_arr = np.array(image.convert("RGB")) / 255.0
    overlay = (1 - alpha) * img_arr + alpha * heat
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n",              type=int,   default=8)
    p.add_argument("--mmhal_json",     default=_MMHAL_JSON)
    p.add_argument("--mmhal_images",   default=_MMHAL_IMAGES)
    p.add_argument("--output",         default="results/saliency_heatmaps/mmhal")
    p.add_argument("--full_img_thresh", type=float, default=0.25)
    p.add_argument("--patch_thresh",    type=float, default=0.27)
    p.add_argument("--backup",          default="none",
                   choices=["none", "cross_scale", "blur_delta", "raw_entropy"])
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.mmhal_json) as f:
        records = json.load(f)
    samples = records[:args.n]
    print(f"Loaded {len(samples)} MMHal-Bench samples from {args.mmhal_json}")

    print("Loading CLIP ViT-B/32…")
    cs._load_clip()

    for idx, rec in enumerate(samples):
        img_fname = rec["image_src"].split("/")[-1]
        img_path  = os.path.join(args.mmhal_images, img_fname)
        image     = Image.open(img_path).convert("RGB")

        # Use first image_content item as CLIP noun (the primary object in the image)
        noun = rec["image_content"][0] if rec["image_content"] else rec["question"]
        qtype = rec["question_type"]
        question = rec["question"]
        content_str = ", ".join(rec["image_content"])

        print(f"\n[{idx+1:02d}/{len(samples)}] type={qtype}  content=[{content_str}]")
        print(f"  noun='{noun}'")
        print(f"  Q: {question}")

        result = cs.compute_clip_salience_full_gate_v3(
            image=image,
            text=noun,
            grid_h=LLAVA_GRID_H,
            grid_w=LLAVA_GRID_W,
            top_k_pct=0.3,
            coarse_scales=(3, 5, 7),
            backup=args.backup,
            full_img_thresh=args.full_img_thresh,
            patch_thresh=args.patch_thresh,
        )

        print(f"  full_img_sim={result.full_img_sim:.3f}  patch_max={result.max_sim:.3f}"
              f"  present={result.object_present}"
              f"  gate_full={result.gate_full}  gate_patch={result.gate_patch}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        axes[0].imshow(image)
        axes[0].set_title(
            f"[{qtype}] {content_str}\nQ: {question[:80]}{'…' if len(question)>80 else ''}",
            fontsize=7, wrap=True)
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
            f"MMHal-Bench sample {idx+1} "
            f"(full_thresh={args.full_img_thresh}  patch_thresh={args.patch_thresh})",
            fontsize=8)
        fig.tight_layout()

        fname = f"mmhal_{idx+1:02d}_{qtype}_{img_fname.split('.')[0]}.png"
        out_path = os.path.join(args.output, fname)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_path}")

    print(f"\nDone. {len(samples)} heatmaps saved to {args.output}/")


if __name__ == "__main__":
    main()
