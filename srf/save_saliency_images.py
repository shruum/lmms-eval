#!/usr/bin/env python3
"""
Save saliency images with Question and GT captions for analysis.

This script runs SRF on VLM Bias samples and saves saliency images
with detailed captions including the question and ground truth answer.

Usage:
    python srf/save_saliency_images.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --n_per_cat 10 \
        --output results/saliency_analysis/
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from collections import defaultdict

_SRF_DIR = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR / "saliency"))
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import config as CFG
os.environ.setdefault("HF_HOME", CFG.HF_HOME)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Import dataset loader
import eval_datasets
from saliency.clip_salience import compute_clip_salience


def parse_args():
    p = argparse.ArgumentParser(description="Save saliency images with captions")
    p.add_argument("--model", default=CFG.DEFAULT_MODEL)
    p.add_argument("--n_per_cat", type=int, default=10,
                   help="Samples per category")
    p.add_argument("--output", default="results/saliency_with_captions/")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(model_id: str):
    """Load model for saliency computation (not needed for saliency only)."""
    print(f"Skipping model loading (CLIP will be loaded automatically)…")
    return None, None, None


def create_captioned_saliency_image(
    image: Image.Image,
    saliency_map: np.ndarray,
    question: str,
    ground_truth: str,
    category: str,
    sample_idx: int,
    save_path: str
):
    """
    Create and save a saliency visualization with detailed caption.

    Args:
        image: Original PIL image
        saliency_map: Saliency heatmap (H, W) with values 0-1
        question: Question text
        ground_truth: Ground truth answer
        category: Category name
        sample_idx: Sample index
        save_path: Where to save the image
    """
    # Create figure with image and saliency side by side
    img_w, img_h = image.size
    fig_w = img_w * 2 + 100  # Two images + margin
    fig_h = img_h + 150      # Image + caption space

    # Create white background
    fig_img = Image.new('RGB', (fig_w, fig_h), 'white')
    draw = ImageDraw.Draw(fig_img)

    # Paste original image
    fig_img.paste(image, (20, 20))

    # Create and paste saliency heatmap
    saliency_resized = Image.fromarray((saliency_map * 255).astype(np.uint8)).resize((img_w, img_h))
    saliency_colored = Image.new('RGB', saliency_resized.size)
    for x in range(img_w):
        for y in range(img_h):
            val = saliency_resized.getpixel((x, y))
            # Red heatmap: black -> red -> yellow -> white
            if val < 128:
                r, g, b = val * 2, 0, 0
            else:
                r, g, b = 255, (val - 128) * 2, (val - 128) * 2
            saliency_colored.putpixel((x, y), (r, g, b))

    fig_img.paste(saliency_colored, (img_w + 50, 20))

    # Add caption with all details
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()

    caption_y = img_h + 40
    line_height = 20

    # Category
    draw.text((20, caption_y), f"Category: {category}", fill='black', font=font)
    caption_y += line_height

    # Sample index
    draw.text((20, caption_y), f"Sample: {sample_idx}", fill='black', font=font)
    caption_y += line_height

    # Question (truncate if too long)
    question_short = question[:80] + "..." if len(question) > 80 else question
    draw.text((20, caption_y), f"Q: {question_short}", fill='blue', font=font)
    caption_y += line_height

    # Ground truth
    draw.text((20, caption_y), f"A: {ground_truth}", fill='green', font=font)

    # Add saliency stats
    mean_sal = saliency_map.mean()
    max_sal = saliency_map.max()
    draw.text((img_w + 50, caption_y), f"Saliency: mean={mean_sal:.2f}, max={max_sal:.2f}",
              fill='red', font=font)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig_img.save(save_path)
    print(f"  Saved: {save_path}")


def main():
    args = parse_args()

    # Load models (CLIP will be loaded automatically)
    load_model(args.model)

    # Load dataset
    print(f"\nLoading VLM Bias dataset ({args.n_per_cat} samples per category)…")
    samples = eval_datasets.load_vlm_bias(
        groups_filter=None,  # All categories
        n_samples=args.n_per_cat
    )

    print(f"\nProcessing {len(samples)} samples…")

    # Process each sample
    for idx, sample in enumerate(samples):
        image = sample["image"]
        question = sample["prompt"]
        ground_truth = sample["ground_truth"]
        category = sample["group"]

        # Extract nouns for CLIP
        import noun_extract
        query = noun_extract.extract_clip_noun(question, mode="vlmbias")

        # Compute CLIP saliency
        print(f"\n[{idx+1}/{len(samples)}] Category: {category}, Query: {query}")
        result = compute_clip_salience(
            image=image,
            text=query,
            grid_h=14,  # Approximate grid height for visualization
            grid_w=14,  # Approximate grid width for visualization
            top_k_pct=0.3,
            coarse_n=7
        )
        saliency_map = result.saliency.reshape(14, 14).cpu().numpy()

        # Save captioned saliency image
        safe_category = category.replace(" ", "_").replace("/", "_")
        filename = f"{safe_category}_{idx:03d}_Q_{ground_truth}.png"
        save_path = os.path.join(args.output, filename)

        create_captioned_saliency_image(
            image=image,
            saliency_map=saliency_map,
            question=question,
            ground_truth=ground_truth,
            category=category,
            sample_idx=idx,
            save_path=save_path
        )

    print(f"\n✅ Saved {len(samples)} saliency images to {args.output}")


if __name__ == "__main__":
    main()