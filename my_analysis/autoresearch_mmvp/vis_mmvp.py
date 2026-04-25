#!/usr/bin/env python3
"""
MMVP visualization: pairs side-by-side with CLIP saliency overlay.

For each selected pair produces a 2-panel image:
  Left:  Image A  + CLIP heatmap + GT / Pred / ✓✗
  Right: Image B  + CLIP heatmap + GT / Pred / ✓✗
  Title: question text

Saves to vis/ directory.

Usage:
    cd /volumes2/mllm/lmms-eval
    python my_analysis/autoresearch_mmvp/vis_mmvp.py
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("HF_HOME", "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SCRIPT_DIR   = pathlib.Path(__file__).parent
ANALYSIS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ANALYSIS_DIR))

import torch
import clip_salience as clip_sal
from noun_extract import extract_clip_noun

VIS_DIR = SCRIPT_DIR / "vis"
VIS_DIR.mkdir(exist_ok=True)

QUESTIONS_CSV = "/volumes2/hugging_face_cache/mmvp_questions/Questions.csv"
LAST_RUN_JSON = SCRIPT_DIR / "last_run.json"

# How many pairs to visualise per outcome group
N_BOTH_CORRECT = 6
N_ONE_CORRECT  = 6
N_BOTH_WRONG   = 5   # all of them


# ---------------------------------------------------------------------------
# Noun extraction (same as srf.py)
# ---------------------------------------------------------------------------
def _extract_noun(question: str) -> str:
    return extract_clip_noun(question, mode="mmvp")


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------
IMG_SIZE = 336   # resize all images to this square for consistency

def _saliency_overlay(image: Image.Image, saliency: torch.Tensor,
                       grid_h: int, grid_w: int) -> Image.Image:
    """Return image with CLIP saliency heatmap blended on top."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    sal_grid = saliency.reshape(grid_h, grid_w).cpu().float().numpy()
    sal_up   = np.array(
        Image.fromarray((sal_grid * 255).astype(np.uint8))
            .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    ).astype(float) / 255.0
    r = (sal_up * 255).astype(np.uint8)
    b = (255 - r)
    g = np.zeros_like(r)
    heatmap = np.stack([r, g, b], axis=-1)
    img_np  = np.array(img).astype(float)
    blended = (0.55 * img_np + 0.45 * heatmap).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def _get_font(size: int = 14):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _wrap_text(text: str, max_chars: int = 55) -> list[str]:
    """Wrap text to lines of at most max_chars characters."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make_pair_figure(
    img_a: Image.Image, img_b: Image.Image,
    sal_a: torch.Tensor | None, sal_b: torch.Tensor | None,
    grid_h: int, grid_w: int,
    question: str, options_raw: str,
    gt_a: str, pred_a: str,
    gt_b: str, pred_b: str,
    noun: str, sim_a: float, sim_b: float,
    out_path: pathlib.Path,
) -> None:
    """Create and save a side-by-side pair figure."""
    PANEL_W  = IMG_SIZE
    PANEL_H  = IMG_SIZE
    TITLE_H  = 72    # top title bar
    INFO_H   = 52    # bottom info bar per panel
    TOTAL_W  = PANEL_W * 2 + 12   # 12px gap
    TOTAL_H  = TITLE_H + PANEL_H + INFO_H

    canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), (30, 30, 30))
    draw   = ImageDraw.Draw(canvas)
    fnt_q  = _get_font(13)
    fnt_l  = _get_font(14)
    fnt_s  = _get_font(11)

    # ── Title bar ──────────────────────────────────────────────────────────
    # Parse "(a) Opt1 (b) Opt2"
    opts = re.findall(r'\([ab]\)\s*([^(]+)', options_raw, re.IGNORECASE)
    opt_a = opts[0].strip() if len(opts) > 0 else "A"
    opt_b = opts[1].strip() if len(opts) > 1 else "B"
    opts_text = f"A. {opt_a}   B. {opt_b}"

    q_lines = _wrap_text(question, 70)
    y = 4
    for line in q_lines[:2]:
        draw.text((6, y), line, font=fnt_q, fill=(240, 240, 100))
        y += 16
    draw.text((6, y), opts_text, font=fnt_s, fill=(180, 220, 180))

    # ── Panels ─────────────────────────────────────────────────────────────
    for side_idx, (img, sal, gt, pred) in enumerate([
        (img_a, sal_a, gt_a, pred_a),
        (img_b, sal_b, gt_b, pred_b),
    ]):
        x_off = side_idx * (PANEL_W + 12)

        # Saliency overlay or plain image
        if sal is not None:
            panel = _saliency_overlay(img, sal, grid_h, grid_w)
        else:
            panel = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        canvas.paste(panel, (x_off, TITLE_H))

        # Info bar
        sim = sim_a if side_idx == 0 else sim_b
        ok  = (pred == gt)
        ok_sym  = "✓" if ok else "✗"
        ok_col  = (80, 220, 80) if ok else (220, 80, 80)
        img_lbl = "Image A" if side_idx == 0 else "Image B"
        y_info  = TITLE_H + PANEL_H + 4

        draw.text((x_off + 4, y_info),
                  f"{img_lbl}  GT:{gt}  Pred:{pred} {ok_sym}",
                  font=fnt_l, fill=ok_col)
        draw.text((x_off + 4, y_info + 18),
                  f"noun='{noun}'  CLIP sim={sim:.3f}",
                  font=fnt_s, fill=(160, 160, 200))

    canvas.save(out_path)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import pandas as pd
    from datasets import load_dataset as hf_load
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    print("Loading last_run.json…")
    with open(LAST_RUN_JSON) as f:
        data = json.load(f)
    samples = data["samples"]

    print("Loading Questions.csv…")
    df = pd.read_csv(QUESTIONS_CSV)

    print("Loading MMVP images…")
    img_ds    = hf_load("MMVP/MMVP", split="train")
    lex_sorted = sorted(range(1, 301), key=str)
    csv_to_hf  = {csv_1idx: hf_idx for hf_idx, csv_1idx in enumerate(lex_sorted)}

    # Group by pair
    pairs = defaultdict(list)
    for s in samples:
        pairs[s["pair_id"]].append(s)

    both_correct = [(pid, p) for pid, p in sorted(pairs.items())
                    if all(s["correct"] for s in p)]
    one_correct  = [(pid, p) for pid, p in sorted(pairs.items())
                    if sum(s["correct"] for s in p) == 1]
    both_wrong   = [(pid, p) for pid, p in sorted(pairs.items())
                    if not any(s["correct"] for s in p)]

    print(f"Both correct: {len(both_correct)}, One correct: {len(one_correct)}, Both wrong: {len(both_wrong)}")

    # Select pairs to visualize
    to_vis: list[tuple[str, int, list]] = []
    to_vis += [("both_correct", pid, p) for pid, p in both_correct[:N_BOTH_CORRECT]]
    to_vis += [("one_correct",  pid, p) for pid, p in one_correct[:N_ONE_CORRECT]]
    to_vis += [("both_wrong",   pid, p) for pid, p in both_wrong[:N_BOTH_WRONG]]

    # Load model & processor (needed for grid dims via get_grid_dims)
    print("Loading model for grid dims…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager"
    ).eval()
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", max_pixels=512 * 28 * 28
    )
    spatial = getattr(model.config.vision_config, "spatial_merge_size", 2)

    print(f"\nGenerating {len(to_vis)} pair visualizations…\n")

    for outcome, pid, pair in to_vis:
        # Sort by csv_1idx so pair[0]=Image A, pair[1]=Image B
        pair = sorted(pair, key=lambda s: s["csv_1idx"])
        s_a, s_b = pair[0], pair[1]

        row = df.iloc[s_a["csv_1idx"] - 1]
        question    = row["Question"]
        options_raw = row["Options"]
        noun        = _extract_noun(question)

        img_a = img_ds[csv_to_hf[s_a["csv_1idx"]]]["image"].convert("RGB")
        img_b = img_ds[csv_to_hf[s_b["csv_1idx"]]]["image"].convert("RGB")

        # Compute grid dims and CLIP saliency for each image
        saliencies = []
        sims       = []
        for img in (img_a, img_b):
            msgs   = [{"role": "user", "content": [{"type": "image", "image": img},
                                                    {"type": "text",  "text": question}]}]
            text   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            vis, _ = process_vision_info(msgs)
            inp    = processor(text=[text], images=vis, return_tensors="pt", padding=True)
            grid_h, grid_w = clip_sal.get_grid_dims(inp, spatial)
            result = clip_sal.compute_clip_salience(img, noun, grid_h, grid_w,
                         top_k_pct=0.30, coarse_n=7)
            saliencies.append(result.saliency if result.max_sim >= 0.20 else None)
            sims.append(result.max_sim)

        fname = f"{outcome}_pair{pid:03d}_{noun.replace(' ','_')}.png"
        make_pair_figure(
            img_a=img_a, img_b=img_b,
            sal_a=saliencies[0], sal_b=saliencies[1],
            grid_h=grid_h, grid_w=grid_w,
            question=question, options_raw=options_raw,
            gt_a=s_a["gt"], pred_a=s_a["pred"],
            gt_b=s_b["gt"], pred_b=s_b["pred"],
            noun=noun, sim_a=sims[0], sim_b=sims[1],
            out_path=VIS_DIR / fname,
        )

    print(f"\nDone. {len(to_vis)} images saved to {VIS_DIR}")


if __name__ == "__main__":
    main()
