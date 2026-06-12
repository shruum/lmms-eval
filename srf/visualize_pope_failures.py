#!/usr/bin/env python3
"""
Debug SRF failures on POPE — 5 FN + 5 FP per split.

For each failure shows a 3-panel figure:
  Panel 1: original image + question + GT label
  Panel 2: CLIP saliency heatmap + CLIP gate decision + all signal values
  Panel 3: SRF model prediction + whether it was correct

Saves to results/saliency_vis_pope/failures/{split}/
  fn_XX_noun.png  — GT=Yes, model said No (missed present object)
  fp_XX_noun.png  — GT=No,  model said Yes (false alarm on absent object)

Usage:
  python srf/visualize_pope_failures.py
  python srf/visualize_pope_failures.py --split adversarial
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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import clip_salience as clip_sal
from noun_extract import extract_clip_noun

_CLIP_MODEL = "openai/clip-vit-base-patch32"
_SPATIAL    = 2
_FULL_JSON  = os.path.join(_REPO, "results", "pope_full_srf_v3.json")
_SPLITS     = ["adversarial", "popular", "random"]


# ── Saliency ──────────────────────────────────────────────────────────────────

def run_clip(image: Image.Image, noun: str):
    """Return ClipSalienceResult or None."""
    try:
        w, h = image.size
        gh = max(int(round((h / 28) / _SPATIAL)), 1)
        gw = max(int(round((w / 28) / _SPATIAL)), 1)
        return clip_sal.compute_clip_salience_full_gate_v3(
            image, noun, gh, gw,
            top_k_pct=0.30, clip_model_name=_CLIP_MODEL, backup="none",
        )
    except Exception as e:
        print(f"      [WARN] CLIP error: {e}")
        return None


def make_heatmap(result, image: Image.Image) -> np.ndarray:
    """Blend saliency over image; return H×W×3 uint8."""
    sal = result.saliency.cpu().float().numpy()
    w, h = image.size
    gh = max(int(round((h / 28) / _SPATIAL)), 1)
    gw = max(int(round((w / 28) / _SPATIAL)), 1)
    if gh * gw != len(sal):
        g = int(round(len(sal) ** 0.5))
        gh = gw = g
    sal2d = sal.reshape(gh, gw)
    sal2d = (sal2d - sal2d.min()) / (sal2d.max() - sal2d.min() + 1e-8)
    sal_up = np.array(Image.fromarray(sal2d).resize((w, h), Image.BILINEAR))
    heat = plt.colormaps["jet"](sal_up)[..., :3]
    img_arr = np.array(image) / 255.0
    overlay = 0.45 * heat + 0.55 * img_arr
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


# ── Figure ────────────────────────────────────────────────────────────────────

def save_figure(image: Image.Image, question: str, gt: str, split: str,
                noun: str, result, srf_pred: str, out_path: str) -> None:
    """3-panel figure: [image+GT] | [heatmap+CLIP stats] | [SRF decision]."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             gridspec_kw={"width_ratios": [3, 3, 2]})
    fig.patch.set_facecolor("#fafafa")

    gt_color  = "#1a7a3c" if gt == "Yes" else "#b22222"
    srf_color = "#1a7a3c" if srf_pred == gt else "#b22222"
    tick = lambda v: "✓" if v else "✗"

    # ── Panel 0: original image ───────────────────────────────────────────────
    axes[0].imshow(image)
    axes[0].axis("off")
    q_disp = question[:65] + ("…" if len(question) > 65 else "")
    axes[0].set_title(
        f"[{split}]  GT = {gt}\n{q_disp}",
        fontsize=9, color=gt_color, pad=5, fontweight="bold",
    )

    # ── Panel 1: CLIP saliency heatmap ────────────────────────────────────────
    if result is None:
        axes[1].imshow(image); axes[1].axis("off")
        axes[1].set_title("CLIP FAILED", fontsize=9, color="#b22222", pad=5)
    else:
        overlay = make_heatmap(result, image)
        axes[1].imshow(overlay); axes[1].axis("off")
        gate_str  = "PRESENT" if result.object_present else "ABSENT"
        gate_col  = "#1a7a3c" if result.object_present == (gt == "Yes") else "#b22222"
        axes[1].set_title(
            f"CLIP gate → {gate_str}   (noun: '{noun}')",
            fontsize=9, color=gate_col, pad=5,
        )
        # Signal values overlaid bottom-left
        sig_lines = (
            f"full_img_sim={result.full_img_sim:.3f} {tick(result.gate_full)}\n"
            f"patch_max   ={result.max_sim:.3f} {tick(result.gate_patch)}\n"
            f"patch_contr ={result.patch_contrast:.2f}x {tick(result.gate_contrast)}\n"
            f"raw_entropy ={result.raw_entropy:.3f} {tick(result.gate_raw_entropy)}\n"
            f"cross_scale ={result.cross_scale_iou:.3f} {tick(result.gate_cross_scale)}\n"
            f"blur_delta  ={result.blur_delta:+.4f} {tick(result.gate_blur_delta)}"
        )
        axes[1].text(0.02, 0.02, sig_lines, transform=axes[1].transAxes,
                     fontsize=7, va="bottom", color="white", fontfamily="monospace",
                     bbox=dict(facecolor="black", alpha=0.65, pad=3, boxstyle="round"))

    # ── Panel 2: SRF prediction + diagnosis ──────────────────────────────────
    axes[2].axis("off")
    correct     = srf_pred == gt
    error_type  = ""
    if not correct:
        error_type = "FN: present object missed" if gt == "Yes" else "FP: absent object hallucinated"

    clip_gate_ok = result is not None and result.object_present == (gt == "Yes")

    diag_lines = [
        ("── SRF Prediction ────────────", "#333"),
        (f"  GT          :  {gt}", gt_color),
        (f"  SRF pred    :  {srf_pred}   {'✓' if correct else '✗'}", srf_color),
        ("", "#333"),
        (f"  {error_type}", "#b22222" if error_type else "#333"),
        ("", "#333"),
        ("── CLIP Gate Diagnosis ───────", "#333"),
    ]
    if result is not None:
        gate_ok_str = "✓ gate correct" if clip_gate_ok else "✗ gate wrong"
        gate_ok_col = "#1a7a3c" if clip_gate_ok else "#b22222"
        diag_lines += [
            (f"  CLIP → {result.object_present and 'PRESENT' or 'ABSENT'}  ({gate_ok_str})", gate_ok_col),
            ("", "#333"),
        ]
        if not clip_gate_ok and gt == "Yes":
            diag_lines += [
                ("  full_img_sim below 0.21", "#b22222"),
                ("  → SRF applies NO spatial boost", "#b22222"),
                ("  → model gets no help → says No", "#b22222"),
            ]
        elif not clip_gate_ok and gt == "No":
            diag_lines += [
                ("  full_img_sim above 0.21 (noise)", "#b22222"),
                ("  → SRF boosts random patches", "#b22222"),
                ("  → model confused → says Yes", "#b22222"),
            ]
        elif clip_gate_ok and not correct:
            diag_lines += [
                ("  CLIP gate correct BUT model fails", "#e67e22"),
                ("  → SRF boost insufficient or", "#e67e22"),
                ("    object too hard for model", "#e67e22"),
            ]
        else:
            diag_lines += [("  gate correct + pred correct", "#1a7a3c")]

    y = 0.97
    for text, color in diag_lines:
        axes[2].text(0.04, y, text, transform=axes[2].transAxes,
                     fontsize=8.5, va="top", color=color, fontfamily="monospace")
        y -= 0.063

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"      → {os.path.basename(out_path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",   default="all",
                        choices=["adversarial", "popular", "random", "all"])
    parser.add_argument("--n_fn",    type=int, default=5, help="FN samples per split")
    parser.add_argument("--n_fp",    type=int, default=5, help="FP samples per split")
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    splits   = _SPLITS if args.split == "all" else [args.split]
    out_root = args.out_dir or os.path.join(_REPO, "results", "saliency_vis_pope", "failures")
    rng      = random.Random(args.seed)

    # ── Load full SRF results ──────────────────────────────────────────────────
    print(f"Loading {_FULL_JSON}…")
    with open(_FULL_JSON) as f:
        full = json.load(f)
    all_records = full["records"]
    seed_used   = full.get("seed", 42)

    # ── Reload POPE with same seed to recover images ──────────────────────────
    print("Reloading POPE dataset (same seed) to recover images…")
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/POPE", split="test")

    # Reproduce exact ordering from eval_pope_val.load_balanced_pope
    ds_rng = random.Random(seed_used)
    buckets: dict = {}
    for sp in _SPLITS:
        for ans in ("Yes", "No"):
            buckets[(sp, ans)] = []
    for r in ds:
        sp  = str(r.get("category", "")).strip().lower()
        ans = str(r.get("answer",   "")).strip().capitalize()
        if sp in _SPLITS and ans in ("Yes", "No"):
            buckets[(sp, ans)].append(r)
    samples = []
    for key in sorted(buckets.keys()):
        rows = buckets[key][:]
        ds_rng.shuffle(rows)
        samples.extend(rows)   # n_per_cell=10000 → takes all
    ds_rng.shuffle(samples)
    print(f"  Recovered {len(samples)} samples (should be 9000)")
    assert len(samples) == len(all_records), \
        f"Mismatch: {len(samples)} samples vs {len(all_records)} records"

    # ── Process each split ────────────────────────────────────────────────────
    for split in splits:
        print(f"\n{'='*60}")
        print(f"  Split: {split}")
        print(f"{'='*60}")

        split_recs = [r for r in all_records if r["split"] == split]
        fn_recs    = [r for r in split_recs if r["gt"] == "Yes" and not r["correct"]]
        fp_recs    = [r for r in split_recs if r["gt"] == "No"  and not r["correct"]]

        rng.shuffle(fn_recs); rng.shuffle(fp_recs)
        chosen = (
            [("fn", r) for r in fn_recs[:args.n_fn]] +
            [("fp", r) for r in fp_recs[:args.n_fp]]
        )
        print(f"  Showing {len([x for x in chosen if x[0]=='fn'])} FN "
              f"+ {len([x for x in chosen if x[0]=='fp'])} FP failures")

        out_dir = os.path.join(out_root, split)

        for tag, rec in chosen:
            idx   = rec["idx"]
            row   = samples[idx]
            image = row["image"].convert("RGB")
            q     = rec["question"]
            gt    = rec["gt"]
            noun  = extract_clip_noun(q, mode="pope")
            pred  = rec["pred"]

            print(f"\n  [{tag.upper()}] idx={idx} GT={gt} pred={pred}  noun='{noun}'")
            print(f"         q='{q}'")

            result = run_clip(image, noun)

            if result:
                gate_str = "PRESENT" if result.object_present else "ABSENT"
                gate_ok  = result.object_present == (gt == "Yes")
                print(f"         CLIP→{gate_str} ({'✓' if gate_ok else '✗'})  "
                      f"full_img={result.full_img_sim:.3f}  patch_max={result.max_sim:.3f}")

            fname = f"{tag}_{idx:05d}_{noun.replace(' ','_')}.png"
            save_figure(image, q, gt, split, noun, result, pred,
                        os.path.join(out_dir, fname))

    print(f"\nAll figures saved to {out_root}/")


if __name__ == "__main__":
    main()
