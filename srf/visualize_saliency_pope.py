#!/usr/bin/env python3
"""
POPE saliency map visualizations — present vs absent objects.

Samples from adversarial, popular, and random splits (balanced Yes/No),
runs clip_full_gate_v3 saliency, and produces:
  - Per-sample figure: image + heatmap overlay + statistics panel
  - Summary grid: 4 rows × N cols arranged by split and GT
  - CSV of all statistics for downstream analysis

Outputs: results/saliency_vis_pope/
  adversarial_present.png, adversarial_absent.png (one sample per page)
  popular_present.png, popular_absent.png
  random_present.png, random_absent.png
  summary_present.png, summary_absent.png   (grid, all splits)
  stats.csv

Usage:
  python srf/visualize_saliency_pope.py           # 20 samples per split per GT label
  python srf/visualize_saliency_pope.py --n 10    # 10 per split per GT label
  python srf/visualize_saliency_pope.py --split adversarial  # one split only
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "srf"))
sys.path.insert(0, os.path.join(_REPO, "srf", "saliency"))

os.environ.setdefault("HF_HOME", "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import clip_salience as clip_sal
from noun_extract import extract_clip_noun

_CLIP_MODEL = "openai/clip-vit-base-patch32"
_SPATIAL    = 2      # Qwen2.5-VL spatial_merge_size
_BAD_NOUNS  = {
    "this", "that", "does", "only", "answer", "appropriate", "according",
    "object", "image", "picture", "taken", "from", "which", "some", "with",
    "right", "left", "true", "false", "sequence", "above", "below",
    "correct", "incorrect", "whether", "what", "where", "when", "have",
    "there", "their", "more", "also", "both", "each", "than",
}


def noun_is_bad(noun: str) -> bool:
    return noun in _BAD_NOUNS or len(noun) <= 2


def load_pope_samples(splits: List[str], n_per_split_per_label: int,
                      seed: int = 42) -> List[Dict]:
    """Load balanced POPE samples — n per (split × GT) giving 2n per split.

    POPE HuggingFace dataset has a single 'test' split with a 'category' field
    that holds adversarial / popular / random.
    """
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset("lmms-lab/POPE", split="test")

    all_samples = []
    for split in splits:
        rows = [r for r in ds if r.get("category") == split]
        yes_rows = [r for r in rows if str(r.get("answer", "")).strip().capitalize() == "Yes"]
        no_rows  = [r for r in rows if str(r.get("answer", "")).strip().capitalize() == "No"]
        rng.shuffle(yes_rows)
        rng.shuffle(no_rows)
        for r in yes_rows[:n_per_split_per_label]:
            all_samples.append({"row": r, "split": split, "gt": "Yes"})
        for r in no_rows[:n_per_split_per_label]:
            all_samples.append({"row": r, "split": split, "gt": "No"})

    return all_samples


def compute_saliency(image: Image.Image, noun: str):
    """Run clip_full_gate_v3 and return the full ClipSalienceResult (or None)."""
    try:
        w, h   = image.size
        grid_h = max(int(round((h / 28) / _SPATIAL)), 1)
        grid_w = max(int(round((w / 28) / _SPATIAL)), 1)
        return clip_sal.compute_clip_salience_full_gate_v3(
            image, noun, grid_h, grid_w,
            top_k_pct=0.30,
            clip_model_name=_CLIP_MODEL,
            backup="none",
        )
    except Exception as e:
        print(f"    [WARN] CLIP failed for '{noun}': {e}")
        return None


def _sal_to_heatmap(result, image: Image.Image) -> Optional[np.ndarray]:
    """Reshape saliency vector to (H, W) float32 in [0,1], upsampled to image size."""
    if result is None:
        return None
    sal = result.saliency.cpu().float().numpy()
    n   = len(sal)
    g   = int(round(n ** 0.5))
    if g * g != n:
        return None
    grid = sal.reshape(g, g)
    w, h = image.size
    up = np.array(Image.fromarray(grid).resize((w, h), Image.BILINEAR))
    mn, mx = up.min(), up.max()
    return (up - mn) / (mx - mn + 1e-8)


def _stats_text(result, noun: str, gt: str) -> str:
    """Format statistics block for the annotation panel."""
    if result is None:
        return f"noun: '{noun}'\nCLIP FAILED"
    lines = [
        f"noun:           '{result.query_noun}'",
        f"GT:             {gt}",
        f"gate decision:  {'PRESENT' if result.object_present else 'ABSENT'}",
        f"",
        f"full_img_sim:   {result.full_img_sim:.4f}",
        f"patch_max_sim:  {result.max_sim:.4f}",
        f"patch_contrast: {result.patch_contrast:.4f}",
        f"contrastive_gap:{result.contrastive_gap:.4f}",
        f"patch_entropy:  {result.patch_entropy:.4f}",
        f"raw_entropy:    {result.raw_entropy:.4f}",
        f"cross_scale_iou:{result.cross_scale_iou:.4f}",
        f"blur_delta:     {result.blur_delta:.4f}",
        f"",
        f"gate_full:      {result.gate_full}",
        f"gate_patch:     {result.gate_patch}",
        f"gate_contrast:  {result.gate_contrast}",
        f"gate_contrastive:{result.gate_contrastive}",
        f"gate_entropy:   {result.gate_entropy}",
        f"gate_raw_entr:  {result.gate_raw_entropy}",
        f"gate_cross_scl: {result.gate_cross_scale}",
        f"gate_blur:      {result.gate_blur_delta}",
    ]
    return "\n".join(lines)


def make_sample_figure(samples: List[Dict], title: str, out_path: str) -> None:
    """One row per sample: [image | heatmap overlay | stats text]."""
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4.5 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, s in enumerate(samples):
        image   = s["image"]
        result  = s.get("result")
        gt      = s["gt"]
        noun    = s["noun"]
        skipped = s["skipped"]
        heatmap = s.get("heatmap")

        # ── Col 0: original image ─────────────────────────────────────────
        axes[i][0].imshow(image)
        axes[i][0].axis("off")
        q_short = s["question"][:100] + ("…" if len(s["question"]) > 100 else "")
        gate_correct = (result is not None and result.object_present == (gt == "Yes"))
        correct_str  = "gate✓" if (result and gate_correct) else "gate✗"
        color_title  = "green" if (result and gate_correct) else "red"
        axes[i][0].set_title(
            f"[{s['split']}] {q_short}\nGT={gt}  {correct_str}",
            fontsize=7, loc="left", pad=3, color=color_title,
        )

        # ── Col 1: saliency heatmap overlay ──────────────────────────────
        axes[i][1].imshow(image)
        if skipped or result is None:
            axes[i][1].text(0.5, 0.5,
                "SRF SKIPPED\n(bad/missing noun)",
                ha="center", va="center", transform=axes[i][1].transAxes,
                fontsize=10, color="white", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="red", alpha=0.7))
        else:
            if heatmap is not None:
                axes[i][1].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            color  = "lime" if result.object_present else "orange"
            status = "PRESENT" if result.object_present else "ABSENT"
            axes[i][1].text(0.02, 0.02, f'CLIP→{status}',
                transform=axes[i][1].transAxes, fontsize=8, color=color,
                bbox=dict(facecolor="black", alpha=0.6, pad=2))
        axes[i][1].axis("off")
        axes[i][1].set_title(f"Saliency heatmap  (noun='{noun}')", fontsize=8, pad=3)

        # ── Col 2: statistics text panel ─────────────────────────────────
        axes[i][2].axis("off")
        stats = _stats_text(result, noun, gt)
        axes[i][2].text(0.03, 0.97, stats,
            transform=axes[i][2].transAxes,
            fontsize=7.5, va="top", ha="left",
            fontfamily="monospace",
            bbox=dict(facecolor="#f0f0f0", alpha=0.8, pad=4, boxstyle="round"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def make_summary_grid(samples_by_split: Dict[str, List[Dict]],
                      gt_label: str, title: str, out_path: str) -> None:
    """Grid: splits as rows, samples as columns, heatmap overlay only."""
    splits = list(samples_by_split.keys())
    max_cols = max(len(v) for v in samples_by_split.values())
    n_rows = len(splits)

    fig, axes = plt.subplots(n_rows, max_cols, figsize=(3.5 * max_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for r, split in enumerate(splits):
        row_samples = samples_by_split[split]
        for c in range(max_cols):
            ax = axes[r][c] if max_cols > 1 else axes[r]
            if c >= len(row_samples):
                ax.axis("off")
                continue
            s       = row_samples[c]
            image   = s["image"]
            result  = s.get("result")
            heatmap = s.get("heatmap")
            ax.imshow(image)
            if result is not None and heatmap is not None and not s["skipped"]:
                ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                color  = "lime" if result.object_present else "orange"
                status = "PRESENT" if result.object_present else "ABSENT"
                ax.text(0.02, 0.02, f'{status}\n{result.full_img_sim:.3f}',
                    transform=ax.transAxes, fontsize=7, color=color,
                    bbox=dict(facecolor="black", alpha=0.6, pad=2))
            elif s["skipped"]:
                ax.text(0.5, 0.5, "SKIPPED", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="white",
                    bbox=dict(facecolor="red", alpha=0.6))
            q_short = s["question"][:50] + "…"
            gate_ok = result is not None and result.object_present == (s["gt"] == "Yes")
            ax.set_title(f"[{split}] {q_short}", fontsize=5.5, pad=2,
                         color="green" if gate_ok else "red")
            ax.axis("off")
        # Row label
        axes[r][0].set_ylabel(split, fontsize=9, rotation=90, labelpad=5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def save_stats_csv(all_processed: List[Dict], out_path: str) -> None:
    fields = [
        "split", "gt", "question", "noun", "skipped",
        "full_img_sim", "patch_max_sim", "patch_contrast",
        "contrastive_gap", "patch_entropy", "raw_entropy",
        "cross_scale_iou", "blur_delta",
        "gate_full", "gate_patch", "gate_contrast",
        "gate_contrastive", "gate_entropy", "gate_raw_entropy",
        "gate_cross_scale", "gate_blur_delta",
        "object_present", "gate_correct",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in all_processed:
            r = s.get("result")
            row = {
                "split":    s["split"],
                "gt":       s["gt"],
                "question": s["question"],
                "noun":     s["noun"],
                "skipped":  s["skipped"],
            }
            if r is not None:
                row.update({
                    "full_img_sim":    f"{r.full_img_sim:.4f}",
                    "patch_max_sim":   f"{r.max_sim:.4f}",
                    "patch_contrast":  f"{r.patch_contrast:.4f}",
                    "contrastive_gap": f"{r.contrastive_gap:.4f}",
                    "patch_entropy":   f"{r.patch_entropy:.4f}",
                    "raw_entropy":     f"{r.raw_entropy:.4f}",
                    "cross_scale_iou": f"{r.cross_scale_iou:.4f}",
                    "blur_delta":      f"{r.blur_delta:.4f}",
                    "gate_full":        r.gate_full,
                    "gate_patch":       r.gate_patch,
                    "gate_contrast":    r.gate_contrast,
                    "gate_contrastive": r.gate_contrastive,
                    "gate_entropy":     r.gate_entropy,
                    "gate_raw_entropy": r.gate_raw_entropy,
                    "gate_cross_scale": r.gate_cross_scale,
                    "gate_blur_delta":  r.gate_blur_delta,
                    "object_present":   r.object_present,
                    "gate_correct":     r.object_present == (s["gt"] == "Yes"),
                })
            else:
                for k in fields[5:]:
                    row[k] = ""
            w.writerow(row)
    print(f"  Stats CSV → {out_path}")


def print_stat_summary(all_processed: List[Dict]) -> None:
    """Print mean statistics split by GT=Yes vs GT=No."""
    def _collect(samples):
        stats = {k: [] for k in [
            "full_img_sim", "patch_max_sim", "patch_contrast",
            "contrastive_gap", "patch_entropy", "raw_entropy",
            "cross_scale_iou", "blur_delta",
        ]}
        for s in samples:
            r = s.get("result")
            if r is None:
                continue
            stats["full_img_sim"].append(r.full_img_sim)
            stats["patch_max_sim"].append(r.max_sim)
            stats["patch_contrast"].append(r.patch_contrast)
            stats["contrastive_gap"].append(r.contrastive_gap)
            stats["patch_entropy"].append(r.patch_entropy)
            stats["raw_entropy"].append(r.raw_entropy)
            stats["cross_scale_iou"].append(r.cross_scale_iou)
            stats["blur_delta"].append(r.blur_delta)
        return stats

    present = [s for s in all_processed if s["gt"] == "Yes" and not s["skipped"]]
    absent  = [s for s in all_processed if s["gt"] == "No"  and not s["skipped"]]
    sp = _collect(present)
    sa = _collect(absent)

    gate_acc_p = sum(1 for s in present if s.get("result") and s["result"].object_present)
    gate_acc_a = sum(1 for s in absent  if s.get("result") and not s["result"].object_present)

    print(f"\n{'='*62}")
    print(f"  Statistics: GT=Yes (n={len(present)})  vs  GT=No (n={len(absent)})")
    print(f"{'='*62}")
    print(f"  {'Metric':<22}  {'GT=Yes mean':>12}  {'GT=No mean':>12}  {'Δ':>8}")
    print(f"  {'-'*58}")
    for k in sp:
        yv = np.mean(sp[k]) if sp[k] else float("nan")
        nv = np.mean(sa[k]) if sa[k] else float("nan")
        d  = yv - nv
        print(f"  {k:<22}  {yv:>12.4f}  {nv:>12.4f}  {d:>+8.4f}")
    print(f"\n  Gate accuracy:")
    print(f"    GT=Yes correct (TPR): {gate_acc_p}/{len(present)}  = {gate_acc_p/len(present):.3f}" if present else "")
    print(f"    GT=No  correct (TNR): {gate_acc_a}/{len(absent)}   = {gate_acc_a/len(absent):.3f}" if absent else "")
    n_skip = sum(1 for s in all_processed if s["skipped"])
    print(f"    Skipped (bad noun):   {n_skip}/{len(all_processed)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=20, help="Samples per split per GT label")
    parser.add_argument("--split",  default=None, help="One split: adversarial|popular|random")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    splits   = [args.split] if args.split else ["adversarial", "popular", "random"]
    out_root = args.out_dir or os.path.join(_REPO, "results", "saliency_vis_pope")

    print(f"Loading POPE samples ({args.n} per split per label)…")
    raw = load_pope_samples(splits, args.n, seed=args.seed)
    print(f"  Loaded {len(raw)} samples total.")

    # Process: extract noun → compute saliency
    all_processed: List[Dict] = []
    for idx, item in enumerate(raw):
        r    = item["row"]
        gt   = item["gt"]
        split = item["split"]
        q    = str(r["question"]).strip()
        noun = extract_clip_noun(q, mode="pope")
        skip = noun_is_bad(noun)

        print(f"  [{idx+1:04d}/{len(raw)}] {split:12s} GT={gt}  noun='{noun}'"
              + (" [SKIP]" if skip else ""))

        result  = None
        heatmap = None
        if not skip:
            img    = r["image"].convert("RGB")
            result = compute_saliency(img, noun)
            if result is not None:
                heatmap = _sal_to_heatmap(result, img)

        all_processed.append({
            "image":    r["image"].convert("RGB"),
            "question": q,
            "gt":       gt,
            "split":    split,
            "noun":     noun,
            "skipped":  skip,
            "result":   result,
            "heatmap":  heatmap,
        })

    # Stats summary
    print_stat_summary(all_processed)

    # Save CSV
    save_stats_csv(all_processed, os.path.join(out_root, "stats.csv"))

    # Per-split per-GT figures
    for split in splits:
        for gt_label in ("Yes", "No"):
            subset = [s for s in all_processed if s["split"] == split and s["gt"] == gt_label]
            if not subset:
                continue
            fname  = f"{split}_{'present' if gt_label=='Yes' else 'absent'}.png"
            label  = "PRESENT (GT=Yes)" if gt_label == "Yes" else "ABSENT (GT=No)"
            make_sample_figure(
                subset,
                f"POPE [{split}] — {label}  (SRF best: clip_full_gate_v3, ls=6, le=12)",
                os.path.join(out_root, split, fname),
            )

    # Summary grids
    for gt_label in ("Yes", "No"):
        by_split = {
            sp: [s for s in all_processed if s["split"] == sp and s["gt"] == gt_label]
            for sp in splits
        }
        label = "PRESENT (GT=Yes)" if gt_label == "Yes" else "ABSENT (GT=No)"
        make_summary_grid(
            by_split,
            gt_label,
            f"POPE — {label} — summary grid across splits",
            os.path.join(out_root, f"summary_{'present' if gt_label=='Yes' else 'absent'}.png"),
        )

    print(f"\nAll outputs saved to {out_root}/")


if __name__ == "__main__":
    main()
