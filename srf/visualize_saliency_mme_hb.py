#!/usr/bin/env python3
"""
Saliency map visualizations for MME and HallusionBench.

For each dataset, samples examples across categories and shows:
  Col 0: original image + question + GT
  Col 1: CLIP saliency heatmap (if noun is good) OR "SKIPPED" banner (bad noun)
  Col 2: noun extracted + gate decision

Saves PNGs to results/saliency_vis_mme/ and results/saliency_vis_hb/

Usage:
  python srf/visualize_saliency_mme_hb.py          # both datasets
  python srf/visualize_saliency_mme_hb.py --mme    # MME only
  python srf/visualize_saliency_mme_hb.py --hb     # HallusionBench only
  python srf/visualize_saliency_mme_hb.py --n 4    # 4 samples per category
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "srf"))
sys.path.insert(0, os.path.join(_REPO, "srf", "saliency"))
sys.path.insert(0, os.path.join(_REPO, "my_analysis"))

os.environ.setdefault("HF_HOME", "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import clip_salience as clip_sal
from noun_extract import extract_clip_noun

_BAD_NOUNS = {
    "this", "that", "does", "only", "answer", "appropriate", "according",
    "object", "image", "picture", "taken", "from", "which", "some", "with",
    "right", "left", "true", "false", "sequence", "above", "below",
    "correct", "incorrect", "whether", "what", "where", "when", "have",
    "there", "their", "more", "also", "both", "each", "than",
}

_CLIP_MODEL = "openai/clip-vit-base-patch32"
_SPATIAL    = 2     # Qwen2.5-VL spatial_merge_size
_GRID_N     = 7     # coarse CLIP grid


def noun_is_bad(noun: str) -> bool:
    return noun in _BAD_NOUNS or len(noun) <= 2


def compute_saliency(image: Image.Image, noun: str) -> Optional[np.ndarray]:
    """Return H×W float32 saliency array in [0,1], or None if CLIP fails."""
    try:
        # Use approximate grid — same as SRF runtime
        w, h   = image.size
        n_img  = int(round((w / 28) / _SPATIAL)) * int(round((h / 28) / _SPATIAL))
        grid_h = int(round((h / 28) / _SPATIAL))
        grid_w = int(round((w / 28) / _SPATIAL))
        grid_h = max(grid_h, 1); grid_w = max(grid_w, 1)

        result = clip_sal.compute_clip_salience_full_gate_v3(
            image, noun, grid_h, grid_w,
            top_k_pct=0.30,
            clip_model_name=_CLIP_MODEL,
            backup="none",
        )
        sal = result.saliency  # float tensor (n_img_tokens,)
        sal_np = sal.cpu().float().numpy()
        # Reshape to grid
        n = len(sal_np)
        g = int(round(n ** 0.5))
        if g * g == n:
            return sal_np.reshape(g, g)
        # fallback: use coarse grid
        result2 = clip_sal.compute_clip_salience(
            image, noun, _GRID_N, _GRID_N,
            top_k_pct=0.30,
            coarse_n=_GRID_N,
            clip_model_name=_CLIP_MODEL,
        )
        sal2 = result2.saliency.cpu().float().numpy()
        return sal2.reshape(_GRID_N, _GRID_N)
    except Exception as e:
        print(f"    [WARN] CLIP failed for noun='{noun}': {e}")
        return None


def overlay_heatmap(ax, image: Image.Image, sal: Optional[np.ndarray],
                    noun: str, skipped: bool, present: bool = True) -> None:
    ax.imshow(image)
    if skipped or sal is None:
        ax.text(0.5, 0.5, "SRF SKIPPED\n(bad noun)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="white", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="red", alpha=0.7))
    else:
        h, w = image.size[1], image.size[0]
        sal_up = np.array(Image.fromarray(sal).resize((w, h), Image.BILINEAR))
        sal_up = (sal_up - sal_up.min()) / (sal_up.max() - sal_up.min() + 1e-8)
        ax.imshow(sal_up, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        color  = "lime" if present else "orange"
        status = "PRESENT" if present else "ABSENT"
        ax.text(0.02, 0.02, f'"{noun}" → {status}',
                transform=ax.transAxes, fontsize=8, color=color,
                bbox=dict(facecolor="black", alpha=0.6, pad=2))
    ax.axis("off")


def make_figure(samples: List[Dict], title: str, out_path: str) -> None:
    n   = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, s in enumerate(samples):
        image   = s["image"]
        noun    = s["noun"]
        skipped = s["skipped"]
        sal     = s.get("saliency")
        present = s.get("present", True)

        # Col 0: image + text
        axes[i][0].imshow(image)
        axes[i][0].axis("off")
        q_short = s["question"][:120] + ("…" if len(s["question"]) > 120 else "")
        axes[i][0].set_title(
            f"[{s['category']}]\n{q_short}\nGT={s['gt']}  pred={s.get('pred','?')}",
            fontsize=7, loc="left", pad=3,
        )

        # Col 1: saliency overlay
        overlay_heatmap(axes[i][1], image, sal, noun, skipped, present)
        noun_label = f"noun: \"{noun}\"" + (" ❌skip" if skipped else " ✓")
        axes[i][1].set_title(noun_label, fontsize=8, pad=3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_mme_samples(n_per_cat: int = 3, seed: int = 42) -> Dict[str, List[Dict]]:
    from datasets import load_dataset
    ds  = load_dataset("lmms-lab/MME", split="test")
    rng = random.Random(seed)

    by_cat: Dict[str, list] = defaultdict(list)
    for r in ds:
        ans = str(r.get("answer", "")).strip().capitalize()
        if ans in ("Yes", "No"):
            by_cat[r["category"]].append(r)

    result = {}
    for cat, rows in sorted(by_cat.items()):
        rng.shuffle(rows)
        selected = rows[:n_per_cat]
        result[cat] = []
        for r in selected:
            noun    = extract_clip_noun(r["question"], mode="pope")
            skipped = noun_is_bad(noun)
            sal     = None
            present = True
            if not skipped:
                sal_result = None
                try:
                    w, h     = r["image"].size
                    grid_h   = max(int(round((h / 28) / _SPATIAL)), 1)
                    grid_w   = max(int(round((w / 28) / _SPATIAL)), 1)
                    sal_result = clip_sal.compute_clip_salience_full_gate_v3(
                        r["image"].convert("RGB"), noun, grid_h, grid_w,
                        top_k_pct=0.30, clip_model_name=_CLIP_MODEL, backup="none",
                    )
                    sal     = sal_result.saliency.cpu().float().numpy()
                    present = sal_result.object_present
                    g = int(round(len(sal) ** 0.5))
                    if g * g == len(sal):
                        sal = sal.reshape(g, g)
                    else:
                        sal = sal.reshape(_GRID_N, _GRID_N) if len(sal) == _GRID_N * _GRID_N else None
                except Exception as e:
                    print(f"    [WARN] {cat} CLIP error: {e}")
            result[cat].append({
                "image":    r["image"].convert("RGB"),
                "question": r["question"],
                "gt":       str(r.get("answer", "")).capitalize(),
                "pred":     "",
                "category": cat,
                "noun":     noun,
                "skipped":  skipped,
                "saliency": sal,
                "present":  present,
            })
    return result


def load_hb_samples(n_per_cat: int = 3, seed: int = 42) -> Dict[str, List[Dict]]:
    from datasets import load_dataset
    ds  = load_dataset("lmms-lab/HallusionBench", split="image")
    rng = random.Random(seed)

    by_cat: Dict[str, list] = defaultdict(list)
    for r in ds:
        if r.get("image") is not None:
            by_cat[r["subcategory"]].append(r)

    result = {}
    for cat, rows in sorted(by_cat.items()):
        rng.shuffle(rows)
        selected = rows[:n_per_cat]
        result[cat] = []
        for r in selected:
            q       = str(r["question"]).strip()
            noun    = extract_clip_noun(q, mode="mmbench")
            skipped = noun_is_bad(noun)
            sal     = None
            present = True
            if not skipped:
                try:
                    img  = r["image"].convert("RGB")
                    w, h = img.size
                    grid_h = max(int(round((h / 28) / _SPATIAL)), 1)
                    grid_w = max(int(round((w / 28) / _SPATIAL)), 1)
                    sal_result = clip_sal.compute_clip_salience_full_gate_v3(
                        img, noun, grid_h, grid_w,
                        top_k_pct=0.30, clip_model_name=_CLIP_MODEL, backup="none",
                    )
                    sal     = sal_result.saliency.cpu().float().numpy()
                    present = sal_result.object_present
                    g = int(round(len(sal) ** 0.5))
                    sal = sal.reshape(g, g) if g * g == len(sal) else None
                except Exception as e:
                    print(f"    [WARN] {cat} CLIP error: {e}")
            gt_raw  = str(r.get("gt_answer", "")).strip()
            gt      = "Yes" if gt_raw == "1" else ("No" if gt_raw == "0" else gt_raw)
            result[cat].append({
                "image":    r["image"].convert("RGB"),
                "question": q,
                "gt":       gt,
                "pred":     "",
                "category": cat,
                "noun":     noun,
                "skipped":  skipped,
                "saliency": sal,
                "present":  present,
            })
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mme",       action="store_true", help="MME only")
    parser.add_argument("--hb",        action="store_true", help="HallusionBench only")
    parser.add_argument("--n",         type=int, default=3, help="Samples per category")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out_dir",   default=None)
    args = parser.parse_args()

    do_mme = args.mme or (not args.mme and not args.hb)
    do_hb  = args.hb  or (not args.mme and not args.hb)

    out_root = args.out_dir or os.path.join(_REPO, "results", "saliency_vis_datasets")

    if do_mme:
        print("\n── MME saliency visualizations ──")
        mme_samples = load_mme_samples(n_per_cat=args.n, seed=args.seed)
        for cat, samples in mme_samples.items():
            n_skip = sum(1 for s in samples if s["skipped"])
            n_good = len(samples) - n_skip
            print(f"  [{cat:28s}]  good={n_good}  skipped={n_skip}")
            out_path = os.path.join(out_root, "mme", f"{cat}.png")
            make_figure(samples, f"MME — {cat}", out_path)

        # Summary figure: one example per category
        summary_samples = [v[0] for v in mme_samples.values() if v]
        make_figure(
            summary_samples,
            "MME — one sample per category (saliency gate overview)",
            os.path.join(out_root, "mme", "_summary.png"),
        )

    if do_hb:
        print("\n── HallusionBench saliency visualizations ──")
        hb_samples = load_hb_samples(n_per_cat=args.n, seed=args.seed)
        for cat, samples in hb_samples.items():
            n_skip = sum(1 for s in samples if s["skipped"])
            n_good = len(samples) - n_skip
            print(f"  [{cat:20s}]  good={n_good}  skipped={n_skip}")
            out_path = os.path.join(out_root, "hallusionbench", f"{cat}.png")
            make_figure(samples, f"HallusionBench — {cat}", out_path)

        summary_samples = [v[0] for v in hb_samples.values() if v]
        make_figure(
            summary_samples,
            "HallusionBench — one sample per subcategory (saliency gate overview)",
            os.path.join(out_root, "hallusionbench", "_summary.png"),
        )

    print(f"\nAll visualizations saved to {out_root}/")


if __name__ == "__main__":
    main()
