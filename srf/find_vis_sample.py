#!/usr/bin/env python3
"""
find_vis_sample.py — rank POPE samples by how well the relevance map localises,
so a figure can be chosen on evidence instead of by guessing.

CPU only. Loads the Qwen processor for exact grid dimensions and runs CLIP on
CPU. No VLM weights, no GPU, so it can run alongside an evaluation.

Reuses noun_extract.extract_clip_noun and
clip_salience.compute_clip_salience_full_gate_v3, so the maps scored here are
the maps the pipeline would actually use.

Ranking criteria, all of which a good figure needs.
  gate fired        otherwise panels (b) and (c) are empty
  peak/mean high    the map concentrates somewhere rather than spreading out
  peak area small   the sharp region is a minority of the image, so the blur
                    in the foveated panel is visible
  background busy   high-frequency detail outside the peak, measured as the
                    mean gradient magnitude, so blurring is visible at all

Usage
-----
  python srf/find_vis_sample.py --n 80 --top 6
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_SRF = pathlib.Path(__file__).parent
sys.path.insert(0, str(_SRF / "saliency"))
sys.path.insert(0, str(_SRF))
sys.path.insert(0, str(_SRF.parent / "my_analysis"))

import config as CFG
os.environ.setdefault("HF_HOME", CFG.HF_HOME)

import clip_salience as clip_sal
from datasets import load_dataset
from noun_extract import extract_clip_noun
from transformers import AutoProcessor


def busyness(img) -> float:
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(g)
    return float(np.hypot(gx, gy).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="Samples to scan.")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--split", default="adversarial")
    ap.add_argument("--out", default="results/vis_candidates/")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    proc = AutoProcessor.from_pretrained(CFG.DEFAULT_MODEL,
                                         max_pixels=CFG.DEFAULT_MAX_PIXELS)
    ds = load_dataset("lmms-lab/POPE", split="test")

    rows = [r for r in ds
            if str(r.get("category", "")).lower() == args.split
            and str(r.get("answer", "")).lower() == "yes"][:args.n]

    scored = []
    for i, r in enumerate(rows):
        image = r["image"].convert("RGB")
        noun = extract_clip_noun(str(r["question"]), mode="pope")
        if not noun:
            continue
        inp = proc(text=["x"], images=[image], return_tensors="pt", padding=True)
        gh, gw = clip_sal.get_grid_dims(inp, 2)
        res = clip_sal.compute_clip_salience_full_gate_v3(image, noun, gh, gw)
        if not res.object_present:
            continue
        sal = res.saliency.float().cpu().numpy()
        k = max(1, int(0.15 * sal.size))
        peak = float(np.sort(sal)[-k:].mean() / (sal.mean() + 1e-9))
        scored.append(dict(qid=r["question_id"], noun=noun, sim=res.full_img_sim,
                           peak=peak, busy=busyness(image), image=image,
                           q=str(r["question"]), sal=sal, gh=gh, gw=gw))
        if (i + 1) % 20 == 0:
            print(f"  scanned {i + 1}/{len(rows)}, kept {len(scored)}")

    # Rank by concentration first, then by how much texture there is to blur.
    scored.sort(key=lambda d: (-d["peak"], -d["busy"]))
    print(f"\n  {'qid':>7}{'noun':>16}{'sim':>8}{'peak/mean':>11}{'busy':>8}")
    for d in scored[:args.top]:
        print(f"  {d['qid']:>7}{d['noun'][:15]:>16}{d['sim']:>8.3f}"
              f"{d['peak']:>11.2f}{d['busy']:>8.3f}")

    n = min(args.top, len(scored))
    fig, axes = plt.subplots(n, 2, figsize=(7.5, 3.1 * n))
    if n == 1:
        axes = axes[None, :]
    for i, d in enumerate(scored[:n]):
        W, H = d["image"].size
        up = torch.nn.functional.interpolate(
            torch.tensor(d["sal"]).reshape(1, 1, d["gh"], d["gw"]),
            size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
        axes[i][0].imshow(d["image"]); axes[i][0].axis("off")
        axes[i][0].set_title(f"qid={d['qid']}  {d['q'][:44]}", fontsize=7)
        axes[i][1].imshow(d["image"])
        axes[i][1].imshow(up, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[i][1].axis("off")
        axes[i][1].set_title(f"noun={d['noun']!r}  peak/mean={d['peak']:.2f}", fontsize=7)
    p = out / "vis_candidates.png"
    fig.tight_layout(); fig.savefig(p, dpi=110, bbox_inches="tight")
    print(f"\n  contact sheet -> {p}")


if __name__ == "__main__":
    main()
