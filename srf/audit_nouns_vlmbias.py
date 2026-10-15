#!/usr/bin/env python3
"""
audit_nouns_vlmbias.py — check the extracted noun and the relevance map on a
sample of VLMBias inputs, before committing GPU time to a full run.

Loads the Qwen processor only, so the grid dimensions are exactly what the
real pipeline uses, and runs CLIP on CPU. No VLM weights, no GPU.

Reuses noun_extract.extract_clip_noun and
clip_salience.compute_clip_salience_full_gate_v3, so what is audited is what
the evaluation actually does.

Reports per sample: the noun, the full-image CLIP similarity against tau, the
gate verdict, and two map-quality numbers.

  peak/mean   ratio of the top decile of patch scores to the mean. A map that
              localises something has a high ratio. A diffuse map sits near 1.
  top-10 area fraction of the image covered by the top decile, for reference.

Also writes a contact sheet so the maps can be eyeballed.

Usage
-----
  python srf/audit_nouns_vlmbias.py --per_topic 2
  python srf/audit_nouns_vlmbias.py --topics Logos Animals --per_topic 3
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

TAU = 0.20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="*", default=None)
    ap.add_argument("--per_topic", type=int, default=2)
    ap.add_argument("--out", default="results/noun_audit/")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    proc = AutoProcessor.from_pretrained(CFG.DEFAULT_MODEL,
                                         max_pixels=CFG.DEFAULT_MAX_PIXELS)
    ds = load_dataset("anvo25/vlms-are-biased", split="main")

    topics = args.topics or ["Animals", "Chess Pieces", "Flags", "Game Boards",
                             "Logos", "Optical Illusion", "Patterned Grid"]
    picks = []
    for t in topics:
        rows = [r for r in ds if str(r["topic"]) == t]
        seen = set()
        for r in sorted(rows, key=lambda r: -r["image"].size[0]):
            key = str(r["prompt"])[:70]
            if key in seen:
                continue
            seen.add(key)
            picks.append((t, r))
            if len([p for p in picks if p[0] == t]) >= args.per_topic:
                break

    print(f"\n{'topic':<18}{'noun':<30}{'sim':>7}{'gate':>9}{'peak/mean':>11}{'top10%':>9}")
    print("-" * 84)

    n = len(picks)
    fig, axes = plt.subplots(n, 2, figsize=(7.5, 3.1 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (topic, r) in enumerate(picks):
        image = r["image"].convert("RGB")
        prompt = str(r["prompt"])
        noun = extract_clip_noun(prompt, mode="vlmbias")

        inp = proc(text=["x"], images=[image], return_tensors="pt", padding=True)
        gh, gw = clip_sal.get_grid_dims(inp, 2)

        res = clip_sal.compute_clip_salience_full_gate_v3(image, noun, gh, gw)
        sal = res.saliency.float().cpu().numpy()
        k = max(1, int(0.10 * sal.size))
        peak = float(np.sort(sal)[-k:].mean() / (sal.mean() + 1e-9))
        gate = "fired" if res.object_present else "rejected"
        print(f"{topic:<18}{noun[:29]:<30}{res.full_img_sim:>7.3f}{gate:>9}"
              f"{peak:>11.2f}{100.0 * k / sal.size:>8.0f}%")

        t2 = torch.tensor(sal).reshape(1, 1, gh, gw)
        W, H = image.size
        up = torch.nn.functional.interpolate(t2, size=(H, W), mode="bilinear",
                                             align_corners=False)[0, 0].numpy()
        axes[i][0].imshow(image); axes[i][0].axis("off")
        axes[i][0].set_title(f"{topic}", fontsize=8)
        axes[i][1].imshow(image)
        axes[i][1].imshow(up, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[i][1].axis("off")
        axes[i][1].set_title(f"noun={noun!r}  sim={res.full_img_sim:.3f}  {gate}",
                             fontsize=8)

    p = out / "noun_audit_vlmbias.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110, bbox_inches="tight")
    print(f"\n  contact sheet -> {p}")
    print(f"  tau = {TAU}.  peak/mean near 1 means the map is diffuse and "
          f"localises nothing.")


if __name__ == "__main__":
    main()
