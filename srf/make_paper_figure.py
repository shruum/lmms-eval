#!/usr/bin/env python3
"""
make_paper_figure.py — stitch the per-model component rows into the single
figure used in the paper.

Produces  PAPER/ICLR/images/srf_components_combined.png

Inputs are the two per-model rows written by srf/visualize_components.py:

  top     images/srf_components_bowl.png    Qwen2.5-VL-3B, POPE question 299
  bottom  images/srf_components_llava.png   LLaVA-1.5-7B, produced on the
                                            cluster, LLaVA weights do not fit
                                            on the local 11 GB card

Both rows must be generated with --no_cbar, because this script supplies one
shared vertical colourbar rather than one per row.

What this script does that the per-row script cannot:
  - crops the bottom row to its panels, so the (a) to (d) titles appear once
  - draws the bottom row's question, which is cropped away with its titles
  - adds the rotated model labels on the left
  - adds a single shared colourbar, centred vertically on the right

Usage
-----
  python srf/make_paper_figure.py
  python srf/make_paper_figure.py --top other_top.png --bottom other_bot.png
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from PIL import Image, ImageDraw, ImageFont

IMG_DIR = pathlib.Path("/volumes2/mllm/PAPER/ICLR/images")
FONT    = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# Shared colour scale. Both rows must be drawn with these same limits.
VMIN, VMAX = 0.0, 0.85
BOTTOM_Q = '"Which company owns the airplane displayed in the back of the image?"'


def trim(im, tol=248):
    a = np.asarray(im.convert("RGB"))
    m = (a < tol).any(axis=2)
    r, c = np.where(m.any(axis=1))[0], np.where(m.any(axis=0))[0]
    return im.crop((c[0], r[0], c[-1] + 1, r[-1] + 1))


def panels_only(im, right=3078):
    """Keep the panels and their axes, drop the header and the panel titles.

    Panel rows are found from the far-left strip, where only panel (a)'s photo
    lives, so centred title text cannot trigger it. `right` drops the per-row
    colourbar strip if one is present.
    """
    a = np.asarray(im.convert("RGB"))
    H, W = a.shape[:2]
    ink = (a < 248).any(axis=2)
    top = int(np.argmax(ink[:, : int(W * 0.10)].mean(axis=1) > 0.80))
    bot = int(np.where(ink.mean(axis=1) > 0.015)[0][-1]) + 1
    return trim(im.crop((0, top, min(right, W), bot)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top",    default="srf_components_bowl.png")
    ap.add_argument("--bottom", default="srf_components_llava.png")
    ap.add_argument("--out",    default="srf_components_combined.png")
    a = ap.parse_args()

    top = trim(Image.open(IMG_DIR / a.top).convert("RGB"))
    bot = panels_only(Image.open(IMG_DIR / a.bottom).convert("RGB"))

    W = max(top.size[0], bot.size[0])
    top = top.resize((W, round(top.size[1] * W / top.size[0])), Image.LANCZOS)
    bot = bot.resize((W, round(bot.size[1] * W / bot.size[0])), Image.LANCZOS)

    LABW, PAD, ROWGAP, QGAP, CBGAP = 132, 18, 46, 6, 30
    fq = ImageFont.truetype(FONT, 52)
    fl = ImageFont.truetype(FONT, 62)          # model labels
    pr = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    bb = pr.textbbox((0, 0), BOTTOM_Q, font=fq)
    qh = bb[3] - bb[1]

    bandH = top.size[1] + ROWGAP + bot.size[1] + QGAP + qh
    cbH = round(bandH * 0.60)
    fig, ax = plt.subplots(figsize=(0.26, cbH / 200))
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(VMIN, VMAX),
                                        cmap="viridis"),
                      cax=ax, orientation="vertical")
    cb.set_label("vision attention ratio", fontsize=15)
    ax.tick_params(labelsize=14)
    fig.savefig("/tmp/_cbv.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    cbar = trim(Image.open("/tmp/_cbv.png").convert("RGB"))
    cbar = cbar.resize((round(cbar.size[0] * cbH / cbar.size[1]), cbH), Image.LANCZOS)

    cv = Image.new("RGB", (LABW + W + CBGAP + cbar.size[0] + 2 * PAD,
                           bandH + 2 * PAD), "white")
    d = ImageDraw.Draw(cv)
    y = PAD
    for i, (im_, lab) in enumerate(zip((top, bot),
                                       ("Qwen2.5-VL-3B", "LLaVA-1.5-7B"))):
        cv.paste(im_, (LABW + PAD, y))
        s = Image.new("RGB", (im_.size[1], LABW), "white")
        ds = ImageDraw.Draw(s)
        tw = ds.textbbox((0, 0), lab, font=fl)[2]
        ds.text(((im_.size[1] - tw) // 2, 34), lab, fill=(0, 0, 0), font=fl)
        cv.paste(s.rotate(90, expand=True), (PAD, y))
        y += im_.size[1]
        if i == 1:
            d.text((LABW + PAD, y + QGAP - bb[1]), BOTTOM_Q, fill=(0, 0, 0), font=fq)
        y += ROWGAP
    cv.paste(cbar, (LABW + PAD + W + CBGAP, PAD + (bandH - cbH) // 2))

    out = IMG_DIR / a.out
    cv.save(out, dpi=(200, 200))
    print(f"  MERGED -> {out}  {cv.size[0]}x{cv.size[1]}")


if __name__ == "__main__":
    main()
