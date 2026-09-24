#!/usr/bin/env python3
"""
tune_pope_gate.py — find the ceiling of the CLIP presence gate on POPE, and
the thresholds that reach it, without running the VLM.

Why this exists
---------------
On POPE, SRF uses the CLIP gate as an auxiliary object detector. When the gate
fires the decoder amplifies image attention, and when it rejects the decoder
attenuates it by neg_absent_alpha, which on POPE is 2.0. So SRF pushes the VLM
toward the gate's verdict. If the gate is less accurate than the VLM already
is, SRF can only hurt. That is consistent with the published table, where the
adversarial split gains 0.96 while popular and random lose 0.17 and 0.07 and
the model is already at 87 to 88 percent there.

This measures the gate directly against the POPE labels, on CPU, with no VLM.

The cheap part
--------------
The gate is

    object_present = (full_img_sim >= tau) or (patch_max_sim >= patch_thresh)

Both signals are computed once per image and cached, so sweeping hundreds of
threshold pairs afterwards is pure arithmetic. One CLIP pass buys the whole
grid.

Reuses eval_presence.load_balanced_pope for the sample set,
noun_extract.extract_clip_noun for the noun and
clip_salience.compute_clip_salience_full_gate_v3 for the signals, so what is
tuned is what the pipeline runs.

Usage
-----
  python srf/tune_pope_gate.py --n_per_cell 25
  python srf/tune_pope_gate.py --n_per_cell 25 --scale_combine prod
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

_SRF = pathlib.Path(__file__).parent
sys.path.insert(0, str(_SRF / "saliency"))
sys.path.insert(0, str(_SRF))
sys.path.insert(0, str(_SRF.parent / "my_analysis"))

import config as CFG
os.environ.setdefault("HF_HOME", CFG.HF_HOME)

import clip_salience as clip_sal
from eval_presence import load_balanced_pope
from noun_extract import extract_clip_noun
from transformers import AutoProcessor

SPLITS = ["adversarial", "popular", "random"]
# Published VLM accuracy, for reference when reading the gate ceiling.
VLM_BASELINE = {"adversarial": 86.37, "popular": 87.57, "random": 88.50}


def collect(n_per_cell, seed, scale_combine):
    proc = AutoProcessor.from_pretrained(CFG.DEFAULT_MODEL,
                                         max_pixels=CFG.DEFAULT_MAX_PIXELS)
    samples = load_balanced_pope(n_per_cell, seed, SPLITS)
    recs = []
    for i, s in enumerate(samples):
        noun = extract_clip_noun(s["question"], mode="pope")
        if not noun:
            continue
        inp = proc(text=["x"], images=[s["image"]], return_tensors="pt", padding=True)
        gh, gw = clip_sal.get_grid_dims(inp, 2)
        r = clip_sal.compute_clip_salience_full_gate_v3(
            s["image"], noun, gh, gw, scale_combine=scale_combine)
        recs.append(dict(split=s["split"], yes=(s["gt"] == "Yes"),
                         noun=noun, full=float(r.full_img_sim),
                         patch=float(r.max_sim)))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(samples)}")
    return recs


def score(recs, tau, pth):
    """Accuracy and F1 of the gate verdict against the POPE label."""
    out = {}
    for sp in SPLITS + ["ALL"]:
        rs = recs if sp == "ALL" else [r for r in recs if r["split"] == sp]
        if not rs:
            continue
        pred = np.array([(r["full"] >= tau) or (r["patch"] >= pth) for r in rs])
        gt = np.array([r["yes"] for r in rs])
        tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[sp] = dict(acc=100.0 * (pred == gt).mean(),
                       f1=100.0 * (2 * prec * rec / (prec + rec) if prec + rec else 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_cell", type=int, default=25,
                    help="Samples per (split, yes/no) cell. 25 gives 150 total.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale_combine", default="max",
                    choices=["max", "prod", "min", "mean"])
    ap.add_argument("--out", default="results/pope_gate/")
    args = ap.parse_args()

    recs = collect(args.n_per_cell, args.seed, args.scale_combine)
    print(f"\n  {len(recs)} samples, scale_combine={args.scale_combine}")

    cur = score(recs, 0.20, 0.27)
    print("\n  SHIPPED GATE  tau=0.20  patch_thresh=0.27")
    print(f"  {'split':<14}{'gate acc':>10}{'gate F1':>10}{'VLM baseline':>15}")
    for sp in SPLITS:
        if sp in cur:
            print(f"  {sp:<14}{cur[sp]['acc']:>10.2f}{cur[sp]['f1']:>10.2f}"
                  f"{VLM_BASELINE[sp]:>15.2f}")
    print(f"  {'ALL':<14}{cur['ALL']['acc']:>10.2f}{cur['ALL']['f1']:>10.2f}")

    taus = np.round(np.arange(0.14, 0.34, 0.01), 3)
    pths = np.concatenate([np.round(np.arange(0.18, 0.40, 0.01), 3), [9.0]])
    best = max(((t, p, score(recs, t, p)) for t in taus for p in pths),
               key=lambda x: x[2]["ALL"]["acc"])
    t, p, m = best
    lbl = "disabled" if p > 1 else f"{p:.2f}"
    print(f"\n  BEST ON THIS GRID  tau={t:.2f}  patch_thresh={lbl}")
    print(f"  {'split':<14}{'gate acc':>10}{'gate F1':>10}{'VLM baseline':>15}")
    for sp in SPLITS:
        if sp in m:
            print(f"  {sp:<14}{m[sp]['acc']:>10.2f}{m[sp]['f1']:>10.2f}"
                  f"{VLM_BASELINE[sp]:>15.2f}")
    print(f"  {'ALL':<14}{m['ALL']['acc']:>10.2f}{m['ALL']['f1']:>10.2f}")

    print("\n  Read this as a ceiling. The gate can only help a split where its")
    print("  accuracy exceeds what the VLM already achieves there.")

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    f = out / f"gate_{args.scale_combine}.json"
    f.write_text(json.dumps(dict(scale_combine=args.scale_combine, n=len(recs),
                                 shipped=cur, best_tau=float(t),
                                 best_patch_thresh=float(p), best=m,
                                 records=recs), indent=2))
    print(f"\n  saved -> {f}")


if __name__ == "__main__":
    main()
