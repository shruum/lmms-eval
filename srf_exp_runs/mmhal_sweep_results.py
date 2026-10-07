#!/usr/bin/env python3
"""Aggregate and print all mmhal_sweep results vs baseline."""
import json, pathlib

BASE = pathlib.Path("results/mmhal_sweep")
BASELINE_SCORE = 2.28125
BASELINE_HAL   = 60.4167

GROUPS = {
    "sigma":    ["sig5","sig10","sig15","sig20","sig25","sig30"],
    "alpha":    ["sig20","a0p3","a1p0","a2p0"],
    "k/heads":  ["sig20","k10","k50","k100"],
    "layer_end":["sig20","le16","le24"],
    "phase":    ["sig20","ph_gen"],
}

LABELS = {
    "sig5":   "σ=5",   "sig10":  "σ=10",  "sig15": "σ=15",
    "sig20":  "σ=20 (ref)", "sig25":"σ=25", "sig30":"σ=30",
    "a0p3":   "α=0.3", "a1p0":  "α=1.0", "a2p0": "α=2.0",
    "k10":    "k=10%", "k50":   "k=50%", "k100": "k=100% (all heads)",
    "le16":   "le=16", "le24":  "le=24",
    "ph_gen": "phase=gen",
}

def load(tag):
    sf = BASE / tag / "mmhalbench_scores.json"
    if not sf.exists():
        return None, None, None
    d = json.loads(sf.read_text())
    return d.get("score"), d.get("hal_pct"), d.get("n")

def print_group(name, tags):
    print(f"\n── {name} ──────────────────────────────────────────")
    print(f"  {'Config':<22} {'Score':>6} {'Hal%':>7}  {'Δscore':>7} {'Δhal':>7}  {'n':>4}")
    print(f"  {'Baseline':<22}  {BASELINE_SCORE:.2f}   {BASELINE_HAL:.1f}%      —       —    96")
    for tag in tags:
        s, h, n = load(tag)
        label = LABELS.get(tag, tag)
        if s is None:
            print(f"  {label:<22}  running…")
            continue
        ds = s - BASELINE_SCORE
        dh = h - BASELINE_HAL
        mark = " ← BEST" if s == max(load(t)[0] or 0 for t in tags if load(t)[0]) else ""
        print(f"  {label:<22} {s:6.2f}  {h:7.1f}%  {ds:+7.2f} {dh:+7.1f}pp  {n:3d}{mark}")

print("=" * 65)
print("MMHal-Bench sweep — srffovea, naa=0.0, n=96, gpt-4o")
print("Fixed: layers=8-[le], llava_boost_mode=additive, clip_thresh=0.25/0.27")
print("=" * 65)

for gname, tags in GROUPS.items():
    print_group(gname, tags)

print()
