#!/usr/bin/env python3
"""
significance.py — paired bootstrap and McNemar over saved MMVP per-pair records.

MMVP pair accuracy is a mean over 150 paired Bernoulli outcomes, so a
difference of one pair is 0.67 points and most differences this project has
measured are two to four pairs. Point estimates alone cannot separate those
from noise. This computes what can be claimed.

Reads the "records" block written by eval.run_mmvp under --save_records, so it
consumes the output of the real evaluation rather than recomputing anything.

  paired bootstrap   resample the 150 pair ids with replacement, recompute both
                     configurations on the SAME resample, and take the
                     difference. Pairing removes the between-sample variance
                     that dominates an unpaired comparison.
  McNemar            exact binomial test on the discordant pairs, the pairs one
                     configuration gets right and the other gets wrong. This is
                     the correct test for two methods on identical items.

Usage
-----
  python srf/significance.py results/ci_baseline results/ci_anchor results/ci_sem05 \
      --labels baseline srf_anchor srf_sem05
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np
from scipy import stats


def load_pairs(result_dir: str) -> Dict[str, bool]:
    """Return {pair_id: both_images_correct} from a run's mmvp.json."""
    path = pathlib.Path(result_dir) / "mmvp.json"
    d    = json.loads(path.read_text())
    if "records" not in d:
        raise SystemExit(f"{path} has no 'records' block. Re-run with --save_records.")
    recs = d["records"]
    gamma = sorted(recs)[0]
    out = {}
    for pid, r in recs[gamma].items():
        if "a" in r and "b" in r:
            out[pid] = bool(r["a"]) and bool(r["b"])
    return out


def boot_ci(mask: np.ndarray, idx: np.ndarray) -> tuple:
    draws = mask[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description="Significance testing for MMVP runs.")
    ap.add_argument("dirs", nargs="+", help="Result directories (first is the reference).")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed",   type=int, default=0)
    args = ap.parse_args()

    labels = args.labels or [pathlib.Path(d).name for d in args.dirs]
    if len(labels) != len(args.dirs):
        raise SystemExit("--labels must match the number of directories.")

    per = [load_pairs(d) for d in args.dirs]
    ids = sorted(set(per[0]).intersection(*[set(p) for p in per[1:]]), key=int)
    n   = len(ids)
    if n == 0:
        raise SystemExit("No pair ids in common.")
    M = np.array([[p[i] for i in ids] for p in per], dtype=bool)   # (n_cfg, n_pairs)

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.n_boot, n))

    print("\n" + "=" * 78)
    print(f"MMVP significance — {n} pairs, {args.n_boot} bootstrap resamples")
    print("=" * 78)
    print(f"\n  {'config':<16}{'pair acc':>10}{'95% CI':>20}")
    for lab, row in zip(labels, M):
        lo, hi = boot_ci(row, idx)
        print(f"  {lab:<16}{100*row.mean():>10.2f}   [{100*lo:>5.2f}, {100*hi:>5.2f}]")

    print(f"\n  paired comparisons against '{labels[0]}'")
    print(f"  {'vs':<16}{'delta':>9}{'95% CI':>20}{'b':>5}{'c':>5}{'McNemar p':>12}")
    a = M[0]
    for lab, b_row in zip(labels[1:], M[1:]):
        d      = b_row.astype(int) - a.astype(int)
        dboot  = d[idx].mean(axis=1)
        lo, hi = np.percentile(dboot, 2.5), np.percentile(dboot, 97.5)
        nb     = int(np.sum(a & ~b_row))     # reference right, other wrong
        nc     = int(np.sum(~a & b_row))     # reference wrong, other right
        p      = stats.binomtest(nc, nb + nc, 0.5).pvalue if (nb + nc) else 1.0
        sig    = "" if (lo <= 0 <= hi) else "  *"
        print(f"  {lab:<16}{100*d.mean():>+9.2f}   [{100*lo:>+6.2f}, {100*hi:>+6.2f}]"
              f"{nb:>5}{nc:>5}{p:>12.4f}{sig}")

    print("\n  b = reference correct, other wrong.  c = reference wrong, other correct.")
    print("  * marks a 95% interval excluding zero. Intervals spanning zero mean the")
    print("  difference is not distinguishable from noise at this sample size.")
    print(f"\n  One pair = {100/n:.2f} points.")


if __name__ == "__main__":
    main()
