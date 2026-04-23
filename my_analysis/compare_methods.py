#!/usr/bin/env python3
"""
Compare multiple methods against baseline.

Generates two plots per dataset:
  1. Bar chart — best value per method (avg accuracy)
  2. Sweep chart — accuracy vs. value for each method (overlay)

Usage:
  python compare_methods.py \
      --baseline  ../results/qwen_all/base/results.json \
      --methods   ../results/qwen_all/temp/results.json \
                  ../results/qwen_all/visboost/results.json \
                  ../results/qwen_all/vhr_v2/results.json \
                  ../results/qwen_all/vaf/results.json \
      --output_dir ../results/qwen_all/compare
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["vlm_bias", "pope", "mmbench"]   # overridden at runtime by detect_datasets()
DATASET_LABELS = {"vlm_bias": "VLM Bias", "pope": "POPE", "mmbench": "MMBench",
                  "cv_bench": "CV-Bench Relation"}


def detect_datasets(baseline_records: List[Dict], method_records: List) -> None:
    """Replace DATASETS global with datasets actually present in the results."""
    global DATASETS
    seen = set()
    for r in baseline_records:
        seen.add(r.get("dataset", ""))
    for _, recs in method_records:
        for r in recs:
            seen.add(r.get("dataset", ""))
    seen.discard("")
    # Preserve canonical order for known datasets; append unknown ones
    canonical = ["vlm_bias", "pope", "mmbench", "cv_bench"]
    DATASETS = [d for d in canonical if d in seen] + sorted(seen - set(canonical))


def load_results(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def accuracy_by(records: List[Dict], keys: Tuple[str, ...]) -> Dict:
    """Group records by `keys` and return {key_tuple: accuracy}."""
    buckets: Dict = defaultdict(lambda: {"c": 0, "t": 0})
    for r in records:
        k = tuple(r.get(key) for key in keys)
        buckets[k]["c"] += int(r["correct"])
        buckets[k]["t"] += 1
    return {k: v["c"] / v["t"] for k, v in buckets.items() if v["t"] > 0}


def dataset_overall(records: List[Dict], dataset: str) -> Dict[float, float]:
    """Return {value: overall_accuracy} for a given dataset."""
    out: Dict = defaultdict(lambda: {"c": 0, "t": 0})
    for r in records:
        if r.get("dataset") == dataset:
            v = r.get("value", 1.0)
            out[v]["c"] += int(r["correct"])
            out[v]["t"] += 1
    return {v: d["c"] / d["t"] for v, d in sorted(out.items()) if d["t"] > 0}


def best_value_acc(records: List[Dict], dataset: str) -> float:
    """Best (peak) overall accuracy across all values for a dataset."""
    vals = dataset_overall(records, dataset)
    return max(vals.values()) if vals else float("nan")


def method_label(path: str) -> str:
    method_vals = set()
    data = load_results(path)
    for r in data:
        method_vals.add(r.get("method", "?"))
    return "/".join(sorted(method_vals))


# ---------------------------------------------------------------------------
# Plot 1 — bar chart: baseline vs each method (best value), per dataset
# ---------------------------------------------------------------------------

def plot_comparison_bar(
    baseline_records: List[Dict],
    method_records: List[Tuple[str, List[Dict]]],   # (label, records)
    output_dir: str,
) -> None:
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for ds in DATASETS:
        labels, accs = [], []

        # Baseline (single value)
        base_acc = dataset_overall(baseline_records, ds)
        if not base_acc:
            continue
        labels.append("baseline")
        accs.append(list(base_acc.values())[0])

        for method_label, records in method_records:
            val_acc = dataset_overall(records, ds)
            if val_acc:
                best = max(val_acc.values())
                labels.append(method_label)
                accs.append(best)

        if not labels:
            continue

        colors = ["#888888"] + list(plt.cm.tab10.colors[:len(labels) - 1])  # type: ignore
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
        bars = ax.bar(labels, accs, color=colors, edgecolor="white", linewidth=0.6)
        ax.axhline(accs[0], linestyle="--", color="#888888", linewidth=1.5,
                   label=f"baseline={accs[0]:.3f}")

        for bar, acc in zip(bars, accs):
            delta = acc - accs[0]
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            color = "#27ae60" if delta > 0 else ("#c0392b" if delta < 0 else "#555")
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{acc:.3f}\n({delta_str})",
                    ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

        ax.set_ylim(0, min(1.0, max(accs) + 0.12))
        ax.set_ylabel("Accuracy (best value)")
        ax.set_title(f"{DATASET_LABELS.get(ds, ds)} — method comparison (best value per method)")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        save = os.path.join(plot_dir, f"{ds}_comparison_bar.png")
        fig.savefig(save, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save}")


# ---------------------------------------------------------------------------
# Plot 2 — sweep lines: accuracy vs value, all methods on one plot per dataset
# ---------------------------------------------------------------------------

def plot_comparison_sweep(
    baseline_records: List[Dict],
    method_records: List[Tuple[str, List[Dict]]],
    output_dir: str,
) -> None:
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    colors = list(plt.cm.tab10.colors)  # type: ignore

    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Baseline — flat horizontal line
        base_acc = dataset_overall(baseline_records, ds)
        if base_acc:
            base_val = list(base_acc.values())[0]
            ax.axhline(base_val, color="#444444", linewidth=2.5, linestyle="--",
                       label=f"baseline ({base_val:.3f})", zorder=5)

        for ci, (label, records) in enumerate(method_records):
            val_acc = dataset_overall(records, ds)
            if len(val_acc) < 1:
                continue
            xs = sorted(val_acc.keys())
            ys = [val_acc[x] for x in xs]
            marker = "o" if len(xs) > 1 else "D"
            ax.plot(xs, ys, marker=marker, linewidth=2.0, markersize=7,
                    color=colors[ci % len(colors)], label=label, zorder=4)

        ax.set_xlabel("Intervention value (alpha)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{DATASET_LABELS.get(ds, ds)} — accuracy vs. intervention value")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save = os.path.join(plot_dir, f"{ds}_comparison_sweep.png")
        fig.savefig(save, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save}")


# ---------------------------------------------------------------------------
# Plot 3 — overall summary: all methods × all datasets, grouped bars
# ---------------------------------------------------------------------------

def plot_overall_summary(
    baseline_records: List[Dict],
    method_records: List[Tuple[str, List[Dict]]],
    output_dir: str,
) -> None:
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    all_labels = ["baseline"] + [lbl for lbl, _ in method_records]
    all_recs   = [baseline_records] + [r for _, r in method_records]

    # accs[label][dataset] = best accuracy
    accs: Dict[str, Dict[str, float]] = {}
    for label, records in zip(all_labels, all_recs):
        accs[label] = {}
        for ds in DATASETS:
            va = dataset_overall(records, ds)
            accs[label][ds] = (max(va.values()) if va else float("nan"))

    n_methods = len(all_labels)
    n_ds      = len(DATASETS)
    x         = np.arange(n_ds)
    width     = 0.8 / n_methods
    colors    = ["#888888"] + list(plt.cm.tab10.colors[:n_methods - 1])  # type: ignore

    fig, ax = plt.subplots(figsize=(max(8, n_methods * 2), 5))
    for i, (label, color) in enumerate(zip(all_labels, colors)):
        vals = [accs[label].get(ds, float("nan")) for ds in DATASETS]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="white", linewidth=0.5)
        base_vals = [accs["baseline"].get(ds, 0) for ds in DATASETS]
        for bar, val, bval in zip(bars, vals, base_vals):
            if np.isnan(val):
                continue
            delta = val - bval
            if label != "baseline" and abs(delta) > 0.001:
                clr = "#27ae60" if delta > 0 else "#c0392b"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{'+'if delta>0 else ''}{delta:.2f}",
                        ha="center", va="bottom", fontsize=6, color=clr, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=11)
    ax.set_ylabel("Accuracy (best value per method)")
    ax.set_title("Overall comparison — all methods across all datasets")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save = os.path.join(plot_dir, "overall_summary.png")
    fig.savefig(save, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save}")


# ---------------------------------------------------------------------------
# Plot 4 — category comparison: per-group accuracy + delta, per dataset
# ---------------------------------------------------------------------------

def best_value_for_dataset(records: List[Dict], dataset: str) -> float:
    """Return the sweep value that achieves best overall accuracy for a dataset."""
    va = dataset_overall(records, dataset)
    if not va:
        return float("nan")
    return max(va, key=lambda v: va[v])


def group_accuracy(records: List[Dict], dataset: str, value: float) -> Dict[str, float]:
    """Return {group: accuracy} for a given dataset and sweep value."""
    buckets: Dict[str, Dict] = defaultdict(lambda: {"c": 0, "t": 0})
    for r in records:
        if r.get("dataset") == dataset and abs(r.get("value", 1.0) - value) < 1e-9:
            g = r.get("group", "unknown")
            buckets[g]["c"] += int(r["correct"])
            buckets[g]["t"] += 1
    return {g: v["c"] / v["t"] for g, v in buckets.items() if v["t"] > 0}


def plot_category_comparison(
    baseline_records: List[Dict],
    method_records: List[Tuple[str, List[Dict]]],
    output_dir: str,
) -> None:
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    colors = list(plt.cm.tab10.colors)  # type: ignore

    for ds in DATASETS:
        # Groups from baseline
        base_val  = best_value_for_dataset(baseline_records, ds)
        base_grp  = group_accuracy(baseline_records, ds, base_val)
        groups    = sorted(base_grp.keys())
        if not groups:
            continue

        n_groups  = len(groups)
        n_methods = len(method_records)
        x         = np.arange(n_groups)
        width     = 0.75 / (n_methods + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n_groups * 1.2), 9), sharex=True)

        # --- top panel: absolute accuracy ---
        base_vals = [base_grp.get(g, float("nan")) for g in groups]
        ax1.bar(x - n_methods / 2 * width, base_vals, width,
                label="baseline", color="#888888", edgecolor="white", linewidth=0.5)

        for ci, (label, records) in enumerate(method_records):
            bv   = best_value_for_dataset(records, ds)
            grp  = group_accuracy(records, ds, bv)
            vals = [grp.get(g, float("nan")) for g in groups]
            offset = (ci + 1 - n_methods / 2) * width
            ax1.bar(x + offset, vals, width,
                    label=label, color=colors[ci % len(colors)], edgecolor="white", linewidth=0.5)

        ax1.set_ylabel("Accuracy")
        ax1.set_title(f"{DATASET_LABELS.get(ds, ds)} — per-category accuracy")
        ax1.set_ylim(0, 1.1)
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(axis="y", alpha=0.3)
        ax1.axhline(np.nanmean(base_vals), color="#888888", linewidth=1.2, linestyle="--", alpha=0.6)

        # --- bottom panel: delta vs baseline ---
        for ci, (label, records) in enumerate(method_records):
            bv     = best_value_for_dataset(records, ds)
            grp    = group_accuracy(records, ds, bv)
            deltas = [grp.get(g, float("nan")) - base_grp.get(g, float("nan")) for g in groups]
            bar_colors = ["#27ae60" if d > 0.001 else ("#c0392b" if d < -0.001 else "#aaaaaa")
                          for d in deltas]
            offset = (ci - n_methods / 2 + 0.5) * width
            ax2.bar(x + offset, deltas, width,
                    label=label, color=bar_colors, edgecolor="white", linewidth=0.5)

        ax2.axhline(0, color="black", linewidth=1.0)
        ax2.set_ylabel("Δ accuracy vs baseline")
        ax2.set_title(f"{DATASET_LABELS.get(ds, ds)} — delta vs baseline (green=↑, red=↓)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(groups, rotation=35, ha="right", fontsize=8)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        save = os.path.join(plot_dir, f"{ds}_category_comparison.png")
        fig.savefig(save, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(
    baseline_records: List[Dict],
    method_records: List[Tuple[str, List[Dict]]],
) -> None:
    ds_headers = "  ".join(f"{DATASET_LABELS.get(ds, ds):>12}" for ds in DATASETS)
    print(f"\n{'Method':<22}  {ds_headers}  {'Avg':>10}")
    print("-" * (24 + 14 * len(DATASETS) + 12))

    def row(label, records):
        accs = []
        for ds in DATASETS:
            va = dataset_overall(records, ds)
            accs.append(max(va.values()) if va else float("nan"))
        avg = np.nanmean(accs)
        cols = [f"{a:.3f}" for a in accs] + [f"{avg:.3f}"]
        print(f"{label:<22}  {'  '.join(f'{c:>12}' for c in cols)}")
        return accs, avg

    base_accs, base_avg = row("baseline", baseline_records)

    for label, records in method_records:
        accs, avg = row(label, records)
        deltas = [a - b for a, b in zip(accs, base_accs)]
        d_str = "  ".join(f"{'↑' if d>0.001 else ('↓' if d<-0.001 else '=')}{abs(d):.3f}" for d in deltas)
        d_avg = avg - base_avg
        print(f"  {'Δ vs baseline':<20}  {d_str}  {'↑' if d_avg>0.001 else '↓'}{abs(d_avg):.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    print("Loading results...")
    baseline_records = load_results(args.baseline)
    print(f"  baseline: {len(baseline_records)} records")

    labels = args.labels if args.labels else []
    method_records = []
    for i, path in enumerate(args.methods):
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue
        records = load_results(path)
        label = labels[i] if i < len(labels) else os.path.basename(os.path.dirname(path))
        method_records.append((label, records))
        print(f"  {label}: {len(records)} records")

    detect_datasets(baseline_records, method_records)
    print(f"  Datasets detected: {DATASETS}")

    os.makedirs(args.output_dir, exist_ok=True)
    print_summary(baseline_records, method_records)

    print("\nGenerating plots...")
    plot_comparison_bar(baseline_records, method_records, args.output_dir)
    plot_comparison_sweep(baseline_records, method_records, args.output_dir)
    plot_overall_summary(baseline_records, method_records, args.output_dir)
    plot_category_comparison(baseline_records, method_records, args.output_dir)
    print(f"\nDone. Plots in: {os.path.join(args.output_dir, 'plots')}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare method results against baseline")
    p.add_argument("--baseline", required=True, help="Path to baseline results.json")
    p.add_argument("--methods", nargs="+", required=True, help="Paths to method results.json files")
    p.add_argument("--labels", nargs="+", default=[], help="Optional display labels for each method (same order as --methods)")
    p.add_argument("--output_dir", default="../results/qwen_all/compare")
    main(p.parse_args())
