#!/usr/bin/env python3
"""
LLaVA-1.5 inference-time attention intervention evaluation.

Loads the model once, runs sanity checks, then evaluates every
(dataset, method, value) combination and saves:
  <output_dir>/results.json     — per-sample records
  <output_dir>/summary.csv      — accuracy per (dataset, group, method, value)
  <output_dir>/plots/           — accuracy-vs-value line plots

Architecture notes
------------------
Model: LlavaForConditionalGeneration  (llava-hf/llava-1.5-7b-hf)
  - vision_tower      : CLIPVisionModel (ViT-L/14-336)
  - language_model    : LlamaForCausalLM
  - decoder layers    : model.language_model.model.layers
  - image tokens      : (image_size // patch_size)^2 = 576 for default ViT-L/14-336
  - input_ids layout  : [system/text] [1 IMAGE placeholder] [question text]
    → language model sees: [system/text] [576 image features] [question text]
  - image_token_index : model.config.image_token_index  (32000 for llava-1.5)

Usage
-----
# Baseline on all datasets
python run_llava_eval.py --dataset all --method baseline --n_samples 5

# SRF-CLIP sweep on POPE
python run_llava_eval.py --dataset pope --method srf_clip \\
    --sweep 1.5 2.0 4.0 8.0 --n_samples 50

# 7B model on MMVP
python run_llava_eval.py --model llava-hf/llava-1.5-13b-hf \\
    --dataset mmvp --method srf_hssa --sweep 1.5 2.0 4.0 8.0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_attn_patch as patch
import clip_salience    as srf_clip_basic
import hssa_salience
from eval_datasets import LOADERS, is_correct, SEED

os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

MODEL = "llava-hf/llava-1.5-7b-hf"   # overridden by --model arg

# ---------------------------------------------------------------------------
# Supported attention methods (contrastive methods are Qwen-specific)
# ---------------------------------------------------------------------------

ATTN_METHODS = (
    "baseline", "temperature", "vision_boost", "vhr_boost", "vaf",
    "attn_salience", "srf_clip_basic", "srf_clip", "srf_hssa",
)

_GEN_KWARGS = dict(
    max_new_tokens=16, do_sample=False,
    temperature=None, top_p=None, num_beams=1,
)


# ---------------------------------------------------------------------------
# Persistence helpers (shared logic with run_qwen_eval.py)
# ---------------------------------------------------------------------------

def append_results(records: List[Dict], path: str) -> None:
    existing: List[Dict] = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    with open(path, "w") as f:
        json.dump(existing + records, f, indent=2)


def append_summary_csv(records: List[Dict], path: str) -> None:
    key_records: Dict[Tuple, List[bool]] = defaultdict(list)
    for r in records:
        key = (r["dataset"], r["group"], r["method"], r["value"])
        key_records[key].append(r["correct"])

    rows = [
        {
            "dataset":   ds,
            "group":     grp,
            "method":    meth,
            "value":     val,
            "accuracy":  round(sum(corrects) / len(corrects), 4),
            "n_samples": len(corrects),
        }
        for (ds, grp, meth, val), corrects in sorted(key_records.items())
    ]

    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["dataset", "group", "method", "value", "accuracy", "n_samples"]
        )
        if write_header:
            w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Plotting (mirrors run_qwen_eval.py)
# ---------------------------------------------------------------------------

def plot_bar(all_records: List[Dict], output_dir: str) -> None:
    by_ds_method: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in all_records:
        by_ds_method[(r["dataset"], r["method"])].append(r)

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for (ds, method), records in by_ds_method.items():
        values = sorted({r["value"] for r in records})
        if len(values) != 1:
            continue
        val    = values[0]
        groups = sorted({r["group"] for r in records})
        accs   = []
        for grp in groups:
            subset = [r for r in records if r["group"] == grp]
            accs.append(sum(r["correct"] for r in subset) / len(subset))
        overall = sum(r["correct"] for r in records) / len(records)

        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.7), 5))
        colors  = plt.cm.tab20.colors  # type: ignore[attr-defined]
        bars    = ax.bar(groups, accs,
                         color=[colors[i % len(colors)] for i in range(len(groups))],
                         edgecolor="white", linewidth=0.5)
        ax.axhline(overall, color="black", linewidth=1.8, linestyle="--",
                   label=f"overall={overall:.2f}")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ds}  —  {method}  (value={val})")
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", rotation=45)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{acc:.2f}", ha="center", va="bottom", fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        save_path = os.path.join(plot_dir, f"{ds}_{method}_bar.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot → {save_path}")


def plot_accuracy(all_records: List[Dict], output_dir: str) -> None:
    by_ds_method: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in all_records:
        by_ds_method[(r["dataset"], r["method"])].append(r)

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for (ds, method), records in by_ds_method.items():
        values = sorted({r["value"] for r in records})
        if len(values) < 2:
            continue
        groups = sorted({r["group"] for r in records})
        fig, ax = plt.subplots(figsize=(10, 5))
        colors  = plt.cm.tab20.colors  # type: ignore[attr-defined]

        for gi, grp in enumerate(groups):
            grp_recs = [r for r in records if r["group"] == grp]
            xs, ys   = [], []
            for v in values:
                subset = [r for r in grp_recs if r["value"] == v]
                if subset:
                    xs.append(v)
                    ys.append(sum(r["correct"] for r in subset) / len(subset))
            ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=4,
                    color=colors[gi % len(colors)], alpha=0.6, label=grp)

        xs, ys = [], []
        for v in values:
            subset = [r for r in records if r["value"] == v]
            if subset:
                xs.append(v)
                ys.append(sum(r["correct"] for r in subset) / len(subset))
        ax.plot(xs, ys, marker="D", linewidth=2.5, markersize=7,
                color="black", label="overall", zorder=5)

        ax.set_xlabel("Intervention value")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ds}  —  {method}")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(plot_dir, f"{ds}_{method}.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot → {save_path}")


# ---------------------------------------------------------------------------
# Per-value evaluation pass
# ---------------------------------------------------------------------------

def evaluate(
    model: Any,
    processor: Any,
    samples: List[Dict],
    method: str,
    value: float,
    grid_h: int,
    grid_w: int,
    get_img_range_fn: Callable,
    sal_top_k: float,
) -> List[Dict]:
    """Run ``samples`` under (method, value), return per-sample result dicts.

    Args:
        model            : LlavaForConditionalGeneration (eager attention, eval mode)
        processor        : AutoProcessor
        samples          : list of dicts with keys image/prompt/ground_truth/group
        method           : attention intervention method name
        value            : sweep value (ignored for baseline)
        grid_h / grid_w  : image token grid dimensions (fixed for LLaVA-1.5)
        get_img_range_fn : callable(inputs) → (img_start, img_end)
        sal_top_k        : fraction of image tokens for binary salience mask
    """
    patch.patch_model(model, method, value)
    records: List[Dict] = []

    for i, sample in enumerate(samples):
        # Build prompt using HuggingFace chat template for LLaVA
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": sample["prompt"]},
        ]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            images=sample["image"], text=prompt, return_tensors="pt"
        )
        inputs = inputs.to(next(model.parameters()).device)

        img_start, img_end = get_img_range_fn(inputs)

        # Salience computation — run BEFORE setting method to "srf" so the
        # HSSA forward pass doesn't accidentally trigger the srf intervention.
        sal_top_k_state = patch._STATE.get("_sal_top_k", sal_top_k)

        if method == "attn_salience":
            sal = patch.compute_attention_salience(
                model, inputs, img_start, img_end,
                top_k_pct=sal_top_k_state, normalize=True,
            )
            patch._STATE["salience_mask"] = sal

        elif method == "srf_clip_basic":
            clip_result = srf_clip_basic.compute_clip_salience(
                sample["image"], sample["prompt"], grid_h, grid_w,
                top_k_pct=sal_top_k_state,
            )
            patch._STATE["salience_mask"] = clip_result.mask

        elif method == "srf_clip":
            # Reset to neutral method during CLIP computation, then switch to srf
            patch._STATE["method"] = "srf_clip"
            clip_result = srf_clip_basic.compute_clip_salience(
                sample["image"], sample["prompt"], grid_h, grid_w,
                top_k_pct=sal_top_k_state,
            )
            patch._STATE["salience_mask"] = clip_result.mask
            patch._STATE["method"] = "srf"

        elif method == "srf_hssa":
            # Reset to neutral before HSSA forward pass, then switch to srf
            patch._STATE["method"] = "srf_hssa"
            hssa_layer = patch._STATE.get("_hssa_layer", 16)
            # LLaVA: text after image in input_ids = tokens after the single placeholder
            text_ids   = inputs["input_ids"][0, img_start + 1:].cpu()
            hssa_result = hssa_salience.compute_hssa_salience(
                model, inputs, img_start, img_end,
                layer_idx=hssa_layer, top_k_pct=sal_top_k_state,
                text_ids=text_ids,
                special_token_threshold=None,   # LLaVA: no Qwen special tokens
            )
            patch._STATE["salience_mask"] = hssa_result.mask
            patch._STATE["method"] = "srf"

        else:
            patch._STATE["salience_mask"] = None

        patch.update_sample(img_start, img_end)
        with torch.inference_mode():
            out = model.generate(**inputs, **_GEN_KWARGS)

        pred    = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        correct = is_correct(pred, sample["ground_truth"])

        torch.cuda.empty_cache()

        records.append({
            "group":        sample["group"],
            "prompt":       sample["prompt"][:120],
            "ground_truth": sample["ground_truth"],
            "prediction":   pred,
            "correct":      correct,
            "method":       method,
            "value":        value,
        })

        status = "✓" if correct else "✗"
        print(f"    [{i + 1:03d}/{len(samples):03d}] {sample['group']:30s} "
              f"GT={sample['ground_truth']:6s}  Pred={pred[:14]:14s}  {status}")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    global MODEL
    if args.model:
        MODEL = args.model

    datasets = list(LOADERS.keys()) if args.dataset == "all" else [args.dataset]

    if args.method == "baseline":
        values = [1.0]
    else:
        values = args.sweep if args.sweep else [args.value]

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    summary_path = os.path.join(args.output_dir, "summary.csv")

    # --plot_only
    if args.plot_only:
        if not os.path.exists(results_path):
            print(f"ERROR: {results_path} not found.")
            sys.exit(1)
        with open(results_path) as f:
            all_records = json.load(f)
        plot_bar(all_records, args.output_dir)
        plot_accuracy(all_records, args.output_dir)
        print("Plots regenerated.")
        return

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"\nLoading {MODEL} (eager attention)…")
    _dtype_str = os.environ.get("TORCH_DTYPE", "float16")
    _dtype = torch.float16 if _dtype_str == "float16" else torch.bfloat16
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL, device_map="auto", torch_dtype=_dtype,
        attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL)

    vis_cfg    = model.config.vision_config
    n_img_side = vis_cfg.image_size // vis_cfg.patch_size   # e.g. 336 // 14 = 24
    grid_h     = n_img_side
    grid_w     = n_img_side
    n_img_toks = n_img_side ** 2                            # e.g. 576
    n_layers   = len(patch._get_decoder_layers(model))
    print(f"  image grid: {grid_h}×{grid_w} = {n_img_toks} tokens, "
          f"decoder layers: {n_layers}")

    # Arch-aware image-token range function — used by patch + sanity checks
    get_img_range_fn: Callable = lambda inp: patch.get_image_token_range(inp, model=model)

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------
    print("\n--- Sanity checks ---")
    n_calib     = max(1, min(20, args.n_samples))
    pre_samples = LOADERS[datasets[0]](args.groups, n_samples=n_calib)
    if not pre_samples:
        print("ERROR: no samples for sanity check — check --groups.")
        sys.exit(1)

    device = next(model.parameters()).device
    calib_inputs_list: List[Any] = []
    calib_img_ranges:  List[Any] = []

    for s in pre_samples:
        conv     = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": s["prompt"]},
        ]}]
        prompt   = processor.apply_chat_template(conv, add_generation_prompt=True)
        sc_inp   = processor(images=s["image"], text=prompt, return_tensors="pt")
        sc_inp   = {k: v.to(device) for k, v in sc_inp.items()}
        calib_inputs_list.append(sc_inp)
        calib_img_ranges.append(get_img_range_fn(sc_inp))

    sc_inputs = calib_inputs_list[0]

    # Choose sanity-check method proxy
    if args.method == "baseline":
        sc_method, sc_value = "temperature", 1.5
    elif args.method in ("vhr_boost", "vaf", "attn_salience",
                         "srf_clip_basic", "srf_clip", "srf_hssa"):
        sc_method, sc_value = "vision_boost", 2.0
    else:
        sc_method = args.method
        sc_value  = next((v for v in values if v != 1.0), 1.5)

    patch.run_sanity_checks(model, sc_inputs, get_img_range_fn, sc_method, sc_value)
    patch.unpatch_model()

    # VHR calibration for head-selective methods
    if args.method in ("vhr_boost", "vaf", "attn_salience",
                       "srf_clip_basic", "srf_clip", "srf_hssa"):
        torch.cuda.empty_cache()
        print(f"\n--- VHR head identification (used by {args.method}) ---")
        patch.run_vhr_sanity_checks(
            model,
            calib_inputs_list,
            calib_img_ranges,
            sc_inputs,
            get_img_range_fn,
            top_k_pct=args.vhr_top_k,
        )
        n_heads    = len(patch._STATE["head_mask"])
        n_selected = int(patch._STATE["head_mask"].sum().item())
        print(f"  Using {n_selected}/{n_heads} vision-aware heads "
              f"(top {args.vhr_top_k * 100:.0f}%)")

    # VAF / SRF layer range + suppression
    if args.method in ("vaf", "srf_clip", "srf_hssa"):
        patch._STATE["vaf_beta"]        = args.vaf_beta
        patch._STATE["vaf_layer_start"] = args.vaf_layer_start
        patch._STATE["vaf_layer_end"]   = args.vaf_layer_end
        if args.method == "vaf":
            print(f"  VAF: alpha=value (sweep), beta={args.vaf_beta}, "
                  f"layers={args.vaf_layer_start}-{args.vaf_layer_end}")

    # Salience top-k
    if args.method in ("attn_salience", "srf_clip_basic", "srf_clip", "srf_hssa"):
        patch._STATE["_sal_top_k"] = args.sal_top_k
        print(f"  {args.method}: sal_top_k={args.sal_top_k:.0%}, "
              f"using {'vision-aware heads' if patch._STATE['head_mask'] is not None else 'all heads'}")

    # SRF-specific params
    if args.method in ("srf_clip", "srf_hssa"):
        patch._STATE["srf_background_eps"] = args.srf_background_eps
        patch._STATE["_hssa_layer"]        = args.hssa_layer
        print(f"  SRF: boost=log(value), background_eps={args.srf_background_eps}, "
              f"beta={args.vaf_beta}, layers={args.vaf_layer_start}-{args.vaf_layer_end}"
              + (f", hssa_layer={args.hssa_layer}" if args.method == "srf_hssa" else ""))

    del calib_inputs_list, calib_img_ranges
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Main evaluation loop
    # -----------------------------------------------------------------------
    all_records: List[Dict] = []

    for ds_name in datasets:
        print(f"\n{'=' * 65}")
        print(f"  Dataset : {ds_name}")
        print(f"{'=' * 65}")

        samples = LOADERS[ds_name](args.groups, args.n_samples)
        if not samples:
            print(f"  No samples found for {ds_name}. Skipping.")
            continue

        for val in values:
            print(f"\n  method={args.method}  value={val}")

            records = evaluate(
                model, processor, samples,
                args.method, val,
                grid_h, grid_w,
                get_img_range_fn,
                sal_top_k=args.sal_top_k,
            )

            for r in records:
                r["dataset"] = ds_name

            n_correct = sum(r["correct"] for r in records)
            print(f"\n  → Overall: {n_correct}/{len(records)} "
                  f"= {n_correct / len(records):.3f}")

            all_records.extend(records)
            append_results(records, results_path)
            append_summary_csv(records, summary_path)

    # -----------------------------------------------------------------------
    # Plots and cleanup
    # -----------------------------------------------------------------------
    patch.unpatch_model()
    plot_bar(all_records, args.output_dir)
    plot_accuracy(all_records, args.output_dir)

    print(f"\n{'=' * 65}")
    print(f"Done.")
    print(f"  results  → {results_path}")
    print(f"  summary  → {summary_path}")
    print(f"  plots    → {os.path.join(args.output_dir, 'plots')}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="LLaVA-1.5 attention intervention evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset", default="all",
        choices=["all"] + list(LOADERS.keys()),
        help="Dataset(s) to evaluate (default: all).",
    )
    p.add_argument(
        "--model", default=None,
        help="HuggingFace model ID (default: llava-hf/llava-1.5-7b-hf). "
             "E.g. llava-hf/llava-1.5-13b-hf",
    )
    p.add_argument(
        "--method", default="baseline",
        choices=list(ATTN_METHODS),
        help="Attention intervention method.",
    )
    p.add_argument(
        "--vhr_top_k", type=float, default=0.50,
        help="Fraction of heads selected as vision-aware (default: 0.50).",
    )
    p.add_argument(
        "--sal_top_k", type=float, default=0.3,
        help="Fraction of image tokens selected as salient (default: 0.3).",
    )
    p.add_argument(
        "--srf_background_eps", type=float, default=0.1,
        help="SRF suppression amount for non-salient image tokens (default: 0.1).",
    )
    p.add_argument(
        "--hssa_layer", type=int, default=16,
        help="SRF-HSSA: decoder layer index for hidden-state saliency (default: 16).",
    )
    p.add_argument(
        "--vaf_beta", type=float, default=0.1,
        help="VAF/SRF suppression coefficient β for system-prompt attention (default: 0.1).",
    )
    p.add_argument(
        "--vaf_layer_start", type=int, default=9,
        help="First decoder layer for VAF/SRF intervention "
             "(default: 9, matching ClearSight for LLaVA-1.5 / 32-layer LLaMA).",
    )
    p.add_argument(
        "--vaf_layer_end", type=int, default=14,
        help="Last decoder layer for VAF/SRF intervention inclusive "
             "(default: 14, matching ClearSight for LLaVA-1.5 / 32-layer LLaMA).",
    )
    p.add_argument(
        "--value", type=float, default=1.0,
        help="Single intervention value (use --sweep for multiple).",
    )
    p.add_argument(
        "--sweep", type=float, nargs="+", default=None,
        help="Sweep over multiple values, e.g. --sweep 1.5 2.0 4.0 8.0",
    )
    p.add_argument(
        "--n_samples", type=int, default=5,
        help="Samples per group/category (default: 5).",
    )
    p.add_argument(
        "--groups", nargs="*", default=None,
        help="Groups/categories to include (default: all).",
    )
    p.add_argument(
        "--output_dir", default="output/llava_eval",
        help="Root output directory (default: output/llava_eval).",
    )
    p.add_argument(
        "--plot_only", action="store_true",
        help="Skip evaluation; regenerate plots from existing results.json.",
    )
    main(p.parse_args())
