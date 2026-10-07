#!/usr/bin/env python3
"""
LLaVA-1.5-7B attention analysis — Fig 2 equivalent for the paper.

Measures Vision Token Attention Ratio (VTAR) on POPE adversarial samples
and produces the two-panel figure (bar chart + layer-head heatmap).

Output: results/llava_attn_analysis/
  llava_analysis_figure.png    — 2-panel paper figure (bar + heatmap)
  summary.json                 — all numeric results
  plots/                       — individual plots

Usage (on GPU node):
  cd /home/sgowda/workspace/SRF/lmms-eval
  conda activate mllm
  python srf/analysis/llava_attn_analysis.py --n 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.environ["HF_HOME"] = "/home/sgowda/.cache/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"
LLAVA_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: <image>\n{question}\nASSISTANT:"
)
# LLaVA-1.5 fixed image grid: 336px / 14px patch = 24×24 = 576 tokens
LLAVA_GRID_H = 24
LLAVA_GRID_W = 24


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_lm(model):
    """Return the language model submodule (handles Transformers 4.x and 5.x layouts)."""
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    raise AttributeError(f"Cannot find language_model in {type(model)}")


def load_model(device="cuda"):
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    print(f"Loading {LLAVA_MODEL} (eager attention)…")
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL)
    print(f"  n_layers={len(get_lm(model).layers)}")
    return model, processor


def encode(processor, model, image: Image.Image, question: str, device="cuda"):
    import torch
    prompt = LLAVA_PROMPT.format(question=question)
    inp = processor(text=prompt, images=[image], return_tensors="pt", padding=True).to(device)
    img_token_id = model.config.image_token_index
    vis_cfg = model.config.vision_config
    n_img_tok = (vis_cfg.image_size // vis_cfg.patch_size) ** 2  # 576
    ids = inp["input_ids"][0].tolist()
    img_start = next(i for i, t in enumerate(ids) if t == img_token_id)
    img_end = img_start + n_img_tok - 1
    return inp, img_start, img_end


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def get_attention_stats(model, inp, img_start: int, img_end: int):
    """Extract per-layer per-head VTAR via forward hooks.

    Returns dict:
        per_layer_per_head_vtar  List[List[float]]  (n_layers, n_heads)
        per_layer_vtar           List[float]         (n_layers,)
        overall_vtar             float
        img_attn_frac            float  — mean attention fraction to image tokens
        pre_img_attn_frac        float  — mean attention to template/sys tokens (before image)
        post_img_attn_frac       float  — mean attention to query tokens (after image)
        n_img, n_pre, n_post     int token counts
    """
    import torch
    seq_len = inp["input_ids"].shape[1]
    last_pos = seq_len - 1

    img_mask = torch.zeros(seq_len, dtype=torch.bool)
    img_mask[img_start:img_end + 1] = True
    pre_mask = torch.zeros(seq_len, dtype=torch.bool)
    pre_mask[:img_start] = True
    post_mask = torch.zeros(seq_len, dtype=torch.bool)
    post_mask[img_end + 1:] = True

    n_img  = img_mask.sum().item()
    n_pre  = pre_mask.sum().item()
    n_post = post_mask.sum().item()

    captured = []

    def make_hook(storage):
        def hook_fn(module, _inp, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_w = output[1]  # (1, n_heads, seq_len, seq_len)
                storage.append(attn_w[0, :, last_pos, :].detach().cpu())
                return (output[0], None) + output[2:]
            return output
        return hook_fn

    layers = get_lm(model).layers
    hooks = [lay.self_attn.register_forward_hook(make_hook(captured)) for lay in layers]
    try:
        with torch.inference_mode():
            model(**inp, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    per_layer_per_head = []
    per_layer = []
    img_attn_list, pre_attn_list, post_attn_list = [], [], []

    for attn_slice in captured:   # (n_heads, seq_len)
        total = attn_slice.sum(dim=-1).clamp(min=1e-9)
        vtar_ph = (attn_slice[:, img_mask].sum(dim=-1) / total).tolist()
        pre_ph  = (attn_slice[:, pre_mask].sum(dim=-1) / total).tolist()
        post_ph = (attn_slice[:, post_mask].sum(dim=-1) / total).tolist()
        per_layer_per_head.append(vtar_ph)
        per_layer.append(float(np.mean(vtar_ph)))
        img_attn_list.append(float(np.mean(vtar_ph)))
        pre_attn_list.append(float(np.mean(pre_ph)))
        post_attn_list.append(float(np.mean(post_ph)))

    return {
        "per_layer_per_head_vtar": per_layer_per_head,
        "per_layer_vtar": per_layer,
        "overall_vtar": float(np.mean(per_layer)),
        "img_attn_frac":  float(np.mean(img_attn_list)),
        "pre_img_attn_frac":  float(np.mean(pre_attn_list)),
        "post_img_attn_frac": float(np.mean(post_attn_list)),
        "n_img": int(n_img),
        "n_pre": int(n_pre),
        "n_post": int(n_post),
    }


def get_spatial_attention(model, inp, img_start: int, img_end: int):
    """Return (n_img_tokens,) mean attention from last pos to each image token, averaged over all layers & heads."""
    import torch
    seq_len = inp["input_ids"].shape[1]
    last_pos = seq_len - 1
    n_img = img_end - img_start + 1
    captured = []

    def make_hook(storage):
        def hook_fn(module, _inp, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_w = output[1]
                storage.append(attn_w[0, :, last_pos, img_start:img_end+1].detach().cpu())
                return (output[0], None) + output[2:]
            return output
        return hook_fn

    layers = get_lm(model).layers
    hooks = [lay.self_attn.register_forward_hook(make_hook(captured)) for lay in layers]
    try:
        with torch.inference_mode():
            model(**inp, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    if not captured:
        return np.zeros(n_img, dtype=np.float32)
    stacked = np.stack([t.float().numpy() for t in captured])  # (n_layers, n_heads, n_img)
    mean_attn = stacked.mean(axis=(0, 1))  # (n_img,)
    s = mean_attn.sum()
    if s > 1e-9:
        mean_attn /= s
    return mean_attn


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_pope_samples(n: int, split: str = "adversarial", seed: int = 42):
    from datasets import load_dataset
    import random
    print(f"Loading POPE {split} from HuggingFace…")
    ds = load_dataset("lmms-lab/POPE", split="test")
    rows = [r for r in ds if str(r.get("category", "")).lower() == split]
    random.Random(seed).shuffle(rows)
    samples = rows[:n]
    print(f"  Using {len(samples)} samples")
    return [
        {
            "image": r["image"].convert("RGB"),
            "question": str(r["question"]).strip() + " Answer with Yes or No only.",
            "gt": str(r.get("answer", "")).strip().lower(),
        }
        for r in samples
    ]


# ---------------------------------------------------------------------------
# Figure: 2-panel paper figure (bar + heatmap), matching analysis2_figure.png
# ---------------------------------------------------------------------------

def plot_paper_figure(records, out_path: str):
    """
    2-panel figure:
      (a) Token count vs attention received for Image / Query / Template token groups
      (b) Vision token attention per (layer, head)
    """
    n_tot_img  = np.mean([r["n_img"]  for r in records])
    n_tot_pre  = np.mean([r["n_pre"]  for r in records])
    n_tot_post = np.mean([r["n_post"] for r in records])
    n_total    = n_tot_img + n_tot_pre + n_tot_post

    pct_img_tok  = n_tot_img  / n_total * 100
    pct_pre_tok  = n_tot_pre  / n_total * 100
    pct_post_tok = n_tot_post / n_total * 100

    attn_img  = np.mean([r["img_attn_frac"]      for r in records]) * 100
    attn_pre  = np.mean([r["pre_img_attn_frac"]  for r in records]) * 100
    attn_post = np.mean([r["post_img_attn_frac"] for r in records]) * 100

    # Head-layer heatmap
    all_ph = np.array([r["per_layer_per_head_vtar"] for r in records])  # (N, n_layers, n_heads)
    mean_heatmap = all_ph.mean(axis=0) * 100  # (n_layers, n_heads)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    matplotlib.rcParams.update({"font.size": 11})

    # --- Panel (a): bar chart ---
    ax = axes[0]
    ax.set_title("(a) Token count vs attention received", fontsize=12, fontweight="bold", pad=8)

    groups = ["Vision\ntokens", "Query\ntext", "System\nprompt"]
    tok_pcts  = [pct_img_tok,  pct_post_tok, pct_pre_tok]
    attn_pcts = [attn_img,     attn_post,    attn_pre]

    x = np.arange(len(groups))
    w = 0.35
    bars_tok  = ax.bar(x - w/2, tok_pcts,  w, label="Token count", color="#6baed6", hatch="//", edgecolor="white", linewidth=0.5)
    bars_attn = ax.bar(x + w/2, attn_pcts, w, label="Attention received", color="#fd8d3c", edgecolor="white", linewidth=0.5)

    for b, v in zip(bars_tok,  tok_pcts):
        ax.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for b, v in zip(bars_attn, attn_pcts):
        ax.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_ylim(0, max(max(tok_pcts), max(attn_pcts)) * 1.20)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel (b): layer-head heatmap ---
    ax2 = axes[1]
    ax2.set_title(
        "(b) Vision token attention per (layer, head)\ndarker = more attention directed to visual content",
        fontsize=12, fontweight="bold", pad=8,
    )

    n_heads = mean_heatmap.shape[1]
    im = ax2.imshow(mean_heatmap, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=mean_heatmap.max())
    cbar = plt.colorbar(im, ax=ax2, shrink=0.85)
    cbar.set_label("Attention to vision tokens (%)", fontsize=9)
    ax2.set_xlabel("Attention head", fontsize=11)
    ax2.set_ylabel("Decoder layer", fontsize=11)
    ax2.set_xticks(range(0, n_heads, max(1, n_heads // 8)))
    ax2.set_xticklabels([f"H{i}" for i in range(0, n_heads, max(1, n_heads // 8))], fontsize=8)

    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    print(f"\n=== Key numbers ===")
    print(f"  Vision tokens   : {pct_img_tok:.0f}% of tokens → {attn_img:.0f}% of attention")
    print(f"  Query text      : {pct_post_tok:.0f}% of tokens → {attn_post:.0f}% of attention")
    print(f"  System/template : {pct_pre_tok:.0f}% of tokens → {attn_pre:.0f}% of attention")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",    type=int,  default=100, help="POPE samples to analyse")
    p.add_argument("--split", default="adversarial")
    p.add_argument("--seed", type=int,  default=42)
    p.add_argument("--output", default="results/llava_attn_analysis")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "plots"), exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(device)

    samples = load_pope_samples(args.n, args.split, args.seed)

    records = []
    for i, s in enumerate(samples):
        inp, img_start, img_end = encode(processor, model, s["image"], s["question"], device)
        stats = get_attention_stats(model, inp, img_start, img_end)
        records.append(stats)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}]  VTAR={stats['img_attn_frac']*100:.1f}%", flush=True)
        torch.cuda.empty_cache()

    # Summary JSON
    summary = {
        "model": LLAVA_MODEL,
        "n_samples": len(records),
        "mean_img_attn_pct":   float(np.mean([r["img_attn_frac"]      for r in records]) * 100),
        "mean_pre_attn_pct":   float(np.mean([r["pre_img_attn_frac"]  for r in records]) * 100),
        "mean_post_attn_pct":  float(np.mean([r["post_img_attn_frac"] for r in records]) * 100),
        "mean_n_img_tokens":   float(np.mean([r["n_img"]  for r in records])),
        "mean_n_pre_tokens":   float(np.mean([r["n_pre"]  for r in records])),
        "mean_n_post_tokens":  float(np.mean([r["n_post"] for r in records])),
        "head_layer_mean_vtar": np.mean(
            np.array([r["per_layer_per_head_vtar"] for r in records]), axis=0
        ).tolist(),
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary.json")

    # Paper figure
    plot_paper_figure(records, os.path.join(args.output, "llava_analysis_figure.png"))

    # Standalone heatmap
    all_ph = np.array([r["per_layer_per_head_vtar"] for r in records]) * 100
    mean_hm = all_ph.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(mean_hm, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="VTAR (%)")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_title(f"LLaVA-1.5-7B — Mean VTAR per Head × Layer (n={len(records)}, POPE {args.split})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "plots", "heatmap_layers_heads.png"), dpi=150)
    plt.close()

    print(f"\nDone. Output at {args.output}/")


if __name__ == "__main__":
    main()
