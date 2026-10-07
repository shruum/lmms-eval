"""
Modality-fusion attention analysis — LLaVA-1.5-7B (Fig 2 equivalent).

STAGE 1 (GPU): collect attention stats from MMHal-Bench → save .npy files
STAGE 2 (CPU): load .npy files → generate figure

Run with --collect to redo the GPU pass, otherwise just re-plots.

Outputs (in OUT_DIR):
  attn_by_layer_query.npy     (n_layers, 3)        — mean attn FROM query TO [vision, query, sys]
  token_counts.npy            (3,)                  — mean token counts [vis, query, sys]
  rho_matrix.npy              (n_layers, n_heads)   — vision attn fraction per head×layer
  llava_analysis_figure.png   — 3-panel figure (line + bar + heatmap)

Usage:
  cd /home/sgowda/workspace/SRF/lmms-eval
  conda activate mllm
  python srf/analysis/llava_modality_attention.py --collect
  python srf/analysis/llava_modality_attention.py             # re-plot from saved .npy
"""
from __future__ import annotations

import argparse
import json
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/home/sgowda/.cache/huggingface"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR  = "/home/sgowda/workspace/SRF/lmms-eval/analysis/llava_attn_data"
OUT_FIG  = "/home/sgowda/workspace/SRF/lmms-eval/analysis/llava_analysis_figure.png"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ID      = "llava-hf/llava-1.5-7b-hf"
LLAVA_PROMPT  = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: <image>\n{question}\nASSISTANT:"
)
MMHAL_JSON    = "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json"
MMHAL_IMG_DIR = "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/images"
FUSE_START    = 8
FUSE_END      = 20   # SRF fusion zone
SMOOTH_WIN    = 3

VIS_C = "#4C9EAF"
TXT_C = "#6B8E7B"
SYS_C = "#E07B54"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_lm(model):
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    raise AttributeError(f"Cannot find language_model in {type(model)}")


def load_mmhal(seed: int = 42, n: int = -1) -> list[dict]:
    with open(MMHAL_JSON) as f:
        data = json.load(f)
    rows = []
    for r in data:
        fname = r["image_src"].rstrip("/").split("/")[-1]
        img_path = os.path.join(MMHAL_IMG_DIR, fname)
        if os.path.exists(img_path):
            rows.append({"img_path": img_path, "question": r["question"],
                         "question_type": r.get("question_type", "other")})
    random.Random(seed).shuffle(rows)
    if n > 0:
        rows = rows[:n]
    print(f"  {len(rows)} MMHal samples selected")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — GPU data collection
# ══════════════════════════════════════════════════════════════════════════════

def collect(n: int = -1):
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from PIL import Image

    print("Loading LLaVA-1.5-7B (eager attention)…")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).eval().cuda()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    lm         = get_lm(model)
    n_layers   = len(lm.layers)
    n_heads    = model.config.text_config.num_attention_heads
    img_tok_id = model.config.image_token_index
    vis_cfg    = model.config.vision_config
    n_img_tok  = (vis_cfg.image_size // vis_cfg.patch_size) ** 2   # 576
    print(f"  {n_layers} layers, {n_heads} heads, {n_img_tok} image tokens")

    samples = load_mmhal(n=n)

    attn_accum    = np.zeros((n_layers, 3),       dtype=np.float64)
    rho_vis_accum = np.zeros((n_layers, n_heads), dtype=np.float64)
    count_accum   = np.zeros(3,                   dtype=np.float64)
    n_valid = 0

    for i, s in enumerate(samples):
        image  = Image.open(s["img_path"]).convert("RGB")
        prompt = LLAVA_PROMPT.format(question=s["question"])
        inp    = processor(text=prompt, images=[image],
                           return_tensors="pt", padding=True).to("cuda")

        ids     = inp["input_ids"][0]
        seq_len = ids.shape[0]

        vis_pos   = (ids == img_tok_id).nonzero(as_tuple=True)[0]
        img_start = vis_pos[0].item()
        img_end   = img_start + n_img_tok   # exclusive

        vis_mask   = torch.zeros(seq_len, dtype=torch.bool)
        vis_mask[img_start:img_end] = True
        sys_mask   = torch.zeros(seq_len, dtype=torch.bool)
        sys_mask[:img_start] = True
        query_mask = torch.zeros(seq_len, dtype=torch.bool)
        query_mask[img_end:seq_len - 1] = True

        n_vis   = int(vis_mask.sum())
        n_sys   = int(sys_mask.sum())
        n_query = int(query_mask.sum())
        if n_query == 0:
            continue

        count_accum += [n_vis, n_query, n_sys]

        with torch.inference_mode():
            out = model(**inp, output_attentions=True)

        vis_idx   = vis_mask.numpy().nonzero()[0]
        query_idx = query_mask.numpy().nonzero()[0]
        sys_idx   = sys_mask.numpy().nonzero()[0]

        for li, attn_layer in enumerate(out.attentions):
            a_heads  = attn_layer[0].float().cpu().numpy()   # (n_heads, seq, seq)
            a_from_q = a_heads[:, query_idx, :]              # (n_heads, n_query, seq)
            a_mean_q = a_from_q.mean(axis=1)                 # (n_heads, seq)
            rho_vis_accum[li] += a_mean_q[:, vis_idx].sum(axis=1)

            a_mean = a_mean_q.mean(axis=0)                   # (seq,)
            attn_accum[li, 0] += a_mean[vis_idx].sum()
            attn_accum[li, 1] += a_mean[query_idx].sum()
            attn_accum[li, 2] += a_mean[sys_idx].sum()

        n_valid += 1
        del out
        torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}]", flush=True)

    print(f"Processed {n_valid}/{len(samples)} samples")
    attn_by_layer = attn_accum    / n_valid
    token_counts  = count_accum   / n_valid
    rho_matrix    = rho_vis_accum / n_valid

    np.save(os.path.join(OUT_DIR, "attn_by_layer_query.npy"), attn_by_layer)
    np.save(os.path.join(OUT_DIR, "token_counts.npy"),        token_counts)
    np.save(os.path.join(OUT_DIR, "rho_matrix.npy"),          rho_matrix)
    print("Saved .npy files.")
    return attn_by_layer, token_counts, rho_matrix


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — figure generation
# ══════════════════════════════════════════════════════════════════════════════

def plot(attn_by_layer, token_counts, rho_matrix):
    import matplotlib.colors as mcolors

    n_layers = attn_by_layer.shape[0]

    fig = plt.figure(figsize=(11, 5.0))
    gs  = fig.add_gridspec(
        1, 2,
        width_ratios=[1.4, 3.2],
        wspace=0.30,
        left=0.07, right=0.97,
        top=0.88, bottom=0.13,
    )
    ax_bar  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    # ── Panel (a): token count vs attention received ──────────────────────────
    tok_pct   = token_counts / token_counts.sum() * 100
    attn_mean = attn_by_layer.mean(axis=0)
    attn_pct  = attn_mean / attn_mean.sum() * 100

    def _round100(pcts):
        """Largest-remainder rounding so displayed integers sum to exactly 100."""
        floored = np.floor(pcts).astype(int)
        deficit = 100 - floored.sum()
        bumps   = np.argsort(pcts - floored)[::-1][:deficit]
        floored[bumps] += 1
        return floored

    tok_lbl  = _round100(tok_pct)
    attn_lbl = _round100(attn_pct)

    x      = np.arange(3)
    bw     = 0.38
    colors = [VIS_C, TXT_C, SYS_C]
    labels = ["Vision\ntokens", "Query\ntext", "System\nprompt"]

    b1 = ax_bar.bar(x - bw/2, tok_pct,  width=bw, color=colors,
                    alpha=0.45, edgecolor="#555555", linewidth=0.8, hatch="///")
    b2 = ax_bar.bar(x + bw/2, attn_pct, width=bw, color=colors,
                    alpha=1.00, edgecolor="white",   linewidth=0.8)

    for bar, v in zip(list(b1) + list(b2), list(tok_lbl) + list(attn_lbl)):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{v}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax_bar.legend(
        handles=[
            mpatches.Patch(facecolor="#AAAAAA", edgecolor="#555555",
                           hatch="///", label="Token count"),
            mpatches.Patch(facecolor="#AAAAAA", edgecolor="white",
                           label="Attention received"),
        ],
        fontsize=9, framealpha=0.85, loc="upper right",
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=10)
    ax_bar.set_ylim(0, 112)
    ax_bar.set_title("(a) Token count vs\nattention received", fontsize=10, fontweight="bold")
    ax_bar.yaxis.grid(True, ls="--", alpha=0.4)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ── Panel (b): vision attention heatmap ──────────────────────────────────
    # Skip layer 0: near-uniform attention makes 576/628 ≈ 92% go to image tokens
    # by default — this reflects sequence composition, not vision-responsiveness.
    vis_attn = rho_matrix[1:] * 100
    n_heads  = vis_attn.shape[1]

    # vmax=p98 keeps the full range; PowerNorm(gamma=0.65) gently stretches
    # low values so the 5–25% band shows colour gradients rather than flat yellow.
    # gamma=1 → linear (too yellow); gamma=0.5 → too red; 0.65 is the sweet spot.
    _vmax = np.percentile(vis_attn, 98)
    im = ax_heat.imshow(
        vis_attn, aspect="auto", cmap="YlOrRd",
        norm=mcolors.PowerNorm(gamma=0.65, vmin=0, vmax=_vmax),
        interpolation="nearest", origin="upper",
    )
    ax_heat.set_xlabel("Attention head", fontsize=10)
    ax_heat.set_ylabel("Decoder layer",  fontsize=10)
    n_shown = vis_attn.shape[0]   # 31 (layers 1–31)
    ax_heat.set_xticks(range(0, n_heads, max(1, n_heads // 8)))
    ax_heat.set_xticklabels(
        [f"H{h}" for h in range(0, n_heads, max(1, n_heads // 8))], fontsize=9)
    ytick_pos = range(0, n_shown, 4)
    ax_heat.set_yticks(list(ytick_pos))
    ax_heat.set_yticklabels([str(i + 1) for i in ytick_pos], fontsize=9)
    ax_heat.set_title(
        "(b) Visual attention per (layer, head) — "
        "darker = more attention to visual tokens",
        fontsize=10, fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.88, pad=0.012, extend="max")
    cbar.set_label("Attention to vision tokens (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(OUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_FIG}")
    PAPER_FIG = "/home/sgowda/workspace/SRF/paper/images/llava_head.png"
    plt.savefig(PAPER_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {PAPER_FIG}")

    print("\n=== Key numbers ===")
    print(f"  Vision tokens:  {tok_pct[0]:.0f}% of tokens → {attn_pct[0]:.0f}% of attention")
    print(f"  Query text:     {tok_pct[1]:.0f}% of tokens → {attn_pct[1]:.0f}% of attention")
    print(f"  System prompt:  {tok_pct[2]:.0f}% of tokens → {attn_pct[2]:.0f}% of attention")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true",
                        help="Re-run GPU collection (otherwise load saved .npy)")
    parser.add_argument("--n", type=int, default=-1,
                        help="Number of MMHal samples (-1 = all 96)")
    args = parser.parse_args()

    npy_a = os.path.join(OUT_DIR, "attn_by_layer_query.npy")
    npy_c = os.path.join(OUT_DIR, "token_counts.npy")
    npy_r = os.path.join(OUT_DIR, "rho_matrix.npy")

    if args.collect or not all(os.path.exists(p) for p in [npy_a, npy_c, npy_r]):
        attn_by_layer, token_counts, rho_matrix = collect(n=args.n)
    else:
        print("Loading saved .npy files…")
        attn_by_layer = np.load(npy_a)
        token_counts  = np.load(npy_c)
        rho_matrix    = np.load(npy_r)

    plot(attn_by_layer, token_counts, rho_matrix)
