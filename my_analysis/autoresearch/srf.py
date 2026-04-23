#!/usr/bin/env python3
"""
SRF method — AGENT MODIFIES THIS FILE.

The problem is split into two stages. Modify each section independently.
One stage change per commit.

Public interface (called by pope_eval.py — do not rename these functions):
    setup(model, processor)                                    → called once
    prepare_sample(inputs, img_start, img_end,
                   image, question, model, processor)          → called per sample
    cleanup()                                                  → called per sample
"""
from __future__ import annotations

import pathlib
import random
import sys

# ── path setup: reach my_analysis/ siblings ───────────────────────────────
_ANALYSIS_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ANALYSIS_DIR))

import torch
import qwen_attn_patch as patch
import clip_salience   as clip_sal
import hssa_salience   as hssa_sal


# =============================================================================
# STAGE 1 — SALIENCY
# Goal: identify which image tokens are relevant to the question.
#
# Two sources:
#   "clip"      — CLIP ViT-L/14: patch-level cosine sim between image crops
#                 and the queried noun. Qualitatively better spatial precision.
#   "hssa"      — Hidden-State Semantic Alignment: cosine sim between image
#                 token hidden states and question token hidden states at a
#                 middle decoder layer. Uses model's own representations.
#   "clip_hssa" — Weighted ensemble of both.
#
# Saliency type:
#   soft  (use_soft=True)  → continuous [0,1] per token. Smoother gradient.
#   binary(use_soft=False) → top-k tokens = 1.0, rest = 0.0. Sharper.
#
# Key parameters to sweep:
#   CLIP: coarse_grid {5,7,9}, top_k_pct {0.15,0.20,0.30,0.40}, absence_thresh {0.15,0.20,0.25}
#   HSSA: layer {8,10,12,14,16,18,20}, top_k_pct {0.20,0.30,0.40}
#   Ensemble: clip_weight + hssa_weight (should sum to 1.0)
# =============================================================================

SALIENCY = {
    "source":            "clip",   # "clip" | "hssa" | "clip_hssa"

    # CLIP params
    "clip_coarse_grid":   7,       # N×N grid for CLIP patch extraction
    "clip_top_k_pct":     0.30,    # fraction of img tokens to mark as salient
    "clip_absence_thresh": 0.20,   # max_sim below this → object probably absent
    "clip_use_soft":      True,    # True = soft [0,1] saliency; False = binary top-k

    # HSSA params
    "hssa_layer":         16,      # decoder layer to extract hidden states from
    "hssa_top_k_pct":     0.30,
    "hssa_use_soft":      True,

    # Ensemble weights (only used when source == "clip_hssa")
    "clip_weight":        0.7,
    "hssa_weight":        0.3,
}


# =============================================================================
# STAGE 2 — ATTENTION BIASING
# Goal: use the saliency map to redirect attention toward query-relevant tokens.
#
# WHERE to bias:
#   layer_start / layer_end — decoder layer range. Middle layers (8-15) are
#                             the visual-language fusion zone (ClearSight finding).
#   head_top_k_pct          — only bias vision-aware heads (calibrated offline).
#                             0.0 = apply to all heads.
#
# HOW to bias — choose one bias_mode:
#
#   "additive_logit"  (pre-softmax)
#       logit[img_i] += boost_alpha * sal[i]  (+ background suppression via background_eps)
#       Equivalent to multiplying unnormalized prob by exp(alpha*sal). Exponential effect.
#       Strong, can dominate; good when model barely looks at image.
#
#   "prob_interp"  (post-softmax)
#       p_img_new = (1-λ)·p_orig + λ·sal_norm·total_img_weight
#       Redistributes existing image attention budget; total attention to text unchanged.
#       λ = interp_lambda. Pure redistribution — does NOT increase total img attention.
#       Best when model already attends to image but at wrong tokens.
#
#   "prob_scale"  (post-softmax)
#       p_img_i *= (1 + boost_alpha·sal_i), then renorm all rows.
#       Linear multiplicative boost. Salient tokens grow proportionally; renorm
#       reduces non-image attention slightly. Gentler than additive_logit.
#
#   "attn_floor"  (post-softmax)
#       p_img_i = max(p_img_i, prob_floor) for salient tokens, then renorm.
#       Guarantees minimum attention regardless of how low model drove it.
#       Best when model completely ignores the queried region.
#
# sys_beta is always applied pre-softmax for ALL modes (confirmed to help).
#
# Key parameters to sweep:
#   bias_mode: "additive_logit" | "prob_interp" | "prob_scale" | "attn_floor"
#   layer range: (6,13) (8,15) (10,18) (12,20)
#   head_top_k_pct: 0.30, 0.50, 0.70
#   boost_alpha: 1.0, 1.5, 2.0, 3.0, 4.0  (additive_logit and prob_scale)
#   interp_lambda: 0.3, 0.5, 0.7, 1.0     (prob_interp only)
#   prob_floor: 0.002, 0.005, 0.01, 0.02  (attn_floor only)
#   background_eps: 0.0, 0.05, 0.10, 0.20 (additive_logit only)
#   sys_beta: 0.05, 0.10, 0.15, 0.20
# =============================================================================

BIAS = {
    "layer_start":      8,
    "layer_end":        15,
    "head_top_k_pct":   0.50,          # top-50% vision-aware heads
    "sys_beta":         0.10,          # system prompt suppression (all modes)

    # Bias mode — pick one:
    "bias_mode":        "prob_interp", # "additive_logit" | "prob_interp" | "prob_scale" | "attn_floor" | "global_redistribute"

    # additive_logit params:
    "boost_alpha":      1.5,
    "background_eps":   0.0,

    # prob_interp params:
    "interp_lambda":    1.0,           # 0 = no-op, 1 = full redistribution

    # prob_scale params:
    # (uses boost_alpha)

    # attn_floor params:
    "prob_floor":       0.005,         # min attention per salient token

    # global_redistribute params:
    "img_scale":        2.0,           # multiply current total img attention fraction
}


# =============================================================================
# VISUALIZATION CONFIG
# Saves saliency overlays for the first VIS_SAMPLES samples each run so you
# can visually inspect whether the saliency map is localising correctly.
# Output: my_analysis/autoresearch/vis/sample_NN.png  (overwritten each run)
# Set VIS_SAMPLES = 0 to disable.
# =============================================================================

VIS_SAMPLES = 5
VIS_DIR     = pathlib.Path(__file__).parent / "vis"


# =============================================================================
# IMPLEMENTATION — you may also modify this section if needed
# =============================================================================

_model     = None
_processor = None
_spatial   = 2   # spatial_merge_size for Qwen (default 2)
_vis_count = 0   # reset in setup() each run


def setup(model, processor) -> None:
    """Called once after model load. Calibrates heads and registers hooks."""
    global _model, _processor, _spatial, _vis_count
    _model     = model
    _processor = processor
    _spatial   = getattr(model.config.vision_config, "spatial_merge_size", 2)
    _vis_count = 0   # reset per run so each experiment gets fresh vis

    _calibrate_heads()

    # Register the patch using "vaf" to satisfy patch_model's valid-method check.
    # prepare_sample() will flip _STATE["method"] to "srf" before each generate().
    patch.patch_model(model, "vaf", BIAS["boost_alpha"])
    patch._STATE["vaf_layer_start"]    = BIAS["layer_start"]
    patch._STATE["vaf_layer_end"]      = BIAS["layer_end"]
    patch._STATE["vaf_beta"]           = BIAS["sys_beta"]
    patch._STATE["srf_background_eps"] = BIAS["background_eps"]
    patch._STATE["srf_bias_mode"]      = BIAS["bias_mode"]
    patch._STATE["srf_interp_lambda"]  = BIAS["interp_lambda"]
    patch._STATE["srf_prob_floor"]     = BIAS["prob_floor"]
    patch._STATE["srf_img_scale"]      = BIAS["img_scale"]


def _calibrate_heads() -> None:
    """Load 20 POPE adversarial samples (seed=0) and calibrate vision-aware heads."""
    if BIAS["head_top_k_pct"] <= 0.0:
        patch._STATE["head_mask"] = None
        print("  [SRF] head_top_k_pct=0 → applying bias to all heads")
        return

    from datasets import load_dataset as hf_load
    from qwen_vl_utils import process_vision_info as pvi

    print("  [SRF] Calibrating vision-aware heads (20 samples, seed=0)…")
    ds       = hf_load("lmms-lab/POPE", split="test")
    rows     = [r for r in ds
                if str(r.get("category", r.get("type", ""))).strip().lower() == "adversarial"]
    rng      = random.Random(0)            # seed=0 ≠ eval seed=42
    rng.shuffle(rows)
    calib_rows = rows[:20]

    img_id    = _processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    device    = next(_model.parameters()).device

    calib_inputs, img_ranges = [], []
    for r in calib_rows:
        img = r["image"].convert("RGB")
        q   = str(r.get("question", "")).strip() + "\nAnswer with Yes or No only."
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text":  q},
        ]}]
        text  = _processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = pvi(msgs)
        inp   = _processor(text=[text], images=vis, return_tensors="pt", padding=True).to(device)
        ids   = inp["input_ids"][0].tolist()
        s     = ids.index(img_id)
        e     = len(ids) - 1 - ids[::-1].index(img_id)
        calib_inputs.append(inp)
        img_ranges.append((s, e))

    patch.identify_visual_heads(_model, calib_inputs, img_ranges, BIAS["head_top_k_pct"])
    n_sel = int(patch._STATE["head_mask"].sum().item())
    print(f"  [SRF] {n_sel} vision-aware heads selected (top {BIAS['head_top_k_pct']*100:.0f}%)")


def prepare_sample(
    inputs:    dict,
    img_start: int,
    img_end:   int,
    image,
    question:  str,
    model,
    processor,
) -> None:
    """Compute saliency map and arm the patch for this sample."""
    patch.update_sample(img_start, img_end)
    patch._STATE["value"]  = BIAS["boost_alpha"]   # used by additive_logit and prob_scale
    patch._STATE["method"] = "srf"
    # Re-sync mode in case it was mutated (e.g. by a previous failed experiment):
    patch._STATE["srf_bias_mode"]     = BIAS["bias_mode"]
    patch._STATE["srf_interp_lambda"] = BIAS["interp_lambda"]
    patch._STATE["srf_prob_floor"]    = BIAS["prob_floor"]
    patch._STATE["srf_img_scale"]     = BIAS["img_scale"]

    # ── Stage 1: compute saliency ─────────────────────────────────────────
    source    = SALIENCY["source"]
    grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)

    sal_clip = sal_hssa = None

    if source in ("clip", "clip_hssa"):
        result = clip_sal.compute_clip_salience(
            image, question,
            grid_h, grid_w,
            top_k_pct = SALIENCY["clip_top_k_pct"],
            coarse_n  = SALIENCY["clip_coarse_grid"],
        )
        sal_clip = result.saliency if SALIENCY["clip_use_soft"] else result.mask

    if source in ("hssa", "clip_hssa"):
        result_h = hssa_sal.compute_hssa_salience(
            model, inputs, img_start, img_end,
            layer_idx = SALIENCY["hssa_layer"],
            top_k_pct = SALIENCY["hssa_top_k_pct"],
        )
        sal_hssa = result_h.saliency if SALIENCY["hssa_use_soft"] else result_h.mask

    if source == "clip":
        sal = sal_clip
    elif source == "hssa":
        sal = sal_hssa
    else:  # clip_hssa ensemble
        wc  = SALIENCY["clip_weight"]
        wh  = SALIENCY["hssa_weight"]
        combined = wc * sal_clip + wh * sal_hssa
        mn, mx   = combined.min(), combined.max()
        sal      = (combined - mn) / (mx - mn + 1e-8)

    patch._STATE["salience_mask"] = sal

    # ── Stage 1 visualization (first VIS_SAMPLES per run) ────────────────
    global _vis_count
    if VIS_SAMPLES > 0 and _vis_count < VIS_SAMPLES:
        _clip_result = result   if source in ("clip", "clip_hssa") else None
        _hssa_result = result_h if source in ("hssa", "clip_hssa") else None
        _save_vis(image, question, sal_clip, sal_hssa, sal,
                  grid_h, grid_w, _vis_count + 1, _clip_result, _hssa_result)
        _vis_count += 1


def _save_vis(image, question, sal_clip, sal_hssa, sal_final,
              grid_h, grid_w, sample_idx, clip_result=None, hssa_result=None) -> None:
    """Save saliency overlay figure to VIS_DIR/sample_NN.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        from PIL import Image as PILImage

        def _overlay(sal_t, image):
            sal_np  = sal_t.float().cpu().numpy().reshape(grid_h, grid_w)
            sal_np  = (sal_np - sal_np.min()) / (sal_np.max() - sal_np.min() + 1e-8)
            sal_img = PILImage.fromarray((sal_np * 255).astype(np.uint8)).resize(
                image.size, PILImage.BILINEAR)
            sal_arr = np.array(sal_img) / 255.0
            heat    = cm.get_cmap("jet")(sal_arr)[..., :3]
            img_arr = np.array(image.convert("RGB")) / 255.0
            return (np.clip(0.45 * img_arr + 0.55 * heat, 0, 1) * 255).astype(np.uint8)

        source = SALIENCY["source"]
        panels = [("Original", np.array(image.convert("RGB")))]

        if sal_clip is not None:
            noun    = getattr(clip_result, "query_noun", "?") if clip_result else "?"
            present = "" if (clip_result and clip_result.object_present) else " (absent)"
            panels.append((f"CLIP '{noun}'{present}", _overlay(sal_clip, image)))

        if sal_hssa is not None:
            layer = SALIENCY["hssa_layer"]
            panels.append((f"HSSA layer={layer}", _overlay(sal_hssa, image)))

        if source == "clip_hssa":
            panels.append(("Ensemble", _overlay(sal_final, image)))

        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, img_arr) in zip(axes, panels):
            ax.imshow(img_arr)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        q_short = question[:80].replace("\n", " ")
        fig.suptitle(f"sample {sample_idx} | {q_short}", fontsize=7)
        fig.tight_layout()

        VIS_DIR.mkdir(parents=True, exist_ok=True)
        out = VIS_DIR / f"sample_{sample_idx:02d}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [vis] {out}")
    except Exception as exc:
        print(f"  [vis] WARNING: visualization failed ({exc})")


def cleanup() -> None:
    """Clear per-sample state after generate()."""
    patch._STATE["salience_mask"] = None
    patch._STATE["method"]        = "vaf"   # safe no-op state between samples
