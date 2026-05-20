#!/usr/bin/env python3
"""
SRF-v2 — AGENT MODIFIES THIS FILE.

New axes vs autoresearch/srf.py:
  (S) Saliency:
      - CLIP model size: ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14-336
      - Multi-scale CLIP: run at multiple coarse grids, average saliency
      - DINOv2: vision-only spatial saliency combined with CLIP text alignment
      - HSSA: existing hidden-state saliency (unchanged)
      - Adaptive alpha mode: binary | linear | sigmoid | zone
        (how confidence in CLIP similarity maps to boost/suppress strength)

  (B) Boost / Suppress:
      - Negative-example handling:
          "suppress_salient"  — when object absent, suppress the near-miss region
          "suppress_text"     — additionally suppress question-noun token attention
          "suppress_both"     — suppress_salient + suppress_text
      - Per-sample head selection: "static" (calibrated once) | "per_sample"
        Per-sample: run identify_visual_heads on THIS sample's inputs (1 extra pass)
      - Logit blending (SLA): blend final-layer logits with penultimate-layer avg

  (L) Logit blending:
      - gamma > 0 blends penultimate-k layer logits with final layer logits
      - Applied INSTEAD of generate() — harness calls get_logits() directly

Public interface (called by harness.py — do not rename):
    setup(model, processor)
    prepare_sample(inputs, img_start, img_end, image, question, model, processor)
    get_logits(model, inputs) -> torch.Tensor   (1, vocab)   ← NEW
    cleanup()
"""
from __future__ import annotations

import pathlib
import random
import sys
from typing import Optional

_V2_DIR       = pathlib.Path(__file__).parent
_ANALYSIS     = _V2_DIR.parent
_REPO_ROOT    = _ANALYSIS.parent          # lmms-eval/
_SRF_SALIENCY = _REPO_ROOT / "srf" / "saliency"
sys.path.insert(0, str(_ANALYSIS))
sys.path.insert(0, str(_SRF_SALIENCY))

import torch
import torch.nn.functional as F
import qwen_attn_patch as patch
import clip_salience   as clip_sal
import hssa_salience   as hssa_sal


# =============================================================================
# CONFIG — sweep this dict via sweep.py
# =============================================================================

CONFIG = {

    # ── Saliency source ───────────────────────────────────────────────────────
    # "clip"       : CLIP only (text-conditioned)
    # "dino"       : DINOv2 only (vision saliency, not text-conditioned)
    # "clip_dino"  : weighted ensemble of CLIP + DINOv2
    # "hssa"       : hidden-state semantic alignment (existing)
    # "clip_hssa"  : CLIP + HSSA ensemble
    "saliency_source": "clip",

    # ── CLIP settings ─────────────────────────────────────────────────────────
    # Model: "openai/clip-vit-base-patch32"     (ViT-B/32, 0.31GB, 32px patches)
    #        "openai/clip-vit-base-patch16"     (ViT-B/16, 0.31GB, 16px patches — same size, 2× finer)
    #        "openai/clip-vit-large-patch14"    (ViT-L/14, 0.86GB, 14px patches — best quality)
    #        "openai/clip-vit-large-patch14-336"(ViT-L/14-336, 0.86GB, higher input res)
    "clip_model": "openai/clip-vit-base-patch32",

    # Multi-scale: list of coarse_n values. Average saliency maps across scales.
    # [7] = single scale (current default)
    # [5, 9] = two-scale: coarse for context + fine for detail
    # [5, 7, 9] = three-scale (slower, most robust)
    "clip_scales": [7],

    "clip_top_k_pct":  0.30,
    "clip_use_soft":   True,

    # ── DINOv2 settings ───────────────────────────────────────────────────────
    # Uses facebook/dinov2-small (86MB) — loads on CPU, borrows GPU for inference
    # DINOv2 attention maps from [CLS] token → patches = vision-only saliency
    # Combined with CLIP text-alignment via ensemble weights below
    "dino_model":      "facebook/dinov2-small",   # 86MB, ViT-S/14 (14px patches)
    "dino_top_k_pct":  0.30,
    "dino_weight":     0.4,   # weight in clip_dino ensemble (CLIP gets 1-dino_weight)

    # ── HSSA settings ─────────────────────────────────────────────────────────
    "hssa_layer":      16,
    "hssa_top_k_pct":  0.30,
    "hssa_use_soft":   True,
    "hssa_weight":     0.3,   # weight in clip_hssa ensemble

    # ── POPE-specific: absence threshold and handling ─────────────────────────
    # CALIBRATED from autoresearch/saliency_quality.py:
    #   present objects (gt=yes): mean max_sim = 0.252
    #   absent  objects (gt=no):  mean max_sim = 0.242
    #   suppress_thresh = 0.248 separates at ~70% accuracy
    "absence_thresh":  0.248,   # below this → object likely absent

    # What to do when object is absent (max_sim < absence_thresh):
    #   "nothing"         — do nothing (fallback, current production behaviour)
    #   "suppress_salient"— suppress the near-miss region (kills visual trigger)
    #   "suppress_text"   — suppress question-noun token attention (kills linguistic trigger)
    #   "suppress_both"   — both of the above (strongest)
    "absent_strategy": "suppress_salient",

    # Alpha for suppressing salient region when absent
    "suppress_alpha":  5.0,

    # ── Adaptive alpha mode ───────────────────────────────────────────────────
    # How CLIP similarity → boost/suppress strength when object IS present:
    #   "fixed"   — always use boost_alpha regardless of sim
    #   "linear"  — alpha_eff = boost_alpha * (sim - absence_thresh) / (1 - absence_thresh)
    #   "sigmoid" — alpha_eff = boost_alpha * sigmoid(k * (sim - midpoint))
    #   "zone"    — three zones: absent (suppress) | uncertain (nothing) | present (boost)
    "adaptive_alpha_mode": "fixed",
    "adaptive_sigmoid_k":   20.0,   # steepness for sigmoid mode

    # ── Boost parameters ─────────────────────────────────────────────────────
    "boost_alpha":     4.0,

    # ── Head selection ────────────────────────────────────────────────────────
    # "static"     — calibrate on N_CALIB samples once, reuse for all (current)
    # "per_sample" — run identify_visual_heads on THIS sample (1 extra forward pass)
    #                More targeted but 2× slower. Good for hard adversarial cases.
    "head_mode":       "static",
    "head_top_k_pct":  0.20,
    "n_calib":         20,

    # ── Bias layer range ─────────────────────────────────────────────────────
    "layer_start":  8,
    "layer_end":    15,

    # ── Bias mode ────────────────────────────────────────────────────────────
    # "additive_logit" | "prob_interp" | "prob_scale" | "attn_floor" | "global_redistribute"
    "bias_mode":       "additive_logit",
    "background_eps":  0.0,
    "sys_beta":        0.10,
    "interp_lambda":   1.0,
    "prob_floor":      0.005,
    "img_scale":       1.5,

    # ── Phase ─────────────────────────────────────────────────────────────────
    # "both" | "prefill" | "generation"
    "phase":           "both",

    # ── Text suppression (for absent-object queries) ──────────────────────────
    # When absent_strategy includes "suppress_text":
    # suppress attention TO question-noun tokens (deep layers where LM priors form)
    "text_suppress_beta":        0.3,
    "text_suppress_layer_start": 20,
    "text_suppress_layer_end":   27,

    # ── Logit blending (SLA — Self-Logits Augmentation from VISTA) ────────────
    # gamma = 0.0  → disabled (use final-layer logits only)
    # gamma > 0.0  → blend: final*(1-γ) + mean(layers[-k:-1])*γ
    # Requires output_hidden_states=True in get_logits() — adds ~20ms overhead.
    # k layers to average for the penultimate blend:
    "logit_blend_gamma": 0.0,
    "logit_blend_k":     3,     # number of penultimate layers to average
}


# =============================================================================
# Module state
# =============================================================================

_model     = None
_processor = None
_spatial   = 2
_dino      = None   # DINOv2 model (lazy-loaded)
_is_absent : Optional[bool] = None   # set per sample

# Whether current get_logits() call needs hidden states
_need_hidden = False


# =============================================================================
# DINOv2 saliency
# =============================================================================

def _load_dino():
    global _dino
    if _dino is not None:
        return _dino
    from transformers import AutoModel
    print(f"  [srf_v2] Loading {CONFIG['dino_model']} on CPU…")
    _dino = AutoModel.from_pretrained(
        CONFIG["dino_model"],
        torch_dtype=torch.float16,
    ).to("cpu").eval()
    return _dino


def _compute_dino_salience(image, grid_h: int, grid_w: int,
                            top_k_pct: float) -> torch.Tensor:
    """
    Compute DINOv2 CLS-to-patches attention saliency.

    DINOv2 ViT-S/14 produces 14×14 = 196 patch tokens for a 224px image.
    We average the [CLS] attention over all heads (last layer) → spatial map.
    Upsample to (grid_h, grid_w) → same shape as CLIP saliency.

    Returns: float tensor (grid_h * grid_w,) in [0,1]
    """
    dino = _load_dino()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Preprocess: DINOv2 expects 224px
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_t = tf(image.convert("RGB")).unsqueeze(0).half().to(device)
    dino.to(device)

    with torch.no_grad():
        out = dino(pixel_values=img_t)

    dino.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    # Use last hidden state patch norms as saliency.
    # out.last_hidden_state: (1, 1+n_patches, hidden_dim) — index 0 is CLS
    hs = out.last_hidden_state[0, 1:, :]       # (n_patches, hidden_dim)
    mean_attn = hs.norm(dim=-1).cpu().float()  # (n_patches,) — high norm = salient

    # DINOv2-small at 224px: 16×16 patch grid (14px patches)
    n_p = mean_attn.shape[0]
    p_side = int(n_p ** 0.5)
    sal_grid = mean_attn.view(1, 1, p_side, p_side).float()

    # Upsample to Qwen token grid
    sal_token = F.interpolate(sal_grid, size=(grid_h, grid_w),
                              mode="bilinear", align_corners=False).squeeze()
    sal_flat  = sal_token.reshape(-1)

    # Normalize to [0,1]
    mn, mx = sal_flat.min(), sal_flat.max()
    sal    = (sal_flat - mn) / (mx - mn + 1e-8)
    return sal


# =============================================================================
# Multi-scale CLIP saliency
# =============================================================================

def _compute_multiscale_clip(image, noun: str, grid_h: int, grid_w: int,
                              scales: list, top_k_pct: float,
                              clip_model: str) -> tuple:
    """
    Average CLIP saliency across multiple coarse grid sizes.
    Returns (combined_saliency tensor, max_sim float)
    """
    sals   = []
    maxsim = 0.0
    for scale in scales:
        res = clip_sal.compute_clip_salience(
            image, noun, grid_h, grid_w,
            top_k_pct=top_k_pct,
            coarse_n=scale,
            clip_model_name=clip_model,
        )
        sals.append(res.saliency)
        maxsim = max(maxsim, res.max_sim)

    combined = torch.stack(sals).mean(0)          # average across scales
    mn, mx   = combined.min(), combined.max()
    combined = (combined - mn) / (mx - mn + 1e-8) # renormalize
    return combined, maxsim


# =============================================================================
# Adaptive alpha
# =============================================================================

def _adaptive_alpha(max_sim: float) -> float:
    """
    Map CLIP max_sim → effective alpha value.
    Returns positive float for boost, negative for suppress, 0 for neutral.
    """
    mode   = CONFIG["adaptive_alpha_mode"]
    thresh = CONFIG["absence_thresh"]
    ba     = CONFIG["boost_alpha"]
    sa     = CONFIG["suppress_alpha"]

    if mode == "fixed":
        # Handled externally (absent strategy handles negatives)
        return ba

    elif mode == "linear":
        if max_sim >= thresh:
            conf = (max_sim - thresh) / max(1.0 - thresh, 1e-6)
            return ba * min(conf * 2.0, 1.0)
        else:
            # Let absent_strategy handle suppression
            return -sa * min((thresh - max_sim) / max(thresh, 1e-6) * 2.0, 1.0)

    elif mode == "sigmoid":
        k   = CONFIG["adaptive_sigmoid_k"]
        mid = thresh
        sig = torch.sigmoid(torch.tensor(k * (max_sim - mid))).item()
        # map [0,1] sigmoid to [-suppress_alpha, +boost_alpha]
        return -sa + (ba + sa) * sig

    elif mode == "zone":
        lo, hi = thresh - 0.01, thresh + 0.01
        if max_sim < lo:
            return -sa
        elif max_sim > hi:
            return ba
        else:
            return 0.0

    return ba   # fallback


# =============================================================================
# Setup
# =============================================================================

def setup(model, processor) -> None:
    global _model, _processor, _spatial, _need_hidden
    _model, _processor = model, processor
    _spatial = getattr(model.config.vision_config, "spatial_merge_size", 2)
    _need_hidden = CONFIG["logit_blend_gamma"] > 0.0

    _calibrate_heads()

    patch.patch_model(model, "vaf", max(float(CONFIG["boost_alpha"]), 1e-6))
    _sync_patch_state(CONFIG["boost_alpha"])


def _calibrate_heads() -> None:
    """Static head calibration on N_CALIB POPE adversarial samples."""
    if CONFIG["head_top_k_pct"] <= 0.0 or CONFIG["head_mode"] == "per_sample":
        patch._STATE["head_mask"] = None
        if CONFIG["head_mode"] != "per_sample":
            print("  [srf_v2] head_top_k_pct=0 → all heads active")
        else:
            print("  [srf_v2] head_mode=per_sample → calibrating per query (no global mask)")
        return

    from datasets import load_dataset as hf_load
    from qwen_vl_utils import process_vision_info as pvi

    n      = CONFIG["n_calib"]
    print(f"  [srf_v2] Calibrating heads on {n} POPE adversarial samples…")
    ds     = hf_load("lmms-lab/POPE", split="test")
    rows   = [r for r in ds
              if str(r.get("category", r.get("type", ""))).strip().lower() == "adversarial"]
    rng    = random.Random(0)
    rng.shuffle(rows)
    rows   = rows[:n]

    img_id  = _processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    device  = next(_model.parameters()).device
    calib_inputs, img_ranges = [], []

    for r in rows:
        img  = r["image"].convert("RGB")
        q    = str(r["question"]).strip() + "\nAnswer with Yes or No only."
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                              {"type": "text",  "text":  q}]}]
        text = _processor.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        vis, _ = pvi(msgs)
        inp  = _processor(text=[text], images=vis, return_tensors="pt",
                          padding=True).to(device)
        ids  = inp["input_ids"][0].tolist()
        s    = ids.index(img_id)
        e    = len(ids) - 1 - ids[::-1].index(img_id)
        calib_inputs.append(inp)
        img_ranges.append((s, e))

    patch.identify_visual_heads(_model, calib_inputs, img_ranges,
                                 CONFIG["head_top_k_pct"])
    n_sel = int(patch._STATE["head_mask"].sum().item())
    print(f"  [srf_v2] {n_sel} vision-aware heads (top "
          f"{CONFIG['head_top_k_pct']*100:.0f}%)")


def _sync_patch_state(alpha: float, absent: bool = False) -> None:
    """Push current CONFIG + computed alpha into shared patch _STATE."""
    patch._STATE["vaf_layer_start"]    = CONFIG["layer_start"]
    patch._STATE["vaf_layer_end"]      = CONFIG["layer_end"]
    patch._STATE["vaf_beta"]           = CONFIG["sys_beta"]
    patch._STATE["srf_background_eps"] = CONFIG["background_eps"]
    patch._STATE["srf_bias_mode"]      = CONFIG["bias_mode"]
    patch._STATE["srf_interp_lambda"]  = CONFIG["interp_lambda"]
    patch._STATE["srf_prob_floor"]     = CONFIG["prob_floor"]
    patch._STATE["srf_img_scale"]      = CONFIG["img_scale"]
    patch._STATE["srf_apply_phase"]    = CONFIG["phase"]
    patch._STATE["value"]              = alpha

    # Text suppression: only enable when absent + absent_strategy includes it
    strat = CONFIG["absent_strategy"]
    if absent and strat in ("suppress_text", "suppress_both"):
        patch._STATE["srf_text_beta"]         = CONFIG["text_suppress_beta"]
        patch._STATE["srf_text_layer_start"]  = CONFIG["text_suppress_layer_start"]
        patch._STATE["srf_text_layer_end"]    = CONFIG["text_suppress_layer_end"]
    else:
        patch._STATE["srf_text_beta"]         = 0.0


# =============================================================================
# prepare_sample
# =============================================================================

def prepare_sample(inputs, img_start: int, img_end: int,
                   image, question: str, model, processor) -> None:
    global _is_absent

    patch.update_sample(img_start, img_end)
    patch._STATE["method"] = "srf"

    source   = CONFIG["saliency_source"]
    grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)

    sal_clip = sal_dino = sal_hssa = None
    max_sim  = 0.0

    # ── CLIP saliency ─────────────────────────────────────────────────────────
    if source in ("clip", "clip_dino", "clip_hssa"):
        sal_clip, max_sim = _compute_multiscale_clip(
            image, question, grid_h, grid_w,
            scales    = CONFIG["clip_scales"],
            top_k_pct = CONFIG["clip_top_k_pct"],
            clip_model= CONFIG["clip_model"],
        )

    # ── DINOv2 saliency ───────────────────────────────────────────────────────
    if source in ("dino", "clip_dino"):
        sal_dino = _compute_dino_salience(
            image, grid_h, grid_w, CONFIG["dino_top_k_pct"]
        )

    # ── HSSA saliency ─────────────────────────────────────────────────────────
    if source in ("hssa", "clip_hssa"):
        res_h    = hssa_sal.compute_hssa_salience(
            model, inputs, img_start, img_end,
            layer_idx = CONFIG["hssa_layer"],
            top_k_pct = CONFIG["hssa_top_k_pct"],
        )
        sal_hssa = res_h.saliency if CONFIG["hssa_use_soft"] else res_h.mask

    # ── Combine saliency sources ──────────────────────────────────────────────
    if source == "clip":
        sal = sal_clip
    elif source == "dino":
        sal = sal_dino
    elif source == "hssa":
        sal = sal_hssa
    elif source == "clip_dino":
        w_d = CONFIG["dino_weight"]
        raw = (1.0 - w_d) * sal_clip + w_d * sal_dino
        sal = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    elif source == "clip_hssa":
        w_h = CONFIG["hssa_weight"]
        raw = (1.0 - w_h) * sal_clip + w_h * sal_hssa
        sal = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    else:
        sal = sal_clip

    # ── Absent / present detection ────────────────────────────────────────────
    thresh  = CONFIG["absence_thresh"]
    strat   = CONFIG["absent_strategy"]
    absent  = (source in ("clip", "clip_dino", "clip_hssa")) and (max_sim < thresh)
    _is_absent = absent

    if absent:
        # Object likely absent: decide what to do
        if strat == "nothing":
            # No targeted intervention; fall back to static head boost only
            patch._STATE["salience_mask"] = None
            alpha = CONFIG["boost_alpha"]
        elif strat in ("suppress_salient", "suppress_both"):
            # Suppress the near-miss region (the visual hallucination trigger)
            patch._STATE["salience_mask"] = sal
            alpha = -abs(CONFIG["suppress_alpha"])
        else:
            patch._STATE["salience_mask"] = None
            alpha = CONFIG["boost_alpha"]
    else:
        # Object likely present: adaptive alpha + boost salient region
        alpha = _adaptive_alpha(max_sim)
        patch._STATE["salience_mask"] = sal

    # ── Per-sample head selection ──────────────────────────────────────────────
    if CONFIG["head_mode"] == "per_sample":
        patch.identify_visual_heads(
            model, [inputs], [(img_start, img_end)],
            CONFIG["head_top_k_pct"]
        )

    _sync_patch_state(alpha, absent=absent)


# =============================================================================
# get_logits  — replaces model.generate() for single-token prediction
# =============================================================================

def get_logits(model, inputs: dict) -> torch.Tensor:
    """
    Run one forward pass and return final-token logits (1, vocab).

    If logit_blend_gamma > 0 (SLA — Self-Logits Augmentation):
        logits_out = (1-γ) * final_logits + γ * mean(penultimate_k_logits)
    This brings back visual signal from layers before final-layer language drift.

    The patched softmax is active during this forward pass (SRF intervention applied).
    """
    gamma = CONFIG["logit_blend_gamma"]
    device = next(model.parameters()).device
    model_inp = {k: v.to(device) if hasattr(v, "to") else v
                 for k, v in inputs.items()}

    if gamma <= 0.0:
        with torch.inference_mode():
            out = model(**model_inp)
        return out.logits[:, -1, :].float()

    # SLA: need hidden states. Use no_grad (not inference_mode) so the hidden
    # state tensors can be re-used as inputs to lm_head outside the context.
    with torch.no_grad():
        out = model(**model_inp, output_hidden_states=True)

    final_logits = out.logits[:, -1, :].float()   # (1, vocab)

    # Penultimate layers: average over last-k hidden states
    k        = CONFIG["logit_blend_k"]
    lm_head  = model.lm_head   # Qwen2_5_VLForConditionalGeneration.lm_head
    lm_dtype = next(lm_head.parameters()).dtype   # bfloat16
    hs       = out.hidden_states   # tuple (embed, layer_0, ..., layer_N)
    # Take last k hidden states before the final (skip the very last = final)
    prev_k   = hs[-(k + 1) : -1]   # k entries
    with torch.no_grad():
        prev_logits = torch.stack([
            lm_head(h[:, -1, :].to(lm_dtype)).float() for h in prev_k
        ]).mean(0)   # (1, vocab)

    return (1.0 - gamma) * final_logits + gamma * prev_logits


# =============================================================================
# cleanup
# =============================================================================

def cleanup() -> None:
    global _is_absent
    patch._STATE["salience_mask"] = None
    patch._STATE["method"]        = "vaf"
    patch._STATE["srf_text_beta"] = 0.0
    _is_absent = None
