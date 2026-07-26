"""
CLIP-based per-token salience for image-token boosting.

Key design decisions vs naive approach:
  1. Coarse grid (COARSE_GRID = 4×4 or 5×5) instead of Qwen's full token grid.
     Qwen's token grid is 15-23 cells wide — patches are ~15px, too small for
     CLIP to extract meaningful features. We use a coarser grid, compute CLIP
     similarities, then upsample back to the token grid with bilinear
     interpolation.

  2. Object noun extraction: POPE questions are "Is there a <noun> in the image?"
     We extract just the noun for CLIP similarity — "chair" works much better
     than "Is there a chair in the image?\nAnswer with Yes or No only."

  3. Absence detection: if max patch similarity < ABSENCE_THRESHOLD the object
     is likely not in the image. We return a uniform mask (no targeted boost)
     rather than boosting the "least wrong" region.
     The caller can check `clip_result.object_present` to decide strategy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

_CLIP_MODEL      = None
_CLIP_PROCESSOR  = None
_CLIP_MODEL_NAME = None   # track which model is currently loaded
_CLIP_MODEL_KIND = None   # "clip" or "siglip"
# Use ViT-B/32 (0.31 GB fp16) instead of ViT-L/14 (0.86 GB) to fit alongside Qwen 3B (7.5 GB).
# Borrow-GPU strategy: load on CPU, move to CUDA for inference, move back after.
# ViT-B/32 on GPU: ~0.07s/sample vs ~40s/sample on CPU = 570x speedup.
_CLIP_STORAGE_DEVICE = "cpu"
_CLIP_INFER_DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
_CLIP_DEFAULT_MODEL  = "openai/clip-vit-base-patch32"  # 0.31 GB fp16 — fits with Qwen

COARSE_GRID    = 7       # NxN coarse grid for CLIP — 7×7 gives ~50px patches, better spatial precision
ABSENCE_THRESH = 0.20    # default patch max_sim below this → object probably absent

# ── Prompt-ensemble templates ─────────────────────────────────────────────────
# Averaging text embeddings from multiple templates consistently outperforms a
# single template (CLIP paper: +3.5pp on ImageNet with 80 templates).
# These 5 were chosen to cover: bare noun, photo framing (a/the), existence
# framing, and scene-container framing — all POPE-relevant phrasings.
# Applied to just the extracted noun (e.g. "person"), not the full question.
_PRESENCE_TEMPLATES = [
    "{noun}",
    "a photo of a {noun}",
    "a photo of the {noun}",
    "there is a {noun} in this photo",
    "an image containing a {noun}",
]

# Per-model calibrated patch absence thresholds
_MODEL_ABSENCE_THRESH = {
    "openai/clip-vit-base-patch32":   0.20,
    "openai/clip-vit-base-patch16":   0.20,
    "openai/clip-vit-large-patch14":  0.20,
    "google/siglip-base-patch16-224": 0.06,   # SigLIP sims are ~0.02–0.08
}

# Full-image similarity thresholds (whole uncropped image vs noun).
# Full-image sims are higher + more discriminating than patch max:
#   present → ~0.25–0.45  |  absent → ~0.15–0.28  (ViT-B/32 POPE)
# Threshold sits in the overlap zone; combined gate handles ambiguity.
_FULL_IMG_ABSENCE_THRESH = {
    "openai/clip-vit-base-patch32":   0.24,
    "openai/clip-vit-base-patch16":   0.24,
    "openai/clip-vit-large-patch14":  0.28,
    "google/siglip-base-patch16-224": 0.08,
}

# Spatial contrast threshold: top-30% patch mean / all-patches mean.
# Present objects create concentrated high-sim patches (contrast > 1.4).
# Absent: uniform low sims → low contrast (~1.2–1.4).
_CONTRAST_THRESH = 1.40

# Contrastive gap threshold: sim("a photo with noun") − sim("a photo without noun").
# Present → gap typically > 0.005; absent → gap ≈ 0 or slightly negative.
_CONTRASTIVE_GAP_THRESH = 0.005

# Entropy threshold: entropy of softmax(saliency) in [0,1]; 0=spike, 1=flat.
# From val-set data: GT-Yes mean=0.593, GT-No mean=0.763. Midpoint ≈ 0.68.
_ENTROPY_THRESH = 0.68

# ── v3 thresholds ─────────────────────────────────────────────────────────────
# Full-image threshold: optimal from val-set sweep (60 samples, seed=42).
#   t=0.21 → acc=0.817 F1=0.800 TPR=0.733 FPR=0.100  (gate_full alone, no fallbacks)
#   t=0.24 → acc=0.633 F1=0.421 TPR=0.267 FPR=0.000  (current/was too conservative)
# NOTE: This constant is the fallback default only. All callers in srf.py pass
# full_img_thresh=SALIENCY["clip_fallback_thresh"] (config default 0.20) so this
# constant is only used if clip_fallback_thresh is somehow None. Keep in sync with
# config.py SRF_ARCH_PARAMS["clip_fallback_thresh"] = 0.20.
_FULL_IMG_THRESH_V3 = 0.20

# Raw-sim entropy: computed on unnormalized patch cosine sims (not saliency [0,1]).
# Temperature=0.02 because raw sims span ~0.10 (present peak ~0.22 vs bg ~0.17).
# Present: one or few patches peak → peaked softmax → low entropy.
# Absent:  all patches similar low sim → uniform softmax → high entropy (→ 1.0).
_RAW_ENTROPY_THRESH       = 0.95    # < this → peaked → present
_RAW_ENTROPY_TEMPERATURE  = 0.02

# Cross-scale localization agreement: Jaccard of top-30% patches, coarsest vs finest.
# Present: object at same location across scales → high overlap.
# Absent:  noise peaks at different spots each scale → low overlap.
# Random baseline Jaccard ≈ 0.09 (30% top-k on 49-token grid); 0.30 is well above noise.
_CROSS_SCALE_THRESH = 0.30

# Blur delta: sim(real_image, noun) - sim(GaussianBlur(image, r=15), noun).
# Present: high-frequency object features lost in blur → large positive delta.
# Absent:  no object to lose → delta ≈ 0.
_BLUR_DELTA_THRESH = 0.005
_BLUR_RADIUS       = 15.0

# GradCAM absence thresholds: use full-image similarity (not patch max)
# Full-image sim is higher and cleaner: ~0.5-0.8 present vs 0.2-0.4 absent for CLIP
_GRADCAM_ABSENCE_THRESH: dict = {
    "openai/clip-vit-base-patch32":   0.17,   # full-img sims ~0.17-0.26; 0.17 keeps most samples
    "openai/clip-vit-base-patch16":   0.18,
    "openai/clip-vit-large-patch14":  0.22,
    "google/siglip-base-patch16-224": 0.10,
}


def _model_kind(model_name: str) -> str:
    """Return 'siglip' or 'clip' based on model name."""
    return "siglip" if "siglip" in model_name.lower() else "clip"


@dataclass
class ClipSalienceResult:
    mask: torch.Tensor          # float (n_img_tokens,) — binary top-k (0.0 or 1.0)
    saliency: torch.Tensor      # float (n_img_tokens,) — soft continuous [0,1], for visualisation
    max_sim: float              # highest patch similarity (multi-scale max)
    object_present: bool        # combined gate decision
    query_noun: str             # extracted noun used for CLIP
    # ── Multi-signal presence diagnostics (optional, default 0/False) ─────────
    full_img_sim:    float = 0.0    # full-image CLIP similarity (uncropped)
    patch_contrast:  float = 0.0    # top-30% mean / all-patches mean
    gate_full:       bool  = False  # full_img_sim >= full_img_thresh
    gate_patch:      bool  = False  # patch max_sim >= patch_thresh
    gate_contrast:   bool  = False  # patch_contrast >= _CONTRAST_THRESH
    # ── v2 signals ────────────────────────────────────────────────────────────
    contrastive_gap:  float = 0.0   # sim("with noun") − sim("without noun")
    patch_entropy:    float = 0.0   # entropy of softmax(saliency): 0=spike, 1=flat
    gate_contrastive: bool  = False # contrastive_gap >= _CONTRASTIVE_GAP_THRESH
    gate_entropy:     bool  = False # patch_entropy < _ENTROPY_THRESH
    # ── v3 signals ────────────────────────────────────────────────────────────
    raw_entropy:      float = 0.0   # entropy of softmax(raw_patch_sims/t): 0=spike, 1=uniform
    cross_scale_iou:  float = 0.0   # Jaccard overlap of top-30% patches: coarsest vs finest scale
    blur_delta:       float = 0.0   # sim(real_image, noun) - sim(GaussianBlur(image), noun)
    gate_raw_entropy: bool  = False  # raw_entropy < _RAW_ENTROPY_THRESH
    gate_cross_scale: bool  = False  # cross_scale_iou >= _CROSS_SCALE_THRESH
    gate_blur_delta:  bool  = False  # blur_delta >= _BLUR_DELTA_THRESH


def _load_clip(model_name: str = _CLIP_DEFAULT_MODEL) -> tuple:
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_MODEL_NAME, _CLIP_MODEL_KIND
    if _CLIP_MODEL is not None and _CLIP_MODEL_NAME == model_name:
        return _CLIP_MODEL, _CLIP_PROCESSOR   # already loaded
    if _CLIP_MODEL is not None and _CLIP_MODEL_NAME != model_name:
        # Unload old model to free memory before loading new one
        _CLIP_MODEL = None
        _CLIP_PROCESSOR = None
        if _CLIP_INFER_DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()

    kind = _model_kind(model_name)
    print(f"  [clip_salience] Loading {model_name} ({kind}) on {_CLIP_STORAGE_DEVICE}…")
    dtype = torch.float16 if _CLIP_INFER_DEVICE == "cuda" else torch.float32

    if kind == "siglip":
        from transformers import SiglipModel, SiglipProcessor
        _CLIP_MODEL     = SiglipModel.from_pretrained(model_name, torch_dtype=dtype,
                                                       ignore_mismatched_sizes=True,
                                                       ).to(_CLIP_STORAGE_DEVICE).eval()
        _CLIP_PROCESSOR = SiglipProcessor.from_pretrained(model_name)
    else:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP_MODEL     = CLIPModel.from_pretrained(model_name, torch_dtype=dtype,
                                                     ).to(_CLIP_STORAGE_DEVICE).eval()
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)

    _CLIP_MODEL_NAME = model_name
    _CLIP_MODEL_KIND = kind
    return _CLIP_MODEL, _CLIP_PROCESSOR


def _clip_model_to_infer() -> None:
    """Move CLIP to inference device (called just before CLIP forward pass)."""
    global _CLIP_MODEL
    if _CLIP_MODEL is not None and _CLIP_INFER_DEVICE != _CLIP_STORAGE_DEVICE:
        _CLIP_MODEL.to(_CLIP_INFER_DEVICE)


def _clip_model_to_storage() -> None:
    """Move CLIP back to storage device (CPU) after inference to free VRAM for Qwen."""
    global _CLIP_MODEL
    if _CLIP_MODEL is not None and _CLIP_INFER_DEVICE != _CLIP_STORAGE_DEVICE:
        _CLIP_MODEL.to(_CLIP_STORAGE_DEVICE)
        if _CLIP_INFER_DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()


def extract_query_noun(prompt: str) -> str:
    """
    Extract object noun phrase for CLIP. Handles POPE + MME question formats.

    Priority order (most specific first):
      1. POPE/MME existence  — "Is there a/an X in the/this image?"
      2. MME count           — "Is there only one X?" / "Are there N X?"
      3. MME position        — "Is X to the left/right of Y?" → "X and Y"
      4. MME color/attribute — "Is the X [color]?" → "[color] X"
      5. Generic existence   — "Is there a/an X?"
      6. Fallback            — first sentence stripped of "?"
    """
    q = prompt.split("\n")[0].strip()
    # Accept "image" OR any similar word — tolerates dataset typos like "imange"
    _IN_IMG = r"in\s+(?:the|this)\s+\w+"

    # 1 — POPE / MME existence
    m = re.search(r"is there an?\s+([\w\s]+?)\s+" + _IN_IMG, q, re.IGNORECASE)
    if m:
        return m.group(1).lower().strip()
    m = re.search(r"do you see an?\s+([\w\s]+?)\s+" + _IN_IMG, q, re.IGNORECASE)
    if m:
        return m.group(1).lower().strip()

    # 2 — MME count
    m = re.search(r"is there only (?:one|1)\s+([\w\s]+?)\s+" + _IN_IMG, q, re.IGNORECASE)
    if m:
        return m.group(1).lower().strip()
    m = re.search(
        r"are there (?:only\s+)?(?:\d+|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"([\w\s]+?)\s+" + _IN_IMG,
        q, re.IGNORECASE,
    )
    if m:
        noun = m.group(1).lower().strip()
        if noun.endswith("s") and len(noun) > 3:
            noun = noun[:-1]
        return noun
    m = re.search(r"is there only (?:one|1)\s+([\w\s]+?)[\?\.!\n]", q, re.IGNORECASE)
    if m:
        return m.group(1).lower().strip()

    # 3 — MME position (MUST precede color to avoid greedy mismatch)
    m = re.search(
        r"is the\s+([\w\s]+?)\s+"
        r"(?:to the\s+|on the\s+)?(?:left|right|above|below|on top|in front|behind|next\s+to)"
        r"[\w\s]*?(?:of\s+the\s+|the\s+)([\w\s]+?)"
        r"(?:\s+" + _IN_IMG + r")?\s*\?",
        q, re.IGNORECASE,
    )
    if m:
        return f"{m.group(1).lower().strip()} and {m.group(2).lower().strip()}"

    # 4 — MME color/attribute: "Is the X yellow?" → "yellow X"
    m = re.search(
        r"^is the\s+([\w][\w\s]*?)\s+([\w]+(?:\s+and\s+[\w]+)?)\s*\?",
        q, re.IGNORECASE,
    )
    if m:
        return f"{m.group(2).lower().strip()} {m.group(1).lower().strip()}"

    # 5 — Generic existence (no "in the image")
    m = re.search(r"is there an?\s+([\w\s]+?)[\?\.!\n]", q, re.IGNORECASE)
    if m:
        return m.group(1).lower().strip()

    # 6 — Fallback
    return q.split("?")[0].strip()


def compute_clip_salience(
    image: Image.Image,
    text: str,
    grid_h: int,
    grid_w: int,
    top_k_pct: float = 0.3,
    coarse_n: int = COARSE_GRID,
    clip_model_name: str = _CLIP_DEFAULT_MODEL,
) -> ClipSalienceResult:
    """
    Compute CLIP-based salience mask at coarse grid resolution, then upsample
    to Qwen token grid (grid_h × grid_w).

    Returns ClipSalienceResult with:
      .mask           : float tensor (grid_h * grid_w,), binary top-k
      .saliency       : float tensor (grid_h * grid_w,), soft continuous [0,1]
      .max_sim        : float, highest patch-text similarity
      .object_present : bool, False if max_sim < ABSENCE_THRESH
      .query_noun     : str, the noun used for CLIP lookup

    Changes vs v1:
      - Default model: ViT-L/14 (stronger than ViT-B/32)
      - Default grid: 7×7 (finer spatial resolution)
      - Now also returns .saliency for continuous heatmap visualisation
    """
    model, processor = _load_clip(clip_model_name)
    noun = extract_query_noun(text)
    kind = _CLIP_MODEL_KIND or _model_kind(clip_model_name)
    absence_thresh = _MODEL_ABSENCE_THRESH.get(clip_model_name, ABSENCE_THRESH)

    W, H = image.size
    ph = H / coarse_n
    pw = W / coarse_n

    patches = []
    for row in range(coarse_n):
        for col in range(coarse_n):
            y0, y1 = int(row * ph), int((row + 1) * ph)
            x0, x1 = int(col * pw), int((col + 1) * pw)
            patch = image.crop((x0, y0, x1, y1))
            patches.append(patch)

    _clip_model_to_infer()   # move CLIP to GPU for fast inference
    with torch.no_grad():
        if kind == "siglip":
            img_inputs = processor(images=patches, return_tensors="pt",
                                   padding="max_length").to(_CLIP_INFER_DEVICE)
            txt_inputs = processor(text=[noun], return_tensors="pt",
                                   padding="max_length", truncation=True,
                                   max_length=64).to(_CLIP_INFER_DEVICE)
        else:
            img_inputs = processor(images=patches, return_tensors="pt",
                                   padding=True).to(_CLIP_INFER_DEVICE)
            txt_inputs = processor(text=[noun], return_tensors="pt",
                                   padding=True, truncation=True,
                                   max_length=77).to(_CLIP_INFER_DEVICE)

        img_feats  = model.get_image_features(**img_inputs)      # (n_patches, d)
        img_feats  = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feat   = model.get_text_features(**txt_inputs)        # (1, d)
        txt_feat   = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        sims = (img_feats @ txt_feat.T).squeeze(-1).cpu()         # (n_patches,)
    _clip_model_to_storage()  # free VRAM before Qwen forward pass

    max_sim        = float(sims.max())
    object_present = max_sim >= absence_thresh

    # Reshape coarse similarity grid → (1, 1, coarse_n, coarse_n)
    sim_grid = sims.view(1, 1, coarse_n, coarse_n)

    # Upsample to token grid with bilinear interpolation
    sim_token = F.interpolate(
        sim_grid, size=(grid_h, grid_w), mode="bilinear", align_corners=False
    ).squeeze()   # (grid_h, grid_w)

    # Flatten and build top-k binary mask
    sim_flat = sim_token.reshape(-1)   # (grid_h * grid_w,)
    n_tokens = len(sim_flat)

    # Soft saliency: min-max normalise the raw similarities → [0,1]
    s_min, s_max = sim_flat.min(), sim_flat.max()
    saliency = (sim_flat - s_min) / (s_max - s_min + 1e-8)

    if object_present:
        k        = max(1, round(n_tokens * top_k_pct))
        topk_idx = sim_flat.topk(k).indices
        mask     = torch.zeros(n_tokens, dtype=torch.float32)
        mask[topk_idx] = 1.0
    else:
        # Object absent: uniform mask, keep the actual CLIP saliency map.
        # Bug fix: was overwriting saliency with 0.5 ("neutral"), which is NOT neutral —
        # the boost logic computes alpha*0.5 - eps*0.5 = a weak positive boost.
        # Callers must check object_present and decide the absent behaviour themselves
        # (e.g. set salience_mask=None for no boost, or use uniform alpha).
        mask = torch.ones(n_tokens, dtype=torch.float32)

    return ClipSalienceResult(
        mask=mask, saliency=saliency, max_sim=max_sim,
        object_present=object_present, query_noun=noun,
    )


def compute_clip_salience_multiscale(
    image: Image.Image,
    text: str,
    grid_h: int,
    grid_w: int,
    top_k_pct: float = 0.3,
    coarse_scales: tuple = (3, 5, 7),
    clip_model_name: str = _CLIP_DEFAULT_MODEL,
) -> ClipSalienceResult:
    """
    Multi-scale CLIP saliency: run patch-CLIP at multiple grid sizes, upsample all
    to (grid_h, grid_w), take elementwise max.

    Handles objects of varying sizes better than a single fixed grid:
    - Small grid (3×3): large patches (~150px) → better context for small/occluded objects
    - Large grid (7×7): fine spatial resolution → better localization for large objects

    Returns same ClipSalienceResult interface as compute_clip_salience().
    """
    noun = extract_query_noun(text)
    absence_thresh = _MODEL_ABSENCE_THRESH.get(clip_model_name, ABSENCE_THRESH)

    scale_results: list[ClipSalienceResult] = []
    for coarse_n in coarse_scales:
        res = compute_clip_salience(
            image, noun, grid_h, grid_w,
            top_k_pct=top_k_pct,
            coarse_n=coarse_n,
            clip_model_name=clip_model_name,
        )
        scale_results.append(res)

    # Elementwise max of saliency maps across scales
    stacked  = torch.stack([r.saliency for r in scale_results], dim=0)  # (n_scales, n_tokens)
    combined = stacked.max(dim=0).values                                  # (n_tokens,)

    # Re-normalise the combined map
    c_min, c_max = combined.min(), combined.max()
    saliency = (combined - c_min) / (c_max - c_min + 1e-8)

    # max_sim = best across all scales (most context-aware crop)
    max_sim        = max(r.max_sim for r in scale_results)
    object_present = max_sim >= absence_thresh

    n_tokens = grid_h * grid_w
    if object_present:
        k        = max(1, round(n_tokens * top_k_pct))
        topk_idx = saliency.topk(k).indices
        mask     = torch.zeros(n_tokens, dtype=torch.float32)
        mask[topk_idx] = 1.0
    else:
        mask     = torch.ones(n_tokens, dtype=torch.float32)
        saliency = torch.full((n_tokens,), 0.5)

    return ClipSalienceResult(mask=mask, saliency=saliency, max_sim=max_sim,
                               object_present=object_present, query_noun=noun)


def compute_full_image_sim(
    image: Image.Image,
    noun: str,
    clip_model_name: str = _CLIP_DEFAULT_MODEL,
) -> float:
    """
    Single CLIP forward on the UNCROPPED full image → cosine similarity with noun.

    Full-image context gives a much cleaner presence/absence signal than patch max:
    patches lose surrounding context and suffer boundary artifacts. Full-image sim
    typically shows a larger gap between present (0.25–0.45) and absent (0.15–0.28)
    objects for ViT-B/32 on POPE-style questions.
    """
    model, processor = _load_clip(clip_model_name)
    kind = _CLIP_MODEL_KIND or _model_kind(clip_model_name)

    prompts = [t.format(noun=noun) for t in _PRESENCE_TEMPLATES]
    _clip_model_to_infer()
    with torch.no_grad():
        if kind == "siglip":
            img_inp = processor(images=[image], return_tensors="pt",
                                padding="max_length").to(_CLIP_INFER_DEVICE)
            txt_inp = processor(text=prompts, return_tensors="pt",
                                padding="max_length", truncation=True,
                                max_length=64).to(_CLIP_INFER_DEVICE)
        else:
            img_inp = processor(images=[image], return_tensors="pt",
                                padding=True).to(_CLIP_INFER_DEVICE)
            txt_inp = processor(text=prompts, return_tensors="pt",
                                padding=True, truncation=True,
                                max_length=77).to(_CLIP_INFER_DEVICE)

        img_feat  = model.get_image_features(**img_inp)
        img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feats = model.get_text_features(**txt_inp)              # (n_templates, d)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        txt_feat  = txt_feats.mean(dim=0, keepdim=True)
        txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sim = float((img_feat @ txt_feat.T).squeeze().cpu())
    _clip_model_to_storage()
    return sim


def _compute_full_image_contrastive(
    image: Image.Image,
    noun: str,
    clip_model_name: str = _CLIP_DEFAULT_MODEL,
) -> tuple:
    """
    One image + three text encodes in a single batched forward pass.

    Returns:
        full_img_sim    — sim(image, noun)
        contrastive_gap — sim(image, "a photo with a {noun}") − sim(image, "a photo without a {noun}")
    """
    model, processor = _load_clip(clip_model_name)
    kind = _CLIP_MODEL_KIND or _model_kind(clip_model_name)

    prompts = [noun, f"a photo with a {noun}", f"a photo without a {noun}"]

    _clip_model_to_infer()
    with torch.no_grad():
        if kind == "siglip":
            img_inp = processor(images=[image], return_tensors="pt",
                                padding="max_length").to(_CLIP_INFER_DEVICE)
            txt_inp = processor(text=prompts, return_tensors="pt",
                                padding="max_length", truncation=True,
                                max_length=64).to(_CLIP_INFER_DEVICE)
        else:
            img_inp = processor(images=[image], return_tensors="pt",
                                padding=True).to(_CLIP_INFER_DEVICE)
            txt_inp = processor(text=prompts, return_tensors="pt",
                                padding=True, truncation=True,
                                max_length=77).to(_CLIP_INFER_DEVICE)

        img_feat = model.get_image_features(**img_inp)
        txt_feat = model.get_text_features(**txt_inp)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat @ txt_feat.T).squeeze(0).cpu()   # shape: (3,)

    _clip_model_to_storage()
    full_img_sim    = float(sims[0])
    contrastive_gap = float(sims[1]) - float(sims[2])     # "with" − "without"
    return full_img_sim, contrastive_gap


def _patch_entropy(saliency: torch.Tensor) -> float:
    """Entropy of a sharpened softmax over saliency values. 0=spike, 1=uniform."""
    import math as _math
    p = torch.softmax(saliency.float() * 20.0, dim=0)
    return float(-(p * (p + 1e-9).log()).sum() / _math.log(len(p)))


def compute_clip_salience_full_gate_v2(
    image: Image.Image,
    text: str,
    grid_h: int,
    grid_w: int,
    top_k_pct: float = 0.3,
    coarse_scales: tuple = (3, 5, 7),
    clip_model_name: str = _CLIP_DEFAULT_MODEL,
) -> "ClipSalienceResult":
    """
    Full-gate v2: five independent presence signals, score-based decision.

    New vs v1:
      4. Contrastive text probe — sim("a photo with noun") − sim("a photo without noun").
         Explicitly tests for vs against presence. Cheap: one extra text encode batched.
      5. Patch entropy gate — entropy of softmax(saliency). Present objects create
         concentrated patches (low entropy); absent → uniform (high entropy).

    Combined decision (score ≥ 2 out of 4 active signals):
      score = gate_full + gate_contrastive + gate_entropy + gate_contrast
      object_present = score >= 2

    gate_patch dropped (accuracy=0.500 on val set = random).
    """
    import math as _math

    noun            = extract_query_noun(text)
    absence_thresh  = _MODEL_ABSENCE_THRESH.get(clip_model_name, ABSENCE_THRESH)
    full_img_thresh = _FULL_IMG_ABSENCE_THRESH.get(clip_model_name, 0.24)

    # ── Signal 1 + 4: full-image sim + contrastive gap (one batched call) ───
    full_img_sim, contrastive_gap = _compute_full_image_contrastive(
        image, noun, clip_model_name
    )

    # ── Signal 2: multi-scale patches → spatial map + patch_max_sim ─────────
    scale_results = []
    for coarse_n in coarse_scales:
        res = compute_clip_salience(
            image, noun, grid_h, grid_w,
            top_k_pct=top_k_pct,
            coarse_n=coarse_n,
            clip_model_name=clip_model_name,
        )
        scale_results.append(res)

    stacked  = torch.stack([r.saliency for r in scale_results], dim=0)
    combined = stacked.max(dim=0).values
    c_min, c_max = combined.min(), combined.max()
    saliency = (combined - c_min) / (c_max - c_min + 1e-8)

    patch_max_sim = max(r.max_sim for r in scale_results)

    # ── Signal 3: patch spatial contrast ────────────────────────────────────
    n_tokens = grid_h * grid_w
    k_top    = max(1, round(n_tokens * 0.30))
    top_mean = float(saliency.topk(k_top).values.float().mean())
    all_mean = float(saliency.float().mean()) + 1e-8
    patch_contrast = top_mean / all_mean

    # ── Signal 5: patch entropy ──────────────────────────────────────────────
    entropy = _patch_entropy(saliency)

    # ── Gate decisions ───────────────────────────────────────────────────────
    gate_full        = full_img_sim    >= full_img_thresh
    gate_patch       = patch_max_sim   >= absence_thresh
    gate_contrast    = patch_contrast  >= _CONTRAST_THRESH
    gate_contrastive = contrastive_gap >= _CONTRASTIVE_GAP_THRESH
    gate_entropy     = entropy         <  _ENTROPY_THRESH

    # Score-based: 2 out of 4 signals must agree (gate_patch excluded — random)
    score = int(gate_full) + int(gate_contrastive) + int(gate_entropy) + int(gate_contrast)
    object_present = score >= 2

    # ── Build mask ───────────────────────────────────────────────────────────
    if object_present:
        k        = max(1, round(n_tokens * top_k_pct))
        topk_idx = saliency.topk(k).indices
        mask     = torch.zeros(n_tokens, dtype=torch.float32)
        mask[topk_idx] = 1.0
    else:
        mask     = torch.ones(n_tokens, dtype=torch.float32)
        saliency = torch.full((n_tokens,), 0.5)

    return ClipSalienceResult(
        mask=mask, saliency=saliency,
        max_sim=patch_max_sim,
        object_present=object_present,
        query_noun=noun,
        full_img_sim=full_img_sim,
        patch_contrast=patch_contrast,
        gate_full=gate_full,
        gate_patch=gate_patch,
        gate_contrast=gate_contrast,
        contrastive_gap=contrastive_gap,
        patch_entropy=entropy,
        gate_contrastive=gate_contrastive,
        gate_entropy=gate_entropy,
    )


def compute_clip_salience_multiscale_full_gate(
    image: Image.Image,
    text: str,
    grid_h: int,
    grid_w: int,
    top_k_pct: float = 0.3,
    coarse_scales: tuple = (3, 5, 7),
    clip_model_name: str = _CLIP_DEFAULT_MODEL,
) -> ClipSalienceResult:
    """
    Multi-scale CLIP patches with a multi-signal presence gate.

    Three presence signals are computed and combined:

      1. Full-image CLIP sim (primary) — uncropped image vs noun.
         Most reliable: full context, no boundary artifacts.
         Gate: full_img_sim >= _FULL_IMG_ABSENCE_THRESH[model]

      2. Multi-scale patch max sim (secondary backup) — best patch across
         3×3 / 5×5 / 7×7 grids.
         Gate: patch_max_sim >= _MODEL_ABSENCE_THRESH[model]

      3. Patch spatial contrast (structural signal) — ratio of top-30%
         patch mean to all-patches mean. Present objects create concentrated
         high-sim patches; absent → uniform low sims.
         Gate: patch_contrast >= _CONTRAST_THRESH (1.4)

    Combined decision:
      object_present = gate_full                           # full-image clear
                       OR (gate_patch AND gate_contrast)   # two weak signals agree
                       OR (gate_full_soft AND gate_patch)  # borderline full + patch

    Spatial saliency (for localization when present):
      Multi-scale elementwise max → min-max normalized (same as clip_improved).
    """
    noun          = extract_query_noun(text)
    absence_thresh     = _MODEL_ABSENCE_THRESH.get(clip_model_name, ABSENCE_THRESH)
    full_img_thresh    = _FULL_IMG_ABSENCE_THRESH.get(clip_model_name, 0.24)

    # ── Signal 1: Full-image CLIP similarity ────────────────────────────────
    full_img_sim = compute_full_image_sim(image, noun, clip_model_name)

    # ── Signal 2: Multi-scale patches — spatial map + max_sim ───────────────
    scale_results: list[ClipSalienceResult] = []
    for coarse_n in coarse_scales:
        res = compute_clip_salience(
            image, noun, grid_h, grid_w,
            top_k_pct=top_k_pct,
            coarse_n=coarse_n,
            clip_model_name=clip_model_name,
        )
        scale_results.append(res)

    stacked  = torch.stack([r.saliency for r in scale_results], dim=0)
    combined = stacked.max(dim=0).values
    c_min, c_max = combined.min(), combined.max()
    saliency = (combined - c_min) / (c_max - c_min + 1e-8)

    patch_max_sim = max(r.max_sim for r in scale_results)

    # ── Signal 3: Patch spatial contrast ────────────────────────────────────
    # Use the finest-scale (7×7 or last) raw similarities for contrast measure
    # — we want the sharpest spatial signal.
    all_sims = torch.stack([r.saliency for r in scale_results], dim=0).max(dim=0).values
    n_tokens = grid_h * grid_w
    k_top    = max(1, round(n_tokens * 0.30))
    top_mean = float(all_sims.topk(k_top).values.float().mean())
    all_mean = float(all_sims.float().mean()) + 1e-8
    patch_contrast = top_mean / all_mean

    # ── Combined gate ────────────────────────────────────────────────────────
    gate_full     = full_img_sim  >= full_img_thresh
    gate_full_soft = full_img_sim >= 0.85 * full_img_thresh   # borderline
    gate_patch    = patch_max_sim >= absence_thresh
    gate_contrast = patch_contrast >= _CONTRAST_THRESH

    object_present = (
        gate_full                            # full-image clearly present
        or (gate_patch and gate_contrast)    # two weak patch signals agree
        or (gate_full_soft and gate_patch)   # borderline full + patch backup
    )

    # ── Build mask ───────────────────────────────────────────────────────────
    if object_present:
        k        = max(1, round(n_tokens * top_k_pct))
        topk_idx = saliency.topk(k).indices
        mask     = torch.zeros(n_tokens, dtype=torch.float32)
        mask[topk_idx] = 1.0
    else:
        mask     = torch.ones(n_tokens, dtype=torch.float32)
        saliency = torch.full((n_tokens,), 0.5)

    return ClipSalienceResult(
        mask=mask, saliency=saliency,
        max_sim=patch_max_sim,
        object_present=object_present,
        query_noun=noun,
        full_img_sim=full_img_sim,
        patch_contrast=patch_contrast,
        gate_full=gate_full,
        gate_patch=gate_patch,
        gate_contrast=gate_contrast,
    )


def compute_clip_salience_soft_gate(
    image: Image.Image,
    text: str,
    grid_h: int,
    grid_w: int,
    top_k_pct: float = 0.3,
    coarse_scales: tuple = (3, 5, 7),
    clip_model_name: str = _CLIP_DEFAULT_MODEL,
) -> ClipSalienceResult:
    """
    Multi-scale CLIP with soft presence gate.

    Instead of a hard binary present/absent decision, blend spatial saliency
    with a uniform neutral baseline weighted by presence confidence:

        w = clip(full_img_sim / full_img_thresh, 0, 1)
        saliency_out = w * spatial + (1 - w) * 0.5

    - High sim (clearly present): w≈1 → purely spatial localization
    - Low sim (clearly absent):   w≈0 → uniform 0.5 (no strong bias)
    - Borderline:                 gradual blend — no hard cut-off

    This avoids two failure modes of the hard gate:
      1. False absent: misses present object → soft gate still provides some localization
      2. False present: tries to localize absent object → soft gate dilutes the map

    After blending, re-normalize so the final saliency is in [0, 1].
    """
    noun            = extract_query_noun(text)
    full_img_thresh = _FULL_IMG_ABSENCE_THRESH.get(clip_model_name, 0.24)
    n_tokens        = grid_h * grid_w

    # ── Full-image CLIP similarity (presence confidence) ──────────────────────
    full_img_sim = compute_full_image_sim(image, noun, clip_model_name)

    # ── Multi-scale spatial map ────────────────────────────────────────────────
    scale_results: list[ClipSalienceResult] = []
    for coarse_n in coarse_scales:
        res = compute_clip_salience(
            image, noun, grid_h, grid_w,
            top_k_pct=top_k_pct,
            coarse_n=coarse_n,
            clip_model_name=clip_model_name,
        )
        scale_results.append(res)

    stacked  = torch.stack([r.saliency for r in scale_results], dim=0)
    spatial  = stacked.max(dim=0).values
    s_min, s_max = spatial.min(), spatial.max()
    spatial  = (spatial - s_min) / (s_max - s_min + 1e-8)

    patch_max_sim = max(r.max_sim for r in scale_results)

    # ── Soft blend ────────────────────────────────────────────────────────────
    w       = min(float(full_img_sim) / (full_img_thresh + 1e-8), 1.0)
    neutral = torch.full((n_tokens,), 0.5, dtype=spatial.dtype)
    saliency = w * spatial + (1.0 - w) * neutral

    # Re-normalise to [0, 1] so downstream boosting is consistent
    b_min, b_max = saliency.min(), saliency.max()
    saliency = (saliency - b_min) / (b_max - b_min + 1e-8)

    # Mask: top-k of blended saliency
    k        = max(1, round(n_tokens * top_k_pct))
    topk_idx = saliency.topk(k).indices
    mask     = torch.zeros(n_tokens, dtype=torch.float32)
    mask[topk_idx] = 1.0

    # Treat as "present" when w > 0.5 (more spatial than neutral)
    object_present = w > 0.5

    # Use same gate signals as full_gate for diagnostics
    absence_thresh = _MODEL_ABSENCE_THRESH.get(clip_model_name, ABSENCE_THRESH)
    gate_full      = full_img_sim >= full_img_thresh
    gate_patch     = patch_max_sim >= absence_thresh

    all_sims      = torch.stack([r.saliency for r in scale_results], dim=0).max(dim=0).values
    k_top         = max(1, round(n_tokens * 0.30))
    top_mean      = float(all_sims.topk(k_top).values.float().mean())
    all_mean      = float(all_sims.float().mean()) + 1e-8
    patch_contrast = top_mean / all_mean
    gate_contrast  = patch_contrast >= _CONTRAST_THRESH

    return ClipSalienceResult(
        mask=mask, saliency=saliency,
        max_sim=patch_max_sim,
        object_present=object_present,
        query_noun=noun,
        full_img_sim=full_img_sim,
        patch_contrast=patch_contrast,
        gate_full=gate_full,
        gate_patch=gate_patch,
        gate_contrast=gate_contrast,
    )


def compute_clip_gradcam(
    image: Image.Image,
    text: str,
    grid_h: int,
    grid_w: int,
    top_k_pct: float = 0.3,
    clip_model_name: str = _CLIP_DEFAULT_MODEL,
) -> ClipSalienceResult:
    """
    GradCAM-based saliency: full image → CLIP → cosine sim → backprop through ViT.

    Fixes the two main weaknesses of patch-crop CLIP:
      1. Low similarity: full-image context gives 0.5–0.8 (present) vs 0.2–0.4 (absent)
      2. High entropy: gradient is concentrated on the object → focused, low-entropy map

    Returns same ClipSalienceResult interface as compute_clip_salience() — drop-in.
    max_sim is the full-image cosine similarity (not patch max).
    """
    model, processor = _load_clip(clip_model_name)
    noun = extract_query_noun(text)
    kind = _CLIP_MODEL_KIND or _model_kind(clip_model_name)

    _clip_model_to_infer()

    # Prepare inputs for the full image (single forward, not patches)
    if kind == "siglip":
        img_inputs = processor(images=[image], return_tensors="pt",
                               padding="max_length").to(_CLIP_INFER_DEVICE)
        txt_inputs = processor(text=[noun], return_tensors="pt",
                               padding="max_length", truncation=True,
                               max_length=64).to(_CLIP_INFER_DEVICE)
    else:
        img_inputs = processor(images=[image], return_tensors="pt").to(_CLIP_INFER_DEVICE)
        txt_inputs = processor(text=[noun], return_tensors="pt",
                               truncation=True, max_length=77).to(_CLIP_INFER_DEVICE)

    # Hook: capture second-to-last encoder layer activations
    # (last layer is more task-specific; -2 gives more stable spatial gradients)
    _acts: dict = {}

    def _fwd_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h.retain_grad()
        _acts["h"] = h

    handle = model.vision_model.encoder.layers[-2].register_forward_hook(_fwd_hook)

    try:
        # Text features — no gradient needed
        with torch.no_grad():
            txt_feat = model.get_text_features(**txt_inputs).float()
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        # Image features — with gradient tracking for GradCAM
        model.zero_grad()
        with torch.enable_grad():
            img_feat   = model.get_image_features(**img_inputs).float()
            img_feat_n = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim        = (img_feat_n * txt_feat.detach()).sum()
            sim.backward()
    finally:
        handle.remove()

    _clip_model_to_storage()

    full_sim       = float(sim.detach().cpu().item())
    absence_thresh = _GRADCAM_ABSENCE_THRESH.get(clip_model_name, 0.22)
    object_present = full_sim >= absence_thresh
    n_tokens       = grid_h * grid_w

    h    = _acts.get("h")
    grad = h.grad if h is not None else None

    if h is None or grad is None:
        # Fallback: uniform if hook failed
        saliency = torch.full((n_tokens,), 0.5)
        mask     = torch.ones(n_tokens, dtype=torch.float32)
        return ClipSalienceResult(mask=mask, saliency=saliency, max_sim=full_sim,
                                   object_present=object_present, query_noun=noun)

    # GradCAM: relu(grad * activation), sum over feature dim → (n_seq,)
    # h, grad shape: (1, n_seq, d)
    cam = torch.relu((grad.float() * h.detach().float()).sum(dim=-1)).squeeze(0)  # (n_seq,)

    # Drop CLS token for CLIP (prepended); SigLIP uses mean pooling — no CLS token
    if kind == "clip":
        cam = cam[1:]   # → (n_patches,)

    # Determine native patch grid from n_patches
    # ViT-B/32: 49 → 7×7; ViT-L/14: 256 → 16×16; SigLIP-B/16: 196 → 14×14
    n_patches   = cam.shape[0]
    native_size = int(round(n_patches ** 0.5))
    if native_size * native_size == n_patches:
        native_h = native_w = native_size
    else:
        native_h = native_size
        native_w = (n_patches + native_size - 1) // native_size

    # Bilinear upsample native patch grid → target token grid (grid_h × grid_w)
    cam_2d = cam.reshape(1, 1, native_h, native_w).cpu().float()
    cam_up = F.interpolate(cam_2d, size=(grid_h, grid_w),
                            mode="bilinear", align_corners=False).squeeze()
    cam_flat = cam_up.reshape(-1)   # (n_tokens,)

    c_min, c_max = cam_flat.min(), cam_flat.max()
    saliency = (cam_flat - c_min) / (c_max - c_min + 1e-8)

    if object_present:
        k        = max(1, round(n_tokens * top_k_pct))
        topk_idx = saliency.topk(k).indices
        mask     = torch.zeros(n_tokens, dtype=torch.float32)
        mask[topk_idx] = 1.0
    else:
        mask     = torch.ones(n_tokens, dtype=torch.float32)
        saliency = torch.full((n_tokens,), 0.5)

    return ClipSalienceResult(mask=mask, saliency=saliency, max_sim=full_sim,
                               object_present=object_present, query_noun=noun)


def compute_clip_salience_full_gate_v3(
    image: Image.Image,
    text: str,
    grid_h: int,
    grid_w: int,
    top_k_pct: float = 0.3,
    coarse_scales: tuple = (3, 5, 7),
    clip_model_name: str = _CLIP_DEFAULT_MODEL,
    backup: str = "none",
    full_img_thresh: Optional[float] = None,
    patch_thresh: float = 0.27,
) -> ClipSalienceResult:
    """
    Full-gate v3: three new independent presence signals, single GPU block.

    Changes from v1 (clip_full_gate):
      - full_img_thresh: 0.24 → 0.21 (optimal from val-set sweep: acc=0.817, F1=0.800)
      - gate_patch REMOVED from all gate logic (fires 100% on all samples = no signal)
      - All CLIP passes done in one GPU block (no repeated CPU↔GPU transfers)

    Three new signals:
      1. raw_entropy   — entropy of softmax(raw_patch_sims / t) at finest scale.
                         Unlike old patch_entropy (computed on normalized [0,1] saliency,
                         which destroys absolute-level information), this preserves the
                         real spread of similarities.
                         Present: peaked sims → low entropy.
                         Absent:  all patches uniformly low → high entropy (→ 1.0).

      2. cross_scale_iou — Jaccard overlap of top-30% patches between coarsest (3×3)
                           and finest (7×7) scale (both upsampled to token grid).
                           Present: real object at a stable location → high overlap.
                           Absent:  CLIP attends to random noise at each scale → low overlap.

      3. blur_delta — sim(real_image, noun) - sim(GaussianBlur(r=15)(image), noun).
                      Blurring destroys high-frequency object features while keeping
                      global scene statistics.
                      Present: delta is large and positive (object was contributing).
                      Absent:  delta ≈ 0 (no object features to destroy).

    Gate: gate_full (t=0.21) OR (gate_full_soft AND any_v3_signal)
    gate_patch NOT used in any decision.
    """
    import math as _math

    noun            = extract_query_noun(text)
    model, proc     = _load_clip(clip_model_name)
    kind            = _CLIP_MODEL_KIND or _model_kind(clip_model_name)
    absence_thresh  = _MODEL_ABSENCE_THRESH.get(clip_model_name, ABSENCE_THRESH)
    n_tokens        = grid_h * grid_w
    W, H            = image.size

    # ── Build patch crops at all scales ───────────────────────────────────────
    scale_patches: dict = {}
    for coarse_n in coarse_scales:
        ph, pw = H / coarse_n, W / coarse_n
        scale_patches[coarse_n] = [
            image.crop((int(c * pw), int(r * ph), int((c + 1) * pw), int((r + 1) * ph)))
            for r in range(coarse_n)
            for c in range(coarse_n)
        ]

    # ── Single GPU block: text + full images + all patch scales ───────────────
    _clip_model_to_infer()

    def _enc_imgs(imgs):
        with torch.no_grad():
            if kind == "siglip":
                inp = proc(images=imgs, return_tensors="pt",
                           padding="max_length").to(_CLIP_INFER_DEVICE)
            else:
                inp = proc(images=imgs, return_tensors="pt",
                           padding=True).to(_CLIP_INFER_DEVICE)
            f = model.get_image_features(**inp).float()
            return (f / f.norm(dim=-1, keepdim=True)).cpu()

    def _enc_text_ensemble(n: str) -> torch.Tensor:
        """Encode all presence templates for noun n, return mean-pooled unit vector (1, d)."""
        prompts = [t.format(noun=n) for t in _PRESENCE_TEMPLATES]
        with torch.no_grad():
            if kind == "siglip":
                inp = proc(text=prompts, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=64).to(_CLIP_INFER_DEVICE)
            else:
                inp = proc(text=prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(_CLIP_INFER_DEVICE)
            f = model.get_text_features(**inp).float()          # (n_templates, d)
            f = f / f.norm(dim=-1, keepdim=True)
            f_mean = f.mean(dim=0, keepdim=True)                # (1, d)
            f_mean = f_mean / f_mean.norm(dim=-1, keepdim=True) # re-normalize
        return f_mean.cpu()

    txt_feat = _enc_text_ensemble(noun)                 # (1, d)

    # Full-image: encode real image (+ blurred only when needed for blur_delta backup)
    if backup == "blur_delta":
        from PIL import ImageFilter as _ImageFilter
        blurred     = image.filter(_ImageFilter.GaussianBlur(radius=_BLUR_RADIUS))
        full_feats  = _enc_imgs([image, blurred])        # (2, d)
        full_img_sim = float((full_feats[0:1] @ txt_feat.T).item())
        sim_blurred  = float((full_feats[1:2] @ txt_feat.T).item())
    else:
        full_feats   = _enc_imgs([image])                # (1, d)
        full_img_sim = float((full_feats[0:1] @ txt_feat.T).item())
        sim_blurred  = 0.0                               # unused

    # Patch scales: raw cosine sims (unnormalized) for each scale
    scale_raw_sims: dict = {}
    for coarse_n, patches in scale_patches.items():
        patch_feats = _enc_imgs(patches)                 # (coarse_n^2, d)
        sims        = (patch_feats @ txt_feat.T).squeeze(-1)   # (coarse_n^2,)
        scale_raw_sims[coarse_n] = sims

    _clip_model_to_storage()

    # ── Build multi-scale normalized saliency (upsampled to token grid) ───────
    scale_saliency: dict = {}
    for coarse_n, raw_sims in scale_raw_sims.items():
        sim_grid  = raw_sims.view(1, 1, coarse_n, coarse_n)
        sim_up    = F.interpolate(sim_grid.float(), size=(grid_h, grid_w),
                                   mode="bilinear", align_corners=False).squeeze()
        sim_flat  = sim_up.reshape(-1)
        s_min, s_max = sim_flat.min(), sim_flat.max()
        scale_saliency[coarse_n] = (sim_flat - s_min) / (s_max - s_min + 1e-8)

    stacked  = torch.stack(list(scale_saliency.values()), dim=0)
    combined = stacked.max(dim=0).values
    c_min, c_max = combined.min(), combined.max()
    saliency = (combined - c_min) / (c_max - c_min + 1e-8)

    patch_max_sim = max(float(s.max()) for s in scale_raw_sims.values())

    # ── Patch spatial contrast (existing signal, kept for diagnostics) ────────
    k_top      = max(1, round(n_tokens * 0.30))
    top_mean   = float(saliency.topk(k_top).values.float().mean())
    all_mean   = float(saliency.float().mean()) + 1e-8
    patch_contrast = top_mean / all_mean

    # ── Gate decisions ────────────────────────────────────────────────────────
    # backup="none" (default): gate is purely gate_full — no backup signals needed.
    # Only compute backup signals when the relevant backup mode is active.
    _thresh   = full_img_thresh if full_img_thresh is not None else _FULL_IMG_THRESH_V3
    gate_full = full_img_sim >= _thresh

    # Defaults for diagnostic fields (populated only when backup is active)
    raw_entropy     = 0.0
    cross_scale_iou = 0.0
    blur_delta      = 0.0
    gate_raw_entropy = False
    gate_cross_scale = False
    gate_blur_delta  = False

    if backup == "none":
        gate_patch_presence = patch_max_sim >= patch_thresh
        object_present = gate_full or gate_patch_presence

    elif backup == "cross_scale":
        finest_n     = max(coarse_scales)
        coarsest_n   = min(coarse_scales)
        sal_coarsest = scale_saliency[coarsest_n]
        sal_finest   = scale_saliency[finest_n]
        k_iou        = max(1, round(n_tokens * 0.30))
        top_coarsest = set(sal_coarsest.topk(k_iou).indices.tolist())
        top_finest_s = set(sal_finest.topk(k_iou).indices.tolist())
        inter        = len(top_coarsest & top_finest_s)
        union        = len(top_coarsest | top_finest_s)
        cross_scale_iou  = inter / (union + 1e-8)
        gate_cross_scale = cross_scale_iou >= _CROSS_SCALE_THRESH
        gate_full_soft   = full_img_sim >= 0.85 * _thresh
        object_present   = gate_full or (gate_full_soft and gate_cross_scale)

    elif backup == "blur_delta":
        blur_delta      = full_img_sim - sim_blurred   # sim_blurred set above when backup=="blur_delta"
        gate_blur_delta = blur_delta >= _BLUR_DELTA_THRESH
        gate_full_soft  = full_img_sim >= 0.85 * _thresh
        object_present  = gate_full or (gate_full_soft and gate_blur_delta)

    elif backup == "raw_entropy":
        finest_n     = max(coarse_scales)
        raw_finest   = scale_raw_sims[finest_n]
        p_raw        = torch.softmax(raw_finest.float() / _RAW_ENTROPY_TEMPERATURE, dim=0)
        raw_entropy      = float(-(p_raw * (p_raw + 1e-9).log()).sum() / _math.log(len(p_raw)))
        gate_raw_entropy = raw_entropy < _RAW_ENTROPY_THRESH
        gate_full_soft   = full_img_sim >= 0.85 * _thresh
        object_present   = gate_full or (gate_full_soft and gate_raw_entropy)

    else:
        object_present = gate_full

    gate_patch    = patch_max_sim  >= absence_thresh    # diagnostics only
    gate_contrast = patch_contrast >= _CONTRAST_THRESH  # diagnostics only

    # ── Build mask ────────────────────────────────────────────────────────────
    if object_present:
        k        = max(1, round(n_tokens * top_k_pct))
        topk_idx = saliency.topk(k).indices
        mask     = torch.zeros(n_tokens, dtype=torch.float32)
        mask[topk_idx] = 1.0
    else:
        mask     = torch.ones(n_tokens, dtype=torch.float32)
        saliency = torch.full((n_tokens,), 0.5)

    return ClipSalienceResult(
        mask=mask, saliency=saliency,
        max_sim=patch_max_sim,
        object_present=object_present,
        query_noun=noun,
        full_img_sim=full_img_sim,
        patch_contrast=patch_contrast,
        gate_full=gate_full,
        gate_patch=gate_patch,
        gate_contrast=gate_contrast,
        raw_entropy=raw_entropy,
        cross_scale_iou=cross_scale_iou,
        blur_delta=blur_delta,
        gate_raw_entropy=gate_raw_entropy,
        gate_cross_scale=gate_cross_scale,
        gate_blur_delta=gate_blur_delta,
    )


def get_grid_dims(inputs: dict, spatial_merge_size: int = 2) -> tuple[int, int]:
    """Extract (grid_h, grid_w) from Qwen processor inputs."""
    thw    = inputs["image_grid_thw"][0]
    grid_h = int(thw[1].item()) // spatial_merge_size
    grid_w = int(thw[2].item()) // spatial_merge_size
    return grid_h, grid_w
