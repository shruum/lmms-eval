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

_CLIP_MODEL = None
_CLIP_PROCESSOR = None
# Use ViT-B/32 (0.31 GB fp16) instead of ViT-L/14 (0.86 GB) to fit alongside Qwen 3B (7.5 GB).
# Borrow-GPU strategy: load on CPU, move to CUDA for inference, move back after.
# ViT-B/32 on GPU: ~0.07s/sample vs ~40s/sample on CPU = 570x speedup.
_CLIP_STORAGE_DEVICE = "cpu"
_CLIP_INFER_DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
_CLIP_DEFAULT_MODEL  = "openai/clip-vit-base-patch32"  # 0.31 GB fp16 — fits with Qwen

COARSE_GRID    = 7       # NxN coarse grid for CLIP — 7×7 gives ~50px patches, better spatial precision
ABSENCE_THRESH = 0.20    # max sim below this → object probably absent


@dataclass
class ClipSalienceResult:
    mask: torch.Tensor          # float (n_img_tokens,) — binary top-k (0.0 or 1.0)
    saliency: torch.Tensor      # float (n_img_tokens,) — soft continuous [0,1], for visualisation
    max_sim: float              # highest patch similarity
    object_present: bool        # False if max_sim < ABSENCE_THRESH
    query_noun: str             # extracted noun used for CLIP


def _load_clip(model_name: str = _CLIP_DEFAULT_MODEL) -> tuple:
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None:
        from transformers import CLIPModel, CLIPProcessor
        print(f"  [clip_salience] Loading {model_name} on {_CLIP_STORAGE_DEVICE} (one-time, infers on {_CLIP_INFER_DEVICE})…")
        _CLIP_MODEL     = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if _CLIP_INFER_DEVICE == "cuda" else torch.float32,
        ).to(_CLIP_STORAGE_DEVICE).eval()
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
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
    Extract the object noun phrase from a POPE-style question.
    "Is there a dining table in the image?" → "dining table"
    "Is there a chair in the image?"        → "chair"
    Falls back to the full first sentence if extraction fails.
    """
    # Match: "is there a/an <phrase> in the image" — captures multi-word nouns
    m = re.search(r"is there an?\s+([\w\s]+?)\s+in\s+the\s+image", prompt, re.IGNORECASE)
    if m:
        return m.group(1).lower().strip()
    # Match: "do you see a/an <phrase> in the image"
    m = re.search(r"do you see an?\s+([\w\s]+?)\s+in\s+the\s+image", prompt, re.IGNORECASE)
    if m:
        return m.group(1).lower().strip()
    # Match: "is there a/an <phrase>?" (no "in the image")
    m = re.search(r"is there an?\s+([\w\s]+?)[\?\.!\n]", prompt, re.IGNORECASE)
    if m:
        return m.group(1).lower().strip()
    # Fallback: first sentence only
    first = prompt.split("\n")[0].split("?")[0]
    return first.strip()


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
        img_inputs = processor(images=patches, return_tensors="pt",
                               padding=True).to(_CLIP_INFER_DEVICE)
        img_feats  = model.get_image_features(**img_inputs)      # (n_patches, d)
        img_feats  = img_feats / img_feats.norm(dim=-1, keepdim=True)

        txt_inputs = processor(text=[noun], return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=77).to(_CLIP_INFER_DEVICE)
        txt_feat   = model.get_text_features(**txt_inputs)        # (1, d)
        txt_feat   = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        sims = (img_feats @ txt_feat.T).squeeze(-1).cpu()         # (n_patches,)
    _clip_model_to_storage()  # free VRAM before Qwen forward pass

    max_sim        = float(sims.max())
    object_present = max_sim >= ABSENCE_THRESH

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
        # Object absent: uniform mask — no targeted boost
        mask     = torch.ones(n_tokens, dtype=torch.float32)
        saliency = torch.full((n_tokens,), 0.5)   # neutral when absent

    return ClipSalienceResult(
        mask=mask, saliency=saliency, max_sim=max_sim,
        object_present=object_present, query_noun=noun,
    )


def get_grid_dims(inputs: dict, spatial_merge_size: int = 2, model_type: str = "qwen") -> tuple[int, int]:
    """Extract (grid_h, grid_w) from processor inputs (Qwen or LLaVA)."""
    if model_type == "llava":
        # LLaVA uses pixel_values with shape [batch, num_patches, channels, height, width]
        # For LLaVA-1.5, default is 336x336 images with 24x24 patch grid
        pixel_values = inputs.get("pixel_values", inputs.get("image_features"))
        if pixel_values is not None and len(pixel_values) > 0:
            # LLaVA typically uses 24x24 grid for 336px images
            # We'll use the clip_coarse_grid parameter value instead
            return 6, 6  # Default for LLaVA (6x6 as specified in scripts)
        return 6, 6  # Fallback
    else:
        # Qwen format - check if image_grid_thw exists (Qwen2-VL)
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            thw    = inputs["image_grid_thw"][0]
            grid_h = int(thw[1].item()) // spatial_merge_size
            grid_w = int(thw[2].item()) // spatial_merge_size
            return grid_h, grid_w
        else:
            # Qwen-VL v1 doesn't have image_grid_thw - use default 7x7 grid
            return 7, 7
