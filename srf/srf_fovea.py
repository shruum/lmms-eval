"""
srf_fovea.py — SRF-Fovea: pre-encoder foveal blur + SRF attention routing.

Wraps srf.py exactly. The only difference from SRF base:

  prepare_sample() additionally:
    1. Reads CLIP saliency mask computed by srf.prepare_sample
    2. Upsamples it to full image resolution → per-pixel blend weight W ∈ [0,1]
    3. Foveated image = W * original + (1−W) * GaussianBlur(original, σ)
    4. Re-processes foveated image through the ViT image processor
    5. Replaces inp["pixel_values"] (and inp["image_grid_thw"] if shape changes)

SRF attention boost remains active during inference — this is the full combined
method: blur non-salient regions at the encoder input AND boost salient tokens
in the decoder attention.

Best σ from sweep (2026-07-31): σ=20 → +3.33pp MMVP, no POPE regression.

Interface identical to srf.py — eval.py dispatches with --method srffovea.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import ImageFilter

import qwen_attn_patch as patch
import srf as _srf
import clip_salience as clip_sal

# ---------------------------------------------------------------------------
# Re-export full SRF interface — everything except prepare_sample is identical
# ---------------------------------------------------------------------------

setup             = _srf.setup
reset_for_dataset = _srf.reset_for_dataset
cleanup           = _srf.cleanup

# ---------------------------------------------------------------------------
# Hyperparameter
# ---------------------------------------------------------------------------

SIGMA: float = 20.0    # Gaussian blur radius in pixels (best from sweep)

# ---------------------------------------------------------------------------
# Foveal blur helpers (same as test_srffovea_mmvp.py)
# ---------------------------------------------------------------------------

_SPATIAL = 2   # Qwen2.5-VL spatial merge factor


def _saliency_to_weight(
    mask: torch.Tensor | None,
    image_w: int,
    image_h: int,
    grid_h: int,
    grid_w: int,
) -> np.ndarray | None:
    """
    Upsample token-grid saliency (n_tokens,) to full-image weight map (H, W).
    High weight = keep sharp; low weight = blur.
    Returns None if mask is None (absent object / bad noun → no blur).
    """
    if mask is None:
        return None
    import torch.nn.functional as F
    sal_2d   = mask.reshape(grid_h, grid_w).unsqueeze(0).unsqueeze(0).float()
    sal_full = F.interpolate(sal_2d, size=(image_h, image_w),
                             mode="bilinear", align_corners=False)
    return sal_full.squeeze().cpu().numpy()   # (H, W) in [0, 1]


def _apply_foveal_blur(image, weight: np.ndarray, sigma: float):
    """
    Foveal composite: W * original + (1−W) * GaussianBlur(original, σ).
    """
    from PIL import Image
    blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    img_arr = np.array(image).astype(np.float32)
    blr_arr = np.array(blurred).astype(np.float32)
    w       = weight[:, :, np.newaxis]   # (H, W, 1)
    result  = w * img_arr + (1.0 - w) * blr_arr
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Public API — prepare_sample overrides srf's
# ---------------------------------------------------------------------------

def prepare_sample(
    inp,
    img_start: int,
    img_end: int,
    image,
    question: str,
    model,
    processor,
    **kwargs,
) -> None:
    """
    1. Run SRF prepare (CLIP saliency + attention state configured).
    2. Extract salience_mask, build per-pixel blur weight.
    3. Apply foveal blur to PIL image.
    4. Replace inp["pixel_values"] with foveated image pixels.
    """
    # Step 1: run SRF prepare — sets patch._STATE (salience_mask, img range, etc.)
    _srf.prepare_sample(inp, img_start, img_end, image, question, model, processor, **kwargs)

    saved_mask = patch._STATE.get("salience_mask")
    if saved_mask is None:
        return   # CLIP gate failed or absent object — no blur, SRF attn still active

    # Step 2: grid dims + weight map
    try:
        grid_h, grid_w = clip_sal.get_grid_dims(inp, _SPATIAL)
    except Exception:
        return   # can't determine grid — skip blur safely

    image_w, image_h = image.size
    blur_weight = _saliency_to_weight(saved_mask, image_w, image_h, grid_h, grid_w)
    if blur_weight is None:
        return

    # Step 3: apply foveal blur
    fovea_img = _apply_foveal_blur(image, blur_weight, SIGMA)

    # Step 4: re-process foveated image → replace pixel_values in inp
    _replace_pixel_values(inp, fovea_img, processor)


def _replace_pixel_values(inp: dict, fovea_img, processor) -> None:
    """
    Process foveated PIL image through the ViT image processor and replace
    inp["pixel_values"] (and "image_grid_thw" if present) in-place.
    Skips silently if shape differs from original (safety).
    """
    try:
        from qwen_vl_utils import process_vision_info
        msgs = [{"role": "user", "content": [{"type": "image", "image": fovea_img}]}]
        vis_inputs, _ = process_vision_info(msgs)
        # Use dummy text to get pixel processing only
        dummy_text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        out = processor(
            text=[dummy_text],
            images=vis_inputs,
            return_tensors="pt",
            padding=False,
        )
    except Exception:
        return   # processing failed — keep original pixel_values

    device = inp["input_ids"].device
    dtype  = inp["pixel_values"].dtype

    new_pv = out.get("pixel_values")
    if new_pv is None:
        return

    new_pv = new_pv.to(device=device, dtype=dtype)

    # Only replace if shape matches exactly (same resolution → same patch count)
    if new_pv.shape != inp["pixel_values"].shape:
        return

    inp["pixel_values"] = new_pv

    # Update image_grid_thw if present (should be identical, but keep consistent)
    new_grid = out.get("image_grid_thw")
    if new_grid is not None and "image_grid_thw" in inp:
        inp["image_grid_thw"] = new_grid.to(device=device)
