"""
srf_fovea.py — SRF-Fovea: pre-encoder foveal blur + SRF attention routing.
Model-agnostic: supports LLaVA-1.5 and Qwen2.5-VL.

Wraps srf.py exactly. The only difference from SRF base:

  prepare_sample() additionally:
    1. Reads CLIP saliency mask stored by srf.prepare_sample in patch._STATE
    2. Upsamples it to full image resolution → per-pixel blend weight W ∈ [0,1]
    3. Foveated image = W * original + (1−W) * GaussianBlur(original, σ)
    4. Re-processes foveated image through the ViT image processor
    5. Replaces inp["pixel_values"] (and inp["image_grid_thw"] for Qwen) in-place

SRF attention boost remains active during inference — combined method:
blur non-salient regions at encoder input AND boost salient tokens in decoder attention.

Best σ from Qwen2.5-VL MMVP sweep (2026-07-31): σ=20 → +3.33pp pair accuracy.

Interface identical to srf.py — eval.py dispatches with --method srffovea.
eval.py injects: method_mod.patch = patch  (before setup())
eval.py sets:   method_mod.SIGMA = args.fovea_sigma
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import ImageFilter

import srf as _srf
import clip_salience as clip_sal

# patch injected by eval.py before setup() (same pattern as srf.py)
patch = None

# Re-export full SRF interface — everything except prepare_sample is identical
setup             = _srf.setup
reset_for_dataset = _srf.reset_for_dataset
cleanup           = _srf.cleanup

# Gaussian blur radius in pixels. Override via eval.py: method_mod.SIGMA = args.fovea_sigma
SIGMA: float = 20.0

_SPATIAL = 2   # Qwen2.5-VL spatial merge factor (ignored for LLaVA)


def _model_type() -> str:
    return "llava" if "llava" in _srf._model_id.lower() else "qwen"


def _saliency_to_weight(
    mask: torch.Tensor,
    image_w: int,
    image_h: int,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """Upsample token-grid saliency (n_tokens,) to full-image weight map (H, W)."""
    import torch.nn.functional as F
    sal_2d   = mask.reshape(grid_h, grid_w).unsqueeze(0).unsqueeze(0).float()
    sal_full = F.interpolate(sal_2d, size=(image_h, image_w),
                             mode="bilinear", align_corners=False)
    return sal_full.squeeze().cpu().numpy()   # (H, W) in [0, 1]


def _apply_foveal_blur(image, weight: np.ndarray, sigma: float):
    """Foveal composite: W * original + (1−W) * GaussianBlur(original, σ)."""
    from PIL import Image
    blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    img_arr = np.array(image).astype(np.float32)
    blr_arr = np.array(blurred).astype(np.float32)
    w       = weight[:, :, np.newaxis]   # (H, W, 1) broadcast
    result  = w * img_arr + (1.0 - w) * blr_arr
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def _replace_pixel_values(inp: dict, fovea_img, processor, mt: str) -> None:
    """Re-process foveated image and replace inp['pixel_values'] in-place."""
    device = inp["input_ids"].device
    dtype  = inp["pixel_values"].dtype

    try:
        if mt == "llava":
            out = processor(text="placeholder", images=[fovea_img], return_tensors="pt")
        else:
            from qwen_vl_utils import process_vision_info
            msgs = [{"role": "user", "content": [{"type": "image", "image": fovea_img}]}]
            vis_inputs, _ = process_vision_info(msgs)
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
        return

    new_pv = out.get("pixel_values")
    if new_pv is None:
        return
    new_pv = new_pv.to(device=device, dtype=dtype)
    if new_pv.shape != inp["pixel_values"].shape:
        return   # resolution changed — skip to avoid token count mismatch

    inp["pixel_values"] = new_pv

    # Qwen only: keep image_grid_thw consistent
    if mt != "llava":
        new_grid = out.get("image_grid_thw")
        if new_grid is not None and "image_grid_thw" in inp:
            inp["image_grid_thw"] = new_grid.to(device=device)


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
    1. Run SRF prepare_sample (sets CLIP saliency + attention state in patch._STATE).
    2. Extract salience_mask → per-pixel blur weight map.
    3. Apply foveal blur to PIL image.
    4. Replace inp['pixel_values'] with foveated image pixels.

    If CLIP gate fails (absent object / bad noun), no blur is applied;
    SRF attention state is still active (as set by _srf.prepare_sample).
    """
    _srf.prepare_sample(inp, img_start, img_end, image, question, model, processor, **kwargs)

    saved_mask = patch._STATE.get("salience_mask")
    if saved_mask is None:
        return   # CLIP gate failed — no blur, SRF attn state unchanged

    mt = _model_type()
    try:
        grid_h, grid_w = clip_sal.get_grid_dims(inp, _SPATIAL, mt)
    except Exception:
        return

    image_w, image_h = image.size
    blur_weight = _saliency_to_weight(saved_mask, image_w, image_h, grid_h, grid_w)
    fovea_img   = _apply_foveal_blur(image, blur_weight, SIGMA)
    _replace_pixel_values(inp, fovea_img, processor, mt)
