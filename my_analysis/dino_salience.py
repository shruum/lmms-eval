"""
DINO-based per-token salience for image-token boosting.

DINO (self-distillation with no labels) excels at foreground/background separation
without any supervision, making it ideal for saliency detection in vision models.

Key advantages over CLIP for saliency:
  1. No text query needed — works purely from image features
  2. Better at capturing object boundaries and fine-grained details
  3. Self-supervised — no language-vision bias that might conflict with VLM

Design:
  - Use DINO ViT-B/16 or ViT-S/16 for saliency maps
  - Extract attention rollouts from the last layer
  - Create coarse grid saliency, upsample to VLM token grid
  - Return both soft saliency and binary top-k mask
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

_DINO_MODEL = None
_DINO_PROCESSOR = None
_DINO_STORAGE_DEVICE = "cpu"
_DINO_INFER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DINO_DEFAULT_MODEL = "facebook/dino-vitb16"  # Good balance of quality/speed

# DINO ViT-B/16: 0.34 GB fp16 — fits alongside LLaVA 7B (13 GB)
# DINO ViT-S/16: 0.66 GB fp16 — better quality, more memory


@dataclass
class DinoSalienceResult:
    mask: torch.Tensor          # float (n_img_tokens,) — binary top-k (0.0 or 1.0)
    saliency: torch.Tensor      # float (n_img_tokens,) — soft continuous [0,1]
    attention_map: torch.Tensor # (H, W) — raw DINO attention rollout
    max_attention: float        # highest attention value (for thresholding)
    query_noun: str             # placeholder for API compatibility


def _load_dino(model_name: str = _DINO_DEFAULT_MODEL) -> tuple:
    """Load DINO model and processor."""
    global _DINO_MODEL, _DINO_PROCESSOR
    if _DINO_MODEL is None:
        from transformers import AutoImageProcessor, AutoModel
        print(f"  [dino_salience] Loading {model_name} on {_DINO_STORAGE_DEVICE} "
              f"(infers on {_DINO_INFER_DEVICE})…")

        _DINO_MODEL = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if _DINO_INFER_DEVICE == "cuda" else torch.float32,
        ).to(_DINO_STORAGE_DEVICE).eval()

        _DINO_PROCESSOR = AutoImageProcessor.from_pretrained(model_name)
    return _DINO_MODEL, _DINO_PROCESSOR


def _dino_model_to_infer() -> None:
    """Move DINO to inference device."""
    global _DINO_MODEL
    if _DINO_MODEL is not None and _DINO_INFER_DEVICE != _DINO_STORAGE_DEVICE:
        _DINO_MODEL.to(_DINO_INFER_DEVICE)


def _dino_model_to_storage() -> None:
    """Move DINO back to storage device to free VRAM."""
    global _DINO_MODEL
    if _DINO_MODEL is not None and _DINO_INFER_DEVICE != _DINO_STORAGE_DEVICE:
        _DINO_MODEL.to(_DINO_STORAGE_DEVICE)
        if _DINO_INFER_DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()


def _compute_attention_rollout(attentions: List[torch.Tensor], head_fusion: str = "mean") -> torch.Tensor:
    """
    Compute attention rollout from DINO attention maps.

    Args:
        attentions: List of attention tensors from DINO outputs
        head_fusion: How to fuse multi-head attentions ("mean", "max", "min")

    Returns:
        (H, W) attention rollout map
    """
    # Average over heads (or use max/min)
    if head_fusion == "mean":
        attn = torch.stack(attentions).mean(dim=0)  # (layers, heads, seq_len, seq_len)
    elif head_fusion == "max":
        attn = torch.stack(attentions).max(dim=0)[0]
    elif head_fusion == "min":
        attn = torch.stack(attentions).min(dim=0)[0]
    else:
        raise ValueError(f"Unknown head_fusion: {head_fusion}")

    # Remove CLS token and compute rollout
    # DINO uses [CLS] token as special token, we want spatial token attentions
    attn = attn[:, :, 0, 1:]  # (layers, heads, seq_len-1) - attention from [CLS] to spatial tokens

    # Reshape to spatial grid (assumes square grid for DINO)
    seq_len = attn.shape[-1]
    grid_size = int(np.sqrt(seq_len))

    if grid_size * grid_size != seq_len:
        # Handle non-square grids (use first square-ish reshape)
        grid_size = int(np.ceil(np.sqrt(seq_len)))

    # Reshape and compute rollout through layers
    attn = attn.mean(dim=1)  # Average over heads: (layers, seq_len)
    attn = attn.reshape(-1, grid_size, grid_size)  # (layers, H, W)

    # Rollout: multiply attentions through layers
    rollout = attn[0]  # First layer
    for i in range(1, len(attn)):
        rollout = torch.matmul(rollout, attn[i])

    return rollout


def _extract_dino_features(model, image: torch.Tensor) -> torch.Tensor:
    """Extract DINO features for saliency."""
    outputs = model(image, output_attentions=True)
    attention_rollout = _compute_attention_rollout(outputs.attentions)

    return attention_rollout


def compute_dino_salience(
    image: Image.Image,
    query: str = "",  # Not used for DINO, but kept for API compatibility
    coarse_n: int = 8,
    top_k_pct: float = 0.30,
    use_soft: bool = True,
    model_name: str = _DINO_DEFAULT_MODEL,
    target_img_size: tuple = (24, 24),  # LLaVA's image token grid
) -> DinoSalienceResult:
    """
    Compute DINO-based saliency for an image.

    Args:
        image: PIL image
        query: Text query (ignored for DINO, kept for API compatibility)
        coarse_n: Coarse grid size (N×N)
        top_k_pct: Top-k percentage for binary mask
        use_soft: Return soft saliency or hard mask
        model_name: DINO model to use
        target_img_size: Target image token grid size for upsampling

    Returns:
        DinoSalienceResult with mask, saliency, and metadata
    """
    # Load model
    model, processor = _load_dino(model_name)
    _dino_model_to_infer()

    try:
        # Process image
        inputs = processor(images=image, return_tensors="pt").to(_DINO_INFER_DEVICE)

        # Extract DINO features
        with torch.no_grad():
            dino_features = _extract_dino_features(model, inputs.pixel_values)

        # Normalize to [0, 1]
        dino_features = (dino_features - dino_features.min()) / (dino_features.max() - dino_features.min() + 1e-8)

        # Create coarse grid
        h, w = dino_features.shape
        coarse_features = F.interpolate(
            dino_features.unsqueeze(0).unsqueeze(0),
            size=(coarse_n, coarse_n),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # Upsample to target grid size
        saliency = F.interpolate(
            coarse_features.unsqueeze(0).unsqueeze(0),
            size=target_img_size,
            mode='bilinear',
            align_corners=False
        ).squeeze().flatten()

        # Create binary top-k mask
        k = int(saliency.numel() * top_k_pct)
        topk_values, topk_indices = torch.topk(saliency, k)
        mask = torch.zeros_like(saliency)
        mask[topk_indices] = 1.0

        # Return soft or hard saliency
        final_saliency = saliency if use_soft else mask

        result = DinoSalienceResult(
            mask=mask,
            saliency=final_saliency,
            attention_map=dino_features.cpu(),
            max_attention=dino_features.max().item(),
            query_noun=query,  # Placeholder for compatibility
        )

        return result

    finally:
        _dino_model_to_storage()


def compute_dino_salience_batch(
    images: List[Image.Image],
    queries: List[str] = None,
    **kwargs
) -> List[DinoSalienceResult]:
    """Compute DINO saliency for a batch of images."""
    if queries is None:
        queries = [""] * len(images)

    results = []
    for img, query in zip(images, queries):
        result = compute_dino_salience(img, query, **kwargs)
        results.append(result)

    return results


# ============================================================================
# Convenience functions matching CLIP API
# ============================================================================

def get_dino_model_name() -> str:
    """Get default DINO model name."""
    return _DINO_DEFAULT_MODEL


def set_dino_devices(storage: str = "cpu", infer: str = "cuda"):
    """Set DINO storage and inference devices."""
    global _DINO_STORAGE_DEVICE, _DINO_INFER_DEVICE
    _DINO_STORAGE_DEVICE = storage
    _DINO_INFER_DEVICE = infer


if __name__ == "__main__":
    # Test DINO salience
    from PIL import Image

    # Create a test image
    test_img = Image.new('RGB', (224, 224), color='red')

    result = compute_dino_salience(
        test_img,
        query="test",
        coarse_n=8,
        top_k_pct=0.3,
    )

    print(f"DINO saliency test successful!")
    print(f"  Max attention: {result.max_attention:.4f}")
    print(f"  Saliency shape: {result.saliency.shape}")
    print(f"  Mask sum: {result.mask.sum().item()}")
