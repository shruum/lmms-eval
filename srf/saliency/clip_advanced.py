"""
Advanced CLIP salience with hidden layer features and bigger models.
Uses intermediate layer representations for better object localization.
"""
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional, Literal
import sys
sys.path.insert(0, 'srf/saliency')
from clip_salience import _load_clip, _clip_model_to_infer, _clip_model_to_storage, extract_query_noun, ABSENCE_THRESH, ClipSalienceResult

_ADVANCED_CLIP_MODEL = None
_ADVANCED_CLIP_PROCESSOR = None


def _load_advanced_clip(model_name: str = "openai/clip-vit-large-patch14"):
    """Load bigger CLIP model (ViT-L/14)."""
    global _ADVANCED_CLIP_MODEL, _ADVANCED_CLIP_PROCESSOR
    if _ADVANCED_CLIP_MODEL is None:
        from transformers import CLIPModel, CLIPProcessor
        print(f"  [clip_advanced] Loading {model_name} on CPU (one-time, infers on CUDA)…")
        _ADVANCED_CLIP_MODEL = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to("cpu").eval()
        _ADVANCED_CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
    return _ADVANCED_CLIP_MODEL, _ADVANCED_CLIP_PROCESSOR


def compute_clip_with_hidden_layers(
    image: Image.Image,
    text: str,
    grid_h: int,
    grid_w: int,
    top_k_pct: float = 0.3,
    coarse_n: int = 7,
    model_size: Literal["base", "large"] = "large",
    hidden_layer: int = -1,  # -1 for final layer, 0-N for intermediate
    aggregation: Literal["mean", "max", "attention"] = "attention",
    target_n_tokens: int = None,
) -> ClipSalienceResult:
    """
    Compute CLIP salience using hidden layer features.

    Args:
        model_size: "base" for ViT-B/32, "large" for ViT-L/14
        hidden_layer: Which layer to use (-1 for final, 0-N for intermediate)
        aggregation: How to aggregate hidden features
            - "mean": Average all hidden states
            - "max": Take maximum across hidden states
            - "attention": Use attention weights to weight features
    """
    # Load appropriate model
    if model_size == "large":
        model, processor = _load_advanced_clip("openai/clip-vit-large-patch14")
    else:
        model, processor = _load_clip("openai/clip-vit-base-patch32")

    noun = extract_query_noun(text)
    W, H = image.size
    ph = H / coarse_n
    pw = W / coarse_n

    # Create patches
    patches = []
    for row in range(coarse_n):
        for col in range(coarse_n):
            y0, y1 = int(row * ph), int((row + 1) * ph)
            x0, x1 = int(col * pw), int((col + 1) * pw)
            patch = image.crop((x0, y0, x0, y1))
            patches.append(patch)

    # Move model to GPU for inference
    model.to("cuda")
    with torch.no_grad():
        # Process images
        img_inputs = processor(images=patches, return_tensors="pt", padding=True).to("cuda")

        if model_size == "large" and aggregation == "attention":
            # For ViT-L, use attention-based aggregation
            outputs = model.vision_model(**img_inputs, output_attentions=True)
            hidden_states = outputs.last_hidden_state  # (n_patches, hidden_dim)
            attentions = outputs.attentions[-1]  # Last layer attention (n_patches, n_patches)

            # Aggregate using attention weights
            # Take attention to CLS token as importance weights
            cls_attn = attentions[:, :, 0, :].mean(dim=1)  # (n_patches, n_patches)
            img_feats = (hidden_states * cls_attn).sum(dim=1) / cls_attn.sum(dim=1, keepdim=True)
        elif hidden_layer != -1:
            # Use intermediate hidden layer
            outputs = model.vision_model(**img_inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[hidden_layer]  # (n_patches, hidden_dim)

            if aggregation == "mean":
                img_feats = hidden_states.mean(dim=1)
            elif aggregation == "max":
                img_feats = hidden_states.max(dim=1)[0]
            else:
                img_feats = hidden_states[:, 0, :]  # Use [CLS] token
        else:
            # Use final layer [CLS] token
            img_feats = model.get_image_features(**img_inputs)

        # Normalize
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        # Process text
        txt_inputs = processor(text=[noun], return_tensors="pt", padding=True, truncation=True, max_length=77).to("cuda")
        txt_feat = model.get_text_features(**txt_inputs)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        # Compute similarities
        sims = (img_feats @ txt_feat.T).squeeze(-1).cpu()

    # Move model back to CPU
    model.to("cpu")
    torch.cuda.empty_cache()

    max_sim = float(sims.max())
    object_present = max_sim >= ABSENCE_THRESH

    # Reshape and upsample
    sim_grid = sims.view(1, 1, coarse_n, coarse_n)
    sim_token = F.interpolate(sim_grid, size=(grid_h, grid_w), mode="bilinear", align_corners=False).squeeze()
    sim_flat = sim_token.reshape(-1)

    n_tokens = len(sim_flat)

    # Soft saliency
    s_min, s_max = sim_flat.min(), sim_flat.max()
    saliency = (sim_flat - s_min) / (s_max - s_min + 1e-8)

    if object_present:
        k = max(1, round(n_tokens * top_k_pct))
        topk_idx = sim_flat.topk(k).indices
        mask = torch.zeros(n_tokens, dtype=torch.float32)
        mask[topk_idx] = 1.0
    else:
        mask = torch.ones(n_tokens, dtype=torch.float32)
        saliency = torch.full((n_tokens,), 0.5)

    # Optional upsampling
    if target_n_tokens is not None and target_n_tokens != n_tokens:
        saliency_2d = saliency.view(1, 1, grid_h, grid_w)
        mask_2d = mask.view(1, 1, grid_h, grid_w)
        target_side = int(target_n_tokens ** 0.5)
        saliency = F.interpolate(saliency_2d, size=(target_side, target_side), mode='bilinear', align_corners=False).squeeze()
        mask = F.interpolate(mask_2d, size=(target_side, target_side), mode='bilinear', align_corners=False).squeeze()
        saliency = saliency.reshape(-1)
        mask = mask.reshape(-1)

    return ClipSalienceResult(
        mask=mask,
        saliency=saliency,
        max_sim=max_sim,
        object_present=object_present,
        query_noun=noun,
    )


def compute_clip_with_text_attention(
    image: Image.Image,
    text: str,
    grid_h: int,
    grid_w: int,
    top_k_pct: float = 0.3,
    coarse_n: int = 7,
    model_size: Literal["base", "large"] = "large",
    target_n_tokens: int = None,
) -> ClipSalienceResult:
    """
    Compute CLIP salience using text-to-image cross-attention.
    Uses how much each image patch attends to the text token.
    """
    model, processor = _load_advanced_clip(model_size="large" if model_size == "large" else "base")
    noun = extract_query_noun(text)

    W, H = image.size
    ph = H / coarse_n
    pw = W / coarse_n

    patches = []
    for row in range(coarse_n):
        for col in range(coarse_n):
            y0, y1 = int(row * ph), int((row + 1) * ph)
            x0, x1 = int(col * pw), int((col + 1) * pw)
            patch = image.crop((x0, y0, x0, y1))
            patches.append(patch)

    model.to("cuda")
    with torch.no_grad():
        # Process images
        img_inputs = processor(images=patches, return_tensors="pt", padding=True).to("cuda")

        # Process text
        txt_inputs = processor(text=[noun], return_tensors="pt", padding=True, truncation=True, max_length=77).to("cuda")

        # Get vision model outputs with hidden states
        vision_outputs = model.vision_model(**img_inputs, output_hidden_states=True)
        vision_hidden = vision_outputs.last_hidden_state  # (n_patches, hidden_dim)

        # Get text model outputs
        text_outputs = model.text_model(**txt_inputs, output_hidden_states=True)
        text_hidden = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token (1, hidden_dim)

        # Compute cross-attention: how much each patch attends to text
        # Similarity between patch features and text [CLS]
        sims = (vision_hidden @ text_hidden.T).squeeze(-1).cpu()

    model.to("cpu")
    torch.cuda.empty_cache()

    max_sim = float(sims.max())
    object_present = max_sim >= ABSENCE_THRESH

    # Rest is same as above
    sim_grid = sims.view(1, 1, coarse_n, coarse_n)
    sim_token = F.interpolate(sim_grid, size=(grid_h, grid_w), mode="bilinear", align_corners=False).squeeze()
    sim_flat = sim_token.reshape(-1)
    n_tokens = len(sim_flat)

    s_min, s_max = sim_flat.min(), sim_flat.max()
    saliency = (sim_flat - s_min) / (s_max - s_min + 1e-8)

    if object_present:
        k = max(1, round(n_tokens * top_k_pct))
        topk_idx = sim_flat.topk(k).indices
        mask = torch.zeros(n_tokens, dtype=torch.float32)
        mask[topk_idx] = 1.0
    else:
        mask = torch.ones(n_tokens, dtype=torch.float32)
        saliency = torch.full((n_tokens,), 0.5)

    if target_n_tokens is not None and target_n_tokens != n_tokens:
        saliency_2d = saliency.view(1, 1, grid_h, grid_w)
        mask_2d = mask.view(1, 1, grid_h, grid_w)
        target_side = int(target_n_tokens ** 0.5)
        saliency = F.interpolate(saliency_2d, size=(target_side, target_side), mode='bilinear', align_corners=False).squeeze()
        mask = F.interpolate(mask_2d, size=(target_side, target_side), mode='bilinear', align_corners=False).squeeze()
        saliency = saliency.reshape(-1)
        mask = mask.reshape(-1)

    return ClipSalienceResult(
        mask=mask,
        saliency=saliency,
        max_sim=max_sim,
        object_present=object_present,
        query_noun=noun,
    )
