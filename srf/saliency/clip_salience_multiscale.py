"""
Multi-scale CLIP ensemble for better saliency detection.
Combines 5×5, 7×7, and 9×9 grids for robust object localization.
"""
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple
import sys
sys.path.insert(0, 'srf/saliency')
from clip_salience import _load_clip, _clip_model_to_infer, _clip_model_to_storage, extract_query_noun, ABSENCE_THRESH, ClipSalienceResult

def compute_multiscale_clip_salience(
    image: Image.Image,
    text: str,
    grid_h: int,
    grid_w: int,
    top_k_pct: float = 0.3,
    scales: List[int] = [5, 7, 9],  # Multiple grid sizes
    ensemble_method: str = "average",  # "average", "max", or "weighted"
    target_n_tokens: int = None,
) -> ClipSalienceResult:
    """
    Compute CLIP salience using multiple grid scales and ensemble them.

    Args:
        scales: List of grid sizes to use (e.g., [5, 7, 9])
        ensemble_method: How to combine results
            - "average": Mean of all scales
            - "max": Maximum across scales (most conservative)
            - "weighted": Weighted average (finer scales get more weight)
    """
    model, processor = _load_clip()
    noun = extract_query_noun(text)

    W, H = image.size
    scale_results = []

    for coarse_n in scales:
        ph = H / coarse_n
        pw = W / coarse_n

        patches = []
        for row in range(coarse_n):
            for col in range(coarse_n):
                y0, y1 = int(row * ph), int((row + 1) * ph)
                x0, x1 = int(col * pw), int((col + 1) * pw)
                patch = image.crop((x0, y0, x0, y1))
                patches.append(patch)

        _clip_model_to_infer()
        with torch.no_grad():
            img_inputs = processor(images=patches, return_tensors="pt",
                                   padding=True).to("cuda")
            img_feats = model.get_image_features(**img_inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            txt_inputs = processor(text=[noun], return_tensors="pt",
                                   padding=True, truncation=True,
                                   max_length=77).to("cuda")
            txt_feat = model.get_text_features(**txt_inputs)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            sims = (img_feats @ txt_feat.T).squeeze(-1).cpu()
        _clip_model_to_storage()

        # Reshape and upsample to token grid
        sim_grid = sims.view(1, 1, coarse_n, coarse_n)
        sim_token = F.interpolate(
            sim_grid, size=(grid_h, grid_w), mode="bilinear", align_corners=False
        ).squeeze()

        scale_results.append(sim_token.reshape(-1))

    # Ensemble across scales
    if ensemble_method == "average":
        ensemble_sim = torch.stack(scale_results).mean(dim=0)
    elif ensemble_method == "max":
        ensemble_sim = torch.stack(scale_results).max(dim=0)[0]
    elif ensemble_method == "weighted":
        # Finer scales get more weight
        weights = torch.tensor([1.0 / s for s in scales])
        weights = weights / weights.sum()
        ensemble_sim = torch.stack([w * s for w, s in zip(weights, scale_results)]).sum(dim=0)
    else:
        raise ValueError(f"Unknown ensemble_method: {ensemble_method}")

    # Build result from ensemble similarity
    max_sim = float(ensemble_sim.max())
    object_present = max_sim >= ABSENCE_THRESH

    sim_flat = ensemble_sim
    n_tokens = len(sim_flat)

    # Soft saliency: min-max normalise
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
        saliency = F.interpolate(saliency_2d, size=(target_side, target_side),
                                 mode='bilinear', align_corners=False).squeeze()
        mask = F.interpolate(mask_2d, size=(target_side, target_side),
                            mode='bilinear', align_corners=False).squeeze()
        saliency = saliency.reshape(-1)
        mask = mask.reshape(-1)

    return ClipSalienceResult(
        mask=mask,
        saliency=saliency,
        max_sim=max_sim,
        object_present=object_present,
        query_noun=noun,
    )
