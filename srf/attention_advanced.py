"""
Advanced attention boosting and suppression strategies.
Multiple mechanisms for modulating attention beyond simple multiplicative scaling.
"""
import torch
import torch.nn.functional as F
from typing import Literal, Optional

_ADVANCED_STATE = {
    "method": "baseline",
    "boost_strategy": "multiplicative",  # multiplicative, additive, temperature, hybrid
    "suppress_strategy": "multiplicative",  # multiplicative, subtractive, hard
    "boost_strength": 2.0,
    "suppress_strength": 0.1,
}


def apply_advanced_attention_boost(
    attn_weights: torch.Tensor,  # (batch, heads, q_len, kv_len)
    img_start: int,
    img_end: int,
    sys_end: Optional[int],
    saliency_mask: Optional[torch.Tensor],  # (n_img_tokens,)
    boost_strategy: Literal["multiplicative", "additive", "temperature", "hybrid"] = "multiplicative",
    suppress_strategy: Literal["multiplicative", "subtractive", "hard"] = "multiplicative",
    boost_strength: float = 2.0,
    suppress_strength: float = 0.1,
    layer_idx: int = 0,
) -> torch.Tensor:
    """
    Apply advanced attention boosting and suppression strategies.

    Args:
        attn_weights: Original attention weights (pre-softmax or post-softmax)
        boost_strategy: How to boost image token attention
            - "multiplicative": Multiply by factor (current default)
            - "additive": Add logit boost before softmax
            - "temperature": Divide by temperature (sharpen distribution)
            - "hybrid": Combine multiple strategies
        suppress_strategy: How to suppress system prompt attention
            - "multiplicative": Multiply by factor (current default)
            - "subtractive": Subtract logits before softmax
            - "hard": Zero out attention completely

    Returns:
        Modified attention weights
    """
    result = attn_weights.clone()

    # Handle system prompt suppression
    if sys_end is not None and sys_end >= 0 and suppress_strength != 1.0:
        if suppress_strategy == "multiplicative":
            result[:, :, :, :sys_end+1] *= suppress_strength
        elif suppress_strategy == "subtractive":
            # Convert to logits, subtract, convert back
            result_logits = torch.log(result + 1e-8)
            result_logits[:, :, :, :sys_end+1] -= torch.log(torch.tensor(suppress_strength + 1e-8))
            result = F.softmax(result_logits, dim=-1)
        elif suppress_strategy == "hard":
            result[:, :, :, :sys_end+1] *= 0.0
            # Renormalize
            result = result / result.sum(dim=-1, keepdim=True)

    # Handle image token boosting
    if saliency_mask is not None and img_end is not None and img_start is not None:
        img_tokens = img_end - img_start + 1

        if saliency_mask.numel() == img_tokens:
            salience = saliency_mask.to(attn_weights.device)
        else:
            # Dimension mismatch - use uniform boosting
            salience = torch.ones(img_tokens, device=attn_weights.device)

        if boost_strategy == "multiplicative":
            # Current SRF approach
            scaling = 1.0 + (boost_strength - 1.0) * salience
            result[:, :, :, img_start:img_end+1] *= scaling.unsqueeze(0).unsqueeze(0)

        elif boost_strategy == "additive":
            # Add logit boost before softmax
            # Assuming attn_weights are post-softmax, convert to logits first
            result_logits = torch.log(result + 1e-8)

            # Add per-token boost based on saliency
            boost_per_token = torch.log(torch.tensor(boost_strength, device=attn_weights.device)) * saliency
            result_logits[:, :, :, img_start:img_end+1] += boost_per_token.unsqueeze(0).unsqueeze(0)

            result = F.softmax(result_logits, dim=-1)

        elif boost_strategy == "temperature":
            # Sharpen distribution for image tokens (divide by T < 1)
            # Using temperature = 1/boost_strength
            temp = 1.0 / boost_strength

            # Apply temperature scaling per-image-token
            for i in range(img_tokens):
                token_idx = img_start + i
                # Sharpen attention to this token
                result[:, :, :, token_idx] = F.softmax(
                    torch.log(result[:, :, :, token_idx] + 1e-8) / temp,
                    dim=-1
                )

        elif boost_strategy == "hybrid":
            # Combine multiplicative + additive
            scaling = 1.0 + (boost_strength ** 0.5 - 1.0) * saliency
            result[:, :, :, img_start:img_end+1] *= scaling.unsqueeze(0).unsqueeze(0)

            # Additional logit boost
            result_logits = torch.log(result + 1e-8)
            boost_per_token = torch.log(torch.tensor(1.2, device=attn_weights.device)) * salience
            result_logits[:, :, :, img_start:img_end+1] += boost_per_token.unsqueeze(0).unsqueeze(0) * 0.1
            result = F.softmax(result_logits, dim=-1)

    # Renormalize to maintain probability distribution
    result = result / result.sum(dim=-1, keepdim=True)

    return result


def apply_gradient_based_attention(
    attn_weights: torch.Tensor,
    img_start: int,
    img_end: int,
    gradients: torch.Tensor,  # Gradients of loss w.r.t. attention
    boost_strength: float = 2.0,
) -> torch.Tensor:
    """
    Boost attention based on gradient signal (gradient-based saliency).
    Boost tokens that have higher gradients more.
    """
    # Compute gradient magnitude per token
    grad_mag = gradients.abs()

    # Normalize to [0, 1]
    grad_mag = grad_mag / (grad_mag.max() + 1e-8)

    # Apply gradient-weighted boosting
    img_tokens = img_end - img_start + 1
    if grad_mag.numel() == img_tokens:
        scaling = 1.0 + (boost_strength - 1.0) * grad_mag
        result = attn_weights.clone()
        result[:, :, :, img_start:img_end+1] *= scaling.unsqueeze(0).unsqueeze(0)
        result = result / result.sum(dim=-1, keepdim=True)
        return result
    else:
        return attn_weights


def apply_entropy_based_attention(
    attn_weights: torch.Tensor,
    img_start: int,
    img_end: int,
    target_entropy: float = 2.0,  # Target entropy in nats
    boost_strength: float = 1.5,
) -> torch.Tensor:
    """
    Modulate attention to achieve target entropy.
    Boost/suppress to make distribution more peaked or uniform.
    """
    result = attn_weights.clone()

    # Compute current entropy for image token attention
    img_attn = result[:, :, :, img_start:img_end+1]
    current_entropy = -(img_attn * torch.log(img_attn + 1e-8)).sum(dim=-1).mean()

    # If current entropy is too high (too uniform), boost (peaking)
    # If current entropy is too low (too peaked), suppress (smoothing)
    if current_entropy > target_entropy:
        # Too uniform - boost to peak
        temp = 0.8  # Sharpen
    else:
        # Too peaked - suppress to smooth
        temp = 1.2  # Soften

    # Apply temperature scaling
    result_logits = torch.log(result + 1e-8)
    result = F.softmax(result_logits / temp, dim=-1)

    return result


def apply_head_specific_modulation(
    attn_weights: torch.Tensor,
    img_start: int,
    img_end: int,
    head_importance: torch.Tensor,  # (n_heads,) - importance weights per head
    boost_strength: float = 2.0,
) -> torch.Tensor:
    """
    Boost/suppress attention on a per-head basis.
    Different heads can have different modulation strengths.
    """
    result = attn_weights.clone()
    n_heads = attn_weights.shape[1]

    if head_importance.numel() == n_heads:
        # Normalize head importance to [0, 1]
        head_importance = head_importance / (head_importance.max() + 1e-8)

        # Apply head-specific scaling
        for head_idx in range(n_heads):
            scaling = 1.0 + (boost_strength - 1.0) * head_importance[head_idx]
            result[:, head_idx, :, img_start:img_end+1] *= scaling

        result = result / result.sum(dim=-1, keepdim=True)

    return result
