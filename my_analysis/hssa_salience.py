"""
Hidden State Semantic Alignment (HSSA) saliency for Qwen2.5-VL.

At a target middle decoder layer, computes cosine similarity between each
image-token hidden state and the actual question-word token hidden states.

Key fixes vs v1:
  - Filters out Qwen special tokens (IDs >= 151644) from the text side —
    excludes <|vision_end|>, <|im_end|>, <|im_start|>, assistant, newlines.
    Only actual question word tokens are used.
  - Default layer changed to 16 (highest saliency spread for Qwen2.5-VL-3B-36L).
  - Percentile normalisation (5th-95th) instead of strict min-max — more contrast.

Usage:
    result = compute_hssa_salience(model, inputs, img_start, img_end)
    # result.saliency : float tensor (n_img_tokens,) in [0,1]  — soft, continuous
    # result.mask     : float tensor (n_img_tokens,)           — binary top-k
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass

# All Qwen2.5-VL special tokens have IDs >= this threshold
QWEN_SPECIAL_TOKEN_THRESHOLD = 151644


@dataclass
class HssaSalienceResult:
    saliency:      torch.Tensor   # float (n_img_tokens,) in [0,1], percentile-normalised
    mask:          torch.Tensor   # float (n_img_tokens,) binary top-k
    layer_idx:     int            # which decoder layer was used
    n_query_toks:  int            # number of actual question tokens used


def compute_hssa_salience(
    model,
    inputs: dict,
    img_start: int,
    img_end:   int,
    layer_idx: int   = 16,
    top_k_pct: float = 0.3,
) -> HssaSalienceResult:
    """
    Run one forward pass with output_hidden_states=True.
    Extract hidden states at layer_idx for image tokens and QUESTION-WORD tokens.

    Question tokens = tokens after img_end that are NOT Qwen special tokens
    (filters out <|vision_end|>, <|im_end|>, <|im_start|>, 'assistant', newlines).

    Parameters
    ----------
    model       : Qwen2.5-VL model (eval mode, on device)
    inputs      : processor output dict — must include 'input_ids' on CPU
    img_start   : inclusive start of image tokens in input_ids
    img_end     : inclusive end   of image tokens in input_ids
    layer_idx   : decoder layer to extract (0-indexed; default 16 for 36-layer model)
    top_k_pct   : fraction of image tokens in binary top-k mask

    Returns
    -------
    HssaSalienceResult with .saliency (soft [0,1]) and .mask (binary top-k)
    """
    device = next(model.parameters()).device

    # Move inputs to device
    model_inputs = {k: v.to(device) if hasattr(v, "to") else v
                    for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # hidden_states: tuple (embedding, layer_0, ..., layer_N)
    # index layer_idx+1 to get the output of decoder layer `layer_idx`
    hs = outputs.hidden_states[layer_idx + 1][0].float()   # (seq_len, d_model)

    # Image hidden states
    h_img = hs[img_start : img_end + 1]                    # (n_img, d)

    # Text hidden states — filter to actual question words only
    ids_cpu = inputs["input_ids"][0].cpu()                  # (seq_len,)
    txt_ids = ids_cpu[img_end + 1 :]                        # IDs after image
    keep    = txt_ids < QWEN_SPECIAL_TOKEN_THRESHOLD        # True = real word token
    h_txt   = hs[img_end + 1 :][keep]                      # (n_query, d)

    # Fallback: if everything got filtered, use all post-image tokens
    if h_txt.shape[0] == 0:
        h_txt = hs[img_end + 1 :]

    n_query_toks = h_txt.shape[0]

    # Normalised cosine similarity
    h_img_n = F.normalize(h_img, dim=-1)                   # (n_img, d)
    h_txt_n = F.normalize(h_txt, dim=-1)                   # (n_query, d)
    sims    = (h_img_n @ h_txt_n.T)                        # (n_img, n_query)
    raw_sal = sims.max(dim=-1).values.cpu()                 # (n_img,) max over query tokens

    # Percentile normalisation (5th–95th) for better contrast than strict min-max
    p5  = torch.quantile(raw_sal, 0.05)
    p95 = torch.quantile(raw_sal, 0.95)
    saliency = (raw_sal - p5) / (p95 - p5 + 1e-8)
    saliency = saliency.clamp(0.0, 1.0)

    # Binary top-k mask
    n_tokens = len(saliency)
    k        = max(1, round(n_tokens * top_k_pct))
    topk_idx = saliency.topk(k).indices
    mask     = torch.zeros(n_tokens, dtype=torch.float32)
    mask[topk_idx] = 1.0

    return HssaSalienceResult(
        saliency=saliency, mask=mask,
        layer_idx=layer_idx, n_query_toks=n_query_toks,
    )
