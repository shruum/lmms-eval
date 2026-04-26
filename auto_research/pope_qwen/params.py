"""
Agent-editable parameters for Qwen-VL-Chat autoresearch.
Edit this file, then run: CUDA_VISIBLE_DEVICES=1 python eval.py > run.log
"""

PARAMS = {
    # Method: "srf" or "srfe" (contrastive)
    "method": "srf",

    # Saliency (CLIP-guided)
    "clip_coarse_grid": 7,        # 7x7 patches
    "clip_top_k_pct": 0.25,       # Experiment: more focused saliency
    "clip_fallback_thresh": 0.20, # CLIP similarity threshold

    # Layer range (Qwen-VL has 32 layers, fusion in middle)
    "layer_start": 8,             # start of fusion zone
    "layer_end": 14,              # end of fusion zone

    # Head selection
    "head_top_k_pct": 0.20,       # top 20% vision-aware heads

    # SRF biasing
    "alpha": 2.0,                 # boost strength
    "eps": 0.0,                   # background suppression (0 = no suppression)
    "bias_mode": "additive_logit",

    # SRF-E (only used if method="srfe")
    "beta": None,                 # contrastive strength (None = SRF-only)
}
