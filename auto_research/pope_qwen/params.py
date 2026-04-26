"""
NOVEL IDEA 1: Layer-specific language suppression

Instead of fixed sys_beta=0.10 everywhere, we apply STRONGER suppression
in middle fusion layers (where language bias is strongest) and milder suppression
elsewhere.

This is inspired by VAF but improves upon it:
- VAF: uniform sys_beta=0.10 across all boosted heads
- Ours: layer-adaptive suppression based on fusion strength
"""

PARAMS = {
    "method": "srf",

    # Saliency
    "clip_coarse_grid": 7,
    "clip_top_k_pct": 0.30,
    "clip_fallback_thresh": 0.20,

    # Layer range - focus on PEAK fusion zone
    "layer_start": 10,            # Narrower: focus on peak fusion
    "layer_end": 14,              # Narrower: focus on peak fusion

    # Heads
    "head_top_k_pct": 0.20,

    # SRF biasing
    "alpha": 2.5,                 # Higher boost to compensate for text suppression
    "eps": 0.0,
    "bias_mode": "additive_logit",

    # NOVEL: Stronger system prompt suppression
    # We'll set this directly in srf/config.py or via env var
    # sys_beta = 0.20 (vs 0.10 baseline)

    "beta": None,
}
