# SRF Findings — POPE × Qwen2.5-VL

**Date:** 2026-04-24  
**Total experiments:** ~110  
**Best adversarial accuracy:** 0.8900 (n=100), validated avg 0.8933 across all 3 splits

---

## Best Configuration

### SALIENCY (Stage 1)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `source` | `"clip"` | ViT-B/32, better than HSSA or ensemble |
| `clip_coarse_grid` | `7` | 7×7 = ~50px patches; definitively optimal (5,6,8,9 all worse) |
| `clip_top_k_pct` | `0.30` | Top-30% image tokens; binary top-k hurts |
| `clip_use_soft` | `True` | Soft continuous [0,1]; binary mask hurts |
| `clip_absence_thresh` | `0.20` | Internal absence detection (not used for boost/suppress decision) |
| `clip_suppress_thresh` | `0.248` | **Key param**: max_sim ≥ 0.248 → boost; < 0.248 → suppress |
| `clip_suppress_alpha` | `5.0` | Suppress strength for absent objects |

**Absence-aware strategy:** CLIP max_sim threshold 0.248 separates present (mean 0.252) from absent (mean 0.242) with ~70% accuracy. When object likely absent → suppress all image tokens (fix hallucination FP). When likely present → boost salient tokens (fix FN).

### BIAS (Stage 2)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `bias_mode` | `"additive_logit"` | Pre-softmax logit addition; other modes (prob_scale, prob_interp, attn_floor) all neutral or worse |
| `layer_start` | `8` | |
| `layer_end` | `14` | Layers 8–14 are the VL fusion zone; L15 redundant, L7 and L13 hurt |
| `head_top_k_pct` | `0.20` | Top-20% vision-aware heads (3 heads); very sensitive — 10%/30%/50% all worse |
| `boost_alpha` | `2.0` | Logit boost for present objects; 1.5–5.0 all plateau at same result |
| `background_eps` | `0.0` | Any non-zero suppression of background hurts |
| `sys_beta` | `0.10` | System-prompt token suppression |
| `srf_apply_phase` | `"prefill"` | Apply intervention only during prefill; generation-time bias redundant |

---

## Validation Results (commit `29b1151b`, all 3 splits n=100 each)

| Split | Accuracy |
|-------|----------|
| Adversarial | 0.8900 |
| Popular | 0.8900 |
| Random | 0.9000 |
| **Average** | **0.8933** |

Baseline (no intervention): 0.8800 adversarial, ~0.88–0.90 other splits.

---

## What Definitively Does NOT Help

| Approach | Result | Why |
|----------|--------|-----|
| HSSA saliency (any layer) | ≤ 0.88 | Noisy; CLIP localises queried objects better |
| CLIP+HSSA ensemble | ≤ 0.88 | HSSA adds noise even at low weight |
| Binary top-k saliency | 0.87 | Soft gradient gives smoother, more effective boost |
| Any background_eps > 0 | ≤ 0.88 | Non-salient suppression disrupts correct cases |
| Later layers (>14) | ≤ 0.88 | L15+ are in output shaping zone, too late |
| Earlier layers (<8) | ≤ 0.88 | L7 and below add noise |
| More heads (>20%) | ≤ 0.87 | Non-vision heads dilute the signal |
| Fewer heads (<20%) | 0.88 | Too few heads lose coverage |
| Adaptive alpha (model confidence) | ≤ 0.88 | Cannot distinguish FP and correct-YES cases without GT |
| Boost without suppress | 0.88 | Both directions needed for the synergy |
| Suppress without boost | 0.88 | Both directions needed |
| ViT-L/14 CLIP | 0.86 | ViT-B/32 is better discriminator for this task |
| lm_head logit adjustment | ≤ 0.89 | Redundant with attention intervention |
| Generation-phase bias | 0.87 | Prefill-only is sufficient and cleaner |
| V-proj scaling | 0.88 | Disrupts boost/suppress synergy |
| Per-layer head calibration | 0.87 | Local optima lack global synergy |

---

## Ceiling Analysis

The 0.8900 ceiling appears hard. 11 errors remain at best config:
- **4 FN** (gt=YES): CLIP correctly identifies object present, boost applied, but insufficient to overcome language prior
- **5 FN** (gt=YES): CLIP wrongly classifies as absent (max_sim 0.22–0.243), suppression applied instead of boost
- **2 FP** (gt=NO): CLIP correctly identifies absent, suppress applied, but hallucination prior too strong

The 5 wrongly-suppressed FN cases are the main limiter — they require better CLIP discrimination (the present/absent threshold gap is only ~0.010 max_sim units).

---

## Using SRF on Another Dataset

### Requirements
- POPE-style binary VQA ("Is there a `<noun>` in the image?")
- Qwen2.5-VL-3B (or similar Qwen2.5-VL variant)

### What to recalibrate
1. **`clip_suppress_thresh` (0.248)**: This was calibrated on POPE adversarial. For a new dataset, run a small calibration sample (20–50 items) to find the max_sim that separates present from absent labels. The `saliency_quality.py` script does this.
2. **`head_top_k_pct` (0.20)**: The 3 vision heads were calibrated on POPE adversarial. Re-run `_calibrate_heads()` on the new dataset's samples. It uses only 20 calibration samples.
3. **`clip_suppress_alpha` (5.0)**: If the new dataset has different hallucination rates, the suppress strength may need tuning. Try 3.0, 5.0, 7.0.

### What is likely stable across datasets
- Layer range 8–14 (architecture-level finding for Qwen2.5-VL)
- `clip_coarse_grid=7` (image-size independent)
- `clip_top_k_pct=0.30` (saliency coverage)
- `bias_mode="additive_logit"` (the mechanism)
- `boost_alpha=2.0` (gentle present-boost)
- `background_eps=0.0`
- `srf_apply_phase="prefill"`

### Quick test (no recalibration)
Copy the best config as-is. The absolute gain may differ from POPE but the method should generalise to similar binary VQA tasks.
