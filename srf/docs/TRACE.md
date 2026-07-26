# SRF Single-Sample Trace
> Model: Qwen/Qwen2.5-VL-3B-Instruct
> Dataset: POPE (1 adversarial sample)
> Date: 2026-06-18

## 1. Model & Setup

- Model architecture: Qwen2.5-VL-3B-Instruct
- Decoder layers: 28  (index 0–27)
- Attention heads per layer: 16 (Q/K), 8 (V, GQA)
- Image token: `<|image_pad|>` (id=151655)
- Device: cuda:0

## 2. SRF Calibration

### BIAS config (pope)
| Parameter | Value |
|-----------|-------|
| `layer_start` | 8 |
| `layer_end` | 12 |
| `boost_alpha` (α) | 2.0 |
| `background_eps` (ε) | 0.2 |
| `neg_absent_alpha` | 2.0 |
| `sys_beta` | 0.3 |
| `text_beta` | 0.0 |
| `phase` | `generation` |
| `bias_mode` | `additive_logit` |
| `head_top_k_pct` | 0.2 |

### Head calibration
- Total heads in model: 16 per layer
- `head_top_k_pct` = 0.2 → top 3 heads selected
- **Vision-aware head indices**: [1, 3, 4]
- Calibration: 20 POPE samples, seed=0. Scores each head by mean attention to image tokens from text query positions.

### SALIENCY config
| Parameter | Value |
|-----------|-------|
| `saliency_mode` | `clip_full_gate_v3` |
| `clip_model` | `openai/clip-vit-base-patch32` |
| `clip_coarse_grid` | 7 |
| `clip_top_k_pct` | 0.3 |
| `clip_fallback_thresh` | 0.2 |

## 3. Sample

- **Category**: adversarial
- **Question**: "Is there a backpack in the image?"
- **GT answer**: No  *(model should say No — language prior might say Yes)*
- **Image size**: 640×427 px

## 4. Tokenization & Token Layout

- Total input tokens: **381**
- Image token range: **[15, 359]** (inclusive) = 345 image tokens
- System/prompt tokens: [0, 14] = 15 tokens
- Post-image text tokens: [360, 380] = 21 tokens

```
Token layout (input_ids):
  [0 … 14]            system + question prefix   (15 tokens)
  [15 … 359]  image tokens              (345 tokens)
  [360 … 380]  question suffix + gen prompt (21 tokens)
```

## 5. CLIP Saliency Map

- **Question**: "Is there a backpack in the image?
Answer with Yes or No only."
- **Extracted CLIP noun**: `backpack`

- **Saliency mode**: `clip_full_gate_v3`
- **CLIP grid**: 15×23 = 345 coarse patches → 345 image tokens
- **CLIP full-image similarity**: 0.2070  (threshold: 0.2)
- **Object present**: True  (above threshold)

- **CLIP confidence**: 1.0000  (full_img_sim / thresh)
- **Effective α** = α × conf = 2.0 × 1.0000 = **2.0000**

### Saliency distribution
| Stat | Value |
|------|-------|
| n_img_tokens | 345 |
| top_k_pct | 0.3 → top 103 tokens boosted |
| sal min | 0.0000 |
| sal max | 1.0000 |
| sal mean | 0.5266 |
| sal std | 0.2033 |

Top-10 most salient image token positions (0-indexed within image range):
```
  rank  1: token  218  grid (9,11)  sal=1.0000  logit_boost=+2.0000
  rank  2: token  172  grid (7,11)  sal=1.0000  logit_boost=+2.0000
  rank  3: token  241  grid (10,11)  sal=1.0000  logit_boost=+2.0000
  rank  4: token  171  grid (7,10)  sal=0.9738  logit_boost=+1.9477
  rank  5: token  195  grid (8,11)  sal=0.9478  logit_boost=+1.8956
  rank  6: token  170  grid (7,9)  sal=0.9477  logit_boost=+1.8954
  rank  7: token  173  grid (7,12)  sal=0.9349  logit_boost=+1.8698
  rank  8: token  169  grid (7,8)  sal=0.9215  logit_boost=+1.8430
  rank  9: token  194  grid (8,10)  sal=0.9151  logit_boost=+1.8303
  rank 10: token  149  grid (6,11)  sal=0.9147  logit_boost=+1.8293
```

Background tokens (sal≈0) get logit delta ≈ −ε = **−0.2**
Salient tokens (sal=1.0) get logit delta ≈ **+2.0000**
Net spread (salient vs background): **2.2000 logit units**

## 6. Forward Pass — Per-Layer Logit Modifications

Instrumenting the attention hook to capture exact logit deltas per layer...

### patch._STATE after prepare_sample()
| Key | Value |
|-----|-------|
| `method` | `srf` |
| `value` (boost_alpha effective) | 2.0 |
| `img_start` / `img_end` | 15 / 359 |
| `sys_end` | 14 |
| `srf_apply_phase` | `generation` |
| `srf_bias_mode` | `additive_logit` |
| `srf_background_eps` | 0.2 |
| `vaf_beta` (sys suppression) | 0.3 |
| `vaf_layer_start` | 8 |
| `vaf_layer_end` | 12 |
| `salience_mask` is None | False |
| `salience_mask` shape | [345] |
| `salience_mask` min/max | 0.0000 / 1.0000 |
| `head_mask` active heads | [1, 3, 4] |

### Phase gate check
- **phase** = `generation`
- Prefill step: q_len = 381 > 1  →  `_is_gen = False`
  → phase_ok = **False** during prefill (q_len>1). Boost fires ONLY at generation steps (q_len=1)
- Generation step: q_len = 1  →  `_is_gen = True`
  → phase_ok = **True**. All boosts apply.

### Prediction
- Generated tokens: `no`
- Prediction: **no**  |  GT: **no**  |  Correct: **True**

## 7. Per-Layer Intervention Summary

Layer range: [8, 12]  (5 layers active out of 28)

| Layer | Step | q_len | phase_ok | Active heads | Img Δ mean | Img Δ max | Img Δ min | Sys Δ/head |
|-------|------|-------|----------|--------------|-----------|-----------|-----------|-----------|
|    10 | generation |     1 | ✓ FIRES   |  3 of 16 | +0.9585 | +2.0000 | -0.2000 | -0.3000 |
|    10 | prefill    |   381 | ✗ skipped |  3 of 16 | +0.0000 | +0.0000 | +0.0000 | 0.0000 |
|    11 | generation |     1 | ✓ FIRES   |  3 of 16 | +0.9585 | +2.0000 | -0.2000 | -0.3000 |
|    11 | prefill    |   381 | ✗ skipped |  3 of 16 | +0.0000 | +0.0000 | +0.0000 | 0.0000 |
|    12 | generation |     1 | ✓ FIRES   |  3 of 16 | +0.9585 | +2.0000 | -0.2000 | -0.3000 |
|    12 | prefill    |   381 | ✗ skipped |  3 of 16 | +0.0000 | +0.0000 | +0.0000 | 0.0000 |
|     8 | generation |     1 | ✓ FIRES   |  3 of 16 | +0.9585 | +2.0000 | -0.2000 | -0.3000 |
|     8 | prefill    |   381 | ✗ skipped |  3 of 16 | +0.0000 | +0.0000 | +0.0000 | 0.0000 |
|     9 | generation |     1 | ✓ FIRES   |  3 of 16 | +0.9585 | +2.0000 | -0.2000 | -0.3000 |
|     9 | prefill    |   381 | ✗ skipped |  3 of 16 | +0.0000 | +0.0000 | +0.0000 | 0.0000 |

## 8. Suppression Mechanisms

### System-prompt suppression (sys_beta)
- **β = 0.3** applied to tokens [0, 14] (15 system tokens)
- Applied in **vision-aware heads only** ([1, 3, 4])
- Gated by phase: only fires when `phase_ok=True`
- Effect: reduces model's ability to rely on system-prompt context when answering

### Background suppression (eps)
- **ε = 0.2** subtracted from non-salient image tokens
- Tokens with saliency ≈ 0 get logit −ε; tokens with saliency 1.0 get +α
- Net contrast between salient and background: **2.2000** logit units

### Text-token suppression (text_beta)
- **text_beta = 0.0** → **disabled** (0.0)
- Would suppress question tokens in layers [20, 27]
- Not used: experiments showed no gain; language prior is in MLP, not attention

## 9. End-to-End Method Summary


SRF (Semantic Re-Focus) modifies attention logits **pre-softmax** in a single forward pass:

```
For each token t at query position q (in decoder layers [layer_start, layer_end]):

  logit[h, q, img_i] += α × sal[i]          if token i is an image token, h ∈ vision-heads
  logit[h, q, img_i] -= ε × (1 - sal[i])    background image tokens suppressed
  logit[h, q, sys_j] -= β                    system-prompt tokens suppressed
```

Where:
  α (boost_alpha)     = CLIP confidence × base alpha  (presence-gated boost strength)
  sal[i] ∈ [0,1]      = CLIP patch similarity of image token i to the query noun
  ε (background_eps)  = background suppression (non-salient image tokens)
  β (sys_beta)        = system-prompt suppression
  h ∈ vision-heads    = top-k% heads by image attention score (calibrated once)
  phase gate          = only during generation steps (q_len=1) for POPE

### Numeric summary for this sample
| | |
|---|---|
| CLIP noun | `backpack` |
| CLIP full-image sim | 0.2070 |
| Object present | True |
| Effective α | 2.0000 |
| ε (background) | 0.2 |
| β (sys suppress) | 0.3 |
| Active layers | [8, 12] = 5 layers |
| Active heads | [1, 3, 4] (3/16) |
| Image tokens | 345 |
| Salient tokens (top-30%) | 103 |
| Prediction | **no** (GT=no, ✓ correct) |
