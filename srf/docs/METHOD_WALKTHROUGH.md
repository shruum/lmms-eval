# SRF Method Walkthrough — Code, Correctness & Parameter Audit

> Purpose: precise code-level walkthrough for debugging, paper writing, and onboarding.
> Cross-referenced against Figure 3 of the paper (SRF.pdf).
> Last verified: 2026-07-24.

---

## Overview

Two variants share the same pipeline up to the final decode step:

| Variant | Decode | Use case |
|---------|--------|----------|
| **SRF** (base) | single forward pass | any dataset |
| **SRF-E** (evidence-amplified) | two-pass contrastive | single-token answers only (POPE, MMVP) |

---

## Phase 0 — One-Time Setup

**`srf.py::setup(model, processor, calib_dataset)`** — called once after model load.

### Step 0a: Detect architecture
```
model_id = model.config._name_or_path
arch     = config.SRF_ARCH_PARAMS[model_id]   # or SRF_ARCH_FALLBACK if unknown
_spatial = model.config.vision_config.spatial_merge_size
```

### Step 0b: Build calibration inputs
**`srf.py::_build_calib_inputs(dataset, n=20, seed=0)`**

For each of 20 random samples from the calibration dataset:
1. Format as chat message (image + question)
2. Tokenize via `processor.apply_chat_template()`
3. Find `img_start`, `img_end` by scanning `input_ids` for `<|image_pad|>` token ID:
   ```python
   img_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
   s = ids.index(img_id)
   e = len(ids) - 1 - ids[::-1].index(img_id)
   ```

### Step 0c: Identify vision-aware heads
**`qwen_attn_patch.py::identify_visual_heads(model, calib_inputs, img_ranges, top_k_pct=0.20)`**

1. Run 20 forward passes in `method="baseline"` mode (no intervention)
2. After each softmax call inside the LM decoder, capture:
   ```
   result[0, :, text_q_start:, img_start:img_end+1]  # (n_heads, n_text_q, n_img)
   ```
   — attention FROM text query positions TO image key positions
3. Average per-head scores across all 20 samples
4. Select top-20% heads by score → store as `bool head_mask (n_heads,)` in `patch._STATE`

> **Figure 3 alignment (①):** This matches "Grounding-Head Calibration — semantically-responsive heads". ✓

### Step 0d: Patch the model
**`qwen_attn_patch.py::patch_model(model, "vaf", alpha)`**

Replaces `torch.nn.functional.softmax` globally with `_patched_softmax`.
Registers three types of forward hooks:
- `model.language_model` pre-hook → sets `_STATE["in_decoder"] = True`
- `model.language_model` post-hook → sets `_STATE["in_decoder"] = False`
- Each `layer.self_attn` pre-hook → sets `_STATE["current_layer"] = idx`

The patched softmax is a no-op unless `enabled=True`, `in_decoder=True`, `tensor.dim()==4`, and `method != "baseline"`.

---

## Phase 1 — Per-Dataset Reset

**`srf.py::reset_for_dataset(dataset, **overrides)`** — called once per dataset (or per hyperparam sweep step).

Builds two config dicts from a 4-level priority stack:

```
Priority (highest wins): CLI override → SRF_ARCH_PARAMS → SRF_DATASET_PARAMS → SRF_DEFAULTS
```

**`BIAS` dict** (controls attention intervention):
- `layer_start`, `layer_end`: which decoder layers receive the intervention
- `boost_alpha` (α): logit boost for salient tokens
- `background_eps` (ε): logit suppression for background tokens
- `sys_beta` (β): logit suppression for system-prompt tokens
- `srf_apply_phase`: "generation" / "prefill" / "both"
- `bias_mode`: "additive_logit" (default) or post-softmax variants

**`SALIENCY` dict** (controls CLIP):
- `saliency_mode`: which saliency function to call (default: `clip_full_gate_v3`)
- `clip_top_k_pct`: fraction of image tokens to boost
- `clip_fallback_thresh`: full-image similarity threshold for presence gate (0.21)
- `clip_model`: which CLIP model to use
- `clip_coarse_grid`: coarse patch grid size (7 for Qwen/448px)

Both dicts are pushed into `patch._STATE` via `_sync_patch_state()`.

Re-calibration only triggers if `head_top_k_pct` or `layer_end` changed (expensive).

---

## Phase 2 — Per-Sample Preparation

**`srf.py::prepare_sample(inputs, img_start, img_end, image, question, model, processor)`**

Called before **every** `model.generate()` or `model(**inp)`. Saliency is fully per-sample — nothing is shared between samples.

### Step 2a: Update image token range
**`qwen_attn_patch.py::update_sample(img_start, img_end)`**
```python
_STATE["img_start"] = img_start
_STATE["img_end"]   = img_end
_STATE["sys_end"]   = img_start - 1   # last system-prompt token
```

### Step 2b: Noun extraction
**`noun_extract.py::extract_clip_noun(question, mode)`**

Mode is dataset-specific:
- `pope`: "Is there a **chair** in the image?" → "chair"
- `mmvp`: "Is the **butterfly**'s wings open?" → "butterfly"
- `vlmbias`: "How many **logos** are on this image?" → "logos"
- `mmbench`/`hallusionbench`: MCQ-aware extraction

**Bad-noun gate** (defined inline in `srf.py:560`): if the extracted noun is in a hardcoded stop-word set (`_BAD_NOUNS`) or `len(noun) <= 2`, set `method="baseline"` and **return early** — no intervention for this sample.

> This fires for MME samples like "Does this image show…" → noun="does" → baseline.

### Step 2c: CLIP saliency (default: `clip_full_gate_v3`)
**`clip_salience.py::compute_clip_salience_full_gate_v3(image, noun, grid_h, grid_w, ...)`**

`grid_h, grid_w` are obtained from `clip_salience.get_grid_dims(inputs, _spatial)`:
```python
thw    = inputs["image_grid_thw"][0]   # [T, H_patches, W_patches] — pre-merge ViT grid
grid_h = int(thw[1]) // spatial_merge_size
grid_w = int(thw[2]) // spatial_merge_size
# grid_h * grid_w == img_end - img_start + 1  (for T=1 images)
```

Inside `compute_clip_salience_full_gate_v3`:

1. **Move CLIP to GPU** (stored on CPU between samples)
2. **Encode text — 5-template ensemble**: instead of encoding the bare noun, encode all five templates and mean-pool then re-normalize:
   ```python
   templates = [
       "{noun}",
       "a photo of a {noun}",
       "a photo of the {noun}",
       "there is a {noun} in this photo",
       "an image containing a {noun}",
   ]
   text_feats = stack([CLIP_text(t) for t in templates])  # (5, D)
   noun_feat  = F.normalize(text_feats.mean(dim=0), dim=-1)  # mean-pool → re-normalize
   ```
3. **Encode image** + patch crops at 3×3, 5×5, 7×7 grids — all in one GPU block
4. **Compute full-image similarity**: `full_img_sim = cosine(CLIP(image), noun_feat)`
5. **Compute patch similarities**: for each grid scale, cosine-sim between each patch and `noun_feat` → `patch_max_sim = max(all_patch_sims across all scales)`
6. **Presence gate (OR)**: `object_present = (full_img_sim >= 0.21) OR (patch_max_sim >= 0.27)`
   - Rationale: small or partially-visible objects may score below 0.21 on the full image but exceed 0.27 on the most-matching patch.
7. **Build multi-scale spatial saliency**:
   - Upsample each coarse grid to `(grid_h, grid_w)` via bilinear interpolation
   - Take elementwise max across scales → min-max normalize → saliency tensor `(grid_h * grid_w,)`
6. **Move CLIP back to CPU** (free VRAM for VLM forward pass)

Additionally computed (diagnostics / backup experiments — **not used in default gate**):
- `blur_delta = full_img_sim - CLIP(blurred_image, noun)` ← requires extra CLIP forward
- `raw_entropy`: entropy of softmax(patch_sims / 0.02) at finest scale
- `cross_scale_iou`: Jaccard of top-30% patch locations between 3×3 and 7×7 scales

> **Fixed (2026-07-24)**: Backup signals are now gated on `backup` mode. With default `backup="none"`, only `gate_full` is evaluated — no blurred image, no entropy, no cross-scale IoU. Blurred image encoding (which required an extra CLIP forward pass per sample) is now skipped entirely. Backup modes (`"blur_delta"`, `"cross_scale"`, `"raw_entropy"`) remain available for experiments but are confirmed dead-ends (all hit same 0.900 val ceiling).

### Step 2d: Write result to patch state
Back in `srf.py::prepare_sample()`:

If **object present**:
```python
clip_conf = min(full_img_sim / clip_fallback_thresh, 1.0)   # confidence-capped
_STATE["value"]         = boost_alpha * clip_conf            # effective α
_STATE["salience_mask"] = saliency_tensor                    # (n_img_tokens,) float [0,1]
```

If **object absent** (gate fails):
```python
_STATE["salience_mask"] = None
_STATE["value"]         = -neg_absent_alpha   # default neg_absent_alpha=2.0 for pope
                                               # (negative → suppression when object absent)
```

> **Figure 3 alignment (top section):** Text Query → Semantic Subject Extraction (`noun_extract.py`) → Cross-Modal Alignment (`clip_salience.py`) → Semantic Relevance Map (`salience_mask`). ✓

---

## Phase 3 — Forward Pass with Attention Intervention

**`qwen_attn_patch.py::_patched_softmax(input, dim, dtype)`**

Fires in place of `F.softmax` for every attention layer. Active only when `enabled=True`, `in_decoder=True`, `input.dim()==4`, `method != "baseline"`.

For `method="srf"`, only layers in `[layer_start, layer_end]` are modified:

### Step 3a: Phase gate
```python
_is_gen   = (input.shape[2] == 1)   # q_len==1 → generation step; q_len>1 → prefill
_phase_ok = (
    phase == "both"
    or (phase == "generation" and _is_gen)
    or (phase == "prefill"    and not _is_gen)
)
```
If `_phase_ok` is False for this step → no modification, proceed to original softmax.

Best config uses `phase="both"` for POPE/MMVP/VLIND (fixed 2026-07-26 — `phase="generation"` was a no-op during prefill evaluation, so POPE was receiving zero intervention) and `phase="generation"` for VLMBias/MME (intervene only at decode steps).

### Step 3b: System-prompt suppression (pre-softmax)
```python
# All tokens from position 0 to img_start-1 (system message)
input[..., :sys_end+1] -= sys_beta    # applied only to vision-aware heads
```

### Step 3c: Image-token boost (pre-softmax, `bias_mode="additive_logit"`)

With saliency:
```python
bias_row[i] = alpha * sal[i] - eps * (1 - sal[i])
# salient tokens:    +alpha  (positive; attend more)
# background tokens: -eps    (negative; attend less)
```

Without saliency (`salience_mask=None`, object absent):
```python
bias_row = full_uniform value in _STATE["value"]
# If neg_absent_alpha>0: value is negative → suppress image attention
```

Apply to vision-aware heads only:
```python
full_bias[0, head_mask, 0, :] = bias_row   # shape: (1, n_heads, 1, n_img)
input[..., img_start:img_end+1] += full_bias
```

### Step 3d: Original softmax runs
```python
result = _ORIGINAL_SOFTMAX(input, dim=dim, dtype=dtype)
# → attention probabilities (batch, n_heads, q_len, kv_len)
```

> **Figure 3 alignment (② ③):** ② Fusion-Layer Targeting matches `[layer_start, layer_end]` gating. ③ Query-Grounded Token Amplification matches steps 3b and 3c. ✓

---

## Phase 4 — Cleanup

**`srf.py::cleanup()`**

```python
_STATE["salience_mask"]   = None    # clear per-sample saliency
_STATE["method"]          = "srf"   # restore (dynlayer may have changed layer range)
_STATE["vaf_layer_start"] = BIAS["layer_start"]
_STATE["vaf_layer_end"]   = BIAS["layer_end"]
```

---

## SRF-E Extension (Two-Pass Contrastive)

**`srf_e.py::get_contrastive_logits(model, inp, gamma)`** — for single-token answers.

Replaces the single `model(**inp)` call:

```
Pass 1: method="srf",      full image    → logits_full  = model(**inp)[:, -1, :]
Pass 2: method="baseline", zeroed image  → logits_noval = model(**inp_noval)[:, -1, :]
        where inp_noval["pixel_values"] = zeros_like(inp["pixel_values"])

logits_final = logits_full + gamma * (logits_full - logits_noval)
predicted_token = argmax(logits_final)
```

**`srf_e.py::generate_contrastive()`** — for multi-token answers (step-by-step with two KV caches). Currently broken for VLM Bias: contrastive pass suppresses the `{` format token.

> **γ (gamma)** is the only SRF-E-specific hyperparameter. Best value: γ=3.0 (tuned on MMVP val).

---

## Figure 3 vs Code — Discrepancy Log

| # | Paper Figure 3 says | Code reality | Impact |
|---|---|---|---|
| 1 | Fusion layers **L/3 to 2L/3** | Tuned best: **ls=6, le=12** (≈L/4.5 to L/2.3 for 28-layer model) | Paper figure was drawn before layer sweep. Figure needs updating. |
| 2 | Single symbol **−β** for both CTX suppression and background image suppression | Two separate params: `sys_beta=0.30` (CTX) and `eps=0.20` (background image tokens) | Paper should distinguish β (CTX) and ε (background). |
| 3 | Figure implies backup gate signals are part of the method | Backup signals (`raw_entropy`, `cross_scale_iou`, `blur_delta`) are dead experiments — computed but never used with default `backup="none"` | Remove from paper description; fix wasted compute. |

---

## Parameter Audit

### ✅ Fully controlled (config or CLI)

| Parameter | Config key | Default |
|---|---|---|
| `layer_start` | `SRF_ARCH_PARAMS[model]["layer_start"]` | 6 (3B, tuned) |
| `layer_end` | `SRF_ARCH_PARAMS[model]["dataset_layer_end"][dataset]` | 12 (POPE), 16 (MMVP) |
| `head_top_k_pct` | `SRF_ARCH_PARAMS[model]["head_top_k_pct"]` | 0.20 |
| `alpha` (boost) | `SRF_DATASET_PARAMS[dataset]["alpha"]` | 4.0 |
| `eps` (background) | `SRF_DATASET_PARAMS[dataset]["eps"]` | 0.20 |
| `sys_beta` | `SRF_DEFAULTS["sys_beta"]` | 0.30 |
| `phase` | `SRF_DATASET_PARAMS[dataset]["phase"]` | "both" (POPE, MMVP, VLIND); "generation" (VLMBias, MME) |
| `bias_mode` | `SRF_DEFAULTS["bias_mode"]` | "additive_logit" |
| `clip_top_k_pct` | `SRF_ARCH_PARAMS[model]["clip_top_k_pct"]` | 0.30 |
| `clip_fallback_thresh` | `SRF_ARCH_PARAMS[model]["clip_fallback_thresh"]` | 0.21 |
| `clip_model` | `SRF_ARCH_PARAMS[model]["clip_model"]` | ViT-B/32 |
| `saliency_mode` | `SRF_ARCH_PARAMS[model]["saliency_mode"]` | "clip_full_gate_v3" |
| `gamma` (SRF-E) | `SRFE_DEFAULT_GAMMA` / CLI `--gamma` | 3.0 |
| `calib_n` | `SRF_DEFAULTS["calib_n"]` | 20 |
| `calib_seed` | `SRF_DEFAULTS["calib_seed"]` | 0 |
| `neg_absent_alpha` | `SRF_DATASET_PARAMS[dataset]["neg_absent_alpha"]` | 2.0 (pope) |
| `spatial_merge_size` | `SRF_ARCH_PARAMS[model]["spatial_merge_size"]` | 2 |

### ⚠️ Hardcoded — not in config

| Parameter | Location | Value | Matters when |
|---|---|---|---|
| `patch_thresh` | `clip_salience.py` | `0.27` | OR gate: object flagged present if any patch exceeds this threshold |
| `coarse_scales` | `clip_salience.py` function default args | `(3, 5, 7)` | Always (affects spatial saliency map) |
| `gate_full_soft` factor | `clip_salience.py:691,1090` | `0.85 * thresh` | Only when `backup != "none"` (dead experiment) |
| `_BLUR_RADIUS` | `clip_salience.py` module const | `15.0` | Only when `backup="blur_delta"` (dead) |
| `_BLUR_DELTA_THRESH` | `clip_salience.py` module const | `0.005` | Only when `backup="blur_delta"` (dead) |
| `_CROSS_SCALE_THRESH` | `clip_salience.py` module const | `0.30` | Only when `backup="cross_scale"` (dead) |
| `_RAW_ENTROPY_THRESH` | `clip_salience.py` module const | `0.95` | Only when `backup="raw_entropy"` (dead) |
| `_RAW_ENTROPY_TEMPERATURE` | `clip_salience.py` module const | `0.02` | Only when `backup="raw_entropy"` (dead) |
| `_CONTRAST_THRESH` | `clip_salience.py` module const | `1.40` | Diagnostics only, not in gate |
| contrast `k_top` fraction | function bodies | `0.30` (hardcoded, ignores `clip_top_k_pct`) | Diagnostics only |
| `max_pixels` VLMBias calib | `srf.py::_build_calib_inputs` | `400 * 400` | VLMBias calibration only |
| `_BAD_NOUNS` stop-word set | `srf.py:560` | hardcoded set | Every sample (OK — stop-words don't need tuning) |

**Bottom line:** For the default `clip_full_gate_v3` + `backup="none"` code path, the only hardcoded value that materially affects results is `coarse_scales=(3,5,7)`. Everything else is in config or is a dead-experiment artifact.

---

## Saliency: Per-Sample Confirmation

Yes — saliency is fully per-sample:
- `prepare_sample()` runs CLIP on the current image with the current noun
- `patch._STATE["salience_mask"]` is overwritten for each sample
- `cleanup()` resets it to `None` after generation
- No saliency state persists between samples

**Head mask** is per-dataset (set during calibration). **BIAS/SALIENCY config** is per-dataset-reset.

---

## Debug Checklist

Run these checks in order when something seems wrong:

| Check | How | Expected |
|---|---|---|
| Patch is active | `patch._STATE["enabled"]` | `True` |
| Method is SRF | `patch._STATE["method"]` after `prepare_sample()` | `"srf"` (not `"baseline"`) |
| Head mask set | `patch._STATE["head_mask"].sum()` | ~3 heads for 16-head model at 20% |
| Saliency set | `patch._STATE["salience_mask"]` | tensor `(n_img_tokens,)` or `None` if absent |
| Saliency dimension | `len(salience_mask) == img_end - img_start + 1` | Must match — this is the active bug |
| Layer range active | `patch._STATE["vaf_layer_start/end"]` | Match `BIAS["layer_start/end"]` |
| Noun extracted | print `_noun` inside `prepare_sample()` | Object noun, not a stop-word |
| Phase gate fires | add print in `_patched_softmax` at phase check | Should fire at `q_len==1` for `phase="generation"` |
| Intervention changes output | `sanity_check_intervention_changes_attention()` in `qwen_attn_patch.py` | Pass |

---

## Known Issues (as of 2026-07-24)

1. ~~**Dimension mismatch bug**~~ **Fixed 2026-07-26**: `get_grid_dims()` now returns correct values (was dividing by `spatial_merge_size` twice). An explicit assertion `assert len(saliency) == img_end - img_start + 1` has been added in `prepare_sample()` to catch any future regressions.

2. ~~**Backup signal wasted compute**~~ **Fixed 2026-07-24**: `compute_clip_salience_full_gate_v3` now only computes backup signals when the relevant `backup` mode is active. Default `backup="none"` skips blur, entropy, and cross-scale IoU entirely.

3. **SRF-E broken for VLM Bias**: `generate_contrastive()` suppresses the `{` format token. `content_offset=1` workaround exists but doesn't fully fix it.

4. ~~**config.py header vs RESEARCH_STATUS mismatch**~~ **Fixed 2026-07-26**: config.py header comment updated to match current best values (`ls=8, le=12, α=2.0` for POPE).
