# `llava_attn_patch` vs `qwen_attn_patch` — full comparison

Written 2026-09-20. Reference: `SRF_details.md` section 7, `BRANCH_DIFF.md`.

This document records every functional difference between the two patches, which
bugs need fixing before LLaVA numbers are meaningful, and what the correct
implementation looks like in each case.

---

## Overview

| | `qwen_attn_patch.py` (`autoresearch/mmvp-srf`) | `llava_attn_patch.py` (`srf-llava`) |
|---|---|---|
| Lines | 1114 | 277 |
| Intervention approach | pre-softmax additive logit | post-softmax multiplicative (default), additive available |
| SRF equation | `alpha*sal - eps*(1-sal)` | `enh_para*sal` only — **eps missing** |
| System suppression | pre-softmax, phase-gated, head-masked | post-softmax multiplicative, **discarded in additive mode** |
| Head masking | applied to all methods via `full_bias` | **not applied in additive mode** |
| Phase gate | yes (`srf_apply_phase`) | **missing** |
| Calibration score | VTAR (ratio) in `head_calibration.py` | per-token mean in `identify_visual_heads` |
| `clip_full_gate_v3_paper` | yes | **no** |
| `saliency_mode="random"` | yes | **no** |
| `_STATE["head_weight"]` (soft weights) | yes | no (not needed for `ratio_topk`) |
| `_captured` (VTAR capture) | yes | no |
| Architecture dispatch | Qwen only | LLaVA + Qwen via `in_language_model` hook |

---

## Bug 1 — `lambda_bg` not applied in additive mode

**Where:** `llava_attn_patch._patched_softmax`, lines 114-125.

**Qwen (correct):**
```python
bias_row = alpha_val * sal_dev - eps * (1.0 - sal_dev)
```

**LLaVA (wrong):**
```python
modified[:, :, :, img_start:img_end + 1] += enh_para * sal_dev
```

No `-eps*(1-sal)` term. `srf_background_eps` is stored in `_STATE` but never read in
the additive path. `lambda_bg=0.2` is silently zero.

**Fix:** Read `eps = float(_STATE.get("srf_background_eps", 0.0))`, build
`bias_row = enh_para * sal_dev - eps * (1.0 - sal_dev)`, apply via `full_bias`
(see Bug 3 fix).

---

## Bug 2 — System suppression (`lambda_sys`) is a no-op in additive mode

**Where:** `llava_attn_patch._patched_softmax`, lines 99-125.

**Sequence of events in additive mode:**
1. Line 99: `attn_weights = _ORIGINAL_SOFTMAX(input, ...)` — post-softmax weights computed
2. Line 110: `attn_weights[:, :, :, :sys_end+1] *= sup_para` — suppression applied to `attn_weights`
3. Line 119-125: `modified = input.clone(); ...; return _ORIGINAL_SOFTMAX(modified, ...)` — **a fresh softmax is returned; `attn_weights` is discarded**

`sup_para` multiplies a tensor that is never used. `lambda_sys=0.30` is silently a no-op.

**Qwen (correct):**
System suppression is pre-softmax: `input[..., :sys_end+1] -= beta` before the single
`_ORIGINAL_SOFTMAX(input)` call. It is inside the phase gate and head-masked.

**Fix:** In the additive path, apply system suppression to `modified` (pre-softmax logits)
before returning, not to `attn_weights`:
```python
modified = input.clone()
# system suppression — pre-softmax
if _phase_ok and sys_end is not None and sys_end > 0 and beta > 0:
    if head_mask is not None:
        sup_bias = modified.new_zeros(1, n_heads, 1, 1)
        sup_bias[0, mask_dev, 0, 0] = -beta
        modified[..., :sys_end + 1] = modified[..., :sys_end + 1] + sup_bias
    else:
        modified[..., :sys_end + 1] -= beta
# image boost ...
return _ORIGINAL_SOFTMAX(modified, ...)
```

---

## Bug 3 — `head_mask` not applied in additive mode

**Where:** `llava_attn_patch._patched_softmax`, line 122.

**LLaVA (wrong):**
```python
modified[:, :, :, img_start:img_end + 1] += enh_para * sal_dev.unsqueeze(0).unsqueeze(0)
```
The second dimension is `:` — ALL heads are boosted. `head_mask` is stored but never
consulted.

**Qwen (correct):**
```python
full_bias = input.new_zeros(1, n_heads, 1, n_img)
full_bias[0, mask_dev, 0, :] = bias_row   # only selected heads
input[..., s:e+1] = input[..., s:e+1] + full_bias
```

**Impact.** The Qwen k-sweep at band 6-31 gives:
k=0.2 → 45.33, k=1.0 (all heads) → 38.67, BELOW baseline 40.00.
With Bug 3 present, the LLaVA patch runs the equivalent of k=1.0 regardless of the
configured `head_top_k_pct`.

**Fix:** Build `full_bias` with zeros, fill only the masked head indices, add to `modified`.

---

## Bug 4 — Phase gate missing

**Where:** `llava_attn_patch._patched_softmax`, additive branch.

`_STATE["srf_apply_phase"]` is stored (default `"both"`) but never read. The Qwen patch
has a `_phase_ok` boolean that gates both the image boost and system suppression:

```python
_phase    = _STATE.get("srf_apply_phase", "both")
_is_gen   = (input.shape[2] == 1)
_phase_ok = (_phase == "both"
             or (_phase == "generation" and _is_gen)
             or (_phase == "prefill"    and not _is_gen))
```

**Impact.** With the default `"both"`, no behavior difference. But `phase="generation"` is
load-bearing on VLMBias (+0.90pp vs `phase="both"`, see `SRF_details.md` 5.5), so
this must be fixed before running VLMBias.

**Fix:** Add the `_phase_ok` check before any modification, identical to the Qwen patch.

---

## Bug 5 — Calibration score: per-token mean instead of VTAR

**Where:** `llava_attn_patch.identify_visual_heads`, line 72.

**LLaVA:**
```python
img_attn = attn_weights[:, :, :, img_start:img_end + 1].mean(dim=-1)
img_attn = img_attn.mean(dim=(0, 2))  # (n_heads,)
```
This divides by n_img and by (batch × q_len). The score scales as 1/n_img, so it cannot
be thresholded and varies with image resolution.

**Qwen `head_calibration.py` (VTAR / `ratio_topk`):**
```python
# attention to image keys summed, divided by attention to ALL keys
# = vision fraction in [0,1]
img_sum  = attn[:, :, :, img_start:img_end+1].sum(dim=-1)   # (batch, heads, q)
all_sum  = 1.0   # attention rows already sum to 1
vtar_h   = img_sum.mean(dim=(0, 2))   # (n_heads,)
```

**Impact.** When `head_calibration.py` is ported (the main missing piece), it replaces
`identify_visual_heads` entirely, so this bug becomes moot. However, any run that uses
the existing `llava_attn_patch.identify_visual_heads` function produces per-token-mean
scores, not VTAR. They rank heads the same within a layer (constant divisor), so the
top-k selection is equivalent — but the absolute values differ.

---

## What is correct in `llava_attn_patch`

- Hook installation on `transformer` (LlamaModel) and per-layer `self_attn`: correct
- `get_image_token_range` using `model.config.image_token_index`: correct
- `attn_implementation="eager"` enforcement: correct
- Layer range via `vaf_layer_start/end`: correct
- Multiplicative mode (VAF baseline): correct and untouched
- `_calib_head_acc` accumulation structure: correct (same bug as Qwen pre-VTAR, same behaviour)

---

## What is missing from `srf-llava` `srf.py`

| Feature | Status |
|---|---|
| `clip_full_gate_v3_paper` mode | absent — `clip_full_gate_v3` shipped default is what all LLaVA numbers will use |
| `saliency_mode="random"` | absent — random-map ablation control cannot be run |

Neither blocks initial LLaVA MMVP/POPE runs. Add before writing the random-controls
appendix table.

---

## Fix priority

| # | Bug | Blocks |
|---|---|---|
| 1 | Bug 3: head_mask not applied | every run, worst accuracy impact |
| 2 | Bug 2: lambda_sys no-op | lambda_sys=0.30 costs 3 pairs on Qwen, same expected on LLaVA |
| 3 | Bug 1: lambda_bg missing | smaller, but needed for paper-correct ablation |
| 4 | Bug 4: phase gate | VLMBias only; fix before that run |
| 5 | Bug 5: calibration VTAR | moot once head_calibration.py is ported |

All four code bugs are in the same 30-line block of `_patched_softmax`. Fix them together.

---

## Sanity test

Run `srf/tests/test_llava_patch_sanity.py` before and after any patch change. It tests
all four bugs with controlled tensors — no model load required.
