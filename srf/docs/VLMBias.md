# VLMBias — SRF Evaluation Documentation

> Model: Qwen2.5-VL-3B-Instruct  
> Dataset: VLM Bias (anvo25/vlms-are-biased), 7 categories, n=2784 full  
> Last updated: 2026-06-18

---

## What is VLMBias?

VLMBias tests whether VLMs rely on **language priors instead of visual evidence**.
Each sample shows an image and asks a counting/attribute question. The model should
answer from the image, but language priors dominate (e.g. "animals have 4 legs" → always predict 4).

**7 categories** (≈397 samples each in full dataset):

| Category | Task type | Difficulty |
|----------|-----------|-----------|
| Animals | Count legs/features | Hard — off-by-one counting errors dominate |
| Chess Pieces | Count pieces on board | Hard — structurally intractable for 3B model |
| Flags | Count stripes/colors | Medium |
| Game Boards | Count pieces/cells | Medium |
| Logos | Count elements | Medium |
| Optical Illusion | Yes/No presence | Easier — binary question |
| Patterned Grid | Count pattern elements | Hard |

**Answer format**: `{X}` where X is a number or yes/no.  
**Metric**: exact-match accuracy (overall and per-category).

---

## Results Table

| Method | Overall Acc | Notes |
|--------|:-----------:|-------|
| **Baseline** (no SRF) | ~18.1% | from old n=105 subsample |
| SRF broken (buggy code) | 18.86% | `clip_fallback_thresh=1.0` zeroed boost — effectively baseline |
| *(ref) Old autoresearch best* | *(22.86%)* | *n=105 subsample — LUCKY, not representative* |
| **SRF fixed (deep+uniform)** | **19.76%** | bug-fixed code, n=2784 full — canonical result |

### Per-Category Breakdown (SRF fixed, n=2784, 2026-06-18)

| Category | Correct | Total | Acc | Notes |
|----------|:-------:|:-----:|:---:|-------|
| Animals | 0 | 546 | **0.0%** | Structurally intractable — pure counting errors |
| Chess Pieces | 0 | 288 | **0.0%** | Structurally intractable — piece counting |
| Flags | 49 | 240 | 20.4% | Some signal |
| Game Boards | 22 | 168 | 13.1% | Some signal |
| Logos | 51 | 414 | 12.3% | Some signal |
| **Optical Illusion** | **392** | **792** | **49.5%** | **Best category — binary yes/no, not counting** |
| Patterned Grid | 36 | 336 | 10.7% | Weak signal |
| **Total** | **550** | **2784** | **19.76%** | |

**The overall 19.76% is almost entirely carried by Optical Illusion (49.5%).** All other
categories are ≤20.4%, and Animals/Chess are 0%. The "improvement" over baseline is
concentrated in the Optical Illusion yes/no questions where image attention actually helps.

**Key insight**: The n=105 subsample result (22.86%) was driven by 9/15 Optical Illusion correct
(=60%), which is highly noisy at n=15. On full n=2784 the true signal is ~19.76% (+1.7pp vs baseline).

---

## Best Parameters (for `srf/eval.py`)

```bash
cd /volumes2/mllm/lmms-eval
conda run -n mllm python srf/eval.py \
  --method srf --datasets vlmbias \
  --layer_start 20 --layer_end 28 --alpha 8.0 --eps 0.5 \
  --saliency_mode clip --clip_fallback_thresh 1.0 \
  --output results/srf_bugfix/ \
  2>&1 | tee results/srf_bugfix/vlmbias_per_cat.log
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| `layer_start` | 20 | Deep layers — counting/reasoning attention |
| `layer_end` | 28 | Last layer of 28-layer model |
| `alpha` | 8.0 | Higher than MMVP/POPE — task is harder |
| `eps` | 0.5 | Background suppression |
| `phase` | `generation` | Boost during decode only (not prefill) |
| `saliency_mode` | `clip` | Basic CLIP patch similarity |
| `clip_fallback_thresh` | **1.0** | Forces ALL samples to uniform boost (max_sim always < 1.0) |
| `head_top_k_pct` | 0.20 | Same 3 heads as MMVP/POPE |
| Inference | `model.generate()` | `{X}` format, max_new_tokens=20 |

### Why `clip_fallback_thresh=1.0` (uniform boost)?

CLIP guidance is irrelevant for VLMBias — top-10%, top-80%, and uniform boost all give
**identical accuracy**. The bottleneck is counting/enumeration capability, not spatial
localization. Setting thresh=1.0 forces every sample into the "absent fallback" path which
applies uniform `alpha=8.0` to all image tokens.

**Warning**: `clip_fallback_thresh=1.0` only works correctly with the bug-fixed code.  
In the buggy version, absent fallback zeroed out `value` → no boost at all (18.86% = baseline).  
Fixed behavior: absent with `neg_absent_alpha=0.0` → uniform `alpha=8.0` boost.

### Why deep layers (20–28)?

Middle layers (8–15, used for MMVP) gave no improvement or regression on VLMBias.
Deep layers contain the counting/reasoning computation. Boosting image attention
there helps the model count from the image rather than recall from language priors.

---

## Code Flow (SRF on VLMBias)

```
srf/eval.py  →  run_vlmbias()
```

### 1. Setup (once)
```python
srf.setup(model, processor, calib_dataset="vlmbias")
```
- Calibrates on 20 VLMBias samples (seed=0)
- Selects 3 vision-aware heads (top 20%)

### 2. Per-dataset reset
```python
srf.reset_for_dataset("vlmbias", layer_start=20, layer_end=28,
                       alpha=8.0, eps=0.5, phase="generation")
```
- `phase="generation"` — boost only during decode (not prefill)
- Unlike MMVP (`phase="both"`), prefill boost doesn't help for free-form generation

### 3. Per-sample: noun extraction
```python
noun = extract_clip_noun(question, mode="vlmbias")
```
- VLMBias questions: "How many legs does the animal have?" → "legs"
- Noun is used for CLIP similarity but result is discarded (uniform mode)

### 4. Per-sample: CLIP saliency (forced uniform)
```python
result = clip_sal.compute_clip_salience(image, noun, ...)
# result.max_sim always < 1.0 (clip_fallback_thresh=1.0)
→ salience_mask = None
→ value = boost_alpha = 8.0   # uniform boost
```

### 5. Inference
```python
out_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
```
- All image tokens get `+alpha=8.0` in deep layers [20,28]
- `{X}` format answered directly: `{4}`, `{Yes}`, etc.

### 6. Answer extraction
```python
pred = normalise(extract_answer(processor.decode(out_ids)))
```
- Regex extracts content inside `{...}`; falls back to first word
- `normalise()`: strip, lowercase, remove `{}`
- Exact match against GT

---

## Hyperparameter Sweep Results (autoresearch_vlmbias, n=105 subsample)

> **Warning**: All values below are from n=105 (15/category). High variance — treat as directional only.

| Experiment | Acc | Δ | Keep? | Notes |
|------------|:---:|:---:|:-----:|-------|
| Baseline (ls=8, le=14, α=2.0) | 18.10% | — | baseline | POPE warmstart config |
| + `phase=generation` | 20.00% | +1.9pp | ✓ | generation-only better than both |
| + `alpha=8.0, eps=0.5` | 21.90% | +1.9pp | ✓ | higher alpha needed |
| **+ uniform boost (ls=20, le=28)** | **22.86%** | **+0.96pp** | ✓ | **best subsample result** |
| `alpha=12.0` | 20.95% | −1.90pp | ✗ | Flags regression |
| `layer_end=20` | 19.05% | −3.81pp | ✗ | OptIllusion dropped |
| `layer_start=4` | 19.05% | −3.81pp | ✗ | OptIllusion dropped |
| `head_top_k=0.0` (all heads) | 19.05% | −3.81pp | ✗ | OptIllusion dropped |
| `clip_fallback_thresh=0.10` | 20.00% | ±0 | ✗ | no gain over uniform |
| `sys_beta=0.20` | 20.95% | −1.90pp | ✗ | Logos regression |
| `text_beta=0.1` | 20.00% | −1.90pp | ✗ | Flags+Logos regression |
| `text_beta=2.0 all-heads` | 21.90% | ±0 | ✗ | zero effect — language prior is MLP not attention |

---

## Key Findings & Gotchas

1. **CLIP guidance doesn't help VLMBias.** Top-10%, top-80%, and uniform identical.
   The bottleneck is model counting capability, not spatial attention.
   Use `--clip_fallback_thresh 1.0` to force uniform mode.

2. **The old n=105 "best" of 22.86% is not reproducible on full dataset.**
   Full n=2784 gives ~19.76%. The subsample had 9/15 Optical Illusion correct (60%) by luck.
   Always report full-dataset numbers.

3. **`clip_fallback_thresh=1.0` bug**: buggy code (before 2026-06-18) zeroed out the boost
   in the absent fallback path, giving effectively baseline (18.86%). Fixed: absent with
   `neg_absent_alpha=0` now falls back to full uniform alpha.

4. **Deep layers (20–28) vs middle (8–15).** Middle layers (MMVP config) give no gain or
   regression. Counting/reasoning lives in deep transformer layers.

5. **Animals and Chess Pieces are structurally intractable** for Qwen2.5-VL-3B.
   The model consistently predicts 4 legs for any animal regardless of the image.
   SRF cannot fix model capability — these categories won't improve without fine-tuning.

6. **`phase="generation"` is correct (not "both").** Prefill boost doesn't help for
   free-form generation tasks. Opposite of MMVP where prefill boost was critical.

7. **Two separate pipelines needed.** MMVP and VLMBias need different configs —
   do not use a single config for both datasets.

---

## Files

| File | Role |
|------|------|
| `srf/eval.py` | Main eval; `run_vlmbias()`, per-category tracking |
| `srf/srf.py` | SRF prepare_sample; absent fallback fix (2026-06-18) |
| `srf/noun_extract.py` | Noun extraction (mode="vlmbias") |
| `srf/saliency/clip_salience.py` | CLIP saliency (result discarded in uniform mode) |
| `my_analysis/autoresearch_vlmbias/results.tsv` | Full sweep history (n=105) |
| `my_analysis/autoresearch_vlmbias/vlmbias_eval.py` | Old harness (n=105 subsample) |
| `results/srf_bugfix/vlmbias_per_cat.log` | Latest full-dataset run with per-category breakdown |
