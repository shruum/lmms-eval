# SRF Results — All Methods

> Updated: 2026-04-25
> Model: Qwen2.5-VL-3B-Instruct
> Baselines always run on same samples as the method (where noted).

---

## Summary Table

| Method | MMVP pair acc | VLM Bias acc | POPE (adv) | Notes |
|--------|:---:|:---:|:---:|---|
| **Baseline** | 39.33% | 17.14% | 83.33% | n=150pairs / 105 / 100 |
| **SRF-V1 (CLIP)** | **44.00%** | **21.90%** | 83.33% | autoresearch best; same baseline |
| SRF-V2 (V-Amp + Drift-α) | ~38.7% ↓ | ~18.1% ↓ | ~88% (=) | V2 harness baseline=88.7% n=150; regression on all |
| SRF-Contrastive β=0.5 | 46.7% | 3.8% ↓↓ | 87.0% | baseline in this harness differs; see notes |
| SRF-Contrastive β=1.0 | 48.0% | 1.9% ↓↓ | 86.0% | |
| **SRF-Contrastive β=2.0** | **49.3%** ↑↑ | 0.0% ↓↓ | 86.0% | MMVP: +5.3pp vs V1; VLM Bias: broken |

---

## Method Details

### Baseline
- Method: `method="baseline"` — no attention intervention
- MMVP: 39.33% pair acc (n=150 pairs, 300 images, all)
- VLM Bias: 17.14% (n=105, 15/cat × 7 cats, seed=42)
- POPE: 83.33% (n=100 adversarial, seed=42)

### SRF-V1 (CLIP) — current best
- Method: CLIP-guided attention logit boosting in vision-aware heads (layers 8–15)
- Params: `boost_alpha=4.0, background_eps=0.2, head_top_k_pct=20%`
- VLM Bias uses `phase=generation, alpha=8.0, layer_end=14, eps=0.5`
- MMVP: **44.00%** pair acc (+4.67pp vs baseline) — autoresearch best after noun_extract.py refactor
- VLM Bias: **21.90%** (+4.76pp) — autoresearch best
- POPE: 83.33% (no gain — prior-dominated, SRF can't overcome language prior)
- Baselines confirmed on same sample sets

### SRF-V2 (V-Amp + Drift-α) — failed
- Idea 1: Value-vector amplification — scale v_proj of salient tokens by (1 + β·sal)
- Idea 2: Drift-adaptive alpha — budget-neutral per-layer α redistribution
- Params: `value_beta=0.5, drift_scale=0.5`
- Result: **regression on all datasets**
  - MMVP: ~38.7% (−5.3pp vs V1)
  - VLM Bias: ~18.1% (−3.8pp vs V1)
  - POPE: V2 full=88.00% vs V2 baseline=88.67% (−0.67pp); V-Amp and Drift-α both zero-effect individually
- Root cause: V-Amp can't reach KV-cached image values at generation (q_len=1 guard); Drift-α redistributes budget but doesn't help over V1's flat optimum
- Scripts: `srf_v2.py`, `eval_srf_v2.py`, `verify_pope_v2.py`

### SRF-Contrastive (Idea 4) — run 2026-04-25
- Two forward passes: Pass 1 (SRF + full image) vs Pass 2 (baseline + zeroed pixel_values)
- `logits_final = logits_full + β·(logits_full - logits_noval)`
- POPE/MMVP: first-token contrastive (single forward pass, no KV cache)
- VLM Bias: step-by-step contrastive generation (two synchronized KV caches)
- Beta sweep: β ∈ {0.5, 1.0, 2.0}
- Scripts: `srf_contrastive.py`, `eval_contrastive.py`

**MMVP — big win:** 40.0% baseline → **49.3%** at β=2.0 (+9.3pp vs baseline, +5.3pp vs V1)
- Best MMVP result so far
- β=1.0: 48.0%; β=0.5: 46.7% — monotonically better with higher β

**VLM Bias — catastrophic failure:** 19.0% baseline → 3.8% at β=0.5, 0% at β=2.0
- Root cause: step-by-step contrastive destroys answer formatting
- The model generates answers in `{answer}` format; contrastive suppresses format tokens
  (strong prior for `{` in logits_noval → subtracting it breaks the template)
- Only Flags and Optical Illusion (non-formatting categories) score any correct answers
- Fix needed: apply contrastive only at the actual content token, not every step

**POPE — slight loss:** 88.0% baseline (this harness) → 87.0% at β=0.5
- ⚠️ Harness note: baseline here is 88.0%, not 83.3% from V1 reference
  → first-token forward pass gives different baseline than model.generate() in V1 eval
  → POPE comparisons cross-harness are unreliable; use within-harness deltas only
- Contrastive is -1pp vs this harness's baseline → marginal but unhelpful

---

## Ideas Not Yet Tried

| Idea | Description | Expected fit |
|------|-------------|--------------|
| Idea 3 (Q-steering + CLIP) | Steer query vector toward CLIP-salient key centroid at generation | POPE/VLM Bias (acts at q_len=1); blocked by same issue as SRF logit boost |
| CLIP-conditioned contrast | Idea 4 variant: zero only CLIP-salient tokens in Pass 2 (keep background) | More targeted; needs pixel-level CLIP→ViT patch mapping |
| VISTA early excitation | Hook activation norms at intermediate layers to find query-relevant tokens | Single-pass, grounded in ICML 2025 |
| CRG contrastive region | Mask most-relevant region, contrast output distributions | Needs bounding-box-level masking |

---

## Experimental Notes

- **n=100 vs n=150 POPE**: SRF-V1 baseline with same harness gives 83.3% at n=100 seed=42, but 88.67% at n=150 seed=42 — different sample distributions. Always compare within the same n/seed.
- **MMVP pair vs image accuracy**: Pair accuracy (both images in pair correct) is the primary metric; image accuracy ~0.71 for V1.
- **VLM Bias metric**: exact-match accuracy after `normalise(extract_answer(raw))` — `{answer}` format or first word.
- **Budget-neutral drift**: V2 Drift-α normalises so mean α = base_alpha. Original unnormalized formula inflated all layer alphas → catastrophic regression on MMVP.

---

## Run Commands

```bash
cd /volumes2/mllm/lmms-eval

# SRF-Contrastive (Idea 4) — all three datasets, beta sweep
conda run -n mllm python my_analysis/eval_contrastive.py 2>&1 | tee my_analysis/eval_contrastive.log

# SRF-V2 verification (ablation: which component helped POPE?)
conda run -n mllm python my_analysis/verify_pope_v2.py 2>&1 | tee my_analysis/verify_pope_v2.log

# SRF-V2 full eval (MMVP + VLM Bias + POPE)
conda run -n mllm python my_analysis/eval_srf_v2.py 2>&1 | tee my_analysis/eval_srf_v2.log
```
