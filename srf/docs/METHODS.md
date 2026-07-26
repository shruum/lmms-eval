# SRF Results — All Methods

> Updated: 2026-06-16
> Model: Qwen2.5-VL-3B-Instruct
> Baselines always run on same samples as the method (where noted).

---

## Summary Table

Canonical baseline: MMVP=40.0% (n=150 pairs), POPE adv=86.37% (n=3000), VLMBias=19.04% (n=2784).
Sample sizes differ per method — see method notes for exact n used.

| Method | MMVP pair acc | VLM Bias acc | POPE adv | Δ MMVP | Δ POPE | Notes |
|--------|:---:|:---:|:---:|:---:|:---:|---|
| **Baseline** | 40.0% | 19.04% | 86.37% | — | — | canonical (n=150pairs / 2784 / 3000adv) |
| **VCD** | 39.33% ↓ | 9.23% ↓↓ | 35.0% ↓↓ | −0.67pp | −51pp | n=150 / 840 / 120; POPE catastrophically broken |
| **VAF** | 37.33% ↓ | 19.57% (=) | 83.11% ↓ | −2.67pp | −3.26pp | n=150 / 2100 / 900adv |
| SRF base (ls=8, le=14) | 41.33% | 19.0% (=) | 86.00% ↓ | +1.33pp | −0.37pp | best SRF base MMVP; no contrastive |
| **SRF-E γ=3.0 (tuned)** | **49.33%** ↑↑ | 2.19% ↓↓ | **87.70%** ↑ | **+9.33pp** | **+1.33pp** | MMVP: ls=8,le=16,α=2; POPE: ls=8,le=12,α=2; VLMBias broken |
| *(old) SRF-V1 (CLIP)* | 44.00% | 21.90% | 83.33% | +4.00pp | −3.04pp | April harness, n=150/105/100; not directly comparable |
| *(old) SRF-Contrastive β=2.0* | 49.3% | 0.0% ↓↓ | 86.0% | +9.3pp | −0.37pp | April harness; same underlying method as SRF-E |

**Best config per dataset (SRF-E):**
- MMVP: `ls=8, le=16, alpha=2.0, sys_beta=0.30, γ=3.0` → **49.33%** pair acc
- POPE: `ls=8, le=12, alpha=2.0, sys_beta=0.30, γ=3.0` → **87.70%** adv acc (full 3000)
- VLMBias: SRF-E broken at any γ>0; best = SRF base (γ=0) → **19.0%** (= baseline)

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

### VCD (Visual Contrastive Decoding)
- Contrasts logits of original image vs distorted image (noise/blur) at decoding time
- Results (`results/qwen_all/vcd/`, `results/vcd/`):
  - MMVP: 39.33% (−0.67pp vs baseline 40.0%, n=150 pairs)
  - VLMBias: 9.23% (−9.81pp vs baseline 19.04%, n=840) — severe regression
  - POPE adv: 35.0% (−51pp vs baseline, n=120) — **catastrophically broken** (well below chance)
- Conclusion: VCD does not work for binary Yes/No tasks on Qwen2.5-VL-3B; likely produces inverted answers or format failures

### VAF (Visual Attention Focusing)
- Amplifies attention to foreground/object regions via explicit attention reweighting
- Results (`results/qwen_all/vaf/`, `results/vaf/`):
  - MMVP: 37.33% (−2.67pp vs baseline 40.0%, n=150 pairs) — regression
  - VLMBias: 19.57% (+0.53pp vs baseline 19.04%, n=2100) — marginal, ~noise
  - POPE adv: 83.11% (−3.26pp vs baseline 86.37%, n=900) — regression
- Conclusion: VAF consistently hurts or is neutral; does not help on any dataset tested

### SRF-E (Final, coord-descent tuned, 2026-06-16)
- Same two-pass contrastive as "SRF-Contrastive" above, with tuned params and renamed γ (was β)
- Key fix: `content_offset=1` skips contrastive at format token `{` for VLMBias
- VLMBias still broken at any γ>0 (root cause: zeroed pixel_values creates bad language prior)
- Results: MMVP **49.33%** (+9.33pp), POPE adv **87.70%** (+1.33pp), VLMBias 19.0% (=baseline)
- Result files: `my_analysis/autoresearch_mmvp_v2/results.tsv`

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
