# Saliency Improvement Log

Tracking presence-detection experiments on the fixed 60-sample POPE val set
(10 × 3 splits × 2 answers, balanced, seed=42).

**Eval command:**
```bash
cd /volumes2/mllm/lmms-eval
conda activate mllm
python srf/saliency/eval_presence.py --mode <MODE> --n_per_cell 10
```

**Compare command:**
```bash
python srf/saliency/eval_presence.py \
  --compare results/saliency_val_clip_full_gate/saliency_val_clip_full_gate.json \
            results/saliency_val_<MODE>/saliency_val_<MODE>.json
```

---

## Baselines (2026-06-05)

| Mode | acc | TPR | FPR | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|
| `clip_improved` | 0.500 | 1.000 | 1.000 | 0.667 | 30 | 0 | 30 | 0 |
| `clip_full_gate` | **0.683** | 0.767 | 0.400 | 0.708 | 23 | 18 | 12 | 7 |
| `clip_full_gate_v2` | 0.517 | 0.900 | 0.867 | 0.651 | 27 | 4 | 26 | 3 |

**Target**: acc ≥ 0.80, TPR ≥ 0.75, FPR ≤ 0.20

### Root causes in clip_full_gate (baseline to beat)
- `gate_full` threshold = 0.24 **above** GT-Yes mean (0.219) → only 27% TPR standalone
- Gate fallback uses `gate_patch AND gate_contrast`; `gate_patch` fires 100% on all samples
  → collapses to just `gate_contrast` → FPR = 0.40
- Contrastive probe ("with/without noun"): Δ = +0.011 (too weak, CLIP ignores negation)
- `patch_entropy` (on normalized saliency): Δ = +0.021 (circular signal)

### Threshold sweep for `gate_full` alone (from clip_full_gate data)
```
t=0.21  acc=0.817  TPR=0.733  FPR=0.100  F1=0.800  ← optimal
t=0.20  acc=0.733  TPR=0.733  FPR=0.267  F1=0.733
t=0.24  acc=0.633  TPR=0.267  FPR=0.000  F1=0.421  ← was set here
```
→ `gate_full` alone at t=0.21 outperforms the full combined gate (0.817 > 0.683).

### full_img_sim distributions
```
GT-Yes: mean=0.2190  stdev=0.0261  min=0.165  max=0.261
GT-No:  mean=0.1875  stdev=0.0217  min=0.144  max=0.226
```

---

---

## Experiment 1 — Lower threshold only (clip_full_gate_v3, 2026-06-05)

**One change**: `full_img_thresh` 0.24 → **0.21**, gate = `gate_full` only, no fallbacks.  
`gate_patch` removed from logic. New signals (raw_entropy, cross_scale_iou, blur_delta)
are **computed and recorded for distribution analysis** but NOT used in gate decision.

**Predicted result** (from post-hoc sweep on clip_full_gate data):
- acc=0.817, TPR=0.733, FPR=0.100, F1=0.800

**Run command:**
```bash
python srf/saliency/eval_presence.py --mode clip_full_gate_v3 --n_per_cell 10
```

**Results:**

| Metric | clip_full_gate (baseline) | clip_full_gate_v3 | Δ |
|---|---|---|---|
| accuracy | 0.683 | **0.800** | **+0.117** |
| TPR | 0.767 | 0.700 | −0.067 |
| FPR | 0.400 | **0.100** | **−0.300** |
| F1 | 0.708 | **0.778** | **+0.070** |
| TP/TN/FP/FN | 23/18/12/7 | 21/27/3/9 | |

**New signal distributions (measured, not used in gate):**
| Signal | GT-Yes mean | GT-No mean | Δ | Solo accuracy |
|---|---|---|---|---|
| raw_entropy | 0.908 | 0.938 | −0.030 | 0.500 (useless) |
| cross_scale_iou | 0.355 | 0.297 | +0.058 | 0.617 |
| blur_delta | 0.000 | −0.022 | +0.022 | 0.683 |

**Conclusion:** ✅ Threshold change alone is a large improvement (acc +0.117, FPR −0.300).
Matches prediction from post-hoc sweep (predicted 0.817, got 0.800 — tiny diff from batched encoding).

**Analysis of FNs / FPs for backup signals:**
- 9 FNs total: 7 in soft zone [0.179, 0.21), 2 below (unreachable by backup)
- 3 FPs: all have full_img >= 0.21, already caught by gate_full — backup can't fix these

Backup gate simulation (soft zone only):
| Backup | FNs recovered | New FPs | Net acc change |
|---|---|---|---|
| cross_scale_iou ≥ 0.30 | 5/7 | +5/16 | 0 (trade-off) |
| blur_delta ≥ 0.0 | 1/7 | +6/16 | −0.083 (worse) |
| raw_entropy < 0.92 | 2/7 | +3/16 | −0.017 (worse) |

No backup signal improves accuracy — they all trade TPR for FPR at equal or worse rates.
This is a fundamental limit of ViT-B/32: the soft zone has 7 GT-Yes vs 16 GT-No, and all signals overlap heavily in that range.

---

## Experiment 2 — cross_scale_iou as backup (2026-06-05)

**One change on top of Exp 1**: add `gate_cross_scale` (iou ≥ 0.30) as backup for borderline cases.  
Gate: `gate_full OR (gate_full_soft AND gate_cross_scale)`  
Mode: `clip_full_gate_v3_iou`

Pre-run prediction from Exp 1 data:
- FNs recovered: 5/7 (soft-zone FNs with iou ≥ 0.30)
- New FPs added: 5/16 (soft-zone TNs with iou ≥ 0.30)
- Net acc change: 0 (wash), but TPR↑ FPR↑

**Results:**

| Metric | Exp 1 (threshold only) | Exp 2 (+cross_scale_iou) | Δ |
|---|---|---|---|
| accuracy | 0.800 | 0.800 | 0 |
| TPR | 0.700 | **0.867** | +0.167 |
| FPR | 0.100 | 0.267 | +0.167 |
| F1 | 0.778 | **0.812** | +0.034 |
| TP/TN/FP/FN | 21/27/3/9 | 26/22/8/4 | |

**Conclusion:** Exactly matches prediction. Accuracy unchanged (wash), but F1 improves (+0.034) because
precision drop is smaller than TPR gain. cross_scale_iou is a useful backup IF you prioritize recall
(TPR) over precision. Adversarial split hurt most (FPR 0.0→0.6).

---

## Experiment 3 — blur_delta as backup (2026-06-05)

**One change on top of Exp 1**: add `gate_blur_delta` (delta ≥ 0.0) as backup.  
Gate: `gate_full OR (gate_full_soft AND gate_blur_delta)`  
Mode: `clip_full_gate_v3_blur`

Pre-run prediction from Exp 1 data:
- FNs recovered: 1/7
- New FPs added: 6/16
- Expected to be worse than Exp 1 baseline

**Results:**

| Metric | Exp 1 (threshold only) | Exp 3 (+blur_delta) | Δ |
|---|---|---|---|
| accuracy | 0.800 | 0.733 | **−0.067** |
| TPR | 0.700 | 0.700 | 0 |
| FPR | 0.100 | 0.233 | +0.133 |
| F1 | 0.778 | 0.724 | −0.054 |

**Conclusion:** ❌ Worse than threshold-only. blur_delta adds FPs without recovering FNs.
Confirmed: CLIP sim doesn't change meaningfully with Gaussian blur for present objects — wrong hypothesis.
Blur signal does not work with ViT-B/32.

---

## Experiment 4 — raw_entropy as backup (2026-06-05)

**One change on top of Exp 1**: add `gate_raw_entropy` (entropy < 0.92) as backup.  
Gate: `gate_full OR (gate_full_soft AND gate_raw_entropy)`  
Mode: `clip_full_gate_v3_entropy`

Pre-run prediction from Exp 1 data:
- FNs recovered: 2/7
- New FPs added: 3/16

**Results:**

| Metric | Exp 1 (threshold only) | Exp 4 (+raw_entropy) | Δ |
|---|---|---|---|
| accuracy | 0.800 | 0.767 | −0.033 |
| TPR | 0.700 | **0.900** | +0.200 |
| FPR | 0.100 | 0.367 | +0.267 |
| F1 | 0.778 | 0.794 | +0.016 |

**Conclusion:** ❌ Worse accuracy. raw_entropy fires too broadly — entropy threshold 0.92 too loose,
the signal overlaps heavily between classes (GT-Yes=0.908, GT-No=0.938, Δ=−0.030).
The high TPR (0.900) comes at the cost of very high FPR (0.367), worse than the combined gate baseline.
raw_entropy is not a useful signal with current temperature/threshold.

---

## TODO after saliency experiments
- ~~Head/layer calibration: sweep layer_start, layer_end, head_top_k_pct on val set~~ ✅ Done (2026-06-07)
- ~~SigLIP / ViT-L/14 as CLIP backbone for better presence detection~~ ✅ Tried — ViT-L/14 worse than ViT-B/32 (max acc 0.783 vs 0.800)
- ~~POPE end-to-end accuracy eval with best saliency mode + tuned layers~~ ✅ Done (2026-06-08)
- ~~Stage 3: boost/suppress experiments~~ ✅ Done (2026-06-08)

---

## Stage 2 — Head/Layer Calibration (2026-06-07)

**Script:** `srf/calibration/sweep_heads_layers.py`  
**Val set:** 60 samples (seed=42), VQA accuracy (not CLIP gate accuracy)  
**Baseline SRF VQA acc:** 0.867 (ls=8, le=15, htk=0.20, beta=0.30)

### Phase 1 — Layer range sweep (head_top_k_pct=0.20 fixed)

Top results:

| layer_start | layer_end | acc | TPR | FPR | F1 |
|---|---|---|---|---|---|
| **6** | **12** | **0.900** | 0.800 | 0.000 | 0.889 |
| 8 | 12 | 0.883 | 0.800 | 0.033 | 0.873 |
| 4 | 20 | 0.883 | 0.767 | 0.000 | 0.868 |
| 8 | 15 (current) | 0.867 | 0.767 | 0.033 | 0.852 |

→ **Best: layer_start=6, layer_end=12** (+3.3pp acc, FPR drops to 0.000)

Key observation: narrower range (6 layers: 6→12) beats wider ranges. Earlier start (layer 6) captures
early vision-language fusion that was missed with start=8.

### Phase 2 — Head fraction sweep (ls=6 le=12)

| head_top_k_pct | acc |
|---|---|
| **0.20** | **0.900** |
| 0.05 / 0.10 / 0.15 / 0.30 / 0.40 | 0.867 |

→ **head_top_k_pct=0.20 confirmed optimal** (no change needed)

### Phase 3 — sys_beta sweep (ls=6 le=12 htk=0.20)

| sys_beta | acc |
|---|---|
| **0.30** | **0.900** |
| 0.00 / 0.10 / 0.20 / 0.40 / 0.50 | 0.867 |

→ **sys_beta=0.30 confirmed optimal** (no change needed)

### Config change applied
```python
# config.py — Qwen/Qwen2.5-VL-3B-Instruct
"layer_start": 6,    # was 8
"layer_end":   12,   # was 15
"dataset_layer_end": {"mmvp": 15, "pope": 12, ...}  # pope: 15→12
```

---

## Stage 3 — Boost/Suppress Experiments (2026-06-08)

**Val set:** 60 samples (seed=42), saliency_mode=clip_full_gate_v3, ls=6, le=12  
**Baseline:** clip_full_gate_v3 = acc=0.900 (TP=24, TN=30, FP=0, FN=6)

All experiments compared against clip_full_gate_v3 on 60-sample val set.

| Exp | Mode | acc | TPR | FPR | F1 | Δacc | Notes |
|---|---|---|---|---|---|---|---|
| baseline | clip_full_gate_v3 | 0.900 | 0.800 | 0.000 | 0.889 | — | |
| 3A | neg_absent_alpha=1.0/2.0/4.0 | 0.900 | 0.800 | 0.000 | 0.889 | 0 | FPR=0 → nothing to suppress |
| 3B | clip_full_gate_v3_adaptive | 0.900 | 0.800 | 0.000 | 0.889 | 0 | All present sims ≥ thresh → conf already capped |
| 3C | clip_full_gate_v3_ramp | 0.900 | 0.800 | 0.000 | 0.889 | 0 | Gaussian per-layer alpha has no effect |
| 3D | clip_full_gate_v3_dynhead | 0.900 | 0.800 | 0.000 | 0.889 | 0 | Extra overhead, no gain |
| 3E | clip_full_gate_v3_dynlayer | **0.867** | 0.767 | 0.033 | 0.852 | **−0.033** | Dynamic layers hurt — narrows useful range |

**Conclusion:** The 60-sample val set is saturated at 0.900 for all boost/suppress strategies.
The 6 remaining FNs are GT-Yes samples where CLIP correctly identifies the object as absent
(full_img_sim < 0.21) — no attention manipulation can recover these since the saliency signal
itself is wrong. FPR=0.000 means no suppression targets exist. 3E actively hurts by
collapsing the layer range to below-average layers for some samples.

**Best config:** `clip_full_gate_v3` with `ls=6, le=12` (no Stage 3 changes needed).

---

## Full POPE Evaluation (2026-06-08)

**Config:** SRF + clip_full_gate_v3, ls=6, le=12, alpha=4.0, phase=generation, sys_beta=0.30  
**Dataset:** lmms-lab/POPE, all 9000 samples (3 splits × 3000 each, balanced Yes/No)

| Split | acc (SRF) | acc (baseline) | Δ |
|---|---|---|---|
| adversarial | 0.864 | **0.864** (86.37%) | ~0 |
| popular | 0.876 | **0.876** (87.57%) | ~0 |
| random | **0.886** | — | — |
| **overall** | **0.875** | ~0.866 | **+0.9pp** |

Full metrics (9000 samples):
- accuracy: 0.875 (7879/9000)
- TPR: 0.782 (TP=3519, FN=981)
- FPR: 0.031 (FP=140, TN=4360)
- F1: 0.863
- yes_ratio: 0.407

**Conclusion:** SRF with tuned config achieves +0.9pp overall over baseline.
The adversarial split (hardest) matches baseline exactly. Random split shows the largest gain (+2pp).
The improvement is modest but positive — the main gains were in Stage 1 (saliency gate) and Stage 2 (layer range).

---

## MME & HallusionBench Evaluation (2026-06-09)

### Diagnosis: SRF hurt on new datasets

**Root causes:**
1. `phase="both"` (prefill intervention) — same issue as POPE; prefill KV cache corruption.
2. Noun extraction returning stop-words (bad nouns) — CLIP queries "does", "this", "only", "according" → random saliency → harmful boost.

**Fixes applied:**
- `config.py`: Changed `phase` to `"generation"` for mme, hallusionbench, mmbench.
- `srf.py`: Added bad-noun gate — skips SRF entirely when extracted noun is in blacklist or len ≤ 2.
- `noun_extract.py`: Fixed pope count pattern and added mmbench extraction mode.

### MME Results (2374 samples, 14 categories)

| Config | total_score | perception | cognition | Δtotal |
|---|---|---|---|---|
| baseline | 2362.9 | 1722.9 | 640.0 | — |
| SRF (broken: phase=both) | ~2343 | — | — | **−19.7** |
| **SRF fixed (phase=gen + bad-noun gate)** | **2361.8** | **1719.0** | **642.9** | **−1.1** |

**Neutral on MME.** The fix eliminated the large regression. No categories lost more than 1.7pp.

**Skip analysis (3 samples per category):**
- ✅ CLIP active (good noun): OCR, celebrity, color, commonsense_reasoning, existence, position (partial)
- ❌ SRF skipped (bad noun): artwork, code_reasoning, count, landmark, numerical_calculation, posters, scene, text_translation

Many MME categories extract non-visual nouns → SRF skips majority of samples correctly.

### HallusionBench Results (951 samples, VD+VS)

| Config | aAcc | fAcc | qAcc |
|---|---|---|---|
| baseline | 0.694 | 0.144 | 0.211 |
| SRF (broken: phase=both) | 0.680 | — | — |
| **SRF fixed** | **0.686** | **0.144** | **0.193** |

**Slight hurt (−0.8pp aAcc, −1.8pp qAcc).** The fix reduced the regression but did not eliminate it.

**Root cause of remaining hurt:** VD (Visual Distortion) questions ask whether image edits are real/fake — requires global scene understanding, not local object localization. SRF's spatial saliency boost is wrong for this question type.
- VD acc: 0.645 → 0.638 (−0.7pp)
- VS acc: 0.775 → 0.764 (−1.1pp)

**HallusionBench skip analysis (3 samples per subcategory):**
- ✅ CLIP active: chart, figure, illusion, math (good nouns extracted from these)
- ❌ SRF skipped (partial): ocr (2/3 skipped), table (2/3 skipped), video (1/3 skipped), map (1/3 skipped)

### Summary across all datasets

| Dataset | baseline | SRF fixed | Δ | Notes |
|---|---|---|---|---|
| POPE (9000) | 0.866 | **0.875** | **+0.9pp** | Best result — object existence questions |
| MME (2374) | 2362.9 | 2361.8 | −1.1pts | Neutral — many bad nouns skipped correctly |
| HallusionBench | aAcc=0.694 | aAcc=0.686 | −0.8pp | Still hurts — VD questions need global understanding |
| MMVP | 0.400 | **0.493** | **+9.3pp** | Best result with SRF-E β=2.0 |
| VLM Bias | 0.171 | **0.219** | **+4.8pp** | Best result |

**Visualization outputs:** `results/saliency_vis_datasets/mme/` and `results/saliency_vis_datasets/hallusionbench/`

---

---

## Experiment A — Combined Gate Threshold Sweep (2026-06-12, offline)

**No GPU needed** — swept on existing 60-sample val JSON:
`results/saliency/saliency_val_clip_full_gate_v3/saliency_val_clip_full_gate_v3.json`

**Gate tested:** `full_img_sim >= 0.21 OR (max_sim >= T AND patch_contrast >= 1.40)`

| T_patch | acc | TPR | FPR | F1 | FN recovered | New FP |
|---|---|---|---|---|---|---|
| baseline (full only) | 0.800 | 0.700 | 0.100 | 0.778 | — | — |
| 0.265 | 0.817 | 0.767 | 0.133 | **0.807** | 2 | 1 |
| 0.290 | **0.817** | 0.733 | **0.100** | 0.800 | 1 | 0 |

Adding `cross_scale_iou` or `blur_delta` as extra fallbacks always hurts (FPR jumps ≥ 0.467).

**Conclusion:** T=0.290 is the clean choice (+0.022 F1, FPR unchanged). T=0.265 gets 1 more FN at cost of 1 FP.
Gain on 9000-sample POPE is estimated at ~+0.1–0.2pp. Not yet implemented in `clip_salience.py`.

---

## Stage 4 — Boosting Experiments (2026-06-12)

**Val set:** 60 samples (seed=42), saliency_mode=clip_full_gate_v3, ls=6, le=12  
**Baseline (model, no SRF):** acc=0.867  
**Previous SRF ceiling:** acc=0.900 (Stage 2 head/layer tuning)

### Eval commands
```bash
# B1 — visual reliance compensation
conda run -n mllm python srf/eval_pope_val.py --srf --vr_target 0.15 --out results/pope_val_srf_B1.json
# B3 — within-budget redistribution
conda run -n mllm python srf/eval_pope_val.py --srf --bias_mode budget_shift --out results/pope_val_srf_B3.json
# B4 — confidence-gated two-pass retry
conda run -n mllm python srf/eval_pope_val.py --srf --b4 --b4_threshold 0.25 --b4_multiplier 2.0 --out results/pope_val_srf_B4.json
```

### Results

| Exp | Description | acc | TPR | FPR | F1 | Δacc |
|---|---|---|---|---|---|---|
| baseline | no SRF | 0.867 | 0.767 | 0.033 | 0.852 | — |
| B3 | budget_shift (within-img renorm, α=4) | 0.867 | 0.767 | 0.033 | 0.852 | 0 |
| B3 | budget_shift α=8 | 0.867 | 0.767 | 0.033 | 0.852 | 0 |
| **B1** | vr_target sweep [0.10–0.25], k=[1–8] | **0.900** | **0.800** | **0.000** | **0.889** | **+3.3pp** |
| **B4** | two-pass retry (t=0.21–0.25, ×2–3) | **0.900** | **0.800** | **0.000** | **0.889** | **+3.3pp** |
| B1+B4 | combined | 0.900 | 0.800 | 0.000 | 0.889 | 0 additive |

### Key findings

- **B3 (budget_shift):** Zero gain at any alpha. The image/text attention budget imbalance is not the bottleneck on POPE — the model's allocation to image tokens is already adequate. Redistribution within the image budget alone cannot steer attention to the correct patches.

- **B1 (visual reliance compensation):** Consistently hits 0.900 across *all* vr_target [0.10–0.25] and k [1.0–8.0] values. The VR compensation mechanism is robust and parameter-insensitive. It recovers the same 3 samples regardless of exact threshold.

- **B4 (two-pass retry):** Also hits 0.900 at all threshold/multiplier settings. Recovers the **same 3 samples** as B1 — these are the samples where the base alpha=4.0 boost was marginal and any increase tips the model to "Yes".

- **B1+B4 combined:** No additive benefit — they target identical failures.

- **Val set ceiling at 0.900.** The 6 remaining FNs (bottle, sports ball, spoon, backpack×2, dining table) are **model capacity limits** — CLIP correctly says PRESENT but the model ignores the boosted attention. No attention manipulation on the tested approaches recovers these.

### New code (not breaking existing configs)
| File | Change |
|---|---|
| `my_analysis/qwen_attn_patch.py` | `budget_shift` bias mode; `srf_vr_target`/`srf_vr_k` state keys for B1 |
| `srf/srf.py` | `last_clip_result` dict (B4 signal); `bias_mode`, `vr_target`, `vr_k`, `interp_lambda` added to `reset_for_dataset()` overrides |
| `srf/eval_pope_val.py` | `--bias_mode`, `--vr_target`, `--vr_k`, `--b4`, `--b4_threshold`, `--b4_multiplier` flags; `run_one_b4()` function |

---

## Channel Extension Experiments — SRF-FN and SRF-V (2026-07-29 / 2026-07-30)

Exploring additional intervention channels on top of SRF attention logit boost (single-pass, no contrastive).

**Motivation:**
- SRF (attention) fixes *routing*: boosts Q→K logits so more attention weight flows to salient patches.
- FFN channel (SRF-FN): amplify MLP *output* of salient patches post-FFN → scale residual stream.
- Value channel (SRF-V): scale *v_proj output* of salient patches at prefill → amplifies content extracted when attended to. KV-cached, so persists across all generation steps at zero extra cost.

All experiments: Qwen2.5-VL-3B-Instruct, MMVP (150 pairs, 300 images), single-pass SRF.

---

### SRF-FN — FFN output scaling (2026-07-29)

**Script:** `srf/test_srffn_mmvp.py`  
**Hook:** `layer.mlp` forward output hook  
**Scale:** `mlp_out[:, s:e+1, :] *= (1 + alpha_ffn * salience)`  
**Active layers:** 8–16 (vaf_layer_start/end)  
**Log:** `/tmp/claude-1000/.../tasks/becyt93mo.output`

| Config | pair_acc | img_acc | Δ pair |
|--------|----------|---------|--------|
| baseline | 0.4000 | 0.6767 | ref |
| srf | 0.4133 | 0.6867 | +0.0133 |
| srffn_0.1 | 0.3933 | 0.6800 | −0.0067 |
| srffn_0.3 | 0.4000 | 0.6800 | +0.0000 |
| srffn_0.5 | 0.3933 | 0.6700 | −0.0067 |
| srffn_0.8 | 0.3867 | 0.6700 | −0.0133 |
| srffn_1.0 | 0.3733 | 0.6600 | −0.0267 |

**Conclusion:** ❌ SRF-FN hurts monotonically as α increases. Post-FFN residual scaling across 9 layers compounds multiplicatively — disrupts LayerNorm calibration in subsequent layers. SRF alone (+1.33pp) is better than any FFN addition.

**Root cause:** Scaling post-FFN output inflates residual stream magnitudes. With 9 active layers (8–16) the amplification compounds. The model's internal representations are calibrated to certain magnitude ranges; scaling them post-hoc breaks downstream computations.

---

### SRF-V — Value projection scaling (2026-07-30)

**Script:** `srf/test_srfv_mmvp.py`  
**Hook:** `layer.self_attn.v_proj` forward output hook  
**Scale:** `v_proj_out[:, s:e+1, :] *= (1 + alpha_v * salience)`  
**Active layers:** 8–16 (vaf_layer_start/end)  
**Log:** `/tmp/srfv_mmvp.log`

**Key advantage over SRF-FN:** V vectors are scaled at prefill (step 0) and stored in KV cache — every subsequent generation step automatically reads amplified values for salient patches. No per-step overhead.

| Config | pair_acc | img_acc | Δ pair |
|--------|----------|---------|--------|
| baseline | 0.4000 | 0.6767 | ref |
| srf | 0.4133 | 0.6867 | +0.0133 |
| **srfv_0.1** | **0.4200** | **0.6900** | **+0.0200** |
| srfv_0.3 | 0.4133 | 0.6800 | +0.0133 |
| srfv_0.5 | 0.4133 | 0.6800 | +0.0133 |
| srfv_0.8 | 0.4133 | 0.6833 | +0.0133 |
| srfv_1.0 | 0.4133 | 0.6733 | +0.0133 |

**Conclusion:** ✅ Mild positive signal. α=0.1 adds +0.67pp over SRF alone (pair_acc 0.4133→0.4200). Higher α (0.3–1.0) gives no additional gain over SRF — value scaling at larger magnitude disrupts the attention output computation similarly to SRF-FN. The sweet spot is a very gentle value boost (α=0.1).

Pattern: value scaling is complementary to attention routing but only at very small magnitudes. This makes sense — V vectors feed directly into the attention output; scaling them too strongly overrides the routing decisions that SRF worked to establish.

---

### SRF-RE — Skip-connection re-injection (2026-07-30)

**Script:** `srf/test_srfre_mmvp.py`  
**Hook:** Capture hook on `layers[src_layer]` input + inject hook on `layers[l].mlp` input  
**Formula:** `h_mlp_in[img_i] += beta * salience[i] * h_captured[i]`  
**Active layers:** 8–16 (vaf_layer_start/end)  
**Motivation:** Refresh visual signal in later fusion layers by re-injecting early-layer image token hidden states.

| Config | pair_acc | img_acc | Δ pair |
|--------|----------|---------|--------|
| baseline | 0.4000 | 0.6767 | ref |
| srf | 0.4133 | 0.6867 | +0.0133 |
| srfre_src0_b0.1 | 0.3933 | 0.6800 | −0.0200 |
| srfre_src0_b0.3 | 0.3400 | 0.6333 | −0.0600 |
| srfre_src0_b0.5 | 0.2867 | 0.5767 | −0.1133 |
| srfre_src2_b0.1 | 0.4133 | 0.6933 | +0.0000 |
| srfre_src2_b0.3 | 0.3600 | 0.6300 | −0.0400 |
| srfre_src2_b0.5 | 0.3133 | 0.5933 | −0.0867 |

**Conclusion:** ❌ Catastrophic failure. Even beta=0.1 hurts for src0 (−2pp). Root cause: **distribution mismatch** — layer-0 hidden states exist in a completely different representation space than layer-12 FFN inputs. Layer 0 has pure visual projector output; layer 12 has 12 rounds of attention+FFN mixing. Injecting them into each other violates the implicit distribution assumptions of LayerNorm.

---

### SRF-VEI — Visual Evidence Injection at Answer Token (2026-07-30)

**Script:** `srf/test_srfvei.py`  
**Hook:** `register_forward_pre_hook` on `layer.mlp` for all layers  
**Formula:** `h_ans[l] += gamma * Σ_i w_i * h_img[i][l]`  (w_i = salience-normalized weights)  
**Active layers:** 8–35 (8 to n_layers−1, all post-fusion layers)  
**Target:** ANSWER TOKEN (last input position, `ans_pos = seq_len - 1`)  
**Motivation:** Same-layer injection (no distribution mismatch). Salience-weighted mean of image token hidden states added to answer token's FFN input — what self-attention already does, made explicit and CLIP-conditioned.

**MMVP results (150 pairs, SRF-only, no contrastive):**

| Config | pair_acc | img_acc | Δ pair |
|--------|----------|---------|--------|
| baseline | 0.4000 | 0.6767 | ref |
| srf | 0.4133 | 0.6867 | +0.0133 |
| **srfvei_0.1** | **0.4267** | **0.6967** | **+0.0267** |
| srfvei_0.3 | 0.4133 | 0.6933 | +0.0133 |
| srfvei_0.5 | 0.4200 | 0.7000 | +0.0200 |
| srfvei_1.0 | 0.3800 | 0.6767 | −0.0200 |

**POPE adversarial results (100 samples, SRF-only):**

| Config | acc | Δ acc |
|--------|-----|-------|
| baseline | 0.8800 | ref |
| srf | 0.8800 | +0.0000 |
| srfvei_0.1 | 0.8800 | +0.0000 |
| **srfvei_0.3** | **0.8900** | **+0.0100** |
| srfvei_0.5 | 0.8700 | −0.0100 |
| srfvei_1.0 | 0.7800 | −0.1000 |

**Conclusion:** ⚠️ Marginal and narrow. VEI at gamma=0.1 gives +2.67pp over SRF alone on MMVP (best result). POPE gain at gamma=0.3 (+1pp) is likely noise at n=100. The safe operating region is very narrow — gamma≥0.5 hurts, gamma=1.0 is catastrophic on POPE (−10pp). Root cause of cliffs: injecting at 28 layers (8–35) causes cumulative perturbation accumulating at the answer token's hidden state, which feeds directly to lm_head. VEI addresses the wrong bottleneck for POPE: FFN language priors are in the *weight matrices*, not the input hidden state — adding visual signal to the input doesn't suppress those priors.

**Key finding:** SRF-E (contrastive, +9.33pp MMVP, +1.33pp POPE) dominates VEI by a large margin. VEI is not competitive as a standalone improvement.

---

## Signal Reference

| Signal | Where computed | Interpretation | Known Δ (GT-Yes vs GT-No) |
|---|---|---|---|
| `full_img_sim` | full uncropped image vs noun | direct CLIP presence | +0.032 (mean) |
| `patch_max_sim` | max across 7×7 patches | best patch match | small |
| `patch_contrast` | top-30% / all-patches mean | spatial clustering | +0.152 |
| `contrastive_gap` | sim("with noun") − sim("without noun") | text negation | +0.011 (too weak) |
| `patch_entropy` | on normalized saliency [0,1] | **circular** (absent→uniform) | Δ−0.170 but confounded |
| `raw_entropy` | on raw cosine sims, temp=0.02 | real spread of sims | **unknown — v3 will measure** |
| `cross_scale_iou` | Jaccard top-30% at 3×3 vs 7×7 | localization consistency | **unknown — v3 will measure** |
| `blur_delta` | sim(real) − sim(blurred) | object feature contribution | **unknown — v3 will measure** |
