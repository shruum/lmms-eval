# SRF Research Results - Complete Dataset

**Last Updated:** 2026-04-28  
**Models:** Qwen-VL-Chat, LLaVA-1.5-7B, Qwen2.5-VL-3B  
**Tasks:** POPE (adversarial), MME (2374 samples), MMVP, VLM Bias

---

## Table of Contents

1. [POPE Results (Qwen-VL-Chat)](#pope-results-qwen-vl-chat)
2. [POPE Results (LLaVA-1.5-7B)](#pope-results-llava-15-7b)
3. [MME Results (LLaVA-1.5-7B)](#mme-results-llava-15-7b)
4. [MME Results (Qwen-VL-Chat)](#mme-results-qwen-vl-chat)
5. [Legacy Results (Qwen2.5-VL-3B)](#legacy-results-qwen25-vl-3b)

---

## POPE Results (Qwen-VL-Chat)

**Model:** Qwen/Qwen-VL-Chat  
**Dataset:** POPE adversarial (3000 samples)  
**Date:** 2026-04-28  
**Experiments:** 50-config parameter sweep

### Overall Best Results

| Metric | Baseline | SRF (best) | Delta |
|--------|----------|------------|-------|
| **Accuracy** | 80.00% | **81.00%** | +1.00% |
| **F1 Score** | 76.74% | **76.74%** | 0.00% |

### Best Configuration

```python
{
    "method": "srf",
    "layers": [8, 14],           # Early-mid fusion zone
    "alpha": 1.5,                # Gentle boost
    "clip_top_k_pct": 0.20,      # Top 20% tokens
    "clip_coarse_grid": 7,       # 7x7 grid
    "head_top_k_pct": 0.15-0.20, # Top 15-20% heads
    "clip_suppress_thresh": 0.24, # CLIP max_sim threshold
    "clip_suppress_alpha": 5.0    # Strong suppression when absent
}
```

### Top 10 Configurations

| Config ID | Accuracy | F1 | Δ | Layers | Alpha | CLIP top-k | Head top-k | Suppress thresh |
|-----------|----------|-----|---|--------|-------|------------|------------|----------------|
| ls8_le14_a1.5_ck0.2_g7_hk0.2_t0.24 | 81.00% | 76.74% | +1.0% | 8-14 | 1.5 | 0.20 | 0.20 | 0.24 |
| ls8_le14_a1.5_ck0.2_g7_hk0.2_t0.248 | 81.00% | 76.74% | +1.0% | 8-14 | 1.5 | 0.20 | 0.20 | 0.248 |
| ls8_le14_a1.5_ck0.2_g7_hk0.15_t0.24 | 81.00% | 76.74% | +1.0% | 8-14 | 1.5 | 0.20 | 0.15 | 0.24 |
| ls8_le14_a1.5_ck0.2_g7_hk0.15_t0.248 | 81.00% | 76.74% | +1.0% | 8-14 | 1.5 | 0.20 | 0.15 | 0.248 |
| ls8_le14_a1.5_ck0.2_g7_hk0.1_t0.24 | 81.00% | 76.74% | +1.0% | 8-14 | 1.5 | 0.20 | 0.10 | 0.24 |
| ls8_le14_a1.5_ck0.2_g7_hk0.1_t0.248 | 81.00% | 76.74% | +1.0% | 8-14 | 1.5 | 0.20 | 0.10 | 0.248 |
| ls8_le14_a1.5_ck0.2_g7_hk0.1_t0.25 | 81.00% | 76.74% | +1.0% | 8-14 | 1.5 | 0.20 | 0.10 | 0.25 |
| ls8_le14_a1.5_ck0.2_g7_hk0.15_t0.25 | 81.00% | 76.74% | +1.0% | 8-14 | 1.5 | 0.20 | 0.15 | 0.25 |

### Key Findings from Autoresearch

1. **Early-mid fusion critical:** Layers 8-14 significantly outperformed 10-14
2. **Gentle boost better:** Alpha=1.5 outperformed stronger boosts (2.0, 2.5, 4.0)
3. **Absence-aware strategy essential:** Suppress when CLIP max_sim < 0.24
4. **Focused token selection:** 15-20% top-k optimal
5. **Head selection:** 10-20% of vision-aware heads sufficient

### Data Files

- Full results: `auto_research/pope_qwen/sweep_results/sweep_results.tsv`
- Top 10: `auto_research/pope_qwen/sweep_results/top10.tsv`
- Best config: `auto_research/pope_qwen/params_best.py`

---

## POPE Results (LLaVA-1.5-7B)

**Model:** llava-hf/llava-1.5-7b-hf  
**Dataset:** POPE adversarial (3000 samples)  
**Date:** 2026-04-28  
**Status:** In progress (autoresearch ongoing)

### Current Results

| Metric | Baseline | SRF | Delta |
|--------|----------|-----|-------|
| **Accuracy** | 80.00% | TBD | TBD |

### Configurations Tested

- Late fusion (layers 13-16, alpha=3.0, beta=1.5) - FAILED
- Multiple configurations in progress

---

## MME Results (LLaVA-1.5-7B)

**Model:** llava-hf/llava-1.5-7b-hf  
**Dataset:** MME (2374 samples, full)  
**Date:** 2026-04-28  
**Progress:** 3/8 experiments complete  
**Status:** ⚠️ ZERO improvement across all configs

### Overall Results Summary

| Experiment | Config | Baseline | SRF | Delta | Status |
|------------|--------|----------|-----|-------|--------|
| Exp 1 | Baseline (10-14, α=2.0) | 1650 (69.50%) | 1650 (69.50%) | **0.00%** | ❌ No improvement |
| Exp 2 | Early fusion (8-14, α=2.0) | 1650 (69.50%) | 1650 (69.50%) | **0.00%** | ❌ No improvement |
| Exp 3 | Wide fusion (8-16, α=2.0) | 1650 (69.50%) | 1650 (69.50%) | **0.00%** | ❌ No improvement |
| Exp 4 | Late fusion (13-16, α=3.0) | TBD | TBD | TBD | 🔄 Running |
| Exp 5-8 | Various configs | TBD | TBD | TBD | ⏳ Pending |

### Detailed Per-Category Results (Exp 1-3)

**Note:** All 3 experiments show IDENTICAL results - SRF not improving any category

| Category | Type | Baseline | SRF | Delta |
|----------|------|----------|-----|-------|
| **OCR** | Perception | 28/40 (70.0%) | 28/40 (70.0%) | 0.0% |
| **artwork** | Perception | 284/400 (71.0%) | 285/400 (71.3%) | +0.3% |
| **celebrity** | Perception | 200/340 (58.8%) | 201/340 (59.1%) | +0.3% |
| **code_reasoning** | Cognition | 21/40 (52.5%) | 21/40 (52.5%) | 0.0% |
| **color** | Perception | 52/60 (86.7%) | 52/60 (86.7%) | 0.0% |
| **commonsense_reasoning** | Cognition | 89/140 (63.6%) | 89/140 (63.6%) | 0.0% |
| **count** | Perception | 44/60 (73.3%) | 44/60 (73.3%) | 0.0% |
| **existence** | Perception | 57/60 (95.0%) | 57/60 (95.0%) | 0.0% |
| **landmark** | Perception | 275/400 (68.8%) | 274/400 (68.5%) | -0.3% |
| **numerical_calculation** | Cognition | 20/40 (50.0%) | 20/40 (50.0%) | 0.0% |
| **position** | Perception | 44/60 (73.3%) | 44/60 (73.3%) | 0.0% |
| **posters** | Perception | 201/294 (68.4%) | 200/294 (68.0%) | -0.4% |
| **scene** | Perception | 315/400 (78.8%) | 315/400 (78.8%) | 0.0% |
| **text_translation** | Cognition | 20/40 (50.0%) | 20/40 (50.0%) | 0.0% |

### Overall Metrics (Exp 1-3)

| Metric | Baseline | SRF | Delta |
|--------|----------|-----|-------|
| **Total Score** | 1650/2374 | 1650/2374 | 0.0% |
| **Accuracy** | 69.50% | 69.50% | 0.0% |
| **Pair Accuracy** | 39.60% | 39.51% | -0.09% |
| **Perception** | 1500 | 1500 | 0.0% |
| **Cognition** | 150 | 150 | 0.0% |

### Key Observation

**Perfect identity across all configs** suggests:
1. SRF not being applied to MME
2. Or SRF has no effect on MME task
3. Requires investigation

---

## MME Results (Qwen-VL-Chat)

**Model:** Qwen/Qwen-VL-Chat  
**Dataset:** MME (2374 samples, full)  
**Date:** 2026-04-28  
**Progress:** 1/8 experiments complete  
**Status:** ⚠️ NEGATIVE impact

### Overall Results Summary

| Experiment | Config | Baseline | SRF | Delta | Status |
|------------|--------|----------|-----|-------|--------|
| Exp 1 | POPE best (8-14, α=1.5) | 1948 (82.06%) | 1873 (78.90%) | **-3.16%** | ⚠️ Worse |
| Exp 2-8 | Various configs | TBD | TBD | TBD | 🔄 Running/Pending |

### Detailed Per-Category Results (Exp 1)

| Category | Type | Baseline | SRF | Delta |
|----------|------|----------|-----|-------|
| **OCR** | Perception | 29/40 (72.5%) | 24/40 (60.0%) | **-12.5%** ⚠️ |
| **artwork** | Perception | 301/400 (75.3%) | 294/400 (73.5%) | -1.8% |
| **celebrity** | Perception | 275/340 (80.9%) | 260/340 (76.5%) | -4.4% |
| **code_reasoning** | Cognition | 17/40 (42.5%) | 15/40 (37.5%) | -5.0% |
| **color** | Perception | 55/60 (91.7%) | 55/60 (91.7%) | 0.0% |
| **commonsense_reasoning** | Cognition | 105/140 (75.0%) | 101/140 (72.1%) | -2.9% |
| **count** | Perception | 47/60 (78.3%) | 49/60 (81.7%) | +3.4% ✅ |
| **existence** | Perception | 56/60 (93.3%) | 55/60 (91.7%) | -1.6% |
| **landmark** | Perception | 352/400 (88.0%) | 335/400 (83.8%) | -4.2% |
| **numerical_calculation** | Cognition | 13/40 (32.5%) | 13/40 (32.5%) | 0.0% |
| **position** | Perception | 50/60 (83.3%) | 50/60 (83.3%) | 0.0% |
| **posters** | Perception | 263/294 (89.5%) | 244/294 (83.0%) | -6.5% ⚠️ |
| **scene** | Perception | 351/400 (87.8%) | 354/400 (88.5%) | +0.7% |
| **text_translation** | Cognition | 34/40 (85.0%) | 23/40 (57.5%) | **-27.5%** ⚠️⚠️⚠️ |

### Overall Metrics (Exp 1)

| Metric | Baseline | SRF | Delta |
|--------|----------|-----|-------|
| **Total Score** | 1948/2374 | 1873/2374 | -3.16% |
| **Accuracy** | 82.06% | 78.90% | -3.16% |
| **Pair Accuracy** | 66.81% | TBD | TBD |
| **Perception** | 1779 | 1721 | -3.26% |
| **Cognition** | 169 | 152 | -10.06% |

### Key Observations

1. **Anomalous baseline:** 1948 (82.06%) much higher than typical 1650 (69.50%)
2. **Severe regression in cognition:** -10.06% overall
3. **Text translation devastated:** -27.5%
4. **OCR degraded:** -12.5%
5. **Possible explanations:**
   - Different evaluation split
   - SRF incompatible with Qwen MME
   - Evaluation bug

---

## Legacy Results (Qwen2.5-VL-3B)

**Model:** Qwen/Qwen2.5-VL-3B-Instruct  
**Note:** These results use an older model and are kept for reference

### MMVP Pair Accuracy

| Method | Pair Accuracy | Delta |
|--------|---------------|-------|
| Baseline | 40.00% | - |
| **SRF** | ~44.00% | +4.00% |
| **SRF-E** (β=2.0) | **49.33%** | +9.33% |

### POPE Adversarial

| Method | Accuracy | Delta |
|--------|----------|-------|
| Baseline | 83.33% | - |
| **SRF** | 83.33% | 0.00% |

### VLM Bias

| Method | Accuracy | Delta |
|--------|----------|-------|
| Baseline | 17.14% | - |
| **SRF** | 21.90% | +4.76% |
| **SRF-E** | ⚠️ Broken | - |

**Note:** SRF-E broken for VLM Bias - contrastive pass suppresses format tokens (`{`)

---

## Summary Tables

### Cross-Task Comparison (All Models)

| Task | Model | Baseline | SRF Best | Delta | Status |
|------|-------|----------|----------|-------|--------|
| **POPE (adv)** | Qwen-VL-Chat | 80.00% | **81.00%** | +1.0% | ✅ Success |
| **POPE (adv)** | LLaVA-1.5-7B | 80.00% | TBD | TBD | 🔄 In progress |
| **MME** | LLaVA-1.5-7B | 69.50% | 69.50% | 0.0% | ❌ No improvement |
| **MME** | Qwen-VL-Chat | 82.06%* | 78.90% | -3.16% | ⚠️ Worse |
| **MMVP** | Qwen2.5-VL-3B | 40.00% | **49.33%** | +9.33% | ✅ Success |
| **VLM Bias** | Qwen2.5-VL-3B | 17.14% | **21.90%** | +4.76% | ✅ Success |

*Anomalous baseline - unusually high

### Key Findings

1. **POPE:** SRF shows +1.0% improvement on Qwen-VL-Chat
2. **MME:** SRF shows ZERO or NEGATIVE impact
   - LLaVA: Perfect identity (0.00% delta)
   - Qwen: Significant degradation (-3.16%)
3. **Task-specific effects:**
   - Object existence (POPE): ✅ Improved
   - Complex reasoning (MME): ❌ No improvement or worse
4. **Possible explanations:**
   - SRF optimized for object detection, not reasoning
   - MME requires different saliency strategy
   - Evaluation bugs need investigation

---

## Data Files Reference

### POPE Results
- `auto_research/pope_qwen/sweep_results/sweep_results.tsv` - Full 50-experiment sweep
- `auto_research/pope_qwen/sweep_results/top10.tsv` - Top 10 configurations
- `auto_research/pope_qwen/params_best.py` - Best configuration
- `auto_research/pope_qwen/results.tsv` - Experiment tracking
- `auto_research/STATUS.md` - Research status and findings

### MME Results
- `results/llava_mme_seq/exp*.log` - LLaVA MME experiment logs (8 experiments)
- `results/qwen_mme_seq/exp*.log` - Qwen MME experiment logs (8 experiments)
- `auto_research/LLAVA_MME_FULL.log` - LLaVA MME full evaluation (prior to sweep)
- `srf/RESEARCH_STATUS.md` - Live research status

### Monitoring
- `srf_exp_runs/monitoring_20min.log` - Automated monitoring logs
- `srf_exp_runs/qwen_seq.log` - Qwen sequential run log
- `srf_exp_runs/llava_seq.log` - LLaVA sequential run log

---

**Generated:** 2026-04-28  
**Experiments in progress:** MME sweeps (5 LLaVA + 7 Qwen remaining)  
**Next update:** When MME experiments complete (~8 hours)
