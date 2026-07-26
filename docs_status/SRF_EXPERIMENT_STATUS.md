# SRF Hyperparameter Sweep - Status Report

**Date**: 2026-05-16
**Status**: 🔄 **DIAGNOSTIC PHASE** - Testing ultra-low alpha
**Baseline**: 79.30% (COCO Adversarial, sampling decoding)

---

## 📊 Results Summary

### Round 1: Standard Range (Configs 0-16) ✅ COMPLETE

**17 configurations tested** on COCO Adversarial (3000 samples each)

| Metric | Value |
|--------|-------|
| **Completed** | 17/17 (100%) |
| **Best Result** | 78.77% (Config 11) |
| **vs Baseline** | **-0.53%** ❌ |
| **Mean** | 77.58% ± 0.57% |
| **Range** | 76.77% - 78.77% |

**Quality Distribution:**
- ≥79.30% (target): **0/17** ❌
- ≥78.0% (good): **3/17**
- ≥76.0% (acceptable): **17/17**
- <76.0% (poor): **0/17**

### Top 5 Configurations:

| Rank | Config | Accuracy | Alpha | Layers | Heads | Eps |
|------|--------|----------|-------|--------|-------|-----|
| 1 | 11 | 78.77% | 1.0 | 10-18 | 50% | 0.1 |
| 2 | 3 | 78.70% | 2.0 | 12-18 | 90% | 0.1 |
| 3 | 6 | 78.43% | 4.0 | 10-15 | 90% | 0.1 |
| 4 | 13 | 77.77% | 2.0 | 5-10 | 50% | 0.1 |
| 5 | 9 | 77.67% | 1.0 | 10-18 | 50% | 0.2 |

---

## 🔬 Trend Analysis

### 1. Alpha (Boosting Strength)

**Finding**: **Lower is better** (opposite of hypothesis!)

| Alpha | Avg Accuracy | Configs |
|-------|--------------|---------|
| **1.0** | **77.83%** | 4 |
| **2.0** | 77.51% | 9 |
| **4.0** | 77.49% | 4 |

**Implication**: Our hypothesis (stronger boosting needed) was **WRONG**. LLaVA responds better to subtle intervention.

### 2. Layer Ranges

**Finding**: Middle-late layers (10-18) perform best

| Layer Range | Avg Accuracy | Configs |
|-------------|--------------|---------|
| **10-18** | **78.10%** | 2 |
| 5-10 | 77.70% | 2 |
| 10-15 | 77.67% | 4 |
| 12-18 | 77.57% | 4 |
| 18-25 | 76.93% | 2 |

**Implication**: Extended middle range (10-18) better than narrow or very late.

### 3. Head Selection

**Finding**: Minimal impact

| Head % | Avg Accuracy | Configs |
|--------|--------------|---------|
| **50%** | **78.07%** | 3 |
| 90% | 77.81% | 5 |
| 70% | 77.29% | 9 |

**Implication**: Aggressive filtering doesn't help; conservative (50%) works best.

---

## 🚨 Critical Issue

**ALL SRF configurations perform WORSE than baseline!**

- **Baseline**: 79.30% (sampling decoding)
- **Best SRF**: 78.77% (-0.53%)
- **Gap**: Need **+0.53%** just to match baseline

**This suggests:**
1. SRF intervention degrades performance for LLaVA-1.5-7B on POPE
2. CLIP-based saliency may not align with LLaVA's attention
3. LLaVA may need fundamentally different approach than Qwen

---

## 🔄 Diagnostic Round (Configs 200-211)

**Hypothesis**: Ultra-low alpha (0.1-0.5, closer to VAF's 0.15) might work better.

**12 configurations** testing:
- **Alpha**: 0.1, 0.2, 0.5 (vs. 1.0-4.0 before)
- **Suppression**: eps=0 (no suppression) vs eps=0.1
- **Layer ranges**: Narrow (12-16, 14-15) vs broad (10-18)

**Expected completion**: ~3 hours (2026-05-16 ~23:00)

---

## 📋 Next Steps

### If Diagnostic Round Succeeds (≥79.30%):
1. Expand ultra-low alpha search (0.05, 0.15, 0.25, 0.75)
2. Test on all 9 splits
3. Validate against VCD/AIR papers

### If Diagnostic Round Fails (<79.30%):
**Option 1**: Try SRF-E (evidence-amplified, two-pass)
- More aggressive intervention
- Contrastive reasoning
- Higher computational cost

**Option 2**: Abandon SRF for LLaVA-1.5-7B
- Document negative results
- Focus on Qwen models (where SRF showed promise)
- Investigate architectural differences

**Option 3**: Alternative saliency approaches
- Attention rollout (internal) instead of CLIP (external)
- Query-specific vs generic saliency
- Multi-scale saliency fusion

---

## 📁 Results Location

```
results/srf_focused_sweep/
├── coco_adversarial_config{0-16}/  ✅ Complete
│   └── pope_coco_adversarial.json
├── coco_adversarial_config{200-211}/ 🔄 Running
│   └── run.log
└── gqa_adversarial_config{100-104}/  ⚠️ Incomplete
```

---

## 📊 Comparison with Papers

| Method | COCO Adv | A-OKVQA Adv | GQA Adv | Source |
|--------|----------|-------------|---------|--------|
| **Baseline** | 79.30% | 75.80% | 75.50% | Our experiments |
| **VCD Paper** | 78.96% | 74.04% | 75.08% | Table 1 |
| **AIR Paper** | 75.00% | 74.00% | 75.10% | Table 2 |
| **SRF Best** | 78.77% | - | - | Config 11 |
| **SRF Goal** | ≥80.30% | ≥76.80% | ≥76.50% | Baseline+1% |

**Current gap**: SRF is **-1.53%** from goal on COCO Adversarial.

---

*Last updated: 2026-05-16 - Diagnostic round running*
*Next check: 2026-05-16 23:00 (after ~3 hours)*
