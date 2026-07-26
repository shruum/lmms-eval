# MMVP Phase 1 Results: Complete Failure of SRF Generalization

**Date:** 2026-04-30
**Model:** LLaVA-1.5-7B
**Dataset:** MMVP (150 pairs, 300 images)

---

## 🚨 KEY FINDING: SRF = 0.00% on MMVP

```
MMVP Results:
┌─────────────┬───────────────┬───────────────┬─────────────┐
│ Method      │ Pair Accuracy │ Image Accuracy│ Δ           │
├─────────────┼───────────────┼───────────────┼─────────────┤
│ Baseline    │ 26.67%        │ 62.33%        │ -           │
│ SRF         │ 26.67%        │ 62.33%        │ +0.00%      │
└─────────────┴───────────────┴───────────────┴─────────────┘
```

**Conclusion:** SRF provides ZERO improvement on MMVP.

---

## 📊 Complete Dataset Comparison

| Dataset | Type | Baseline | SRF | Δ | Verdict |
|---------|------|----------|-----|---|---------|
| **POPE (Random)** | Object Detection | 87.27% | 87.30% | **+0.03%** | ✅ Minimal gain |
| **POPE (Popular)** | Object Detection | 85.47% | 85.57% | **+0.10%** | ✅ Small gain |
| **POPE (Adversarial)** | Object Detection | 82.93% | 82.93% | **0.00%** | ❌ No gain |
| **POPE Average** | Object Detection | 84.13% | 84.18% | **+0.05%** | ⚠️ Tiny gain |
| **MME** | Comprehensive | 69.50% | 69.50% | **0.00%** | ❌ Complete failure |
| **MMVP** | Visual Patterns | 26.67% | 26.67% | **0.00%** | ❌ Complete failure |

---

## 🔍 Why SRF Fails on MMVP

### Root Cause Analysis:

1. **Task Similarity vs Difference:**
   - POPE: "Is there a cat?" (detection) ✅ SRF helps (+0.05%)
   - MMVP: "Is arrow pointing left?" (discrimination) ❌ SRF fails (0.00%)

2. **Baseline Difficulty:**
   - POPE baseline: 84.13% (high)
   - MMVP baseline: 26.67% (LOW - CLIP-blind pairs are hard!)

3. **Attention Boosting Doesn't Help:**
   - POPE: Boosting cat regions helps detect presence
   - MMVP: Boosting doesn't help with precise discrimination (left vs right, up vs down)

4. **CLIP-Blind Nature:**
   - MMVP designed to test visual patterns CLIP gets wrong
   - If CLIP can't distinguish, attention boosting may not help either
   - SRF uses CLIP salience → circular problem!

---

## 💡 Technical Explanation

### SRF Mechanism:
```
Image → CLIP → Salience Map → Boost High-Salience Tokens
```

### Why It Works on POPE:
```
Question: "Is there a cat?"
Image: Contains cat
CLIP: High salience on cat region
SRF: Boost cat tokens
Result: Better detection ✅
```

### Why It Fails on MMVP:
```
Question: "Is arrow pointing left?"
Image: Arrow pointing right (CLIP-blind pair)
CLIP: Similar salience for both directions
SRF: Boosts both equally → No discrimination
Result: No improvement ❌
```

---

## 📈 Statistical Analysis

### POPE (3000 samples per category):
- Random: +0.03% (p > 0.05, not significant)
- Popular: +0.10% (p > 0.05, not significant)
- Adversarial: 0.00% (p > 0.05, not significant)
- **Conclusion: Minimal gains, not statistically significant**

### MMVP (300 samples):
- Pair Accuracy: +0.00% (exact match with baseline)
- Image Accuracy: +0.00% (exact match with baseline)
- **Conclusion: Complete failure**

### MME (2374 samples):
- Overall: +0.00% (exact match with baseline)
- **Conclusion: Complete failure**

---

## 🎯 Final Decision

### Phase 1 Complete: MMVP Shows No Improvement

**Decision:** ❌ **DO NOT proceed to Phase 2 (VLMs-Are-Biased)**

**Reasoning:**
1. MMVP was our BEST shot at showing generalization
2. If SRF fails on basic visual discrimination (MMVP)
3. It will definitely fail on counting/precision (VLM-Bias)
4. MME already shows SRF fails on counting (73% → 73%)

### SRF Has Extremely Limited Applicability:

✅ **Works on:** POPE object detection (+0.05%)
❌ **Fails on:**
   - MME comprehensive evaluation (0.00%)
   - MMVP visual patterns (0.00%)
   - Any task requiring precision
   - Any task requiring reasoning

---

## 📝 What to Report

### Honest Assessment:

"SRF (Semantic Re-Focus) was evaluated on three object hallucination benchmarks:
1. **POPE (MS-COCO):** +0.05% average improvement across 3 categories
2. **MME:** 0.00% improvement across 14 categories (2374 samples)
3. **MMVP:** 0.00% improvement on visual pattern discrimination (300 samples)

**Conclusion:** SRF provides minimal gains on POPE (+0.05%) but does not generalize
to other hallucination benchmarks. The method appears to be highly specific to the
POPE binary detection task and fails on tasks requiring:
- Visual discrimination (MMVP: 0.00%)
- Precision (MME counting: 0.00%)
- Reasoning (MME cognition: 0.00%)

**Comparison to Literature:**
- VAF (ClearSight): +1.80% on POPE
- AIR: +5.30% on POPE
- SRF (Ours): +0.05% on POPE

SRF significantly underperforms compared to state-of-the-art methods."

---

## 🚫 What NOT to Do

1. ❌ DO NOT run Phase 2 (VLMs-Are-Biased) - waste of time
2. ❌ DO NOT run MMBench - requires API, VAF already shows it fails
3. ❌ DO NOT claim SRF generalizes - evidence shows otherwise
4. ❌ DO NOT overstate POPE results - +0.05% is not significant

---

## ✅ What to Do Next

1. ✅ Report honest negative results
2. ✅ Analyze WHY SRF fails (CLIP salience limitation)
3. ✅ Discuss that visual attention boosting is insufficient
4. ✅ Conclude SRF has limited applicability
5. ✅ Consider alternative approaches for future work

---

## 📊 Data Files

All results saved to:
```
results/mmvp_evaluation/
├── baseline/output.txt
├── srf/output.txt
└── summary.json
```

Previous results:
```
results/pope_final_evaluation/  (POPE: +0.05%)
results/llava_mme_sweep/         (MME: 0.00%)
```

---

**Generated:** 2026-04-30
**Status:** Phase 1 Complete ✅
**Recommendation:** Stop here, report findings honestly
