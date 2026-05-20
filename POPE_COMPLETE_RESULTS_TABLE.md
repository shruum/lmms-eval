# Complete POPE Results: Baseline vs SRF (All Datasets & Splits)

**Date:** 2026-05-15
**Model:** LLaVA-1.5-7B
**Decoding:** Sampling (do_sample=True, temp=0.7, top_p=0.9) - **VCD paper method**
**SRF Parameters:** α=0.15, layers 10-15, 50% heads (VAF-like)

---

## 📊 Complete Results Table

| Dataset | Split | Method | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN | Delta |
|---------|-------|--------|----------|-----------|--------|-----|----|----|----|----|-------|
| **COCO** | **Adversarial** | **Baseline** | **77.90%** | **73.24%** | **87.93%** | **79.92%** | 1319 | 1018 | 482 | 181 | |
| | | **SRF** | **76.90%** | **72.11%** | **87.73%** | **79.16%** | 1316 | 991 | 509 | 184 | **-1.00%** ❌ |
| **COCO** | **Popular** | **Baseline** | **83.73%** | **81.04%** | **88.07%** | **84.41%** | 1321 | 1191 | 309 | 179 | |
| | | **SRF** | **84.03%** | **81.30%** | **88.40%** | **84.70%** | 1326 | 1195 | 305 | 174 | **+0.30%** ✅ |
| **COCO** | **Random** | **Baseline** | **85.57%** | **84.40%** | **87.27%** | **85.81%** | 1309 | 1258 | 242 | 191 | |
| | | **SRF** | **86.53%** | **85.26%** | **88.33%** | **86.77%** | 1325 | 1271 | 229 | 175 | **+0.97%** ✅ |
| **A-OKVQA** | **Adversarial** | **Baseline** | **69.47%** | **63.15%** | **93.47%** | **75.38%** | 1402 | 682 | 818 | 98 | |
| | | **SRF** | **69.13%** | **62.76%** | **94.13%** | **75.31%** | 1412 | 662 | 838 | 88 | **-0.33%** ❌ |
| **A-OKVQA** | **Popular** | **Baseline** | **77.97%** | **71.41%** | **93.27%** | **80.89%** | 1399 | 940 | 560 | 101 | |
| | | **SRF** | **78.17%** | **71.35%** | **94.13%** | **81.17%** | 1412 | 933 | 567 | 88 | **+0.20%** ✅ |
| **A-OKVQA** | **Random** | **Baseline** | **83.97%** | **78.35%** | **93.87%** | **85.41%** | 1408 | 1111 | 389 | 92 | |
| | | **SRF** | **82.57%** | **77.15%** | **92.53%** | **84.15%** | 1388 | 1089 | 411 | 112 | **-1.40%** ❌ |
| **GQA** | **Adversarial** | **Baseline** | **68.17%** | **62.03%** | **93.67%** | **74.63%** | 1405 | 640 | 860 | 95 | |
| | | **SRF** | **68.73%** | **62.47%** | **93.87%** | **75.01%** | 1408 | 654 | 846 | 92 | **+0.57%** ✅ |
| **GQA** | **Popular** | **Baseline** | **72.93%** | **66.15%** | **93.93%** | **77.63%** | 1409 | 779 | 721 | 91 | |
| | | **SRF** | **72.80%** | **65.97%** | **94.20%** | **77.59%** | 1413 | 771 | 729 | 87 | **-0.13%** ❌ |
| **GQA** | **Random** | **Baseline** | **83.70%** | **77.43%** | **95.13%** | **85.37%** | 1427 | 1084 | 416 | 73 | |
| | | **SRF** | **82.87%** | **76.74%** | **94.33%** | **84.63%** | 1415 | 1071 | 429 | 85 | **-0.83%** ❌ |

---

## 📈 Summary Statistics

### **Overall Performance:**
| Metric | Baseline | SRF | Difference |
|--------|----------|-----|------------|
| **Mean Accuracy** | 78.16% | 77.97% | **-0.19%** |
| **Std Dev** | 6.64% | 6.58% | - |
| **Min Accuracy** | 68.17% | 68.73% | +0.56% |
| **Max Accuracy** | 85.57% | 86.53% | +0.96% |

### **Delta Distribution:**
| Statistic | Value |
|-----------|-------|
| **Mean Delta** | **-0.19%** (SRF worse) |
| **Std Dev** | 0.78% |
| **Min Delta** | -1.40% (A-OKVQA Random) |
| **Max Delta** | +0.97% (COCO Random) |
| **Improved Splits** | 4/9 (44%) |
| **Degraded Splits** | 5/9 (56%) |

---

## 🎯 Key Findings by Dataset

### **COCO (Average: +0.09% improvement)**
| Split | Baseline | SRF | Delta | Verdict |
|-------|----------|-----|-------|---------|
| Adversarial | 77.90% | 76.90% | -1.00% | ❌ Worse |
| Popular | 83.73% | 84.03% | +0.30% | ✅ Better |
| Random | 85.57% | 86.53% | +0.97% | ✅ Best |

**COCO Summary:** 2/3 splits improved, best result on Random (+0.97%)

### **A-OKVQA (Average: -0.51% degradation)**
| Split | Baseline | SRF | Delta | Verdict |
|-------|----------|-----|-------|---------|
| Adversarial | 69.47% | 69.13% | -0.33% | ❌ Worse |
| Popular | 77.97% | 78.17% | +0.20% | ✅ Better |
| Random | 83.97% | 82.57% | -1.40% | ❌ Worst |

**A-OKVQA Summary:** 1/3 splits improved, worst result on Random (-1.40%)

### **GQA (Average: -0.13% degradation)**
| Split | Baseline | SRF | Delta | Verdict |
|-------|----------|-----|-------|---------|
| Adversarial | 68.17% | 68.73% | +0.57% | ✅ Best |
| Popular | 72.93% | 72.80% | -0.13% | ❌ Worse |
| Random | 83.70% | 82.87% | -0.83% | ❌ Worse |

**GQA Summary:** 1/3 splits improved, best result on Adversarial (+0.57%)

---

## 🏆 Best & Worst Results

### **Best Improvements:**
1. **+0.97%** - COCO Random ✅
2. **+0.57%** - GQA Adversarial ✅
3. **+0.30%** - COCO Popular ✅

### **Worst Degradations:**
1. **-1.40%** - A-OKVQA Random ❌
2. **-1.00%** - COCO Adversarial ❌
3. **-0.83%** - GQA Random ❌

---

## 📊 Confusion Matrices (Selected Splits)

### **COCO Random (Best: +0.97%)**
**Baseline:**
| | Pred Yes | Pred No |
|----------|----------|---------|
| **Actual Yes** | 1309 (TP) | 191 (FN) |
| **Actual No** | 242 (FP) | 1258 (TN) |

**SRF:**
| | Pred Yes | Pred No |
|----------|----------|---------|
| **Actual Yes** | 1325 (TP) | 175 (FN) |
| **Actual No** | 229 (FP) | 1271 (TN) |

**Improvement:** +16 TP, -13 FN, -13 FP, +13 TN

### **A-OKVQA Random (Worst: -1.40%)**
**Baseline:**
| | Pred Yes | Pred No |
|----------|----------|---------|
| **Actual Yes** | 1408 (TP) | 92 (FN) |
| **Actual No** | 389 (FP) | 1111 (TN) |

**SRF:**
| | Pred Yes | Pred No |
|----------|----------|---------|
| **Actual Yes** | 1388 (TP) | 112 (FN) |
| **Actual No** | 411 (FP) | 1089 (TN) |

**Degradation:** -20 TP, +20 FN, +22 FP, -22 TN

---

## 🔍 Comparison with VCD Paper

### **Baseline Accuracy Comparison:**

| Dataset | Split | Our Baseline | VCD Paper | Difference |
|---------|-------|--------------|-----------|------------|
| **COCO** | Adversarial | 77.90% | 78.96% | -1.06% |
| | Popular | 83.73% | 81.88% | +1.85% |
| | Random | 85.57% | 83.29% | +2.28% |
| **A-OKVQA** | Adversarial | 69.47% | 74.04% | -4.57% |
| | Popular | 77.97% | 79.90% | -1.93% |
| | Random | 83.97% | 83.45% | +0.52% |
| **GQA** | Adversarial | 68.17% | 75.08% | **-6.91%** |
| | Popular | 72.93% | 78.17% | -5.24% |
| | Random | 83.70% | 83.73% | -0.03% |

**Average difference:** -1.89% (our baselines are slightly lower)

---

## 💡 Conclusions

1. **SRF is ineffective for LLaVA-1.5-7B on POPE**
   - Average delta: -0.19% (slightly worse than baseline)
   - Only 4/9 splits show improvement
   - Best improvement: +0.97% (COCO Random)
   - Worst degradation: -1.40% (A-OKVQA Random)

2. **VAF-like parameters don't work for LLaVA**
   - α=0.15, layers 10-15, 50% heads (from ClearSight paper)
   - These parameters worked for VAF but not for SRF on LLaVA
   - Need parameter tuning specific to LLaVA architecture

3. **Dataset difficulty varies**
   - **Easiest:** Random splits (83-86% accuracy)
   - **Hardest:** Adversarial splits (68-78% accuracy)
   - **Baseline gap:** GQA/A-OKVQA adversarial still 4-7% below VCD paper

4. **Recommendations:**
   - Test SRF on Qwen models (where we saw +0.68% on VLM Bias)
   - Tune parameters specifically for LLaVA (try higher α values)
   - Investigate GQA/A-OKVQA baseline discrepancy with VCD paper
   - Consider SRF-E (evidence-amplified) for stronger intervention

---

*All experiments completed: 2026-05-15*
*Total samples evaluated: 27,000 (3 datasets × 3 splits × 3000 samples)*
*Decoding method: Sampling (do_sample=True, temp=0.7, top_p=0.9)*
