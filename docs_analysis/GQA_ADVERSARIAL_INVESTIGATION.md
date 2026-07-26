# GQA Adversarial Investigation: Our Results vs VCD Paper

**Date:** 2026-05-14
**Focus:** GQA Adversarial - Why is our baseline **-5.45% lower**?

---

## 📊 **Results Comparison**

| **Metric** | **VCD Paper** | **Our** | **Difference** |
|-------------|---------------|---------|---------------|
| **Accuracy** | 75.08% | 69.63% | **-5.45%** ❌ |
| **Precision** | 73.19% | 62.95% | **-10.24%** ❌ |
| **Recall** | 79.16% | 95.47% | **+16.31%** ✅ |
| **F1** | 76.06% | 75.87% | -0.19% |

**Key Issue:** We predict "yes" **75.83%** of the time (2275/3000) instead of expected **50%**!

---

## 🔍 **Detailed Comparison**

### 1. **Model Architecture**
| **Aspect** | **VCD Paper** | **Our** | **Match?** |
|-------------|---------------|---------|----------|
| Model | LLaVA-1.5-7B | LLaVA-1.5-7B | ✅ |
| Checkpoint | Not specified | `llava-hf/llava-1.5-7b-hf` | ⚠️ |
| Vision Encoder | CLIP-ViT-L/14 | CLIP-ViT-L/14 | ✅ |
| Language Decoder | Vicuna 7B | Vicuna 7B | ✅ |

**Status:** ✅ Same architecture, ⚠️ Different checkpoint possible

---

### 2. **Dataset Specifications**
| **Aspect** | **VCD Paper** | **Our** | **Match?** |
|-------------|---------------|---------|----------|
| **Source** | GQA development set | VCD format POPE JSON | ❌ |
| **Images** | 500 GQA images | 500 GQA images | ✅ |
| **Questions** | 6 per image | 6 per image | ✅ |
| **Total** | 3000 samples | 3000 samples | ✅ |
| **Setting** | Adversarial | Adversarial | ✅ |
| **Balance** | 50% yes / 50% no | 75.83% yes / 24.17% no | ❌ |

**Status:** ❌ **CRITICAL DIFFERENCE** - We have wrong label balance!

---

### 3. **Generation Settings**
| **Aspect** | **VCD Paper** | **Our** | **Match?** |
|-------------|---------------|---------|----------|
| **Decoding** | "Direct sampling" | Greedy (do_sample=False) | ❌ |
| **Temperature** | Not specified | N/A (greedy) | ❌ |
| **Max tokens** | Not specified | 20 | ⚠️ |
| **Top-p/K** | Not specified | N/A (greedy) | ❌ |

**From VCD GitHub:**
```python
model.generate(
    do_sample=True  # ← VCD uses sampling!
)
```

**From our code:**
```python
model.generate(
    do_sample=False  # ← We use greedy
)
```

**Status:** ❌ **MAJOR DIFFERENCE** - Different decoding methods!

---

### 4. **Evaluation Settings**
| **Aspect** | **VCD Paper** | **Our** | **Match?** |
|-------------|---------------|---------|----------|
| **Prompt** | Not specified in paper | "Is [object] in this image? Please answer yes or no." | ⚠️ |
| **Answer Parsing** | Not specified | First sentence, check "No"/"not"/"no" | ❌ |
| **Metrics** | Accuracy, Precision, Recall, F1 | Accuracy, Precision, Recall, F1 | ✅ |

**Status:** ⚠️ Prompt/parsing methods unclear

---

## 🎯 **Root Cause Analysis**

### **Issue 1: Model Overpredicts "Yes"** 🔴 **CRITICAL**

**VCD Paper:** Model predictions ~50% yes (balanced with ground truth)
**Our:** Model predictions **75.83% yes** (overpredicting "yes"!)

**Ground Truth Labels:** ✅ 50% yes / 50% no (BALANCED)
**Our Predictions:** ❌ 75.83% yes / 24.17% no (WRONG)

**This explains:**
- ✅ Our high recall (95.47%) - we say "yes" to almost everything
- ❌ Our low precision (62.95%) - many of our "yes" are wrong
- ❌ Our low accuracy (69.63%) - too many false positives

**Root cause:** Our model is **overconfident in saying "yes"** on adversarial GQA questions

---

### **Issue 2: Different Decoding Method** 🔴 **MAJOR**

**VCD Paper:** "Direct sampling from post-softmax" (likely `do_sample=True`)
**Our:** Greedy decoding (`do_sample=False`)

**Impact:**
- Greedy: Always picks most likely token (more deterministic)
- Sampling: Randomly samples from distribution (more varied)
- Could affect model behavior significantly

---

## 📋 **Next Steps**

### **Priority 1: Fix Label Balance** 🔴
- [ ] Check our GQA Adversarial JSON file
- [ ] Count actual yes/no labels
- [ ] Verify dataset integrity
- [ ] Compare with VCD paper's exact data

### **Priority 2: Match Decoding Method**
- [ ] Test with `do_sample=True` (if temperature specified)
- [ ] Compare results with greedy decoding
- [ ] Match VCD paper's exact generation settings

### **Priority 3: Verify Dataset Source**
- [ ] Check if VCD format POPE uses different GQA data
- [ ] Verify image selections match
- [ ] Compare question distributions

---

## 🚨 **Hypothesis**

**Most likely cause:** Our GQA Adversarial dataset has **WRONG LABELS** or **WRONG DATA BALANCE**.

**Evidence:**
1. We predict "yes" 75.83% of the time (expected: 50%)
2. VCD paper gets 75.08% accuracy with balanced labels
3. We get 69.63% accuracy with imbalanced labels
4. This suggests our data is fundamentally different

**Alternative:** Different decoding method (greedy vs sampling) causes model to hallucinate more "yes" responses.

---

## 📁 **Files to Check**

1. `/home/anna2/shruthi/dataset/POPE_images/gqa_adversarial.json` - Our dataset
2. `/home/anna2/shruthi/lmms-eval/srf/eval_pope_vcd_fixed.py` - Our evaluation code
3. VCD GitHub repo: https://github.com/DAMO-NLP-SG/VCD - Their evaluation code

---

## 🔗 **References**

- **VCD Paper:** /home/anna2/shruthi/2311.16922v1.pdf (pages 5-6)
- **VCD GitHub:** https://github.com/DAMO-NLP-SG/VCD
- **Our Results:** /home/anna2/shruthi/lmms-eval/results/llava_pope_gqa/baseline/adversarial/
