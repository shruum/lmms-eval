# CRITICAL ISSUE: Baseline Discrepancy Investigation

**Date:** 2026-05-14
**Status:** 🔴 **OPEN - MAJOR ISSUE**

---

## 🚨 **CRITICAL PROBLEM**

**Our baseline is +5.88% HIGHER than VCD/AIR papers on COCO Random**

This is **NOT** a good thing - it means our evaluation setup is **WRONG/DIFFERENT**.

### Expected Behavior:
- All papers evaluating LLaVA-1.5-7B on POPE should have **nearly identical baselines** (±0.2%)
- Same model + Same dataset + Same evaluation = Same results

### Our Results:
| Dataset | Split | VCD Paper | AIR Paper | Our Baseline | Difference |
|---------|-------|-----------|-----------|--------------|------------|
| **COCO** | Random | 83.29% | 83.70% | **89.17%** | **+5.88%** ❌ |
| | Popular | 81.88% | 78.20% | **86.53%** | **+4.65%** ❌ |
| | Adversarial | 78.96% | 75.00% | **79.47%** | +0.51% ⚠️ |

**Average difference: +3.68%** (our baseline is WRONG)

---

## 🔍 **Possible Causes**

### 1. **Dataset Version** ⭐ MOST LIKELY
- **VCD Paper**: Standard POPE (development sets from MSCOCO/A-OKVQA/GQA)
- **Our Evaluation**: VCD format POPE (from AoiDragon/POPE GitHub)
- **Evidence**: Different image selections, different questions

### 2. **Prompt Format**
- **VCD Paper**: Need to verify exact prompt from paper
- **Our Prompt**: "Is [object] in this image? Please answer yes or no."
- **Impact**: Different wording can affect model performance

### 3. **Answer Parsing**
- **VCD Paper**: How do they extract yes/no from model output?
- **Our Method**: First sentence extraction, check for "No"/"not"/"no"
- **Impact**: Different parsing methods can give different results

### 4. **Image Preprocessing**
- **Resolution**: 336×336 (LLaVA-1.5 default)
- **Normalization**: Could affect model performance
- **Impact**: Small but measurable effects

### 5. **Generation Settings**
- **Ours**: do_sample=False, max_new_tokens=20
- **VCD Paper**: Need to verify from paper/repo
- **Impact**: Different settings can affect outputs

### 6. **Model Checkpoint**
- **Ours**: `llava-hf/llava-1.5-7b-hf` (HuggingFace default)
- **VCD Paper**: Specific checkpoint not specified
- **Impact**: Different checkpoints = different performance

### 7. **Evaluation Split**
- **VCD Paper**: "development sets" of these datasets
- **Our Evaluation**: VCD format JSON files
- **Impact**: Different data splits could cause variance

---

## 📋 **Investigation Steps**

### Priority 1: Verify Dataset Version ✅
- [x] Check VCD paper for dataset details
- [ ] Compare VCD format vs Standard POPE
- [ ] Verify image selections match
- [ ] Check question distributions

### Priority 2: Verify Evaluation Methodology
- [ ] Check VCD GitHub repo for evaluation code
- [ ] Compare prompt format with paper
- [ ] Verify answer parsing method
- [ ] Check generation settings

### Priority 3: Test with VCD Exact Setup
- [ ] Run evaluation using VCD paper's exact methodology
- [ ] Compare results with our current setup
- [ ] Identify specific differences causing +5.88% gap

---

## 🎯 **Success Criteria**

**Baseline is CORRECT when:**
- Our baseline matches VCD paper within ±0.5%
- Average difference < 0.5% across all splits
- COCO Random: 83-84% (not 89.17%)

---

## 📊 **Current Status**

**❌ BASELINE IS INCORRECT**
- Our results are systematically HIGHER than papers
- This means our evaluation setup is DIFFERENT
- **SRF improvements cannot be fairly compared** until baseline is fixed

---

## 🔗 **References**

- **VCD Paper**: https://arxiv.org/pdf/2311.16922
- **VCD GitHub**: https://github.com/DAMO-NLP-SG/VCD
- **AIR Paper**: https://arxiv.org/pdf/2602.24041
- **Our Results**: `/home/anna2/shruthi/lmms-eval/results/llava_pope_*/baseline/`

---

## 📝 **Notes**

**CRITICAL INSIGHT:**
The +5.88% difference is NOT because our model is "better" - it's because we're evaluating differently!

Possible scenarios:
1. We're using easier dataset version (different images/questions)
2. Our prompt is giving the model more hints
3. Our answer parsing is more lenient
4. Our preprocessing is improving model performance

**Until baseline matches papers exactly, all SRF results are INVALID for comparison.**
