# POPE Baseline Investigation: Prompt Format Test Results

**Date:** 2026-05-14
**Test:** GQA Random (3000 samples)
**Result:** ❌ **Prompt format change made NO difference**

---

## 🧪 Test Results

### Prompt Conversion:
```python
# OLD: Direct from JSON
"Is there a bird in the image?"

# NEW: AIR paper format  
"Is bird in this image? Please answer yes or no."
```

### GQA Random Baseline Comparison:

| Version | Accuracy | vs AIR Target |
|---------|----------|---------------|
| **Old Prompt** | 85.80% | -2.20% |
| **New Prompt (AIR)** | 85.80% | -2.20% |
| **AIR Paper Target** | 88.00% | 0.00% |

**Change: +0.00%** ❌

---

## 🔍 **Root Cause Analysis: Prompt Format is NOT the Issue**

### What We Tested:
1. ✅ Confirmed prompt conversion logic works correctly
2. ✅ Applied AIR paper prompt format to all 3000 GQA Random samples
3. ✅ Used same generation parameters (do_sample=False, max_new_tokens=20)
4. ✅ Used same model checkpoint (llava-hf/llava-1.5-7b-hf)

### Result:
**NO CHANGE in baseline accuracy** (85.80% → 85.80%)

---

## 🎯 **Next Steps: Investigate Other Differences**

Since prompt format is not the cause, the baseline discrepancy must be from:

### 1. **Dataset Version Difference** ⭐ MOST LIKELY
- **Our evaluation**: VCD format POPE (from AoiDragon/POPE GitHub)
- **AIR paper**: Standard POPE (HuggingFace `lmms-lab/POPE`)
- **Evidence**: VCD uses different image selections
  - Only 11/500 overlap between COCO and A-OKVQA images
  - Different question distributions

### 2. **Model Version/Checkpoint**
- **Our evaluation**: `llava-hf/llava-1.5-7b-hf` (HuggingFace default)
- **AIR paper**: LLaVA-1.5-7B (specific checkpoint not specified)
- **Impact**: Different checkpoints can have 1-3% performance variance

### 3. **Image Preprocessing**
- **Our evaluation**: Default PIL image loading
- **AIR paper**: May use different preprocessing/resizing
- **Impact**: Small but measurable effects on accuracy

### 4. **Tokenization/Decoding Details**
- **Our evaluation**: `do_sample=False, max_new_tokens=20`
- **AIR paper**: `do_sample=False, temperature=0, beam_size=1`
- **Impact**: Should be identical, but worth verifying

---

## 💡 **Recommended Next Steps**

### Priority 1: Test Standard POPE Dataset
```bash
# Compare VCD format vs standard POPE on same model
python -c "
from datasets import load_dataset
# Load standard POPE
ds = load_dataset('lmms-lab/POPE', split='test')
# Check if questions/images differ from VCD format
"
```

### Priority 2: Verify Model Checkpoint
- Check if HuggingFace `llava-hf/llava-1.5-7b-hf` matches AIR paper version
- Try original LLaVA-1.5-7B checkpoint if available

### Priority 3: Compare Question/Image Distributions
- Sample 100 questions from VCD vs standard POPE
- Check if difficulty/distribution differs
- Verify image sources match

---

## 📁 **Current Status**

### ✅ **Completed:**
1. Fixed prompt format in `eval_datasets.py` 
2. Fixed prompt format in `eval_pope_vcd_fixed.py`
3. Tested GQA Random with new prompt (3000 samples)
4. Confirmed prompt conversion works correctly

### ❌ **Result:**
- **NO IMPROVEMENT** from prompt format change
- **Gap to AIR paper remains**: -2.20% (GQA Random)

### 🎯 **Conclusion:**
**The baseline discrepancy is NOT caused by prompt format.**
The issue is likely from using VCD format POPE instead of standard POPE, or model checkpoint differences.

---

## 📊 **Summary of All Baseline Gaps (Current)**

| Dataset | Split | Our Baseline | AIR Target | Gap |
|---------|-------|--------------|------------|-----|
| **COCO** | Random | 89.17% | 88.2% | **+0.97%** |
| | Popular | 86.53% | 86.1% | **+0.43%** |
| | Adversarial | 79.47% | 82.3% | **-2.83%** |
| **A-OKVQA** | Random | 85.47% | 87.6% | **-2.13%** |
| | Popular | 80.07% | 81.9% | **-1.83%** |
| | Adversarial | 69.60% | 74.3% | **-4.70%** |
| **GQA** | Random | 85.80% | 88.0% | **-2.20%** |
| | Popular | 74.63% | 79.4% | **-4.77%** |
| | Adversarial | 69.63% | 76.3% | **-6.67%** |

**Average gap: -2.72%** (our baseline is LOWER)

---

## 🔗 **Related Files**

- **AIR Paper**: https://arxiv.org/pdf/2602.24041
- **VCD Format POPE**: https://github.com/AoiDragon/POPE
- **Standard POPE**: HuggingFace `lmms-lab/POPE`
- **Our Results**: `results/llava_pope_*/baseline/`
- **Test Results**: `/tmp/gqa_new_prompt_test/`
