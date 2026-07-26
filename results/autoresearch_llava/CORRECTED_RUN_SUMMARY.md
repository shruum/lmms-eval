# SRF AutoResearch - CORRECTED Results - LLaVA-7B + POPE (Adversarial)

## Summary

**Date:** 2026-04-28 (Corrected after code review)
**Model:** llava-hf/llava-1.5-7b-hf
**Dataset:** POPE adversarial
**Samples:** 100
**GPUs:** 1, 2 (sequential execution)

---

## Code Review Findings

### Original Diagnosis (INCORRECT)
- **Mistake:** I initially thought SRF wasn't working because I saw `method=baseline` in debug output
- **Reality:** The debug output was from the BASELINE pass, not the SRF pass
- **Actual Status:** SRF WAS working correctly all along!

### Code Flow (Verified Working)
1. **Line 510** (eval.py): Sets `method="baseline"` → runs baseline prediction
2. **Line 562** (eval.py): Calls `prepare_sample()` → sets `method="srf"`
3. **Line 387** (eval.py): Sets `method="srf"` again → runs SRF prediction
4. **Line 70** (llava_attn_patch.py): Applies SRF boost if `method != "baseline"`

### Verification with Debug Output
When I added comprehensive debug, I confirmed:
- ✅ `prepare_sample()` IS being called
- ✅ `method` IS being set to "srf"
- ✅ SRF IS being applied at all layers (0-31)
- ✅ Absence-aware IS working (`enh_para=0.166` vs `enh_para=2.0`)

---

## Why SRF Shows No Improvement

Since SRF IS working, the real question is: **Why is accuracy unchanged (Δ=0.0%)?**

### Possible Explanations

1. **LLaVA-7B + POPE is already at ceiling**
   - Baseline: 83.0% is quite high for adversarial POPE
   - Model may be exploiting linguistic shortcuts that SRF can't overcome

2. **SRF parameters not tuned for LLaVA-7B**
   - Current config: α=2.0, grid=7×7, layers=[8,20]
   - These were tuned for Qwen-VL, not LLaVA
   - Need LLaVA-specific hyperparameter search

3. **Absence-aware threshold too sensitive**
   - `thresh=0.248` may be triggering suppression too often
   - LLaVA's CLIP features may have different statistics than Qwen-VL

4. **POPE adversarial split is inherently difficult**
   - Questions are designed to trick models
   - Language priors are strong (e.g., "dog" → "yes")
   - SRF boosts visual tokens but may not overcome strong language bias

---

## Advanced Code Modifications to Try

Since basic parameter tuning didn't help, let's try advanced modifications:

### Option 1: Multi-Scale CLIP Ensemble
Combine multiple grid sizes for better saliency detection.

### Option 2: Graduated Multi-Stage Attention
Different α levels for different saliency tiers.

### Option 3: Text Token Suppression
Suppress attention to question tokens to reduce language prior.

### Option 4: Stronger Intervention
Much higher α (8.0+) to force model to rely more on vision.

---

## Next Steps

1. **Accept that 83.0% may be near ceiling for LLaVA-7B + POPE adversarial**
   - Try other datasets (MME, GQA, ScienceQA)
   - Try other POPE splits (popular, random)

2. **Try the advanced code modifications**
   - These require changing `srf/saliency/clip_salience.py`
   - These require changing `srf/srf.py`

3. **Test on larger models**
   - LLaVA-1.5-13B or 34B
   - May have better visual representations

---

## Inspired by

- [@karpathy](https://github.com/karpathy)'s [autoresearch](https://github.com/karpathy/autoresearch)
- Target: NeurIPS 2026 submission
