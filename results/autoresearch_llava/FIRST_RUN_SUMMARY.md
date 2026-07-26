# SRF AutoResearch Results - LLaVA-7B + POPE (Adversarial)

## Summary

**Date:** 2026-04-28
**Model:** llava-hf/llava-1.5-7b-hf
**Dataset:** POPE adversarial
**Samples:** 100
**GPUs:** 1, 2 (sequential execution)

---

## Key Finding: ⚠️ SRF has NO EFFECT on LLaVA-7B + POPE

All 5 experiments returned **identical results**: 83.0% accuracy (Δ=0.0%)

---

## Results Table

| Experiment | GPU | α | eps | grid | Accuracy | Δ vs Baseline | Status |
|------------|-----|---|-----|------|----------|---------------|--------|
| baseline | 1 | 2.0 | 0.0 | 7 | 83.0% | - | ✓ |
| 1A (5×5 grid) | 2 | 2.0 | 0.0 | 5 | 83.0% | 0.0% | ✗ No improvement |
| 1B (9×9 grid) | 1 | 2.0 | 0.0 | 9 | 83.0% | 0.0% | ✗ No improvement |
| 2A (stronger boost) | 2 | 4.0 | 0.2 | 7 | 83.0% | 0.0% | ✗ No improvement |
| 2B (gentle boost) | 1 | 1.5 | 0.1 | 7 | 83.0% | 0.0% | ✗ No improvement |

---

## Analysis

### Why is SRF not working?

**Possible explanations:**

1. **SRF not engaging**: The saliency-based attention manipulation may not be activating for LLaVA-7B on POPE
   - Check: Are vision-aware heads being detected correctly?
   - Check: Is the attention boost actually being applied?

2. **POPE adversarial is too hard**: 83% might be close to the model's ceiling on this split
   - LLaVA-7B has strong language priors that SRF can't overcome
   - Need to test on other datasets (MME, GQA, etc.)

3. **Wrong calibration dataset**: Calibrating on POPE might not generalize to adversarial POPE
   - Try: Calibrate on random or popular split
   - Try: Different calibration size (n=50, n=100)

4. **Implementation bug**: SRF might have a bug specific to LLaVA-7B
   - Check logs for any warnings or errors
   - Verify saliency maps are being computed

---

## Next Steps

### Immediate (High Priority)

1. **Debug SRF engagement**:
   ```bash
   # Run with verbose logging to see if SRF is applying boosts
   python srf/eval.py --method srf --model llava-hf/llava-1.5-7b-hf \
       --datasets pope --pope_splits adversarial --n_pope 10 \
       --output results/debug/ --verbose
   ```

2. **Test on other POPE splits**:
   ```bash
   # Test on popular split (easier)
   python srf/eval.py --method srf --model llava-hf/llava-1.5-7b-hf \
       --datasets pope --pope_splits popular --n_pope 100 \
       --output results/autoresearch_llava/popular_split/
   ```

3. **Test on other datasets**:
   - MME (perception)
   - GQA (reasoning)
   - ScienceQA (knowledge)

### Code Modifications (Medium Priority)

4. **Multi-scale CLIP ensemble**:
   - Combine 5×5, 7×7, 9×9 grids
   - Requires modifying `srf/saliency/clip_salience.py`

5. **Graduated multi-stage attention boost**:
   - Different α for different saliency levels
   - Requires modifying `srf/srf.py`

6. **Text token suppression**:
   - Reduce language prior by suppressing text tokens
   - Requires modifying `srf/srf.py`

---

## Experimental Variations to Test

### Option 1: Better Calibration

- 1A: Calibrate on popular split, test on adversarial
- 1B: Calibrate on random split, test on adversarial
- 1C: Larger calibration set (n=100 instead of n=20)

### Option 2: Better Saliency

- 2A: Multi-scale ensemble (5×5, 7×7, 9×9)
- 2B: Different CLIP model (ViT-L/14 instead of ViT-B/32)
- 2C: Higher top-k percentage (0.5 instead of 0.3)

### Option 3: Stronger Intervention

- 3A: Much stronger boost (α=8.0, eps=0.5)
- 3B: Text token suppression (suppress text attention)
- 3C: Combined approach (strong boost + text suppression)

---

## Logs and Outputs

All results saved to: `results/autoresearch_llava/`

- `baseline/summary.json`
- `1A_5x5_grid/summary.json`
- `1B_9x9_grid/summary.json`
- `2A_stronger_boost/summary.json`
- `2B_gentle_boost/summary.json`

---

## Conclusion

❌ **Initial AutoResearch loop failed to find improvements**

The first round of experiments suggests that SRF is not effectively mitigating hallucinations in LLaVA-7B on POPE adversarial. This is unexpected given previous success on Qwen-VL.

**Recommendation**: Debug why SRF is not engaging before running more experiments. Focus on:
1. Verifying SRF is applying attention boosts
2. Testing on easier datasets/splits
3. Exploring alternative calibration strategies

---

## Inspired by

- [@karpathy](https://github.com/karpathy)'s [autoresearch](https://github.com/karpathy/autoresearch)
- Target: NeurIPS 2026 submission
