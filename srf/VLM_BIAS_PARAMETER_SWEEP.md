# VLM Bias Parameter Sweep - Target >2% Improvement

**Status**: Ready to run
**Model**: Qwen2.5-VL-3B-Instruct
**Dataset**: VLMs-Are-Biased
**Current Best**: 19.68% (+0.68%)
**Target**: >21.00% (+2.0%)

---

## 🎯 Strategy

Based on analysis, the most likely issues are:

1. **Language priors dominate** (40% confidence) - Model ignores visual signal
2. **Wrong layers for counting** (30% confidence) - Counting needs late layers
3. **Boost too weak/strong** (20% confidence) - Alpha not optimal
4. **Model can't count** (10% confidence) - Fundamental limitation

---

## 📋 Two Sweep Options

### **Option 1: Quick Sweep** ⭐ RECOMMENDED FIRST

**Script**: `srf/sweep_vlmbias_quick.py`
**Time**: ~30 minutes
**Configs**: 24 combinations
**Samples**: 10 per category (70 total)

**Parameters**:
- `text_beta`: [0.0, 0.2, 0.5, 0.8, 1.0] - Suppress language priors
- `layer_range`: [(8,15), (16,23), (20,27), (8,27)] - Different fusion zones
- `alpha`: [2.0, 8.0] - Boost strength

**Run**:
```bash
cd /home/anna2/shruthi/lmms-eval
conda activate mllm
python srf/sweep_vlmbias_quick.py
```

**Expected Outcomes**:
- ✅ If >21%: Success! Run full evaluation with best config
- ⚠️ If 20-21%: Good progress, run comprehensive sweep
- ❌ If <20%: Model limitation, consider other datasets

---

### **Option 2: Comprehensive Sweep**

**Script**: `srf/sweep_vlmbias_comprehensive.py`
**Time**: ~12 hours (Phase 1: 8h + Phase 2: 4h)
**Configs**: 96 (Phase 1) + 48 (Phase 2)
**Samples**: 50 per category (350 total)

**Phase 1** (Coarse):
- `alpha`: [2.0, 4.0, 8.0, 16.0]
- `layer_range`: [(8,14), (8,15), (20,27), (8,27)]
- `text_beta`: [0.0, 0.3, 0.6]
- `clip_top_k_pct`: [0.3, 0.7, 1.0]
- `bias_mode`: ["additive_logit", "global_redistribute"]

**Phase 2** (Fine):
- Fine-grained search around best Phase 1 config
- ±25% on alpha, ±1 on layers, ±0.1 on text_beta

**Run**:
```bash
cd /home/anna2/shruthi/lmms-eval
conda activate mllm
python srf/sweep_vlmbias_comprehensive.py
```

---

## 🔍 Key Parameters Explained

### **text_beta** - MOST IMPORTANT ⭐

**What**: Suppress attention to question tokens during generation

**Why**: VLM Bias tests counting, but model uses language priors ("animals have 4 legs")

**Hypothesis**: Stronger text suppression → model relies more on vision

**Expected impact**: HIGH

**Values to test**:
- `0.0` (current) - No suppression
- `0.3` - Gentle suppression
- `0.6` - Moderate suppression
- `1.0` - Strong suppression

---

### **layer_range**

**What**: Which decoder layers to apply SRF

**Why**: Counting might be a late-stage reasoning task

**Hypothesis**: Late layers (20-27) better for counting than middle (8-15)

**Expected impact**: MEDIUM-HIGH

**Ranges to test**:
- `(8, 15)` - Current (middle fusion)
- `(16, 23)` - Late layers (reasoning)
- `(20, 27)` - Very late (final decision)
- `(8, 27)` - All layers

---

### **alpha**

**What**: Strength of attention logit boost

**Why**: Current alpha might be too weak or too strong

**Hypothesis**: Optimal alpha depends on task and text_beta

**Expected impact**: MEDIUM

**Values to test**:
- `2.0` - Gentle boost
- `8.0` - Strong boost (current)
- `16.0` - Very strong boost

---

### **bias_mode**

**What**: How to apply attention modulation

**Options**:
- `additive_logit` - Pre-softmax (current)
- `global_redistribute` - Post-softmax

**Why**: Post-softmax more stable (see Next_steps.md)

**Expected impact**: MEDIUM

---

## 📊 Expected Results Matrix

| text_beta | layer_range | alpha | Expected Result | Why |
|-----------|-------------|-------|-----------------|-----|
| **0.0** | (8,15) | 8.0 | 19.68% | Current best |
| **0.6** | (20,27) | 8.0 | **21-22%** | ✅ Likely best |
| **1.0** | (20,27) | 4.0 | **20-21%** | ✅ Good |
| **0.6** | (8,27) | 8.0 | **20-21%** | ✅ Good |
| **0.0** | (8,15) | 16.0 | 19-20% | Too strong |
| **1.0** | (8,15) | 2.0 | 18-19% | Too weak |

---

## 🚀 Execution Plan

### **Phase 1: Quick Sweep** (30 min)

1. Run `sweep_vlmbias_quick.py`
2. Analyze results:
   - If best >21%: ✅ SUCCESS
   - If best 20-21%: ⚠️ PROMISING
   - If best <20%: ❌ MODEL LIMITATION

### **Phase 2A: If Quick Sweep Successful**

Run full evaluation with best config:
```bash
python srf/eval.py \
  --method srf \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --datasets vlmbias \
  --alpha <best_alpha> \
  --layer_start <best_layer_start> \
  --layer_end <best_layer_end> \
  --text_beta <best_text_beta> \
  --output results/vlmbias_final/
```

### **Phase 2B: If Quick Sweep Promising**

Run comprehensive sweep:
```bash
python srf/sweep_vlmbias_comprehensive.py
```

### **Phase 2C: If Quick Sweep Fails**

Accept that VLM Bias may not be suitable for SRF:
- Model lacks counting capability
- Focus on recognition tasks instead
- Document findings as negative result

---

## 💡 Alternative: Post-Softmax Redistribution

If parameter sweep fails, try mechanism change:

**Approach**: Redistribute post-softmax attention by saliency

**Advantage**: More stable, respects probability constraints

**Implementation**:
```python
# In my_analysis/qwen_attn_patch.py
# Modify patched_softmax() to redistribute
# See Next_steps.md Approach 1 for details
```

**Time to implement**: ~2 hours

---

## 📁 Output Files

Both scripts save results to:
- `results/vlmbias_quick_sweep/` - Quick sweep results
- `results/vlmbias_comprehensive_sweep/` - Full sweep results

Each experiment saves:
- `experiment.json` - Config and accuracy
- `results.json` - All results summary

---

## 🎯 Success Criteria

- **Tier 1**: >21% (+2%) - ✅ Target achieved
- **Tier 2**: 20-21% (+1-2%) - ⚠️ Promising, needs more work
- **Tier 3**: 19-20% (0-1%) - 🔴 Minimal improvement
- **Tier 4**: <19% - ❌ Regression, something wrong

---

## 📚 Related Files

- `INVESTIGATION_WHY_SALIENCY_FAILS.md` - Detailed hypothesis analysis
- `Next_steps.md` - Post-softmax approach details
- `SRF_DEBUG_CONTEXT.md` - Current debug status
- `RESEARCH_STATUS.md` - Overall research status

---

**Last Updated**: 2026-05-02
**Status**: Ready to run
**Recommendation**: Start with quick sweep (30 min)
