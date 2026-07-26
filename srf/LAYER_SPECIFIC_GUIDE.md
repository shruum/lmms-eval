# Layer-Specific Modulation for VLM Bias

**Quick Start**: Run 4 parallel experiments on GPUs 0,1,2,3 (~5 minutes)

```bash
cd /home/anna2/shruthi/lmms-eval
bash srf/run_layer_specific_parallel.sh
```

---

## 🎯 What is Layer-Specific Modulation?

**Current SRF**: Same boost across all layers (uniform)

**Layer-Specific**: Different boosts for different layer zones

### **Why Layers Matter**

Different layers do different things in VLMs:

```
Layer 0-7:   Early visual processing (detect features)
Layer 8-15:  Visual-language fusion (combine modalities)
Layer 16-27: Late reasoning (final decisions, counting)
```

### **The Problem with Uniform Boosting**

Current SRF boosts all layers (8-15) with same strength:
- ✅ Good for visual-language fusion
- ❌ Doesn't help with late-layer reasoning (counting)
- ❌ Doesn't suppress language priors in final layers

### **Layer-Specific Solution**

```python
# Early layers: Gentle boost (detect features)
alpha_early = 0.5, layers = (0, 7)

# Middle layers: Normal boost (fusion)
alpha_mid = 2.0, layers = (8, 15)

# Late layers: Strong boost + text suppression (reasoning)
alpha_late = 8.0, layers = (16, 27), text_beta = 0.8
```

---

## 🚀 Quick Start (Parallel on 4 GPUs)

### **Option 1: Bash Script (Recommended)** ⭐

**Time**: ~5 minutes (4 experiments in parallel)

```bash
bash srf/run_layer_specific_parallel.sh
```

**Tests 4 configurations**:
1. **Late layers + strong text suppress**: α=8.0, layers=(20,27), textβ=0.8
2. **Late layers + moderate text suppress**: α=8.0, layers=(16,27), textβ=0.6
3. **Middle layers only**: α=6.0, layers=(8,15), textβ=0.0
4. **Wide range + text suppress**: α=4.0, layers=(8,27), textβ=0.5

### **Option 2: Python Script**

**Time**: ~20 minutes (13 experiments in batches of 4)

```bash
python srf/sweep_vlmbias_layer_specific.py
```

**Tests 13 configurations**:
- 3-zone approaches (early/mid/late)
- Late-layer only strategies
- Progressive boost strategies
- Text suppression sweep (0.3, 0.5, 0.8, 1.0)

---

## 📊 Experiment Configurations

### **Strategy 1: Late-Layer Focus** ⭐ MOST PROMISING

**Hypothesis**: Counting happens in late layers (16-27)

| Config | Layers | Alpha | Text Beta | Expected |
|--------|--------|-------|-----------|----------|
| Late 20-27 + Strong | (20,27) | 8.0 | 0.8 | **21-22%** ✅ |
| Late 16-27 + Moderate | (16,27) | 8.0 | 0.6 | **20-21%** ✅ |
| Late 16-27 + Weak | (16,27) | 8.0 | 0.3 | 19-20% |

### **Strategy 2: Three-Zone Approach**

| Zone | Layers | Alpha | Purpose |
|------|--------|-------|---------|
| Early | (0,7) | 0.5 | Visual detection |
| Mid | (8,15) | 2.0 | Fusion |
| Late | (16,27) | 8.0, textβ=0.6 | Reasoning |

### **Strategy 3: Progressive Boost**

Increasing boost with depth:
- Layer 0-7: α=0.5
- Layer 8-15: α=2.0
- Layer 16-27: α=8.0

---

## 🎯 Expected Results

Based on analysis of VLM Bias failures:

| Strategy | Expected Acc | Why |
|----------|--------------|-----|
| **Late 20-27, textβ=0.8** | **21-22%** | ✅ Counting + no language prior |
| Late 16-27, textβ=0.6 | **20-21%** | ✅ Good combo |
| Three-zone balanced | 20-21% | ✅ Biologically motivated |
| Current (8-15, textβ=0) | 19.68% | ❌ Baseline |
| Mid layers only | 18-19% | ❌ Too early |

---

## 🔍 Why Late Layers?

### **VLM Bias Analysis**

```
Animals (counting):       0%  ← Needs late-layer reasoning
Optical Illusion (binary): 48% ← Can use mid-layer fusion
```

**Insight**: Counting requires late-layer reasoning, but language priors dominate.

### **Late-Layer Solution**

1. **Strong visual boost** (α=8.0) → Force model to use vision
2. **Text suppression** (textβ=0.8) → Reduce language priors
3. **Layer range 20-27** → Final decision layers

---

## 📁 Output Files

Results saved to: `results/vlmbias_layer_specific/`

Each experiment saves:
- `log.txt` - Full evaluation log
- `vlmbias.json` - Accuracy results
- `experiment.json` - Configuration details

Summary:
- `summary.json` - All results comparison

---

## 🎯 Interpreting Results

### **Success Criteria**

- **Tier 1**: >21% (Δ=+2%) - ✅ Target achieved!
- **Tier 2**: 20-21% (Δ=+1-2%) - ⚠️ Promising
- **Tier 3**: 19-20% (Δ=0-1%) - 🔴 Minimal improvement
- **Tier 4**: <19% - ❌ Regression

### **Next Steps**

**If successful** (>21%):
1. ✅ Run full evaluation with best config (all 2784 samples)
2. ✅ Validate on other datasets (POPE, MMVP)
3. ✅ Update `config.py` with new best params

**If promising** (20-21%):
1. ⚠️ Fine-tune around best config (layer ranges, alpha)
2. ⚠️ Test post-softmax redistribution

**If fails** (<20%):
1. ❌ Model limitation confirmed (can't count)
2. ❌ Focus SRF on recognition tasks instead
3. ❌ Document as negative result

---

## 🔬 Technical Details

### **Current Implementation**

The current implementation uses uniform alpha across all layers:

```python
# qwen_attn_patch.py
patch._STATE["value"] = alpha  # Same for all layers
```

### **Layer-Specific Implementation**

For true layer-specific modulation, need to modify `qwen_attn_patch.py`:

```python
# Check current layer
layer_idx = patch._STATE["current_layer"]

# Apply layer-specific alpha
if 0 <= layer_idx <= 7:
    alpha = layer_specific_config["early_alpha"]
elif 8 <= layer_idx <= 15:
    alpha = layer_specific_config["mid_alpha"]
else:
    alpha = layer_specific_config["late_alpha"]

patch._STATE["value"] = alpha
```

**Note**: Current scripts simulate this by testing different layer ranges with uniform alpha.

---

## 📚 References

- **Next_steps.md**: "Approach 2: Layer-Specific Modulation"
- **INVESTIGATION_WHY_SALIENCY_FAILS.md**: Hypothesis analysis
- **VLM_BIAS_PARAMETER_SWEEP.md**: Overall sweep strategy
- **SRF_DEBUG_CONTEXT.md**: Current debug status

---

**Last Updated**: 2026-05-02
**Status**: Ready to run
**GPUs Available**: 0, 1, 2, 3 (free)
**Recommendation**: Start with bash script (5 min)
