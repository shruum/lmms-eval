# Model Scaling Investigation: Qwen-3B vs Qwen-7B

## 🎯 The Problem

**Qwen-3B Results:**
- Baseline: 84.87% → SRF: 86.73% (+1.86% improvement) ✅

**Qwen-7B Results:**
- Baseline: ?? → SRF: ?? (little or no improvement) ❌

**Question:** Why doesn't the 3B configuration work for 7B?

## 🔍 Common Myths about Model Scaling

### ❌ Myth 1: "Linear scaling assumption"
```
Wrong: If it works for 3B, it should work for 7B
Wrong: Bigger models ≠ same behavior with same parameters
```

### ❌ Myth 2: "Architecture similarity"
```
Wrong: Both are Qwen2.5-VL, so they behave the same
Reality: 3B and 7B have different:
  - Attention patterns (more heads, different specialisation)
  - Feature distributions (different signal-to-noise ratios)
  - Layer functions (object reasoning shifts)
```

## 🧠 Why Model Scaling Breaks SRF

### **1. Layer Function Shift**
```
3B model: Layers 8-14 = object reasoning
7B model: Layers 8-14 might be = something else entirely!

Why? More parameters → different feature hierarchy
```

### **2. Intervention Saturation**
```
3B model: Boost 2.0/Suppress 5.0 = optimal
7B model: Same intervention = too weak OR too strong

Why? 7B has stronger representations, needs different tuning
```

### **3. Head Selection Mismatch**
```
Top 20% heads in 3B ≠ Top 20% heads in 7B

Why? More heads with different specialisation patterns
```

## 🎯 Investigation Strategy

### **Phase 1: Layer Range Analysis** (4 experiments)
Test where object reasoning happens in 7B:
- Early: (6-12) - more visual features
- Baseline: (8-14) - same as 3B
- Late: (14-20) - more reasoning
- Wide: (8-16) - comprehensive coverage

### **Phase 2: Intervention Strength** (4 experiments)
Test scaling of boost/suppress ratios:
- Gentler: (1.0/3.0) - 7B might be more sensitive
- Same: (2.0/5.0) - 3B baseline
- Stronger: (3.0/7.0) - 7B might need more
- Balanced: (1.5/4.0) - middle ground

### **Phase 3: Head Selection** (3 experiments)
Test optimal head percentage:
- Selective: Top 10% (7B has more heads to choose from)
- Baseline: Top 20% (same as 3B)
- Inclusive: Top 30% (more collaborative intelligence)

## 🚀 Expected Findings

### **Most Likely Scenario 1: Layer Shift**
**Hypothesis:** 7B's object reasoning happens in different layers
- **Prediction:** Layers (10-16) or (12-18) work better than (8-14)
- **Reason:** More capacity → reasoning shifts to later layers

### **Most Likely Scenario 2: Intervention Strength**
**Hypothesis:** 7B needs gentler intervention
- **Prediction:** Boost (1.0-1.5)/Suppress (3.0-4.0) works better
- **Reason:** Larger model has stronger signals, less aggressive correction needed

### **Most Likely Scenario 3: Head Selection**
**Hypothesis:** 7B needs more selective head targeting
- **Prediction:** Top 10-15% works better than 20%
- **Reason:** More heads = more noise, need to pick the very best

## 🎯 Running the Investigation

```bash
cd ~/shruthi/lmms-eval/my_analysis

# Run investigation on GPU 3 (or whichever has 7B loaded)
~/miniconda3/envs/mllm/bin/python investigate_qwen7b.py \
    --gpu 3 \
    --max_experiments 11 \
    --output_dir qwen7b_investigation
```

**Expected runtime:** ~2-3 hours (11 experiments × 10-15 min each)

## 📊 What We'll Learn

1. **Layer dynamics:** Where does object reasoning happen in 7B?
2. **Scaling laws:** How should intervention strength scale with model size?
3. **Capacity effects:** Does more capacity help or hurt SRF?

## 💡 Expected Outcomes

### **Best Case:** SRF works for 7B with different parameters
- Find optimal layer range (probably different from 3B)
- Find optimal intervention strength (probably gentler)
- 7B achieves similar or better improvements than 3B

### **Medium Case:** SRF works partially for 7B
- Find some improvements but smaller than 3B
- Learn that scaling effects are non-linear
- Might need model-specific SRF tuning

### **Worst Case:** SRF doesn't work for 7B
- Learn fundamental architectural differences
- SRF might need complete redesign for larger models
- Consider alternative approaches for 7B

## 🔬 Next Steps

1. **Run investigation** → Find what works for 7B
2. **Validate** → Test optimal config on full POPE (500 samples per split)
3. **Compare** → 3B vs 7B: which scales better with SRF?
4. **Generalize** → Can we predict scaling laws for SRF?

## 📈 Hypothesis: Scaling Laws for SRF

**Working hypothesis:** Model size × intervention strength should follow a power law:

```
optimal_boost ∝ model_size^(-0.5)
optimal_layers ∝ log(model_size)
optimal_head_k ∝ model_size^(-0.3)
```

This investigation will help us discover the actual relationship!
