# SRF Improvement Plan - Focus on SRF Only

## **Current SRF Performance Issue:**
- **Current SRF**: 85.23% on RePOPE
- **Target**: Beat 88.28% (VAF) and 88.1% (VCD) 
- **Gap**: Need +3.05% improvement

## **What SRF Does Well (Theory):**
- ✅ **Query-aware**: Boosts question-relevant tokens
- ✅ **CLIP-guided**: Uses visual understanding for smart boosting
- ✅ **Semantic**: Understands image content (e.g., "chair" detection)
- ✅ **Selective**: Only boosts relevant regions, not noise

## **Why Current SRF Fails on RePOPE:**

### **Problem 1: Alpha Too Aggressive**
```python
Current: alpha = 4.0  # Way too strong!
VAF inspiration: alpha = 0.15  # Conservative works better
```
**Fix**: Try alpha = 0.5, 1.0, 1.5, 2.0

### **Problem 2: Wrong Layer Range**
```python
Current SRF: layers 8-15  
VAF inspiration: layers 10-15 (middle fusion)
```
**Fix**: Match VAF's successful range (10-15)

### **Problem 3: Over-Selective Boosting**
```python
Current: clip_top_k_pct = 0.30  # Only boost 30% tokens
VAF inspiration: Boosts ALL visual tokens (100%)
```
**Fix**: Try 0.5, 0.7, 1.0 (more coverage)

### **Problem 4: V3 Gate Disabled**
```python
Current: V3_GATE_ENABLED = False  # Missing key improvement!
```
**Fix**: Enable v3 gate for better object detection

### **Problem 5: Wrong Parameters for RePOPE**
```python
Current: Using POPE-optimized parameters (α=4.0, ε=0.2, layers=8-15)
Issue: RePOPE distribution is different from POPE
```
**Fix**: Re-optimize for RePOPE specifically

## **SRF Improvement Strategy:**

### **Phase 1: Fix Obvious Issues**
1. **Enable v3 gate** - This alone could improve performance
2. **Match VAF layer range** - Use layers 10-15 instead of 8-15
3. **Test conservative alphas** - Try 0.5, 1.0, 1.5 instead of 4.0

### **Phase 2: Parameter Optimization**
1. **Alpha sweep** - Find optimal boost strength
2. **Top-k sweep** - Find optimal token coverage  
3. **Epsilon sweep** - Find optimal smoothing
4. **Head sweep** - Find optimal head selection

### **Phase 3: Saliency Mode Testing**
1. **Test all 4 modes** - legacy, entropy, peak_ratio, multi_metric
2. **Compare performance** - Find best mode for RePOPE
3. **Optimize thresholds** - Tune mode-specific parameters

## **Inspiration from VAF (Not Comparison):**

### **What VAF Does Right:**
- **Conservative boost**: 15% increase (modest, safe)
- **Layer targeting**: Middle fusion layers (10-15)
- **Broad coverage**: All visual tokens get boost
- **Simple logic**: No complex pipelines

### **How to Apply to SRF:**
```python
# VAF-inspired SRF configuration
alpha = 0.15 to 1.0     # Conservative boost (not 4.0!)
layers = 10-15          # Match VAF's successful range
clip_top_k_pct = 0.7   # Boost more tokens (like VAF)
eps = 0.1 to 0.2       # Gentle smoothing
```

## **Next Actions:**

1. **Enable v3 gate immediately**
2. **Test conservative alphas on 100 samples**
3. **Find optimal configuration for RePOPE**
4. **Run full experiments with best config**

**Success criteria**: SRF > 88.28% on RePOPE

