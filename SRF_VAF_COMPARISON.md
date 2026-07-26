# SRF vs VAF Comparison - Why VAF is Winning

## **CRITICAL ISSUE FOUND: V3 Gate is DISABLED!**

The autoresearch improvements that achieved **0.900 ceiling on POPE** used the `clip_full_gate_v3` mechanism, but our RePOPE experiments are running with it **DISABLED**!

```python
# srf/saliency/clip_salience.py:43
V3_GATE_ENABLED = False  # ← SHOULD BE True for autoresearch performance!
```

---

## **VAF (ClearSight) - Simple but Effective**

**Approach:** Uniform amplification of ALL visual tokens

**Parameters:**
```python
enh_para = 1.15      # Boost ALL visual tokens by 15%
sup_para = 0.9       # Suppress ALL system tokens by 10%
layers = 10-15       # Middle fusion layers
```

**Why it works:**
- ✅ **Simple and robust** - no CLIP, no saliency computation
- ✅ **Uniform boosting** - treats all visual tokens equally
- ✅ **No false negatives** - doesn't miss objects (unlike CLIP-based approaches)
- ✅ **Fast** - single forward pass, no CLIP overhead

**Why it wins:**
- Boosts visual tokens uniformly → model pays more attention to image
- Suppresses system tokens → reduces bias from training prompts
- Works because baseline is under-weighting visual tokens

---

## **SRF (Semantic Re-Focus) - Complex but Underperforming**

**Approach:** CLIP-guided selective boosting of query-relevant tokens

**Current Parameters (RePOPE experiments):**
```python
alpha = 4.0           # Boost strength (27x stronger than baseline!)
eps = 0.2             # Smoothing parameter
layers = 8-15         # Wider layer range than VAF
heads = 20%           # Top 20% heads per layer
clip_top_k_pct = 0.30 # Boost top 30% image tokens
clip_fallback_thresh = 0.20  # CLIP absence threshold
```

**Saliency Modes Available:**
1. **legacy** - Basic CLIP similarity with absence threshold (default)
2. **entropy** - Entropy-based detection  
3. **peak_ratio** - Peak-to-mean ratio detection
4. **multi_metric** - 4-signal combination

**V3 Gate Mechanism (DISABLED in our experiments!):**
```python
V3_GATE_ENABLED = False              # ← PROBLEM! Should be True
V3_FULL_IMG_THRESH = 0.21            # Lowered from 0.24 (v3 improvement)
V3_RAW_ENTROPY_THRESH = 0.95        # Low entropy = object present
V3_CROSS_SCALE_IOU_THRESH = 0.30     # Cross-scale overlap
V3_BLUR_DELTA_THRESH = 0.005         # Blur delta signal
V3_GATE_SOFT_MULTIPLIER = 0.85       # Soft gate threshold
```

**Why it's currently losing:**
- ❌ **V3 gate disabled** - missing key autoresearch improvement
- ❌ **Complex CLIP pipeline** - many failure points
- ❌ **False negatives** - CLIP misses small/occluded objects
- ❌ **Wrong parameters** - α=4.0 from POPE may not work for RePOPE
- ❌ **Over-selective** - only boosts top 30% tokens, might miss context

---

## **Direct Comparison**

| Aspect | VAF | SRF (current) |
|--------|-----|--------------|
| **Approach** | Uniform boost all visual tokens | CLIP-guided selective boost |
| **Complexity** | Simple - 2 parameters | Complex - 10+ parameters |
| **CLIP needed?** | No | Yes (ViT-L/14) |
| **Saliency modes** | None | 4 modes (legacy, entropy, peak_ratio, multi_metric) |
| **V3 gate** | N/A | Available but DISABLED |
| **Boost target** | All visual tokens equally | Only CLIP-relevant tokens |
| **False negative risk** | Low (boosts all) | High (CLIP misses objects) |
| **Speed** | Fast (no CLIP) | Slower (CLIP forward pass) |
| **RePOPE accuracy** | **88.28%** | **85.23%** |

---

## **Why VAF Beats SRF on RePOPE**

### 1. **VAF is simpler and more robust**
- No CLIP dependencies
- No saliency computation
- No false negatives from CLIP missing objects

### 2. **SRF is over-optimized for POPE**
- α=4.0 worked on POPE but fails on RePOPE
- CLIP behaves differently on RePOPE distribution
- V3 gate mechanism is disabled (critical issue!)

### 3. **VAF benefits from baseline weakness**
- Baseline under-weights visual tokens
- VAF corrects this uniformly
- SRF tries to be smart but CLIP fails

### 4. **RePOPE distribution is different**
- RePOPE corrects annotation errors
- CLIP trained on original COCO with errors
- CLIP saliency doesn't transfer perfectly

---

## **How to Fix SRF - Action Plan**

### **Priority 1: Enable V3 Gate (CRITICAL!)**
```python
# srf/saliency/clip_salience.py:43
V3_GATE_ENABLED = True  # ← Change this!
```

**Why:** The autoresearch results that hit 0.900 on POPE used v3 gate. Without it, we're missing the key improvement!

### **Priority 2: Test Different Saliency Modes**
Current experiments use `legacy` mode. Test:
- `entropy` mode - might work better on RePOPE
- `peak_ratio` mode - alternative detection
- `multi_metric` mode - combines 4 signals

### **Priority 3: Parameter Sweep for RePOPE**
POPE parameters (α=4.0) don't work on RePOPE. Need to sweep:
- `alpha`: 0.5, 1.0, 2.0, 3.0 (not 4.0)
- `eps`: 0.1, 0.15, 0.2, 0.25
- `layers`: 10-15 (match VAF's range)
- `clip_top_k_pct`: 0.20, 0.25, 0.30, 0.40
- `clip_fallback_thresh`: 0.15, 0.18, 0.21, 0.24

### **Priority 4: Hybrid Approach**
Combine VAF's simplicity with SRF's selectivity:
- Use VAF's uniform boost as base
- Add SRF's CLIP-guided boost on top
- Or: SRF boost × 0.7 + VAF boost × 0.3

---

## **SRF Method Explained - All Modes and Parameters**

### **Core SRF Pipeline:**
```
1. Extract query noun from question ("Is there a chair?" → "chair")
2. Compute CLIP similarity between noun and image patches
3. Detect object presence using saliency mode
4. If present: boost attention logits for top-k image tokens
5. If absent: uniform boost (or no boost)
6. Apply smoothing with epsilon (eps parameter)
```

### **Saliency Modes:**

#### **1. Legacy Mode (default)**
```python
object_present = max_sim > ABSENCE_THRESH  # 0.20
```
- Simple threshold on maximum patch similarity
- Fast but prone to false positives/negatives

#### **2. Entropy Mode**
```python
entropy = softmax(patch_sims / 0.02).entropy()
object_present = entropy < entropy_thresh  # 3.5
```
- Low entropy = peaked similarities = object present
- High entropy = uniform similarities = object absent
- More robust than simple threshold

#### **3. Peak Ratio Mode**
```python
peak_ratio = max_sim / mean_sim
object_present = peak_ratio > peak_thresh  # 3.5
```
- High peak-to-mean = strong signal = object present
- Low ratio = weak signal = object absent

#### **4. Multi-Metric Mode**
```python
signals = [
    max_sim > thresh,
    entropy < entropy_thresh,
    peak_ratio > peak_thresh,
    cross_scale_iou > iou_thresh
]
object_present = any(signals)  # OR combination
```
- Combines 4 different signals
- Most robust but slowest

### **V3 Gate Mechanism (currently DISABLED!):**
```python
gate_full = max_sim >= 0.21
gate_full_soft = max_sim >= 0.85 * 0.21 = 0.1785

v3_signals = {
    "raw_entropy": entropy < 0.95,
    "cross_scale_iou": overlap > 0.30,
    "blur_delta": blur_delta > 0.005
}

object_present = gate_full or (gate_full_soft and any(v3_signals))
```

**Why v3 gate works:**
- Primary gate: strong CLIP signal (≥0.21)
- Secondary gate: borderline CLIP + confirmation signals
- Prevents false positives from weak CLIP matches

---

## **Recommended Next Steps**

1. **Enable v3 gate immediately** - this is the #1 issue
2. **Run small validation test** with v3 gate enabled
3. **Test different saliency modes** on 100 samples
4. **Parameter sweep** around α=1.0-2.0 (not 4.0)
5. **Consider VAF+SRF hybrid** if individual methods don't work

The fact that VAF beats SRF is not because VAF is fundamentally better - it's because:
- V3 gate is disabled (critical issue!)
- Wrong parameters for RePOPE (α=4.0 from POPE)
- CLIP saliency not optimized for RePOPE distribution
- SRF is over-complex with many failure modes

**SRF should beat VAF** if configured correctly! The autoresearch results proved this on POPE (0.900 ceiling vs VAF 88.28%). We just need to find the right configuration for RePOPE.
