# SRF vs VAF Performance Analysis

## **Current Results:**
- **VAF (15% uniform boost)**: 88.28% on RePOPE
- **SRF (CLIP-guided selective boost)**: 85.23% on RePOPE
- **Gap**: VAF beats SRF by 3.05%

## **Core Difference:**

### **VAF Approach (Simple & Effective):**
```python
# VAF uniformly boosts ALL visual tokens by 15%
attn_weights[:, :, :, sys_len:sys_len+img_len] *= 1.15  # Boost
attn_weights[:, :, :, :sys_len] *= 0.9                   # Suppress
```

**Why it works:**
- ✅ **Simple and robust** - no CLIP, no complex logic
- ✅ **Conservative boost** - only 15% increase
- ✅ **No false negatives** - boosts all visual tokens equally
- ✅ **Broad coverage** - doesn't miss any visual regions

### **SRF Approach (Complex & Underperforming):**
```python
# SRF selectively boosts CLIP-relevant tokens
1. Extract query noun: "Is there a chair?" → "chair"
2. Compute CLIP similarity between "chair" and image patches
3. Detect object presence: max_sim > 0.20?
4. If present: boost top 30% tokens with α=4.0
5. If absent: uniform boost (or no boost)
```

**Why it's currently failing:**
- ❌ **Too aggressive** - α=4.0 is 27x stronger than VAF's 15%
- ❌ **Over-selective** - only boosts top 30% tokens
- ❌ **CLIP failures** - misses small/occluded objects
- ❌ **Complex pipeline** - many failure points
- ❌ **Wrong parameters** - optimized for POPE, not RePOPE

## **Hypothesis: VAF Wins Because of Simplicity**

### **VAF Success Factors:**
1. **Conservative boost**: 15% is modest, won't break anything
2. **Broad coverage**: All visual tokens get attention boost
3. **No CLIP dependency**: Works reliably without external model
4. **Layer targeting**: Focuses on middle fusion layers (10-15)

### **SRF Failure Factors:**
1. **Over-boosting**: α=4.0 is too aggressive
2. **Over-selection**: Top 30% might miss context tokens
3. **CLIP bias**: CLIP trained on different data distribution
4. **Complex absence detection**: Multiple failure modes
5. **Parameter mismatch**: POPE parameters don't work for RePOPE

## **Why SRF Should Beat VAF (In Theory):**

**SRF advantages:**
- **Query-aware**: Boosts relevant tokens, suppresses irrelevant
- **Semantic understanding**: Uses CLIP to understand image content
- **Adaptive**: Different boost for different questions
- **Selective**: Doesn't boost noise/background

**VAF disadvantages:**
- **Query-agnostic**: Boosts everything equally
- **No semantic understanding**: Doesn't know what's relevant
- **Fixed boost**: Same 15% for all questions/samples

## **Current SRF Configuration Issues:**

### **Problem 1: Alpha Too High**
```python
alpha = 4.0  # 27x stronger than VAF's 0.15!
```
**Issue**: 4.0 boost is way too aggressive compared to VAF's 15%
**Fix**: Try alpha=0.5, 1.0, 1.5 (more conservative)

### **Problem 2: Top-K Too Restrictive**
```python
clip_top_k_pct = 0.30  # Only boost top 30% tokens
```
**Issue**: Might miss context tokens that VAF boosts
**Fix**: Try 0.5, 0.7, 1.0 (boost more tokens)

### **Problem 3: Layer Range Different**
```python
SRF: layers 8-15
VAF: layers 10-15
```
**Issue**: SRF boosting earlier layers (8-9) that VAF doesn't touch
**Fix**: Match VAF's layer range (10-15)

### **Problem 4: V3 Gate Disabled**
```python
V3_GATE_ENABLED = False  # Should be True!
```
**Issue**: Missing key autoresearch improvement
**Fix**: Enable v3 gate

### **Problem 5: CLIP Threshold Issues**
```python
ABSENCE_THRESH = 0.20  # May be wrong for RePOPE
```
**Issue**: CLIP distribution different on RePOPE
**Fix**: Sweep threshold values

## **Recommended SRF Improvements:**

### **Priority 1: Conservative Alpha**
```python
# Try conservative alphas closer to VAF
alpha_values = [0.5, 1.0, 1.5, 2.0]  # Not 4.0!
```

### **Priority 2: Match VAF Parameters**
```python
# Match VAF's successful setup
layers = [10, 11, 12, 13, 14, 15]  # Same as VAF
heads = 0.50  # Boost more heads than VAF
```

### **Priority 3: Test VAF-Like SRF**
```python
# Make SRF behave like VAF but with CLIP guidance
alpha = 0.15  # Same as VAF
clip_top_k_pct = 1.0  # Boost all tokens (like VAF)
# CLIP only used for gating, not selection
```

### **Priority 4: Enable V3 Gate**
```python
V3_GATE_ENABLED = True
```

## **Experimental Plan:**

### **Phase 1: Conservative Alpha Sweep**
Test: alpha=[0.5, 1.0, 1.5, 2.0] on 100 samples
Target: Find alpha that matches or beats VAF

### **Phase 2: Layer Range Match**
Test: layers=[10-15] (match VAF) vs layers=[8-15] (current)
Target: Confirm optimal layer range

### **Phase 3: Top-K Sweep**
Test: clip_top_k_pct=[0.3, 0.5, 0.7, 1.0]
Target: Find optimal token coverage

### **Phase 4: VAF-Like SRF**
Test: alpha=0.15, clip_top_k_pct=1.0 (VAF-like but with CLIP gating)
Target: SRF that behaves like VAF but smarter

## **Success Criteria:**
- SRF > 88.28% (VAF baseline)
- SRF > 88.1% (VCD baseline)
- SRF > 85.23% (current SRF)

## **Expected Outcome:**
If CLIP guidance is valuable, conservative SRF should beat VAF.
If CLIP guidance is harmful, uniform boost (VAF) is optimal approach.

---

# 🔍 Complete SRF Code Verification (2026-06-18)

**Purpose:** Verify every component of SRF works as intended after dimension bug fix

## **SRF Method Step-by-Step:**

### **1. How many tokens are chosen to boost?**
**Answer: ALL 576 image tokens, but by VARIABLE amounts**

```python
# NOT binary: "boost 173 tokens, ignore 403 tokens"
# ACTUAL: "boost all 576 tokens gradually based on saliency"

saliency = [0.1, 0.8, 0.3, 0.9, ...]  # 576 values (0-1)
boost_amount = [2.5%, 20%, 7.5%, 22.5%, ...]  # Variable boost per token

# High saliency (0.9) → 22.5% boost
# Medium saliency (0.5) → 12.5% boost  
# Low saliency (0.1) → 2.5% boost
```

**Status:** ✅ **CORRECT** - Sophisticated gradual boosting better than binary selection

### **2. Where is it being boosted?**
**Answer: Layers 10-15, top 50% vision-aware heads**

```python
# Layers: Middle fusion layers where vision meets language
layer_start = 10
layer_end = 15  # Applies to layers 10, 11, 12, 13, 14, 15

# Heads: Top 50% most vision-aware (from calibration)
head_mask = [True, True, False, True, ...]  # 16 True, 16 False (out of 32)
# Only applies to heads that responded to vision during calibration
```

**Status:** ✅ **CORRECT** - Targeted modification where it matters

### **3. By how much are they boosted?**
**Answer: 0% to 25% depending on saliency score**

```python
# Formula: boost = 1.0 + (enh_para - 1.0) * saliency
# With enh_para = 1.25 (alpha = 0.25):

saliency = 1.0 → boost = 1.25 (25% increase)
saliency = 0.5 → boost = 1.12 (12% increase)
saliency = 0.1 → boost = 1.02 (2% increase)
saliency = 0.0 → boost = 1.00 (no increase)
```

**Status:** ✅ **CORRECT** - Gradual boost based on saliency

### **4. How do we map from saliency map to tokens?**
**Answer: CLIP 6×6 grid → bilinear upsample → 576 tokens (FIXED)**

```python
# Step 1: CLIP computes 6×6 coarse grid (36 similarity scores)
CLIP_grid = 6×6 = 36 patches

# Step 2: Bilinear interpolation to 24×24 (FIXED - was broken)
upsampled = F.interpolate(6×6, size=(24,24), mode='bilinear')
24×24 = 576 elements

# Step 3: Flatten and trim to exact token count
saliency = upsampled.flatten()[:576]  # (576,)
```

**Status:** ✅ **FIXED** - Dimension bug resolved (36 → 576 upsampling now works)

### **5. How and where do we suppress?**
**Answer: Two-level suppression - system tokens + background**

```python
# Level 1: System token suppression (lines 107-111 in llava_attn_patch.py)
sys_end = 34  # System prompt tokens 0-34
sup_para = 1.0 - sys_beta = 0.85  # 15% reduction
attn_weights[:, :, :, 0:35] *= 0.85

# Level 2: Background suppression (lines 170-176)
background_eps = 0.1  # Up to 10% suppression
suppress_mask = 1.0 - (1.0 - saliency) * 0.1

# Examples:
saliency = 0.9 → suppress = 0.99 (1% suppression)
saliency = 0.1 → suppress = 0.91 (9% suppression)
saliency = 0.0 → suppress = 0.90 (10% suppression)
```

**Status:** ✅ **CORRECT** - Multi-level suppression strategy

---

## **✅ Verification Status - ALL COMPONENTS WORKING**

| Component | Status | Details |
|-----------|--------|---------|
| **Image token detection** | ✅ **CORRECT** | Returns `[35, 610]` for 576 tokens |
| **CLIP saliency upsampling** | ✅ **FIXED** | 36 → 576 elements (was broken) |
| **Token boosting** | ✅ **CORRECT** | All 576 tokens, 0-25% variable boost |
| **Layer selection** | ✅ **CORRECT** | Layers 10-15 (fusion layers) |
| **Head selection** | ✅ **CORRECT** | Top 50% vision-aware heads |
| **System suppression** | ✅ **CORRECT** | 15% reduction for system tokens |
| **Background suppression** | ✅ **CORRECT** | Up to 10% for low saliency |

---

## **🎯 Key Insight: SRF Design is Sophisticated**

**NOT binary boosting** - Instead of "boost these 173 tokens, ignore the rest," SRF uses:
- **Gradual boosting**: High saliency → 22.5% boost, Low saliency → 2.5% boost
- **Multi-level suppression**: System tokens (15%) + background (up to 10%)
- **Targeted application**: Only layers/heads where vision meets language

This is actually **better designed** than simple top-K selection!

---

## **🔧 Critical Bug Fix Applied:**

**Bug**: `_make_saliency()` wasn't reading `clip_upsample_to_tokens` from arch config
**Fix**: Changed line 210 in `srf/srf.py`:
```python
# BEFORE: "clip_upsample_to_tokens": False,  # disabled by default
# AFTER:  "clip_upsample_to_tokens": arch.get("clip_upsample_to_tokens", False)
```

**Impact**: CLIP saliency now properly upsamples from 36 → 576 tokens instead of falling back to uniform enhancement.

**Status**: ✅ **FIXED AND VERIFIED**

---

