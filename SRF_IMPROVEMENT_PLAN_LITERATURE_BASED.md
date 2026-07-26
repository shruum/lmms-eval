# SRF Improvement Plan - Literature-Inspired Strategy

## **Target: Beat VAF 88.28% on RePOPE (Current SRF: 85.23%)**

### **Gap Analysis: VAF vs Current SRF**

| Aspect | VAF (88.28%) | Current SRF (85.23%) | Gap |
|-------|--------------|---------------------|-----|
| **Alpha** | 0.15 (15% boost) | 4.0 (400% boost) | **27x too strong!** |
| **Layers** | 8-15 (middle fusion) | 8-15 | ✅ Match |
| **Heads** | Top 50% vision-aware | Top 20% | Could use more |
| **Suppression** | β=0.1 (10% suppress) | None | Missing component |
| **Adaptivity** | Fixed | Fixed | Both static |

---

## **Key Literature Insights for SRF Improvement:**

### **🎯 Priority 1: Conservative Alpha (VAF-Inspired)**

**Finding:** VAF uses **α=0.15** (15% boost) vs our **α=4.0** (400% boost!)

**Literature Evidence:**
- ClearSight: "α∈(0, 0.25), β∈(0, 0.15)" - **optimal: α=0.15, β=0.1**
- VHR: "α=2 optimal across architectures"
- AdaVBoost: Shows adaptive α works better than fixed

**Action:** Test conservative alphas: **0.15, 0.5, 1.0, 1.5, 2.0**

---

### **🎯 Priority 2: Add System Token Suppression (VAF-Inspired)**

**Finding:** VAF **simultaneously boosts visual AND suppresses system tokens** - we only boost!

**Literature Evidence:**
- ClearSight: "In middle layers (8-15), boost attention to image tokens **and** simultaneously suppress attention to system-prompt tokens"
- PAD: "Compensate by reducing attention to system tokens — those are truly irrelevant"
- Results: +3% POPE, -4.2% CHAIR (better format following)

**Current SRF Issue:** Only boosts visual tokens, doesn't suppress system tokens

**Action:** Implement system token suppression with **β=0.1** (10% suppression)

---

### **🎯 Priority 3: Match VAF's Exact Parameters**

**Literature Recipe (ClearSight):**
```python
alpha = 0.15      # 15% visual enhancement (NOT 4.0!)
beta = 0.1        # 10% system suppression
layers = 8-15      # Middle fusion layers
heads = 0.50      # Top 50% vision-aware heads (we use 20%)
```

**Our Current Issues:**
1. ❌ Alpha = 4.0 (27x too strong)
2. ❌ No beta parameter (no suppression)
3. ❌ Heads = 20% (vs 50% in VAF)

---

### **🎯 Priority 4: Increase Head Coverage**

**Literature Evidence:**
- VAF: Top 50% vision-aware heads
- VHR: "Top ~20% of heads have 3–5× higher visual attention"
- SPIN: "Identify and suppress inattentive heads"

**Current:** We use 20% - could be too selective

**Action:** Test **heads = 0.30, 0.40, 0.50**

---

### **🎯 Priority 5: Query-Conditioned Advantage (Our Strength)**

**Literature Finding:** Most methods are **NOT query-conditioned**:
- AIR: "Query-agnostic: it selects distinctive tokens relative to the visual prototype regardless of what the question asks"
- FlashVLM: "Query-conditioned: Yes - Figure 3 explicitly shows different questions on same image → different kept tokens"
- CAST: "Not query-conditioned: DPP selection doesn't know what question will be asked"

**✅ SRF Advantage:** Our CLIP saliency IS query-conditioned!
- "SRF-CLIP/HSSA is explicitly query-conditioned: different question → different tokens boosted on same image"

**This is our key differentiator** - we should leverage it!

---

## **🎯 Priority 6: Advanced Techniques from Literature**

### **Adaptive Alpha (AdaVBoost-inspired):**
- **Literature:** "AdaVBoost shows VGE-adaptive α at each step"
- **Idea:** Scale boost by model confidence
- **Implementation:** `alpha_adaptive = base_alpha * (1 + confidence_risk)`

### **Exponential Scaling (ILVAD-inspired):**
- **Literature:** ILVAD uses `exp(α·attn)` vs our linear boost
- **Finding:** "Exponential scaling (vs linear in VAF/SRF) creates sharper contrast"
- **Caution:** "Can destabilize attention distribution if over-applied"

### **Internal Saliency Fallback:**
- **Literature:** ILVAD/PAD use internal attention when CLIP fails
- **Idea:** Use inter-layer discrepancy when `max_sim < 0.15`
- **Fallback:** Internal saliency when CLIP uncertain

---

## **🚀 Proposed SRF Improvements:**

### **Phase 1: VAF-Exact Replication (Quick Win)**
```python
alpha = 0.15        # Match VAF exactly
beta = 0.1          # Add system suppression  
layers = 8-15       # Already match VAF
heads = 0.50         # Match VAF's 50%
```

**Expected:** Should match or beat VAF (88.28%)

---

### **Phase 2: Conservative Sweep (Find Optimal)**
```python
# Test conservative alphas
alpha_values = [0.15, 0.25, 0.5, 1.0, 1.5]  # VAF-inspired
# Test with system suppression
beta_values = [0.0, 0.05, 0.1, 0.15]
# Test head coverage
head_values = [0.20, 0.30, 0.40, 0.50]
```

**Target:** Find optimal combination that beats 88.28%

---

### **Phase 3: Advanced Adaptive SRF**
```python
# Adaptive alpha based on CLIP confidence
max_sim = clip_result.max_sim
confidence_risk = 1.0 - max_sim  # Lower sim = higher risk
alpha_adaptive = base_alpha * (1 + confidence_risk)

# Or AdaVBoost-style per-step risk
# (requires implementation of VGE risk scoring)
```

---

## **📊 Expected Results:**

**Conservative estimates:**
- **VAF-exact:** 88.28% (match VAF)
- **With head increase (50%):** 88.5-89.0%
- **With adaptive alpha:** 89.0-90.0%

**Optimistic estimates:**
- **Best combo (α=0.5, β=0.1, heads=0.5):** 90.0-91.0%

---

## **🔬 Experimental Plan:**

### **Step 1: Quick VAF Match Test**
Run: α=0.15, β=0.1, layers=8-15, heads=0.50 on 100 samples
**If this works ≥88%, we've solved it!**

### **Step 2: Conservative Alpha Sweep**  
Test: α=[0.15, 0.5, 1.0, 1.5] + β=0.1 + heads=0.50
**Find optimal alpha for RePOPE**

### **Step 3: Head Coverage Sweep**
Test: α=best_from_step2, β=0.1, heads=[0.30, 0.40, 0.50]
**Find optimal head coverage**

### **Step 4: Adaptive Alpha Implementation**
Add confidence-based adaptive scaling
**Target: 90%+ on RePOPE**

---

## **🎯 Success Criteria:**

- **Minimum:** Beat VAF 88.28% → **Target 89%+**
- **Goal:** Match or exceed VAF +4% = **92%+**
- **Stretch:** Beat best literature results → **93%+**

---

## **💡 Key Takeaway:**

**The literature confirms our diagnosis: Current α=4.0 is 27x too aggressive!**

VAF's success with α=0.15 shows that **conservative boosting + system suppression = better hallucination mitigation** than aggressive boosting alone.

**Our query-conditioned CLIP saliency is our key advantage** - most literature methods aren't query-aware, which explains why SRF should beat VAF when configured correctly.

**Next: Implement VAF-exact parameters and test!**
