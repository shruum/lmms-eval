# SRF Hyperparameter Sweep - Progress & Targets

**Date:** 2026-05-15
**Status:** 🔄 **RUNNING** - 45 experiments on 7 GPUs
**Goal:** Find SRF configuration that beats BOTH our baseline AND AIR/VCD methods

---

## 🎯 **SUCCESS CRITERIA (UPDATED)**

### **Previous Goal: ❌ WRONG**
- Beat our baseline by +1-3%
- This was insufficient because we also need to beat published methods

### **New Goal: ✅ CORRECT**
SRF must:
1. **Improve over our baseline** by ≥1%
2. **Match or exceed VCD paper** results
3. **Match or exceed AIR paper** results

### **Why This Matters:**

If we only beat our baseline but still below VCD/AIR:
- ❌ Not competitive with published methods
- ❌ Doesn't advance the state-of-the-art
- ❌ Shows our evaluation is different but not better

If we beat our baseline AND match/exceed papers:
- ✅ Advances state-of-the-art
- ✅ Shows SRF is effective
- ✅ Publication-worthy results

---

## 📊 **Current Baselines vs Papers**

### **Our Baselines (Sampling Decoding):**

| Dataset | Split | Our Baseline | VCD Paper | AIR Paper | Target (Max+1%) |
|---------|-------|--------------|-----------|-----------|------------------|
| **COCO** | Random | **83.43%** | 83.29% | 83.70% | **≥84.43%** |
| | Popular | **81.23%** | 81.88% | 78.20% | **≥82.23%** |
| | Adversarial | **79.30%** | 78.96% | 75.00% | **≥80.30%** |
| **A-OKVQA** | Random | **83.47%** | 83.45% | 83.40% | **≥84.47%** |
| | Popular | **79.30%** | 79.90% | 79.90% | **≥80.30%** |
| | Adversarial | **75.80%** | 74.04% | 74.00% | **≥76.80%** |
| **GQA** | Random | **82.47%** | 83.73% | 83.70% | **≥83.47%** |
| | Popular | **76.40%** | 78.17% | 78.20% | **≥77.40%** |
| | Adversarial | **75.50%** | 75.08% | 75.10% | **≥76.50%** |

---

## 🔬 **Hyperparameter Sweep Strategy**

### **What We're Testing:**

**45 configurations** across these axes:

1. **α (Boosting strength):**
   - 1.0 (medium)
   - 2.0 (high)
   - 4.0 (very high)

2. **Layer ranges (Fusion zones):**
   - 5-10 (early fusion)
   - 8-12 (early-mid)
   - 10-15 (middle) ← VAF default
   - 12-18 (mid-late)
   - 15-20 (late)
   - 18-25 (very late)

3. **Head selection (Focus):**
   - 50% (conservative)
   - 70% (aggressive)
   - 90% (very aggressive)

4. **Suppression (Background reduction):**
   - 0.1 (minimal)
   - 0.2 (low)

### **Why These Configs?**

Based on **VAF parameters were too weak** theory:
- VAF uses α=0.15 → we test 10-30× stronger (1.0-4.0)
- VAF uses layers 10-15 → we test broader ranges
- VAF uses 50% heads → we test more aggressive selection
- VAF uses eps=0.2 → we test less suppression

---

## 📈 **Success Scenarios**

### **Excellent Outcome (Best):**
- SRF achieves +2-3% over baseline
- **AND matches/exceeds VCD/AIR on 6+/9 splits**
- **Example: COCO Random 86.5% (baseline: 83.43%, VCD: 83.29%, AIR: 83.70%)**

### **Good Outcome:**
- SRF achieves +1-2% over baseline
- **AND matches VCD/AIR on 4+/9 splits**
- **Acceptable for publication**

### **Acceptable Outcome:**
- SRF achieves +0.5-1% over baseline
- **AND comes within 0.5% of VCD/AIR on most splits**
- **Shows promise but needs more tuning**

### **Poor Outcome:**
- SRF degrades performance (current VAF params: -1.69%)
- **OR doesn't match papers**
- **Need fundamental rethinking of approach**

---

## 🎯 **Specific Targets per Split**

### **High Priority (Easiest to beat papers):**
1. **COCO Adversarial:** Target ≥80.30% (our baseline: 79.30%, VCD: 78.96%, AIR: 75.00%)
2. **A-OKVQA Adversarial:** Target ≥76.80% (our baseline: 75.80%, VCD: 74.04%, AIR: 74.00%)
3. **COCO Popular:** Target ≥82.23% (our baseline: 81.23%, VCD: 81.88%, AIR: 78.20%)

### **Medium Priority (Need to match our high baseline):**
1. **COCO Random:** Target ≥84.43% (our baseline: 83.43% - already high!)
2. **A-OKVQA Random:** Target ≥84.47% (our baseline: 83.47% - already high!)
3. **GQA Adversarial:** Target ≥76.50% (our baseline: 75.50%)

### **Challenging (Our baseline is below papers):**
1. **GQA Random:** Target ≥83.73% (our baseline: 82.47%, need +1.26%)
2. **GQA Popular:** Target ≥78.20% (our baseline: 76.40%, need +1.8%)

---

## 🚀 **Current Sweep Status**

### **Phase 1: Quick Screening (Running Now)**
- **45 configs** × COCO Adversarial (3000 samples each)
- **GPUs:** 0-6 (7 GPUs parallel)
- **Timeline:** ~18-21 hours
- **Goal:** Find top 10 configs that beat 80.30% on COCO Adversarial

### **Phase 2: Full Evaluation (Next)**
- **Top 10 configs** × all 9 splits (COCO/AOKVQA/GQA)
- **270 experiments** total
- **Timeline:** ~40 hours
- **Goal:** Validate configs across all datasets

### **Phase 3: Analysis & Selection**
- Identify best config per split
- Generate comparison tables vs VCD/AIR
- Select final SRF configuration

---

## 📊 **What We Hope to Find**

### **Ideal Scenario: Universal Best Config**
- One configuration works well on ALL splits
- Achieves +1-2% over baseline
- Matches/exceeds VCD/AIR on 7+/9 splits

### **Realistic Scenario: Dataset-Specific Best**
- Different configs work best for different datasets
- COCO needs different layers than GQA
- Need ensemble or adaptive approach

### **Fallback Scenario: Modest Improvements**
- Find configs with +0.5-1% on some splits
- Better than current -1.69% degradation
- Shows promise but needs more research

---

## 🔍 **Why Previous SRF Failed**

### **Problem: VAF Parameters Too Weak**
- **VAF:** α=0.15, layers 10-15, 50% heads
- **Our LLaVA results:** -1.69% average degradation
- **Root cause:** VAF designed for different model architecture

### **Hypothesis: LLaVA Needs Stronger Intervention**
- **Test:** Higher alphas (1.0, 2.0, 4.0)
- **Test:** Different layer ranges (5-10, 8-12, 12-18)
- **Test:** More aggressive head selection (70%, 90%)
- **Expected:** Stronger boosting will help

### **Current Sweep: Testing Hypothesis**
- Running 45 configs that test these hypotheses
- Will know in ~18 hours if stronger alphas help

---

## 📁 **Progress Tracking**

### **Monitor Commands:**
```bash
# Quick status
bash check_srf_sweep_status.sh

# Real-time GPU monitoring
watch -n 10 nvidia-smi

# Check specific experiment
tail -f results/srf_focused_sweep/coco_adversarial_config0/run.log
```

### **Key Files:**
- **Sweep progress:** `/home/anna2/shruthi/lmms-eval/SRF_SWEEP_PROGRESS.md`
- **Baseline comparison:** `/home/anna2/shruthi/lmms-eval/POPE_BASELINE_COMPARISON.md`
- **Sweep results:** `/home/anna2/shruthi/lmms-eval/results/srf_focused_sweep/`

---

*Last updated: 2026-05-15 - 45 experiments running, testing hypotheses*
*Goal: Find SRF config that beats baseline AND matches/exceeds VCD/AIR papers*
