# SRF Hyperparameter Sweep - IN PROGRESS

**Date:** 2026-05-15
**Status:** 🔄 **RUNNING**
**Strategy:** Focused sweep on high-potential configurations

---

## 🎯 **Objective**

Find the BEST SRF hyperparameters for LLaVA-1.5-7B on POPE benchmark by systematically testing multiple configurations on all 7 GPUs.

**Problem:** VAF-like parameters (α=0.15, layers 10-15) give -1.69% average degradation

**Goal:** Find configurations that give POSITIVE improvements (+1-3%)

---

## 🔬 **Testing Configurations**

### **Hyperparameter Grid:**
- **α (boosting):** 1.0, 2.0, 4.0 (stronger boosting than VAF's 0.15)
- **Layer ranges:** 5-10, 8-12, 10-15, 12-18, 15-20, 18-25
- **Head selection:** 50%, 70%, 90% (more aggressive than VAF's 50%)
- **Suppression (eps):** 0.1, 0.2 (less suppression than VAF's 0.2)

### **Total Experiments:**
- **Phase 1:** 45 configs × COCO Adversarial = 45 experiments
- **Phase 2:** Top 10 configs × all 9 splits = 90 experiments
- **Total:** 135 experiments

---

## 🚀 **Current Execution**

### **Phase 1: Quick Screening (Running Now)**
- **Dataset:** COCO Adversarial (3000 samples, full evaluation)
- **Configurations:** 45 high-potential combinations
- **GPUs:** 0-6 (7 GPUs parallel)
- **Status:** 30 processes launched, models loading

### **Progress Tracking:**
```bash
bash check_srf_sweep_status.sh
```

### **What's Being Tested:**
1. **Higher alphas** (1.0, 2.0, 4.0) - stronger boosting
2. **Different layer ranges** - find optimal fusion zone
3. **Aggressive head selection** (70%, 90%) - more focused attention
4. **Low suppression** (0.1, 0.2) - less background suppression

---

## 📊 **Experimental Design**

### **Why These Configurations?**

Based on SRF_DEBUG_CONTEXT findings:
- ✅ **Alpha doesn't matter** with current setup → Test HIGHER alphas
- ✅ **Layer range is model-specific** → Test multiple zones
- ✅ **Head selection percentage** → Test more aggressive filtering
- ⚠️ **Need stronger intervention** → VAF params too weak for LLaVA

### **Config Categories:**

**Category 1: High Alpha (2.0) + Middle Layers + Aggressive Heads**
```
α=2.0, layers=10-15, heads=70%, eps=0.1
α=2.0, layers=10-15, heads=90%, eps=0.1
α=2.0, layers=12-18, heads=70%, eps=0.1
α=2.0, layers=12-18, heads=90%, eps=0.1
α=2.0, layers=15-20, heads=70%, eps=0.1
```

**Category 2: Very High Alpha (4.0) + Middle Layers**
```
α=4.0, layers=10-15, heads=70%, eps=0.1
α=4.0, layers=10-15, heads=90%, eps=0.1
α=4.0, layers=12-18, heads=70%, eps=0.1
α=4.0, layers=12-18, heads=90%, eps=0.2
```

**Category 3: Medium Alpha (1.0) + Wide Layers**
```
α=1.0, layers=8-12, heads=50%, eps=0.1
α=1.0, layers=8-12, heads=70%, eps=0.1
α=1.0, layers=10-18, heads=50%, eps=0.1
α=1.0, layers=10-18, heads=70%, eps=0.2
```

**Category 4: Extended Layer Ranges**
```
α=2.0, layers=5-10, heads=50%, eps=0.1
α=2.0, layers=5-10, heads=70%, eps=0.1
α=2.0, layers=18-25, heads=70%, eps=0.1
α=2.0, layers=18-25, heads=90%, eps=0.1
```

---

## ⏱️ **Timeline**

### **Phase 1 (Current):**
- **45 experiments** on COCO Adversarial
- **Parallel:** 7 GPUs (6-7 experiments per GPU)
- **Time per experiment:** ~3 hours
- **Total time:** ~18-21 hours (because parallel)

**Expected completion:** Tomorrow morning (2026-05-16)

### **Phase 2 (Next):**
- **Top 10 configs** × all 9 splits
- **90 experiments** total
- **Parallel:** 7 GPUs
- **Expected time:** ~40 hours

---

## 📈 **Success Criteria**

### **Good Configurations:**
- **Improvement > 0%** on COCO Adversarial
- **Target:** +1% to +3% improvement over baseline (79.30%)

### **Excellent Configurations:**
- **Improvement > 2%** on COCO Adversarial
- **Consistent** across multiple splits
- **Will be tested on all 9 datasets**

### **Poor Configurations:**
- **Degradation < -1%** (worse than baseline)
- **Will be discarded**

---

## 🔍 **Real-Time Monitoring**

### **Check Status:**
```bash
bash check_srf_sweep_status.sh
```

### **Watch Specific Config:**
```bash
tail -f results/srf_focused_sweep/coco_adversarial_config0/run.log
```

### **GPU Usage:**
```bash
watch -n 5 nvidia-smi
```

---

## 📁 **Results Location**

```
/home/anna2/shruthi/lmms-eval/results/srf_focused_sweep/
├── coco_adversarial_config0/
│   ├── run.log
│   └── pope_coco_adversarial_srf.json
├── coco_adversarial_config1/
│   └── ...
└── ... (45 configs total)
```

---

## 🎯 **Next Steps After Phase 1**

1. **Analyze results** from 45 configs
2. **Select top 10** performing configs
3. **Run Phase 2:** Top 10 configs on all 9 splits (COCO/AOKVQA/GQA × 3 splits)
4. **Identify best config** for each dataset/split
5. **Generate final comparison** table

---

## 💡 **Key Insights**

### **Why Previous Results Were Poor:**
1. **VAF params too weak:** α=0.15 designed for VAF, not SRF
2. **Layer range mismatch:** 10-15 may not be optimal for LLaVA-1.5-7B
3. **Head selection:** 50% may not filter enough
4. **Need stronger intervention** for LLaVA vs Qwen

### **Hypothesis:**
LLaVA-1.5-7B needs **stronger boosting** (higher α) and **different layer ranges** than Qwen models.

---

## 📊 **Expected Outcomes**

### **Best Case:**
- Find configs with +2-3% improvement
- Consistent improvements across all splits
- SRF becomes effective for LLaVA-1.5-7B

### **Likely Case:**
- Find configs with +0.5-1.5% improvement
- Some splits improve, others degrade
- Need dataset-specific tuning

### **Worst Case:**
- No config beats baseline
- Need to try completely different approach (e.g., SRF-E)

---

*Last updated: 2026-05-15 - 45 experiments running on 7 GPUs*
