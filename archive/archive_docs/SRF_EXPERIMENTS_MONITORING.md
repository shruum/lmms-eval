# SRF Experiments Monitoring & Status

**Last Updated:** 2026-06-01  
**Current Focus:** Saliency Visualization for Negative/Positive Detection  
**Status:** 🔄 **Visualization Bug Fix Needed**

---

## 🎯 **Current Task: Visualizing Saliency Maps**

### **Goal:**
Understand which metrics best distinguish **present vs absent objects** to guide adaptive SRF intervention.

### **Why This Matters:**
- SRF performs poorly on **negative cases** (when query objects don't exist in images)
- We have CLIP saliency maps that show where query nouns are detected
- Need to identify which saliency metrics reliably signal absence → use for adaptive intervention

### **What We're Doing:**
Creating comprehensive visualizations showing:
- Original images with CLIP saliency maps (6x6 grid)
- Threshold maps at different levels (0.2, 0.3, 0.5)
- All discriminative metrics with classification predictions
- Visual interpretation and insights

---

## 📊 **Current Blocking Issue**

### **Problem:** 
`visualize_saliency_metrics.py` fails with `KeyError: 'present'`

### **Root Cause:**
```python
# Line 262 - WRONG:
for case_type in ['present', 'absent']:
    cases = validation_set[case_type]

# Should be (matching JSON structure):
for case_type in ['present', 'absent']:
    cases = validation_set['positive'] if case_type == 'present' else validation_set['negative']
```

### **Fix Required:**
Change lines 262-263 in `visualize_saliency_metrics.py`:
```python
# OLD (buggy):
cases = validation_set[case_type]

# NEW (correct):
cases = validation_set['positive'] if case_type == 'present' else validation_set['negative']
```

---

## 📁 **Key Files & Their Purposes**

| File | Purpose | Status |
|------|---------|--------|
| `balanced_validation_set.json` | 12 samples (6 present + 6 absent) from all POPE categories | ✅ Complete |
| `visualize_saliency_metrics.py` | Creates comprehensive saliency visualizations | ❌ Bug needs fix |
| `comprehensive_analysis.py` | Systematic metric analysis (Cohen's d, classification accuracy) | ✅ Complete |
| `test_better_metric.py` | Debugging script for metric testing | ✅ Complete |

---

## 🔬 **Key Findings from Analysis**

### **Best Discriminators** (Cohen's d effect size):
1. **num_above_03**: d=0.99, 70.8% accuracy (count of patches > 0.3 threshold)
2. **clip_std_sim**: d=0.83, 70.8% accuracy (std of CLIP saliency)
3. **entropy**: d=0.80, 70.8% accuracy (entropy of saliency distribution)
4. **clip_max_sim**: d=0.74, 75% accuracy (max CLIP similarity)

### **Classification Thresholds:**
- **num_above_03**: <22 → present, >28 → absent, 22-28 → uncertain
- **clip_std_sim**: >0.245 → present (high variance), <0.235 → absent (low variance)
- **entropy**: <3.44 → present (low entropy), >3.48 → absent (high entropy)
- **clip_max_sim**: >0.24 → present (simple threshold, 75% accuracy)

### **Key Insight:**
All metrics show **significant overlap** (54-100%) between present/absent cases, making perfect separation impossible. **CLIP max_sim provides best balanced performance** (75% accuracy overall).

---

## 🎯 **Next Steps**

1. **Fix visualization script** (5 min)
   - Change line 262-263 in `visualize_saliency_metrics.py`
   - Run: `python visualize_saliency_metrics.py`

2. **Review visualizations** (15 min)
   - Check output: `saliency_visualizations/` directory
   - 12 files total (6 present + 6 absent)
   - Look for patterns: do metrics align with visual intuition?

3. **Select criterion for adaptive SRF** (10 min)
   - Decide: single metric (CLIP max_sim) or combination
   - Consider category-specific thresholds (random vs adversarial)
   - Plan intervention strategy (reduce alpha vs reverse attention)

4. **Implement adaptive SRF** (30 min)
   - Modify `srf/srf.py` to use selected metric
   - Add per-sample intervention logic
   - Test on balanced validation set

---

## 🔧 **How to Use This Context in Another Claude Window**

### **Option 1: Copy the key files**
```bash
# In new window, copy the essential files:
cd /home/anna2/shruthi/lmms-eval
cat SRF_EXPERIMENTS_MONITORING.md
cat balanced_validation_set.json
head -50 visualize_saliency_metrics.py
head -100 comprehensive_analysis.py
```

### **Option 2: Quick context loading**
```bash
# In new Claude window, ask:
"Read /home/anna2/shruthi/lmms-eval/SRF_EXPERIMENTS_MONITORING.md and help me fix the visualization script bug"
```

### **Option 3: Direct bug fix**
```bash
# In new window, simply:
cd /home/anna2/shruthi/lmms-eval
# Fix line 262-263 in visualize_saliency_metrics.py
# Then run: python visualize_saliency_metrics.py
```

---

## 📝 **Quick Reference: Balanced Validation Set Structure**

```json
{
  "validation_set": {
    "positive": [
      {"question": "Is there a snowboard...", "answer": "yes", "category": "random"},
      {"question": "Is there a person...", "answer": "yes", "category": "random"},
      {"question": "Is there a snowboard...", "answer": "yes", "category": "popular"},
      {"question": "Is there a person...", "answer": "yes", "category": "popular"},
      {"question": "Is there a snowboard...", "answer": "yes", "category": "adversarial"},
      {"question": "Is there a person...", "answer": "yes", "category": "adversarial"}
    ],
    "negative": [
      {"question": "Is there a car...", "answer": "no", "category": "random"},
      {"question": "Is there a sandwich...", "answer": "no", "category": "random"},
      {"question": "Is there a dining table...", "answer": "no", "category": "popular"},
      {"question": "Is there a car...", "answer": "no", "category": "popular"},
      {"question": "Is there a backpack...", "answer": "no", "category": "adversarial"},
      {"question": "Is there a car...", "answer": "no", "category": "adversarial"}
    ]
  }
}
```

**Total:** 12 samples (2 present + 2 absent per category × 3 categories)

---

## 🔗 **Related Documentation**

- `POPE_BASELINE_COMPARISON.md` - Correct baselines vs papers
- `SRF_TARGET_OBJECTIVES.md` - Success criteria and targets  
- `SRF_EXPERIMENT_STATUS.md` - Hyperparameter sweep status
- `srf/CONTEXT.md` - SRF algorithm details

---

*This document provides concise context for continuing work in another Claude window. Focus: Fix visualization bug → Review saliency patterns → Implement adaptive SRF.*
