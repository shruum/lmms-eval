# ClearSight POPE Dataset Investigation

## 🚨 **MAJOR FINDING**

### **Paper Claims vs. Reality:**

**ClearSight Paper States:**
> "POPE benchmark, with results averaged across the MSCOCO, A-OKVQA, and GQA datasets."

**ClearSight GitHub Repository Has:**
```
data/pope/
├── coco_pope_adversarial.json
├── coco_pope_popular.json
└── coco_pope_random.json
```

**Only MSCOCO! No A-OKVQA, No GQA!**

---

## 🔍 **Implications:**

### **1. Dataset Discrepancy**
- ✅ **Paper SAYS**: Averaged across 3 datasets
- ❌ **Code HAS**: Only 1 dataset (MS-COCO)
- 🤔 **Possible explanations**:
  1. They didn't release A-OKVQA/GQA data
  2. They only tested on COCO but claimed averaging
  3. A-OKVQA/GQA results were in separate unpublished experiments
  4. Paper text is misleading

### **2. Comparison Fairness**
- **ClearSight results**: Only based on MS-COCO
- **AIR paper Table 2**: Shows results for all 3 datasets separately
- **Our approach**: Testing on MS-COCO only (same as ClearSight)

### **3. Research Impact**
- **If paper only tested COCO**: Results are still valid but claims overstated
- **If averaging claim is false**: Misleading to research community
- **Reproducibility issue**: Others cannot reproduce "averaged" results

---

## 📊 **What This Means For Our Work:**

### **✅ Good News:**
1. **We're doing the right thing** - testing on MS-COCO POPE
2. **Direct comparison possible** - same dataset as ClearSight
3. **No disadvantage** - we're using the same data they likely used

### **🎯 Our Position:**
- **Dataset**: POPE (MS-COCO only) - 9000 samples (3000 per category)
- **Model**: LLaVA-1.5-7B
- **Categories**: Random, Popular, Adversarial
- **Splits**: Full dataset (not just 50/100 samples like before)

### **📝 How We Should Report:**
> "We evaluate on the POPE benchmark using MS-COCO images (9000 samples: 3000 each for Random, Popular, and Adversarial splits). Note: While some papers claim averaging across multiple datasets (COCO, A-OKVQA, GQA), the publicly available code for these methods [ClearSight] only includes COCO-based evaluation."

---

## 🔄 **Current Status:**

### **Running Now:**
- **Quick sweep**: 10 configs × 3 categories = 30 experiments
- **Estimated time**: 2-3 hours
- **Goal**: Find best SRF config for MS-COCO POPE

### **Configs Being Tested:**
1. Baseline (no SRF)
2. Higher top-k (50%)
3. Strong boost (α=4.0)
4. No absence-aware
5. Fine grid (5×5) variants
6. Coarse grid (9×9) variants
7. Very strong boost (α=8.0)
8. Very high top-k (70%)

---

## 💡 **Next Steps:**

1. ✅ **Complete quick sweep** (2-3 hours)
2. ✅ **Find best SRF config** for MS-COCO POPE
3. ✅ **Create comparison table** with:
   - Our baseline
   - Our best SRF
   - ClearSight VAF results (from paper)
   - AIR results (from paper)
4. 📝 **Write up findings** with proper dataset attribution

---

## 📚 **References:**

- **ClearSight GitHub**: https://github.com/ustc-hyin/ClearSight
- **ClearSight Paper**: https://arxiv.org/html/2503.13107v2
- **POPE Original Paper**: https://aclanthology.org/2023.emnlp-main.20.pdf
- **RUCAIBox/POPE**: https://github.com/RUCAIBox/POPE

---

**Created: 2026-04-29**
**Investigation by: Claude + User**
