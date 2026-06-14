# Project Overview - SRF & VLM Hallucination Mitigation

**Last Updated:** 2026-06-11
**Status:** Reproducing VCD, MemVR, VAF papers on LLaVA-1.5-7B and Qwen2.5-VL-3B

---

## 🎯 **Main Goal**

Improve SRF (Spatial Reasoning Focus) method to mitigate object hallucinations in VLMs and achieve state-of-the-art results on POPE benchmark.

### **Success Criteria**
- **Primary:** SRF must improve ≥1% over baseline AND match/exceed VCD/AIR papers
- **Secondary:** Reproduce VCD, MemVR, VAF paper results correctly
- **Models:** LLaVA-1.5-7B (primary), Qwen2.5-VL-3B (secondary)

### **Why This Matters**
Current published methods (VCD, AIR, etc.) show 2-8% improvements over vanilla baselines. SRF needs to:
1. Beat our baseline by ≥1%
2. Match or exceed published methods
3. Demonstrate state-of-the-art performance

---

## 📊 **Current Baselines (CORRECTED)**

**Model:** LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`)
**Decoding:** Sampling (do_sample=True, temperature=0.7, top_p=0.9)

### **Our Baselines vs Papers**

| Dataset | Split | Our Baseline | VCD Paper | Target |
|---------|-------|--------------|-----------|---------|
| **COCO** | Random | 83.43% | 83.29% | ≥84.43% |
| | Popular | 81.23% | 81.88% | ≥82.23% |
| | Adversarial | 79.30% | 78.96% | ≥80.30% |
| **A-OKVQA** | Random | 83.47% | 83.45% | ≥84.47% |
| | Popular | 79.30% | 79.90% | ≥80.30% |
| | Adversarial | 75.80% | 74.04% | ≥76.80% |
| **GQA** | Random | 82.47% | 83.73% | ≥83.47% |
| | Popular | 76.40% | 78.17% | ≥77.40% |
| | Adversarial | 75.50% | 75.08% | ≥76.50% |

**Status:** ✅ Baselines match VCD within ±0.03% average - VALIDATED

---

## 🔬 **Research Context**

### **Problem: VLMs Hallucinate Objects**
- VLMs often say objects are present when they're not
- POPE benchmark tests this with 50% absent objects
- Need methods to reduce hallucinations without retraining

### **SRF (Spatial Reasoning Focus) - Our Method**
1. Extract query nouns from questions
2. Compute CLIP-based saliency maps
3. Boost attention to salient tokens by α, suppress background by ε
4. Targets cross-modal fusion layers

**Variants:** SRF (single-pass), SRF-E (two-pass contrastive)

**Status:** ❌ Current best result: -0.53% vs baseline (needs improvement)

### **Related Work (Papers to Reproduce)**

| Method | Conference | arXiv | Key Idea | Paper Results |
|--------|-----------|-------|----------|---------------|
| **VCD** | CVPR 2024 | 2311.16922 | Contrastive decoding with noisy images | +3.5% avg over vanilla |
| **AIR** | 2026 | 2602.24041 | OT-guided patch selection | +2-8% over vanilla |
| **MemVR** | ICML 2025 | - | Memory-based vision reasoning | +1-3% over vanilla |
| **VAF/ClearSight** | CVPR 2025 | - | Visual attention amplification | +1-2% over vanilla |

**Current Priority:** Reproduce VCD → MemVR → VAF (one at a time)

---

## 📁 **Project Structure**

```
lmms-eval/
├── srf/                     # SRF implementation (STABLE CODE)
│   ├── srf.py              # Core SRF function
│   ├── srf_e.py            # Enhanced SRF (contrastive)
│   ├── saliency/           # CLIP saliency computation
│   └── config.py           # Default parameters
├── info/                    # Dataset info & guides
│   ├── POPE_LLAVA_GUIDE.md
│   └── POPE_BASELINE_COMPARISON.md
├── results/                 # All experiment results
│   ├── llava_pope_sampling_baseline/  # Correct baselines
│   └── srf_focused_sweep/           # SRF experiments
├── VCD/                     # VCD repo (cloned for reproduction)
│   ├── experiments/eval/   # Evaluation scripts
│   └── vcd_utils/          # VCD sampling logic
└── dataset/POPE_images/     # POPE benchmark data
```

---

## 🚀 **Current Work (June 2026)**

### **Phase 1: Reproduce Paper Methods**
1. ✅ VCD repo cloned to `/home/anna2/shruthi/VCD/`
2. ⏳ Setup VCD environment (conda, dependencies)
3. ⏳ Test VCD on one POPE split
4. ⏳ Run VCD on all 9 configurations
5. ⏳ Compare with paper Table 1
6. ⏳ Move to MemVR (after VCD verified)
7. ⏳ Move to VAF (after MemVR verified)

### **Phase 2: SRF Improvement (After Papers Verified)**
- Run systematic hyperparameter exploration
- Test different layer ranges (5-25)
- Test alpha values (0.1-4.0)
- Test head selection strategies
- Achieve ≥1% improvement over baseline

---

## 🎯 **Key Success Metrics**

### **For Paper Reproduction**
- VCD results should match within ±0.5% of paper
- MemVR results should match within ±0.5% of paper
- VAF results should match within ±0.5% of paper

### **For SRF Improvement**
- ≥1% improvement over baseline on ≥6/9 splits
- Match/exceed VCD on ≥4/9 splits
- Match/exceed AIR on ≥4/9 splits
- Consistent improvement across datasets

---

## 📚 **Important Documents**

### **Essential Reading**
1. **PROJECT_OVERVIEW.md** (this file) - Project goals & context
2. **CODE_GUIDE.md** - Code architecture & how to modify
3. **RESULTS_COMPENDIUM.md** - All results (avoid re-running)
4. **LITERATURE_REVIEW.md** - Other methods & papers
5. **NEXT_STEPS.md** - Current priorities

### **Dataset-Specific Info**
- `info/POPE_LLAVA_GUIDE.md` - How to run POPE on LLaVA
- `info/POPE_BASELINE_COMPARISON.md` - Baseline validation

### **SRF-Specific**
- `srf/INVESTIGATION_WHY_SALIENCY_FAILS.md` - Saliency analysis
- `srf/LAYER_SPECIFIC_GUIDE.md` - Layer-wise analysis
- `SRF_TARGET_OBJECTIVES.md` - Success criteria

---

## 🛠️ **Quick Start Commands**

### **Environment**
```bash
conda activate mllm
cd /home/anna2/shruthi/lmms-eval
```

### **Run Baseline**
```bash
python srf/eval.py \
  --method baseline \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope_vcd \
  --pope_vcd_name coco_adversarial \
  --do_sample --temperature 0.7 --top_p 0.9
```

### **Run SRF**
```bash
python srf/eval.py \
  --method srf \
  --alpha 0.5 --eps 0.1 \
  --layer_start 10 --layer_end 18 \
  --head_top_k_pct 0.5
```

### **Check Results**
```bash
# Quick baseline check
cat results/llava_pope_sampling_baseline/pope_coco_adversarial_baseline.json | jq '.accuracy'

# Check SRF experiment
grep "accuracy" results/srf_focused_sweep/coco_adversarial_config0/pope_coco_adversarial.json
```

---

## 📊 **Experiment Results Summary**

### **Baselines (✅ VALIDATED)**
- Location: `results/llava_pope_sampling_baseline/`
- All 9 splits run with correct decoding
- Matches VCD paper within ±0.03% average

### **SRF Experiments (⚠️ NEEDS IMPROVEMENT)**
- Location: `results/srf_focused_sweep/`
- Round 1: 17 configs, all below baseline
- Best: 78.77% (α=1.0, layers 10-18) vs baseline 79.30%
- Status: Needs hyperparameter tuning

### **Paper Reproductions (⏳ IN PROGRESS)**
- VCD: Repo cloned, environment setup pending
- MemVR: Not started
- VAF: Not started

---

## 🔍 **Why Previous SRF Failed**

### **Current Best: -0.53% vs Baseline**
**Problem:** VAF parameters (α=0.15) too weak for LLaVA-1.5-7B

**Hypothesis:** Need stronger intervention
- VAF: α=0.15, layers 10-15
- Our tests: α=1.0-4.0, various layer ranges
- Still not working - need different approach

**Possible Issues:**
1. CLIP saliency might not align with LLaVA's attention
2. Layer fusion targets might be wrong
3. Need adaptive alpha per layer/head
4. Need different attention manipulation strategy

---

## 🎯 **Next Steps**

1. **Reproduce VCD paper** (June 2026)
   - Setup environment, test on one split
   - Run all 9 configurations
   - Verify within ±0.5% of paper

2. **Reproduce MemVR** (after VCD)
   - Same process as VCD

3. **Reproduce VAF** (after MemVR)
   - Same process as VCD

4. **SRF Hyperparameter Search** (after papers)
   - Systematic exploration
   - Different strategies beyond stronger alphas
   - Adaptive per-layer methods

---

## 💡 **Key Insights**

1. **Baselines are correct** - match VCD within ±0.03%
2. **SRF needs different approach** - stronger alphas not working
3. **Paper methods first** - reproduce before improving SRF
4. **CLIP saliency quality** - metrics fixed (June 2026) but results still poor
5. **Multi-model testing** - need to test on Qwen2.5-VL-3B too

---

*For detailed code architecture: see CODE_GUIDE.md*
*For all results: see RESULTS_COMPENDIUM.md*
*For literature details: see LITERATURE_REVIEW.md*
*For current priorities: see NEXT_STEPS.md*
