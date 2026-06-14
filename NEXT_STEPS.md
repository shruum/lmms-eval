# Next Steps - Current Priorities and Action Items

**Last Updated:** 2026-06-11
**Purpose:** Clear roadmap of what to work on next - for multiple agents to work in parallel

---

## 🎯 **Current Priority: Reproduce VCD Paper**

### **Status Overview**
- ✅ VCD repo cloned to `/home/anna2/shruthi/VCD/`
- ⏳ Environment setup pending
- ⏳ Experiments not started

### **Why VCD First?**
1. **Best results** - +3.5% average improvement (highest among papers)
2. **Cleanest code** - repo is well-organized and documented
3. **Clear parameters** - documented in code and bash scripts
4. **All 9 splits** - paper reports on all POPE configurations
5. **Understand mechanism** - contrastive decoding might inform SRF improvements

---

## 📋 **Step-by-Step VCD Reproduction Plan**

### **Step 1: Setup VCD Environment** (Priority: HIGH)

**Location:** `/home/anna2/shruthi/VCD/experiments/`

**Commands:**
```bash
cd /home/anna2/shruthi/VCD/experiments
conda create -yn vcd python=3.9
conda activate vcd
pip install -r requirements.txt
```

**What to check:**
- Python version compatibility
- All dependencies installed successfully
- LLaVA-1.5-7B model location (check if already downloaded)

**Model location:** Check `./checkpoints/llava-v1.5-7b` or need to download

**Estimated time:** 10-15 minutes
**Blockers:** None (just conda setup)

---

### **Step 2: Verify Model Location** (Priority: HIGH)

**Check if LLaVA-1.5-7B exists:**
```bash
# Check VCD repo location
ls -la /home/anna2/shruthi/VCD/experiments/checkpoints/

# Check if we have it elsewhere
ls -la ~/.cache/huggingface/hub/ | grep llava-1.5-7b

# Or check lmms-eval location
ls -la /home/anna2/shruthi/lmms-eval/checkpoints/
```

**If model exists:** Create symlink or update model path
**If model doesn't exist:** Download from HuggingFace (instructions in VCD README)

**Estimated time:** 5 minutes
**Blockers:** Model download (if needed) - ~7GB

---

### **Step 3: Test VCD on One Split** (Priority: HIGH)

**Purpose:** Verify VCD works before running full evaluation

**Test on:** MSCOCO Random (smallest split, 3000 samples)

**Command:**
```bash
cd /home/anna2/shruthi/VCD/experiments
bash cd_scripts/llava1.5_pope.sh coco random 1 ./checkpoints/llava-v1.5-7b
```

**Parameters from bash script:**
- `cd_alpha`: 1.0 (from bash script, not code default of 0.5)
- `cd_beta`: 0.2 (from bash script, not code default of 0.1)
- `noise_step`: 500 (diffusion noise strength)

**Expected output:** Files in `./output/llava15_coco_pope_random_answers_no_cd_seed1.jsonl`

**Evaluate:**
```bash
python eval/eval_pope.py \
  --gt_files data/POPE/coco/coco_pope_random.json \
  --gen_files output/llava15_coco_pope_random_answers_no_cd_seed1.jsonl
```

**Expected accuracy:** ~85.4% (VCD paper) vs our baseline 83.43%

**Estimated time:** 15-20 minutes (3000 samples)
**Blockers:** GPU availability

---

### **Step 4: Run VCD on All 9 Configurations** (Priority: MEDIUM)

**Purpose:** Complete VCD reproduction for all POPE splits

**Datasets & Splits:**
1. MSCOCO (random, popular, adversarial)
2. A-OKVQA (random, popular, adversarial)
3. GQA (random, popular, adversarial)

**Commands:**
```bash
# COCO
bash cd_scripts/llava1.5_pope.sh coco random 1 ./checkpoints/llava-v1.5-7b
bash cd_scripts/llava1.5_pope.sh coco popular 1 ./checkpoints/llava-v1.5-7b
bash cd_scripts/llava1.5_pope.sh coco adversarial 1 ./checkpoints/llava-v1.5-7b

# A-OKVQA
bash cd_scripts/llava1.5_pope.sh aokvqa random 1 ./checkpoints/llava-v1.5-7b
bash cd_scripts/llava1.5_pope.sh aokvqa popular 1 ./checkpoints/llava-v1.5-7b
bash cd_scripts/llava1.5_pope.sh aokvqa adversarial 1 ./checkpoints/llava-v1.5-7b

# GQA
bash cd_scripts/llava1.5_pope.sh gqa random 1 ./checkpoints/llava-v1.5-7b
bash cd_scripts/llava1.5_pope.sh gqa popular 1 ./checkpoints/llava-v1.5-7b
bash cd_scripts/llava1.5_pope.sh gqa adversarial 1 ./checkpoints/llava-v1.5-7b
```

**Note:** Can run in parallel if multiple GPUs available

**Expected results:**
- COCO Random: 85.4% (vs baseline 83.43%)
- COCO Popular: 84.3% (vs baseline 81.23%)
- COCO Adversarial: 81.8% (vs baseline 79.30%)
- (See LITERATURE_REVIEW.md for all expected values)

**Estimated time:** 3-4 hours (9 splits × ~20 min each)
**Blockers:** GPU availability

---

### **Step 5: Compare with Paper** (Priority: MEDIUM)

**Purpose:** Verify reproduction quality

**Create comparison table:**
```python
# Python script to compare
import json

# Our results
our_results = {
    "coco_random": 85.4,  # from VCD output
    # ... all 9 splits
}

# Paper results
paper_results = {
    "coco_random": 85.4,
    "coco_popular": 84.3,
    # ... from LITERATURE_REVIEW.md
}

# Calculate differences
for split in our_results:
    delta = our_results[split] - paper_results[split]
    print(f"{split}: Our={our_results[split]}%, Paper={paper_results[split]}%, Delta={delta:.2f}%")
```

**Success criterion:** Within ±0.5% of paper on ≥7/9 splits

**If matches:** ✅ VCD reproduction successful, move to MemVR
**If doesn't match:** ⚠️ Debug parameters, decoding, data differences

**Estimated time:** 30 minutes
**Blockers:** None (just analysis)

---

## 🔄 **After VCD: Next Priorities**

### **Priority 2: Reproduce MemVR** (After VCD Complete)

**Status:** Not started - need to locate paper/repo

**Steps:**
1. Locate MemVR paper (ICML 2025)
2. Find official repo
3. Clone to `/home/anna2/shruthi/MemVR/`
4. Follow same process as VCD (setup → test → full run → compare)

**Expected improvement:** +1-3% over vanilla

**Estimated time:** 1-2 days (including paper analysis and setup)

---

### **Priority 3: Reproduce VAF** (After MemVR Complete)

**Status:** Not started - need to locate paper/repo

**Steps:**
1. Locate VAF/ClearSight paper (CVPR 2025)
2. Find official repo
3. Clone to `/home/anna2/shruthi/VAF/`
4. Follow same process as VCD

**Expected improvement:** +1-2% over vanilla

**Note:** Our tests with VAF parameters (α=0.15) showed -1.69% - need to understand why

**Estimated time:** 1-2 days

---

### **Priority 4: Run on Qwen2.5-VL-3B** (After All Papers Reproduced)

**Status:** Not started

**Purpose:** Verify methods work on different model

**Steps:**
1. Adapt VCD repo for Qwen2.5-VL-3B
2. Run all 9 splits
3. Compare with LLaVA-1.5-7B results
4. Check if SRF works better on Qwen (+0.68% on VLM Bias suggests it might)

**Estimated time:** 2-3 days

---

### **Priority 5: Improve SRF** (After Papers Reproduced and Understood)

**Status:** Not ready - need insights from paper methods first

**Current issues:**
- All 72 configs below baseline on LLaVA-1.5-7B
- Best config: -0.53% vs baseline
- Stronger alphas not working (α=1.0, 2.0, 4.0)
- CLIP saliency quality is good but doesn't help

**Possible approaches** (after understanding VCD/MemVR/VAF):
1. **Contrastive SRF** - use SRF-E (two-pass with noisy images)
2. **Output-level intervention** - like VCD, manipulate logits not attention
3. **Adaptive per-layer alpha** - different strength per layer
4. **Memory-guided SRF** - incorporate MemVR ideas
5. **Different saliency** - try internal attention instead of CLIP

**Estimated time:** 1-2 weeks (systematic exploration)

---

## 👥 **Multi-Agent Collaboration Strategy**

### **How Multiple Agents Can Work in Parallel**

**Agent 1: VCD Reproduction**
- Focus: Setup VCD environment and run experiments
- Timeline: Step 1-5 (this week)
- Deliverables: VCD results on all 9 splits

**Agent 2: MemVR Preparation**
- Focus: Locate MemVR paper and repo, analyze method
- Timeline: Start after Agent 1 completes Step 1
- Deliverables: MemVR repo cloned and ready to run

**Agent 3: VAF Preparation**
- Focus: Locate VAF paper and repo, analyze method
- Timeline: Start after Agent 2 finds MemVR
- Deliverables: VAF repo cloned and ready to run

**Agent 4: Analysis & Documentation**
- Focus: Compare results, update docs, generate tables
- Timeline: Ongoing, update as results come in
- Deliverables: Updated RESULTS_COMPENDIUM.md, comparison tables

### **Communication Protocol**

**Before starting work:**
1. Check NEXT_STEPS.md for current priorities
2. Check RESULTS_COMPENDIUM.md for what's done
3. Assign specific task to avoid duplication

**During work:**
1. Update progress in this file (NEXT_STEPS.md)
2. Add results to RESULTS_COMPENDIUM.md immediately
3. Note any issues or blockers

**After completing work:**
1. Update NEXT_STEPS.md (mark steps complete)
2. Update RESULTS_COMPENDIUM.md (add new results)
3. Document any deviations or issues

---

## 🚀 **Quick Start Commands**

### **For Agent Starting VCD Work:**
```bash
# 1. Setup environment
cd /home/anna2/shruthi/VCD/experiments
conda create -yn vcd python=3.9
conda activate vcd
pip install -r requirements.txt

# 2. Check model
ls checkpoints/llava-v1.5-7b

# 3. Test on one split
bash cd_scripts/llava1.5_pope.sh coco random 1 ./checkpoints/llava-v1.5-7b

# 4. Evaluate results
python eval/eval_pope.py \
  --gt_files data/POPE/coco/coco_pope_random.json \
  --gen_files output/llava15_coco_pope_random_answers_no_cd_seed1.jsonl
```

### **For Agent Analyzing Results:**
```bash
# Check all VCD outputs
ls /home/anna2/shruthi/VCD/experiments/output/

# Parse results
python3 -c "
import json
for split in ['coco_random', 'coco_popular', 'coco_adversarial']:
    # Parse and display results
    print(f'{split}: ...')
"

# Generate comparison table
python generate_vcd_comparison.py
```

---

## 📊 **Progress Tracking**

### **VCD Reproduction Progress**
- [ ] Step 1: Setup environment
- [ ] Step 2: Verify model location
- [ ] Step 3: Test on one split
- [ ] Step 4: Run all 9 configurations
- [ ] Step 5: Compare with paper

### **Overall Project Progress**
- [x] Baselines validated (May 2026)
- [x] SRF sweep completed (June 2026) - failed to improve
- [ ] VCD reproduced
- [ ] MemVR reproduced
- [ ] VAF reproduced
- [ ] All methods compared
- [ ] SRF improved based on insights
- [ ] Qwen2.5-VL-3B experiments

---

## 💡 **Key Success Metrics**

### **For VCD Reproduction**
- ✅ Environment setup successful
- ✅ One split test works
- ✅ All 9 splits run successfully
- ✅ Results match paper within ±0.5%

### **For Overall Project**
- ✅ All paper methods reproduced correctly
- ✅ Understanding of why they work
- ✅ SRF improved to ≥1% over baseline
- ✅ Results consistent across LLaVA and Qwen

---

## 🆘 **Blockers & Issues**

### **Current Blockers**
- None (just starting VCD work)

### **Potential Issues**
1. **Model download** - 7GB, might take time
2. **GPU availability** - need GPU for VCD experiments
3. **Parameter differences** - bash script vs code defaults
4. **Data format** - need to verify POPE data location

### **Solutions**
1. **Model download:** Use HuggingFace cache or create symlink
2. **GPU availability:** Use GPU monitoring, run when free
3. **Parameter differences:** Test both (code defaults and bash script)
4. **Data format:** Check VCD data/POPE/ directory structure

---

## 📞 **Questions & Answers**

### **Q: Why VCD first instead of MemVR or VAF?**
A: VCD has best results (+3.5%), cleanest code, and clear documentation. Good starting point to understand contrastive decoding which might inform SRF improvements.

### **Q: What if VCD results don't match paper?**
A: Debug parameters (try both code defaults and bash script values), check decoding method, verify data format. Document differences and move to MemVR.

### **Q: Can we run VCD and MemVR in parallel?**
A: Yes, if multiple agents available. Agent 1 focuses on VCD reproduction, Agent 2 locates and prepares MemVR repo.

### **Q: What if SRF still fails after understanding paper methods?**
A: Try hybrid approaches: contrastive SRF (SRF-E), output-level intervention, adaptive per-layer methods, or memory-guided SRF.

---

*For project context: see PROJECT_OVERVIEW.md*
*For code details: see CODE_GUIDE.md*
*For all results: see RESULTS_COMPENDIUM.md*
*For literature: see LITERATURE_REVIEW.md*
