# INFO Folder Contents — Quick Reference

## 📁 What's in the `/info` Folder

This folder contains essential documentation for understanding and reproducing the SRF hallucination mitigation research.

---

## 📚 File-by-File Guide

### **CONTEXT.md** 
**Purpose**: Complete SRF algorithm reference  
**Contains**:
- File structure and component map
- Hyperparameter system (3-tier config)
- Algorithm details (SRF base + SRF-E)
- CLI reference with all parameters
- Dataset specifications (MMVP, POPE, MME, VLM Bias)
- Environment setup instructions
- Related work comparison (VAF, AIR, VCD)

**When to read**: First time understanding the system

---

### **pope.md**
**Purpose**: POPE dataset deep-dive and evaluation structure  
**Contains**:
- What papers actually report (3 datasets × 3 splits = 9 combos)
- Table structure from papers (ClearSight, AIR, VCD)
- Annotation file sources and formats
- Image sources and verification
- Download instructions for remote servers
- Image extraction scripts for minimal download

**When to read**: Setting up POPE evaluation or comparing with papers

---

### **POPE_BASELINE_COMPARISON.md**
**Purpose**: Our baselines vs published paper results  
**Contains**:
- Corrected baseline results (using sampling decoding)
- Side-by-side comparison with VCD and AIR papers
- Dataset-by-dataset analysis (COCO, A-OKVQA, GQA)
- Why our baselines differ slightly from papers
- Verification that our evaluation is now correct

**When to read**: Validating results or comparing with published methods

---

### **POPE_LLAVA_GUIDE.md** ⭐ NEW
**Purpose**: Step-by-step guide for running POPE on LLaVA  
**Contains**:
- Complete environment setup
- Dataset download instructions (step-by-step)
- Running commands for all POPE splits
- Parameter explanations
- Troubleshooting common issues
- Current best results and status

**When to read**: Getting started with POPE + LLaVA evaluation

---

### **MMVP.pdf** & **Results_and_Findings.pdf**
**Purpose**: Research papers and result summaries  
**Contains**:
- MMVP benchmark paper
- Comprehensive results and findings from our experiments

**When to read**: Understanding benchmark details or analyzing research results

---

## 🚀 Quick Start Paths

### **Path 1: I'm New to SRF**
1. Read `info/CONTEXT.md` - Understand the system
2. Read `info/pope.md` - Understand the dataset
3. Use `info/POPE_LLAVA_GUIDE.md` - Run your first evaluation

### **Path 2: I Need to Run Experiments**
1. Read `info/POPE_LLAVA_GUIDE.md` - Setup and commands
2. Reference `info/CONTEXT.md` - Parameter details
3. Check `info/POPE_BASELINE_COMPARISON.md` - Expected results

### **Path 3: I'm Analyzing Results**
1. Check `info/POPE_BASELINE_COMPARISON.md` - See paper comparisons
2. Review `info/Results_and_Findings.pdf` - Detailed analysis
3. Reference `info/pope.md` - Understand dataset structure

---

## 📋 Missing Information?

### What We Have ✅
- ✅ Complete algorithm documentation
- ✅ Dataset structure and download instructions
- ✅ Step-by-step running guide for POPE + LLaVA
- ✅ Parameter explanations and CLI reference
- ✅ Baseline comparisons with papers
- ✅ Troubleshooting guide

### What Could Be Added ❓
- ❓ **Model-specific guides** for other VLMs (Qwen, etc.)
- ❓ **MMVP evaluation guide** (currently only POPE)
- ❓ **MME evaluation guide**
- ❓ **VLM Bias evaluation guide**
- ❓ **Automated setup scripts** for datasets
- ❓ **Performance benchmarking guide**

---

## 🎯 Most Common Questions

### **Q: How do I start evaluating POPE on LLaVA?**
**A**: Read `info/POPE_LLAVA_GUIDE.md` - it has everything you need from setup to running commands.

### **Q: Why don't my results match the papers?**
**A**: Check `info/POPE_BASELINE_COMPARISON.md` - you need to use sampling decoding (`--do_sample`).

### **Q: What parameters should I use?**
**A**: Reference `info/CONTEXT.md` for the hyperparameter system, or use the defaults in `info/POPE_LLAVA_GUIDE.md`.

### **Q: Where do I get the datasets?**
**A**: Follow the download instructions in `info/pope.md` or `info/POPE_LLAVA_GUIDE.md`.

### **Q: How does SRF actually work?**
**A**: Read `info/CONTEXT.md` sections on algorithm and CLI reference.

---

## 📊 Current Status Summary

**LLaVA-1.5-7B + POPE Status** (May 2026):
- ✅ Environment setup: Working
- ✅ Dataset download: Documented  
- ✅ Running experiments: Documented
- ❌ SRF improvement: Failed to beat baseline
- ❓ Next steps: Under investigation

**Best Results**:
- COCO Adversarial: 78.77% vs baseline 79.30% (-0.53% gap)
- GQA Adversarial: 69.43% vs baseline 75.50% (-6.07% gap)

---

## 💡 Tips for Using This Folder

1. **Start with the guides**: `POPE_LLAVA_GUIDE.md` is the most practical
2. **Reference CONTEXT.md**: When you need detailed parameter info
3. **Check pope.md**: When setting up datasets or comparing with papers
4. **Use POPE_BASELINE_COMPARISON.md**: When validating your results

---

## 🔄 Keeping This Folder Updated

**When to add new files**:
- New evaluation guides for other datasets (MMVP, MME, VLM Bias)
- Model-specific guides (Qwen, other LLaVA versions)
- Automated setup scripts
- Performance optimization guides

**When to update existing files**:
- New baseline results available
- New parameters discovered
- Better download methods found
- Troubleshooting sections expanded

---

*Last updated: May 2026 - Info folder documentation complete*
*For the latest project status, check root-level files like `SRF_EXPERIMENT_STATUS.md`*