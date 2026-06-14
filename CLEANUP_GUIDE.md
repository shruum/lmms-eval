# Directory Cleanup - What Changed and How to Use

**Date:** 2026-06-11
**Cleanup:** Moved 89 files from root to organized folders

---

## ✅ **BEFORE vs AFTER**

### **BEFORE:**
- ❌ 89 files in root directory (scripts + docs mixed together)
- ❌ Hard to find anything
- ❌ No organization

### **AFTER:**
- ✅ 15 .md files in root (essential docs only)
- ✅ 1 .py file in root (setup.py - standard project file)
- ✅ 0 .sh files in root (all in folders)
- ✅ Organized folders for different purposes

---

## 📂 **New Directory Structure**

```
lmms-eval/
├── ROOT DIRECTORY (Essential Docs Only)
│   ├── README.md                          # Main project README
│   ├── CLAUDE.md                          # Claude Code guidelines
│   ├── PROJECT_OVERVIEW.md               # START HERE - Project goals
│   ├── CODE_GUIDE.md                      # How code works
│   ├── RESULTS_COMPENDIUM.md             # All results
│   ├── LITERATURE_REVIEW.md              # Paper methods
│   ├── NEXT_STEPS.md                     # Current priorities
│   ├── MULTI_AGENT_GUIDE.md              # How to collaborate
│   ├── DOCUMENTATION_CLEANUP_SUMMARY.md  # What was done
│   ├── PROJECT_STRUCTURE.md              # Proposed structure
│   └── [Standard project files: AGENTS.md, CHANGELOG.md, etc.]
│
├── scripts/                             # All utility scripts
│   ├── analysis/                        # Analysis scripts (8 files)
│   │   ├── analyze_saliency_metrics.py
│   │   ├── comprehensive_analysis.py
│   │   └── ...
│   ├── debug/                           # Debug scripts (5 files)
│   │   ├── debug_entropy.py
│   │   ├── debug_saliency_detailed.py
│   │   └── ...
│   ├── experiment_scripts/              # Launch experiments (11 files)
│   │   ├── launch_srf_exploration.sh
│   │   ├── monitor_and_launch.sh
│   │   └── ...
│   ├── monitoring/                      # Monitoring scripts (2 files)
│   │   ├── adaptive_scheduler.py
│   │   └── monitor_adaptive_sweep.py
│   ├── sweep/                           # Sweep scripts (3 files)
│   │   ├── srf_hyperparameter_sweep.py
│   │   └── ...
│   ├── test/                            # Test scripts (17 files)
│   │   ├── test_clip_guided_srf.py
│   │   ├── test_pope_quick.py
│   │   └── ...
│   └── visualization/                  # Visualization scripts (2 files)
│       ├── visualize_saliency_vitl14.py
│       └── ...
│
├── archive/                             # Old archived files
│   ├── archive_docs/                    # Old documentation (moved)
│   └── archive_scripts/                 # Old scripts (if any)
│
├── srf/                                 # SRF implementation (core code)
├── info/                                # Dataset guides
├── results/                             # Experiment results
└── dataset/                             # Data files
```

---

## 📚 **Essential Docs (Root Directory)**

### **Start Here:**
1. **README.md** - Main project README
2. **PROJECT_OVERVIEW.md** - Project goals and context
3. **NEXT_STEPS.md** - Current priorities (VCD reproduction)

### **Reference Docs:**
4. **CODE_GUIDE.md** - Code architecture and how to modify
5. **RESULTS_COMPENDIUM.md** - All experiment results
6. **LITERATURE_REVIEW.md** - Paper methods (VCD, MemVR, VAF)
7. **MULTI_AGENT_GUIDE.md** - How to collaborate with multiple agents

### **Meta Docs:**
8. **DOCUMENTATION_CLEANUP_SUMMARY.md** - What was cleaned up
9. **PROJECT_STRUCTURE.md** - Proposed structure

### **Standard Project Files:**
10. **CLAUDE.md** - Claude Code guidelines
11. **AGENTS.md** - Agent guidelines
12. **CHANGELOG.md** - Project changelog
13. **CODE_OF_CONDUCT.md** - Code of conduct
14. **CONTRIBUTING.md** - Contribution guidelines
15. **SECURITY.md** - Security policy

---

## 🚀 **How to Use the New Structure**

### **Running Scripts**

**Analysis scripts:**
```bash
python scripts/analysis/comprehensive_analysis.py
python scripts/analysis/generate_saliency_report.py
```

**Debug scripts:**
```bash
python scripts/debug/debug_saliency_detailed.py
python scripts/debug/debug_entropy.py
```

**Experiment scripts:**
```bash
bash scripts/experiment_scripts/launch_srf_exploration.sh
bash scripts/experiment_scripts/monitor_and_launch.sh
```

**Monitoring scripts:**
```bash
python scripts/monitoring/adaptive_scheduler.py
```

**Test scripts:**
```bash
python scripts/test/test_clip_guided_srf.py
python scripts/test/test_pope_quick.py
```

**Visualization scripts:**
```bash
python scripts/visualization/visualize_saliency_vitl14.py
```

### **Finding Scripts**

```bash
# List all analysis scripts
ls scripts/analysis/

# Find specific script
find scripts/ -name "*saliency*"

# List all experiment scripts
ls scripts/experiment_scripts/
```

---

## 📂 **Archive Folder**

**Old documentation** moved to `archive/archive_docs/`:
- SRF_EXPERIMENT_STATUS.md
- SRF_TARGET_OBJECTIVES.md
- SRF_DEBUG_CONTEXT.md
- POPE_COMPLETE_RESULTS_TABLE.md
- GQA_DECODING_*.md
- LAYERWISE_HEAD_SELECTION_PLAN.md
- CLEARSIGHT_FINDINGS.md
- PHASE2_AND_COMBINED_READY.md
- MMVP_PHASE1_FINAL_REPORT.md
- POPE_PROMPT_TEST_RESULTS.md
- Next_steps.md (old version)
- SRF_POPE_EXPERIMENTS.md
- SRF_FINAL_RESULTS.md
- SRF_DEBUGGING_REPORT.md
- SRF_EXPERIMENTS_MONITORING.md
- CLEANUP_PLAN.md
- CLEANUP_QUICK_REFERENCE.md
- (20+ other old docs)

**If you need old content:**
```bash
ls archive/archive_docs/
cat archive/archive_docs/SRF_EXPERIMENT_STATUS.md
```

---

## 💡 **Benefits of Cleanup**

### **Before Cleanup:**
- ❌ 89 files in root (impossible to navigate)
- ❌ Scripts mixed with docs
- ❌ No idea where to find anything
- ❌ Difficult to maintain

### **After Cleanup:**
- ✅ 15 docs in root (easy to find)
- ✅ Scripts organized by purpose
- ✅ Clear file locations
- ✅ Easy to maintain and extend

---

## 🎯 **Quick Reference**

| I need to... | Location |
|--------------|----------|
| Run experiments | `scripts/experiment_scripts/` |
| Analyze results | `scripts/analysis/` |
| Debug issues | `scripts/debug/` |
| Test code | `scripts/test/` |
| Visualize data | `scripts/visualization/` |
| Monitor sweeps | `scripts/monitoring/` |
| Find project goals | `PROJECT_OVERVIEW.md` |
| See current priorities | `NEXT_STEPS.md` |
| Check what's been run | `RESULTS_COMPENDIUM.md` |
| Learn paper methods | `LITERATURE_REVIEW.md` |
| Find old docs | `archive/archive_docs/` |

---

## 🔍 **Finding Files**

### **Search for specific script:**
```bash
# Find saliency-related scripts
find scripts/ -name "*saliency*"

# Find experiment scripts
ls scripts/experiment_scripts/

# Find test scripts
ls scripts/test/
```

### **Search for specific content:**
```bash
# Search for "entropy" across all scripts
grep -r "entropy" scripts/

# Search for "launch" in experiment scripts
grep "launch" scripts/experiment_scripts/*.sh
```

---

## 📊 **Summary**

### **Files Moved:**
- **Experiment scripts:** 11 files → `scripts/experiment_scripts/`
- **Analysis scripts:** 8 files → `scripts/analysis/`
- **Debug scripts:** 5 files → `scripts/debug/`
- **Test scripts:** 17 files → `scripts/test/`
- **Visualization scripts:** 2 files → `scripts/visualization/`
- **Monitoring scripts:** 2 files → `scripts/monitoring/`
- **Sweep scripts:** 3 files → `scripts/sweep/`
- **Old documentation:** 40+ files → `archive/archive_docs/`

### **Files Remaining in Root:**
- **Essential docs:** 7 files (PROJECT_OVERVIEW, CODE_GUIDE, etc.)
- **Standard project files:** 8 files (README, CLAUDE, etc.)
- **setup.py:** 1 file (standard Python project file)

**Total:** 15 .md files + 1 .py file = 16 files in root (down from 89!)

---

## 🚀 **Next Steps**

### **Immediate:**
1. ✅ Cleanup complete - start using new structure
2. ✅ Update any scripts that reference old file locations
3. ✅ Point all agents to `PROJECT_OVERVIEW.md` first

### **Ongoing:**
1. Keep new scripts in appropriate folders
2. Keep essential docs at root
3. Archive old docs when no longer needed
4. Maintain clean structure

---

## 📞 **Questions**

**Q: Where did my old script go?**
A: Check `scripts/` folder - organized by purpose (analysis, debug, test, etc.)

**Q: Where did old documentation go?**
A: Moved to `archive/archive_docs/` - can still access if needed

**Q: Where do I put new scripts?**
A: Put in appropriate `scripts/` subfolder (analysis, debug, test, etc.)

**Q: Which docs should I read first?**
A: Start with `PROJECT_OVERVIEW.md` (3 min), then `NEXT_STEPS.md` (2 min)

**Q: How do I run experiments now?**
A: Scripts are in `scripts/experiment_scripts/` - use `bash scripts/experiment_scripts/launch_*.sh`

---

*Directory cleaned: 2026-06-11*
*Files organized: 89 → 16 in root*
*For questions: See PROJECT_OVERVIEW.md or MULTI_AGENT_GUIDE.md*
