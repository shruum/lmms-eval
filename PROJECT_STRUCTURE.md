# Project Documentation Structure

**Last Updated:** 2025-06-15

## **Core Documentation Files** (For Multi-Agent Collaboration)

### **Essential Reading** (Start Here)
1. **PROJECT_OVERVIEW.md** - Project goals, context, success criteria
2. **CODE_GUIDE.md** - Code architecture, important files, how to modify
3. **RESULTS_COMPENDIUM.md** - All results in one place (to avoid re-running)
4. **LITERATURE_REVIEW.md** - Other methods (VCD, MemVR, VAF, etc.)
5. **NEXT_STEPS.md** - Current priorities and what to work on next

### **Archive** (Historical, rarely needed)
- `docs_status/` - Old experiment status files
- `docs_analysis/` - Investigation reports
- `srf/autoresearch_loop/` - Archived research loop docs

---

## **File Organization Strategy**

### **Root Level** (High Visibility)
```
lmms-eval/
├── PROJECT_OVERVIEW.md      # START HERE - Project goals & context
├── CODE_GUIDE.md             # How code works, what to modify
├── RESULTS_COMPENDIUM.md    # All results (avoid re-running)
├── LITERATURE_REVIEW.md     # Other methods & papers
├── NEXT_STEPS.md            # Current priorities
└── CLAUDE.md                # Claude Code instructions
```

### **Task-Specific Folders** (Focused Work)
```
lmms-eval/
├── srf/                     # SRF method implementation
│   ├── srf.py, srf_e.py    # Core code (STABLE)
│   ├── eval.py             # Main evaluation script (STABLE)
│   ├── saliency/           # CLIP saliency computation
│   ├── methods/            # Method implementations
│   │   └── vaf.py         # VAF implementation
│   ├── README.md            # SRF-specific docs
│   └── investigations/      # SRF-specific analysis
├── scripts/                 # All scripts organized by type
│   ├── session_scripts/    # Session-specific validation scripts
│   ├── experiment_scripts/ # Method evaluation scripts
│   └── analysis/           # Analysis and comparison scripts
├── data/                    # Dataset configurations and metadata
│   └── validation_sets/    # Validation set configurations
├── results/                 # All experiment results
│   ├── session_logs/       # Timestamped session results
│   └── saliency_analysis/  # Saliency analysis outputs
└── info/                    # Dataset info & guides
    ├── POPE_LLAVA_GUIDE.md
    └── POPE_BASELINE_COMPARISON.md
```

### **Archive** (Historical)
```
lmms-eval/
├── archive/                 # Old consolidated docs
│   ├── old_srf_experiments.md
│   ├── old_investigations.md
│   └── old_status_reports.md
└── results_archive/        # Old experiment results
```

---

## **Multi-Agent Collaboration Strategy**

### **How Different Agents Should Use These Docs**

**Agent 1: New to Project**
1. Read PROJECT_OVERVIEW.md (5 min)
2. Read CODE_GUIDE.md (10 min)
3. Check NEXT_STEPS.md for what to work on

**Agent 2: Running Experiments**
1. Read RESULTS_COMPENDIUM.md (don't re-run!)
2. Read CODE_GUIDE.md → "Experiment Scripts" section
3. Check NEXT_STEPS.md → "Current Priorities"

**Agent 3: Literature Review**
1. Read LITERATURE_REVIEW.md
2. Read PROJECT_OVERVIEW.md → "Related Work"
3. Add findings to LITERATURE_REVIEW.md

**Agent 4: Debugging Issues**
1. Read CODE_GUIDE.md → "Debugging" section
2. Check RESULTS_COMPENDIUM.md → "Failed Experiments"
3. Archive findings in appropriate folder

---

## **Document Maintenance Rules**

1. **UPDATE RESULTS_COMPENDIUM IMMEDIATELY** after any experiment
2. **KEEP ESSENTIAL 5 FILES UP TO DATE** - they're the single source of truth
3. **ARCHIVE OLD FILES** - don't delete, move to archive/
4. **ONE PLACE FOR RESULTS** - RESULTS_COMPENDIUM.md only
5. **CHECK FOR DUPLICATES** before creating new docs

---

## **File Cleanup Plan**

### **Phase 1: Create Essential 5** (Do this first)
- [ ] PROJECT_OVERVIEW.md
- [ ] CODE_GUIDE.md
- [ ] RESULTS_COMPENDIUM.md
- [ ] LITERATURE_REVIEW.md
- [ ] NEXT_STEPS.md

### **Phase 2: Consolidate & Archive**
- [ ] Merge all SRF experiment status → RESULTS_COMPENDIUM.md
- [ ] Merge all literature info → LITERATURE_REVIEW.md
- [ ] Move old status docs to archive/
- [ ] Move investigation docs to srf/investigations/

### **Phase 3: Remove Duplicates**
- [ ] Delete duplicate files in src/lmms-eval/
- [ ] Consolidate task-specific READMEs (keep only important ones)

---

## **Quick Reference**

**Need to know:** | **Read this**
---|---
Project goals | PROJECT_OVERVIEW.md
How to run SRF | CODE_GUIDE.md → "SRF Usage"
What experiments ran | RESULTS_COMPENDIUM.md
VCD/MemVR details | LITERATURE_REVIEW.md
What to work on | NEXT_STEPS.md
Why SRF failed | srf/investigations/
Baselines | info/POPE_BASELINE_COMPARISON.md
