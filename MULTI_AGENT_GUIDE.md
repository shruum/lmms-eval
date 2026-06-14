# Multi-Agent Collaboration Guide

**Purpose:** How to use the consolidated documentation for parallel work across multiple Claude agents/windows

---

## 🚀 **Quick Start for Any Agent**

### **First 5 Minutes (Essential Reading)**
1. Read **PROJECT_OVERVIEW.md** (3 min) - Understand project goals
2. Read **NEXT_STEPS.md** (2 min) - See current priorities
3. Check **RESULTS_COMPENDIUM.md** (30 sec) - Verify what's already done

**Total:** 5 minutes to onboard any agent

---

## 📚 **The 5 Essential Documents**

### **1. PROJECT_OVERVIEW.md**
**When to read:** Starting fresh, need context
**What it contains:**
- Project goals and success criteria
- Current baselines and targets
- Research context (SRF, VCD, MemVR, VAF)
- Project structure
- Current work status

**Read time:** 3 minutes

---

### **2. CODE_GUIDE.md**
**When to read:** About to modify/run code
**What it contains:**
- Code architecture overview
- Important files and what they do
- How to run experiments
- Critical rules (don't modify stable files)
- Debugging guide
- Best practices

**Read time:** 5 minutes (when needed)

---

### **3. RESULTS_COMPENDIUM.md**
**When to read:** Before running ANY experiment
**What it contains:**
- ALL experiment results (baselines, SRF sweeps, etc.)
- What's been run (don't re-run!)
- Result file locations
- Key insights from results
- Quick reference tables

**Read time:** 2 minutes (scan for relevant section)

---

### **4. LITERATURE_REVIEW.md**
**When to read:** Studying paper methods (VCD, MemVR, VAF)
**What it contains:**
- Detailed analysis of each method
- How each method works
- Parameters and results from papers
- Comparison with SRF
- Reproduction plans

**Read time:** 10 minutes (when studying methods)

---

### **5. NEXT_STEPS.md**
**When to read:** Figuring out what to work on
**What it contains:**
- Current priorities (VCD reproduction)
- Step-by-step plans
- Multi-agent collaboration strategy
- Progress tracking
- Blockers and issues

**Read time:** 5 minutes

---

## 👥 **Scenario-Based Agent Workflows**

### **Scenario 1: New Agent Joining Project**

**Goal:** Get up to speed quickly

**Steps:**
1. Read PROJECT_OVERVIEW.md (3 min)
2. Read NEXT_STEPS.md (2 min)
3. Check RESULTS_COMPENDIUM.md for baselines (1 min)
4. Assign task from NEXT_STEPS.md
5. Start working

**Time:** 5-10 minutes to productive work

---

### **Scenario 2: Agent Running Experiments**

**Goal:** Run VCD experiments without duplicating work

**Steps:**
1. Check NEXT_STEPS.md → "Step-by-Step VCD Reproduction Plan"
2. Check RESULTS_COMPENDIUM.md → "VCD Paper Results" section
3. Verify experiment not already run
4. Follow CODE_GUIDE.md → "How to Run Experiments"
5. Run experiments
6. Update RESULTS_COMPENDIUM.md with new results
7. Update NEXT_STEPS.md progress

**Time:** 15-30 minutes setup, hours for experiments

---

### **Scenario 3: Agent Analyzing Results**

**Goal:** Compare VCD results with paper and baseline

**Steps:**
1. Read LITERATURE_REVIEW.md → "VCD" section (paper results)
2. Read RESULTS_COMPENDIUM.md → "Baselines" section
3. Parse VCD output files
4. Create comparison table
5. Update RESULTS_COMPENDIUM.md with findings
6. Update NEXT_STEPS.md if needed

**Time:** 30-60 minutes

---

### **Scenario 4: Agent Debugging Issues**

**Goal:** Fix VCD setup or SRF code issues

**Steps:**
1. Read CODE_GUIDE.md → "Debugging Guide"
2. Check RESULTS_COMPENDIUM.md → "Key Findings"
3. Identify issue type (environment, code, parameters)
4. Follow debugging steps in CODE_GUIDE.md
5. Document solution in CODE_GUIDE.md if new
6. Update NEXT_STEPS.md blockers resolved

**Time:** Variable (10 min to hours)

---

### **Scenario 5: Agent Studying Paper Methods**

**Goal:** Understand how VCD/MemVR/VAF work

**Steps:**
1. Read LITERATURE_REVIEW.md → specific method section
2. Read PROJECT_OVERVIEW.md → "Research Context"
3. Check RESULTS_COMPENDIUM.md → "Paper Method Results"
4. Locate paper/repo (links in LITERATURE_REVIEW.md)
5. Clone repo and analyze code
6. Document findings in LITERATURE_REVIEW.md
7. Update NEXT_STEPS.md with reproduction plan

**Time:** 1-2 hours

---

## 🔄 **Parallel Work Strategy**

### **Example: 4 Agents Working in Parallel**

**Agent 1: VCD Reproduction**
- Reads: PROJECT_OVERVIEW.md, NEXT_STEPS.md, CODE_GUIDE.md
- Works on: Setup VCD environment, run experiments
- Updates: RESULTS_COMPENDIUM.md, NEXT_STEPS.md

**Agent 2: MemVR Preparation**
- Reads: LITERATURE_REVIEW.md, NEXT_STEPS.md
- Works on: Locate MemVR paper, analyze method
- Updates: LITERATURE_REVIEW.md, NEXT_STEPS.md

**Agent 3: VAF Preparation**
- Reads: LITERATURE_REVIEW.md, NEXT_STEPS.md
- Works on: Locate VAF paper, analyze method
- Updates: LITERATURE_REVIEW.md, NEXT_STEPS.md

**Agent 4: Analysis & Documentation**
- Reads: All 5 docs
- Works on: Compare results, generate tables, update docs
- Updates: All 5 docs as needed

**Communication:** Agents work independently, coordinated through shared docs

---

## 📝 **Documentation Maintenance**

### **When to Update Docs**

**PROJECT_OVERVIEW.md:**
- ✅ Update when project goals change
- ✅ Update when major milestones reached
- ✅ Update when new methods added

**CODE_GUIDE.md:**
- ✅ Update when new important files added
- ✅ Update when debugging solutions found
- ✅ Update when new scripts added

**RESULTS_COMPENDIUM.md:**
- ✅ Update IMMEDIATELY after any experiment
- ✅ Update when new insights discovered
- ✅ Update when paper methods reproduced

**LITERATURE_REVIEW.md:**
- ✅ Update when new papers discovered
- ✅ Update when methods analyzed
- ✅ Update when repos located/cloned

**NEXT_STEPS.md:**
- ✅ Update when priorities change
- ✅ Update when steps completed
- ✅ Update when blockers identified/resolved

### **Update Protocol**

1. **Before updating:** Check current doc content
2. **Make changes:** Edit clearly and concisely
3. **Add timestamp:** Update "Last Updated" date
4. **Note changes:** Add brief comment at top if major change
5. **Sync with team:** Other agents see updates immediately

---

## 🎯 **Quick Reference Table**

| I need to... | Read this | Then this | Update this |
|--------------|------------|------------|-------------|
| Understand project | PROJECT_OVERVIEW.md | NEXT_STEPS.md | (none) |
| Run experiments | CODE_GUIDE.md | RESULTS_COMPENDIUM.md | RESULTS_COMPENDIUM.md |
| Study papers | LITERATURE_REVIEW.md | PROJECT_OVERVIEW.md | LITERATURE_REVIEW.md |
| Debug issues | CODE_GUIDE.md | RESULTS_COMPENDIUM.md | CODE_GUIDE.md |
| Analyze results | RESULTS_COMPENDIUM.md | LITERATURE_REVIEW.md | RESULTS_COMPENDIUM.md |
| Plan next work | NEXT_STEPS.md | PROJECT_OVERVIEW.md | NEXT_STEPS.md |
| Compare methods | LITERATURE_REVIEW.md | RESULTS_COMPENDIUM.md | (none) |

---

## 🔍 **How to Find Information Fast**

### **Ctrl+F Search Terms**

**In PROJECT_OVERVIEW.md:**
- "baseline" - find baseline numbers
- "success criteria" - find targets
- "VCD" "MemVR" "VAF" - find method info
- "parameter" - find hyperparameters

**In CODE_GUIDE.md:**
- "srf.py" - find file description
- "debug" - find debugging guide
- "run" - find how to run experiments
- "rule" - find critical rules

**In RESULTS_COMPENDIUM.md:**
- "coco_adversarial" - find specific split
- "78.77%" - find specific result
- "baseline" - find baseline results
- "VCD" - find VCD results

**In LITERATURE_REVIEW.md:**
- "VCD" "MemVR" "VAF" - find method details
- "+3.5%" - find specific improvements
- "repo" - find GitHub links
- "comparison" - find method comparisons

**In NEXT_STEPS.md:**
- "Step 1" "Step 2" - find specific steps
- "Priority" - find what to work on
- "blocker" - find current issues
- "Agent" - find parallel work assignments

---

## 💡 **Best Practices for Multi-Agent Work**

### **DO:**
1. ✅ Always read essential docs before starting
2. ✅ Check RESULTS_COMPENDIUM.md before running experiments
3. ✅ Update docs immediately after completing work
4. ✅ Use clear, concise language
5. ✅ Add timestamps when updating
6. ✅ Cross-reference related docs
7. ✅ Communicate through doc updates

### **DON'T:**
1. ❌ Don't modify stable code files (see CODE_GUIDE.md)
2. ❌ Don't run experiments without checking results
3. ❌ Don't work on outdated priorities (check NEXT_STEPS.md)
4. ❌ Don't duplicate work (check RESULTS_COMPENDIUM.md)
5. ❌ Don't leave docs outdated after work
6. ❌ Don't make major changes without discussion

---

## 🆘 **Getting Unstuck**

### **If you don't know what to do:**
1. Read NEXT_STEPS.md → "Step-by-Step VCD Reproduction Plan"
2. Find the first uncompleted step
3. Start working on it

### **If results don't make sense:**
1. Check RESULTS_COMPENDIUM.md for similar experiments
2. Check CODE_GUIDE.md for debugging steps
3. Check LITERATURE_REVIEW.md for expected results

### **If you find bugs:**
1. Document in CODE_GUIDE.md → "Debugging Guide"
2. Add solution if you find one
3. Update NEXT_STEPS.md if blocker

### **If docs are unclear:**
1. Note what's confusing
2. Ask for clarification
3. Update doc to be clearer for next agent

---

## 📊 **Doc Health Checklist**

**Weekly check:**
- [ ] All 5 docs updated with latest results
- [ ] NEXT_STEPS.md reflects current priorities
- [ ] RESULTS_COMPENDIUM.md has all experiments
- [ ] No duplicate or outdated info
- [ ] All links and references work
- [ ] Timestamps are current

**Monthly check:**
- [ ] Archive old results to separate file
- [ ] Consolidate redundant sections
- [ ] Update project timeline
- [ ] Remove completed temporary sections
- [ ] Verify all external links still work

---

## 🎓 **Learning Path for New Agents**

**Day 1 (Onboarding):**
1. Read PROJECT_OVERVIEW.md (3 min)
2. Read NEXT_STEPS.md (2 min)
3. Browse CODE_GUIDE.md (10 min)
4. Start simple task (e.g., check GPU status)

**Day 2 (First Task):**
1. Read RESULTS_COMPENDIUM.md (10 min)
2. Read specific section of CODE_GUIDE.md (10 min)
3. Run simple experiment (baseline check)
4. Update RESULTS_COMPENDIUM.md

**Day 3 (Independent Work):**
1. Read LITERATURE_REVIEW.md → VCD section (15 min)
2. Setup VCD environment (30 min)
3. Run VCD test (1 hour)
4. Document results (15 min)

**Day 4+ (Full Productivity):**
1. Check NEXT_STEPS.md for priorities
2. Work on assigned tasks
3. Update relevant docs
4. Coordinate with other agents via docs

---

## 🚀 **Success Metrics**

**For Documentation:**
- ✅ Any agent can start work in <10 minutes
- ✅ No duplicate experiments run
- ✅ All results documented immediately
- ✅ Clear priorities in NEXT_STEPS.md
- ✅ Easy to find any information

**For Collaboration:**
- ✅ Multiple agents work in parallel without conflicts
- ✅ Progress visible through doc updates
- ✅ Clear communication through shared docs
- ✅ Easy to hand off work between agents

---

*Remember: These docs are living documents. Keep them updated, keep them clear, keep them useful!*

**For questions about this guide:** See specific documentation files or check PROJECT_OVERVIEW.md for context.
