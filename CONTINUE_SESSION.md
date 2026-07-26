# CONTINUE SESSION - SRF RePOPE Improvement

**Load this file in new Claude window to continue work**

---

## ✅ COMPLETED WORK

### 1. RePOPE Findings Documented
- File: `REPOPE_FINDINGS.md` (committed to srf-remote)
- Results:
  - **VAF: 88.28%** (best method, +7.88% over baseline)
  - **VCD: 88.1%** (+7.7% over baseline)
  - **Baseline: 80.40%**
- RePOPE corrects 18.1% problematic samples from POPE

### 2. Saliency Improvements Implemented
- File: `srf/saliency/clip_salience.py`
- Changes:
  - **clip_full_gate_v3 mechanism** (backward compatible, disabled by default)
  - **V3_FULL_IMG_THRESH: 0.24 → 0.21** (lowered threshold)
  - **V3_RAW_ENTROPY_THRESH: 0.95** (working signal: low entropy = object present)
  - **Gate logic**: `gate_full OR (gate_full_soft AND any_v3_signal)`
  - Removed non-working signals (cross_scale_iou, blur_delta)

### 3. Validation Script Ready
- File: `scripts/session_scripts/validate_repoe_proper.sh`
- Tests 5 configs on 100 samples
- Environment: `mllm` conda, GPU 0
- Results: `results/session_logs/validation_repope_TIMESTAMP/`

---

## 🎯 CURRENT TASK

**Running two parallel SRF sweeps on RePOPE:**

### **1. Autoresearch Parameter Sweep** (Running now)
**Status:** Full sweep running in background (PID: 744307)
**Experiments:** 8 configurations × 3 splits = 24 total experiments
**GPUs:** 0,1,2 (one per split, parallel execution)
**Focus:** Testing different α, ε, layers, heads combinations
**Saliency:** Using legacy mode with default settings

### **2. Saliency Improvement Sweep** (Ready to launch)
**Purpose:** Test alternative absence detection modes (v3 gate alternatives)
**Script:** `scripts/session_scripts/run_srf_saliency_sweep.sh`
**Experiments:** 8 saliency configs × 3 splits = 24 total experiments
**Focus:** Testing entropy, peak_ratio, multi_metric modes vs legacy
**Base SRF:** α=4.0, ε=0.2, layers=8-15 (autoresearch best)

**Can launch saliency sweep after autoresearch sweep completes** (or run sequentially on GPUs 3,4,5 if available)

**Configurations tested:**
1. Autoresearch best: α=4.0, ε=0.2, layers=8-15, heads=20%
2. Strong boost: α=5.0, ε=0.2, layers=8-15, heads=20%
3. Conservative: α=3.0, ε=0.2, layers=8-15, heads=20%
4. Wide layers: α=4.0, ε=0.2, layers=8-18, heads=20%
5. Narrow layers: α=4.0, ε=0.2, layers=10-15, heads=20%
6. More heads: α=4.0, ε=0.2, layers=8-15, heads=25%
7. High epsilon: α=4.0, ε=0.25, layers=8-15, heads=20%
8. Baseline: α=0.15, ε=0.0, layers=10-15, heads=20%

**Splits:** random, popular, adversarial (from RePOPE COCO)

**Target:** Beat baseline 80.40%, VCD 88.1%, VAF 88.28%

### Decision Criteria
- **>82% accuracy**: Proceed to full experiments on all 3 RePOPE splits
- **<82% accuracy**: Debug parameters, try different configs

### Targets to Beat
- VAF: 88.28% (+7.88% over baseline)
- VCD: 88.1% (+7.7% over baseline)
- Current SRF Config 11: 80.18% (**-0.22% vs baseline**)

---

## 🚀 QUICK START (New Window)

```bash
# 1. Read this file (you're here!)
# 2. Run validation test
bash scripts/session_scripts/validate_repoe_proper.sh

# 3. Check results (when done)
ls results/session_logs/validation_repope_*/
cat results/session_logs/validation_repope_*/validation_summary.txt

# 4. View all results
ls -la results/session_logs/
```

---

## 📊 VALIDATION CONFIGS

Testing 5 configs on 100 samples of RePOPE adversarial split:

### Config 1: Baseline
- **α=0.15, ε=0.0, layers=10-15, heads=20%**
- Expected: ~80.40%

### Config 2: Autoresearch Best
- **α=4.0, ε=0.2, layers=8-15, heads=20%**
- Rationale: From `autoresearch/mmvp-srf` branch, hit 0.900 ceiling on POPE
- 27x stronger than baseline (α=4.0 vs 0.15)

### Config 3: Autoresearch Strong
- **α=5.0, ε=0.2, layers=8-15, heads=20%**
- Rationale: Even stronger boost

### Config 4: Autoresearch Conservative
- **α=3.0, ε=0.15, layers=8-15, heads=20%**
- Rationale: Moderate boost with lower epsilon

### Config 5: Strong Boost
- **α=5.0, ε=0.2, layers=8-15, heads=30%**
- Rationale: Strong boost with more heads

---

## 🔧 TECHNICAL DETAILS

### Environment
- **Conda**: `mllm`
- **GPU**: 0 (free - GPUs 0,1,2,3 available)
- **Model**: `llava-hf/llava-1.5-7b-hf`
- **Data**: `/home/anna2/shruthi/RePOPE/annotations/coco_repope_adversarial.json`

### Command Pattern
```bash
conda run -n mllm python srf/eval.py \
    --method srf \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets pope_vcd \
    --pope_vcd_file /home/anna2/shruthi/RePOPE/annotations/coco_repope_adversarial.json \
    --alpha 4.0 \
    --eps 0.2 \
    --layer_start 8 \
    --layer_end 15 \
    --head_top_k_pct 0.20 \
    --limit 100
```

### Key Files
- **Validation script**: `scripts/session_scripts/validate_repoe_proper.sh`
- **SRF eval**: `srf/eval.py`
- **Saliency**: `srf/saliency/clip_salience.py`
- **RePOPE findings**: `REPOPE_FINDINGS.md`
- **This file**: `CONTINUE_SESSION.md`
- **VAF method**: `srf/methods/vaf.py`
- **Validation sets**: `data/validation_sets/`
- **Saliency analysis**: `results/saliency_analysis/comprehensive_analysis_results.json`

---

## 📝 NEXT STEPS

### If Validation Successful (>82%)
1. Run full autosearch on all 3 RePOPE splits (random, popular, adversarial)
2. Test v3 gate mechanism enabled (set `V3_GATE_ENABLED=True` in `srf/saliency/clip_salience.py`)
3. Sweep around best config (±1 α, ±1 layer, ±0.05 ε)
4. Try Qwen2.5-VL-3B if LLaVA results are good

### If Validation Fails (<82%)
1. Debug why autoresearch configs don't work on RePOPE
2. Try intermediate configs (α=1.0, 2.0, 3.0)
3. Check if v3 gate helps or hurts
4. Review `autoresearch/mmvp-srf` branch for more insights

---

## 🚨 IMPORTANT NOTES

### DO NOT
- ❌ Use all GPUs without checking
- ❌ Run full experiments before validation
- ❌ Reinvent the wheel - use existing scripts

### DO
- ✅ Use `mllm` conda environment
- ✅ Follow existing script patterns
- ✅ Run validation on 100 samples first
- ✅ Check GPU availability before running
- ✅ Document results immediately

### User Feedback
- "remember Cross_scale_iou and blur_delta did not give good results"
- "Why dont you look at your old scripts?"
- "dont keep reinventing the wheel"
- "DONT USE ALL GPUS"

---

## 📂 FILE ORGANIZATION

### Project Structure (Clean)
```
base/                           # Only documentation + setup.py
├── CONTINUE_SESSION.md         # Load this in new window
├── REPOPE_FINDINGS.md          # VCD vs VAF comparison
├── CLAUDE.md                   # Project guidelines
├── setup.py                    # Package setup
└── *.md                        # Other documentation

scripts/
├── session_scripts/
│   └── validate_repoe_proper.sh    # Validation test
├── experiment_scripts/
│   └── eval_vaf_pope.py            # VAF evaluation
└── analysis/
    ├── compare_repope.py            # RePOPE comparison
    └── test_repope_impact.py       # Impact testing

data/
└── validation_sets/
    ├── balanced_validation_set.json
    ├── diverse_validation_set.json
    └── negative_validation_set.json

results/
├── session_logs/
│   └── validation_repope_TIMESTAMP/
│       ├── config_results/          # JSON outputs
│       └── validation_summary.txt
└── saliency_analysis/
    └── comprehensive_analysis_results.json

srf/
└── methods/
    └── vaf.py                        # VAF implementation
```

---

## 🔍 WHY AUTORESEARCH SHOULD WORK

From `autoresearch/mmvp-srf` branch:
- Hit **0.900 accuracy ceiling** on POPE val set
- clip_full_gate_v3 saliency reduced false positives from 12% of failures
- α=4.0 (27x stronger) + ε=0.2 + layers=8-15 was optimal
- Remaining 0.9pp gap due to CLIP missing small/occluded objects

### Why It Might Fail on RePOPE
- RePOPE has different distribution than POPE
- Adversarial split might be harder
- CLIP might behave differently on corrected annotations

---

## 📈 CURRENT SRF PERFORMANCE ON REPOPE

From `scripts/experiment_scripts/run_srf_repoe_all.sh`:
- **Config 11**: α=1.0, layers=10-18, heads=50%, eps=0.1
- **Result**: 80.18% (**-0.22% vs baseline 80.40%**)
- **Problem**: All 72 configs below baseline on LLaVA-1.5-7B

### Why Previous Configs Failed
- Alphas too weak (α=0.15 to 1.0)
- Wrong layer ranges (10-18 vs 8-15)
- No epsilon smoothing (ε=0.0 to 0.1 vs 0.2)
- No v3 gate mechanism

---

## ✅ SUCCESS CRITERIA

### Short-term (This Session)
- ✅ Validation script running
- ✅ Results from 5 configs on 100 samples
- ✅ Decision whether to proceed

### Medium-term (This Week)
- ⏳ Full autosearch on all 3 RePOPE splits
- ⏳ SRF beats VAF 88.28% and VCD 88.1%
- ⏳ Results documented and committed

### Long-term (This Month)
- ⏳ SRF improvements generalized to other datasets
- ⏳ Paper draft ready
- ⏳ Results on Qwen2.5-VL-3B

---

*For project context: See memory files in `.claude/projects/-home-anna2-shruthi-lmms-eval/memory/`*
*For code details: See `srf/` directory*
*For all results: See `results/` directory*
