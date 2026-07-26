# SRF Experiment Status

**Last Updated**: 2026-05-27 10:30 UTC
**Goal**: Improve SRF performance on POPE benchmark → ≥80.30% on COCO Adversarial (baseline: 79.30%)
**Status**: Phase 1 complete, Phase 2 failed to launch

---

## **PHASE 1: LAYER-WISE STRATEGY** ✅ COMPLETE

### Concept
Different alpha multipliers for early/middle/late fusion stages

### Results (100 samples each)
| Config | Early/Mid/Late Multipliers | Accuracy | vs Baseline (83.0%) | Status |
|--------|---------------------------|----------|-------------------|--------|
| **LW-1** | 0.3 / 1.5 / 0.1 | 83.00% | 0.0% | ✅ Complete |
| **LW-2** | 0.5 / 1.2 / 0.2 | 83.00% | 0.0% | ✅ Complete |
| **LW-3** | 0.1 / 2.0 / 0.05 | 83.00% | 0.0% | ✅ Complete |
| **LW-4** | 0.0 / 1.8 / 0.0 | 83.00% | 0.0% | ✅ Complete |
| **LW-5** | 0.5 / 1.0 / 0.5 | 83.00% | 0.0% | ✅ Complete |

### Key Findings
- **All configs showed identical results**: 83.00% accuracy
- **Zero improvement** over baseline on 100 samples
- **Conclusion**: Layer-wise strategy ineffective for this task/model combination

### Result Files
```
results/layerwise_lw1_coco_100/pope.json
results/layerwise_lw2_coco_100/pope.json
results/layerwise_lw3_coco_100/pope.json
results/layerwise_lw4_coco_100/pope.json
results/layerwise_lw5_coco_100/pope.json
```

---

## **PHASE 2: HEAD SELECTION STRATEGY** ❌ FAILED TO LAUNCH

### Concept
Query-conditioned head selection based on query type (count/absence/spatial/attribute)

### Planned Experiments
| Config | Head % | Layers | Target Query Type | Status |
|--------|--------|--------|------------------|--------|
| **HS-1** | 30% | 12-18 | Absence (POPE) | ❌ Failed |
| **HS-2** | 50% | 10-18 | Default/Generic | ❌ Failed |
| **HS-3** | 70% | 8-22 | Count (future) | ❌ Failed |

### Issue
Experiments failed to launch with exit code 1. Possible causes:
- Runtime error in query classifier integration
- Syntax error in prepare_sample function
- Missing dependencies or import errors

### Next Steps
1. Check error logs: `cat results/head_selection_*/log.txt` (no files exist)
2. Test with minimal samples: `--n_pope 2` to isolate issue
3. Verify query classifier independently
4. Fix integration bugs and re-launch

---

## **PHASE 1+2 COMBINED STRATEGY** ⏳ NOT STARTED

### Concept
Combine layer-wise alpha scaling with query-conditioned head selection

### Planned Experiments
| Config | Layer-wise (early/mid/late) | Head % | Layers | Strategy | Status |
|--------|---------------------------|--------|--------|----------|--------|
| **C-1** | 0.3/1.5/0.1 | 30% | 12-18 | Conservative both | ⏳ Not started |
| **C-2** | 0.3/1.5/0.1 | 50% | 10-18 | Conservative layer + default heads | ⏳ Not started |
| **C-3** | 0.1/2.0/0.05 | 30% | 12-18 | Aggressive mid + selective heads | ⏳ Not started |
| **C-4** | 0.5/1.2/0.2 | 30% | 12-18 | Balanced both | ⏳ Not started |

**Base parameters**: α=0.15, POPE adversarial, 100 samples per config

### Dependencies
- Requires Phase 2 completion and working head selection implementation
- Awaiting bug fixes in query classifier integration

---

## **CURRENT GPU STATUS** (2026-05-27 10:30 UTC)

### GPU Utilization
| GPU | Util | Memory Used | Memory Total | Status |
|-----|------|-------------|--------------|--------|
| 0 | 96% | 14GB | 49GB | Available |
| 1 | 100% | 30GB | 49GB | Active |
| 2 | 100% | 14GB | 49GB | Active |
| 3 | 100% | 40GB | 49GB | Active |
| 4 | 100% | 45GB | 49GB | Active |
| 5 | 100% | 47GB | 49GB | Active |
| 6 | 62% | 25GB | 49GB | Available |
| 7 | 72% | 14GB | 49GB | Available |

### Running Processes
- **0** eval.py processes currently running
- GPUs showing memory usage from previous completed experiments (not released)

---

## **IMPLEMENTATION STATUS**

### ✅ Completed Components
1. **Layer-wise Strategy** (`llava_attn_patch.py`)
   - Early/mid/late alpha multipliers
   - CLI integration in eval.py
   - Tested and working (but ineffective)

2. **Query Classifier** (`srf/query_classifier.py`)
   - `classify_query()`: Detects query type
   - `get_head_params()`: Returns query-specific parameters
   - Tested independently and working

3. **CLI Arguments** (`srf/eval.py`)
   - `--use_layerwise`: Enable layer-wise scaling
   - `--early_alpha_mult`, `--mid_alpha_mult`, `--late_alpha_mult`: Multipliers
   - `--use_query_heads`: Enable head selection
   - `--query_type`: Force specific query type (testing)

### ❌ Issues to Fix
1. **Head Selection Integration Bug**
   - Phase 2 experiments fail to launch
   - Need to debug query classifier integration in prepare_sample()
   - Test with minimal samples to isolate error

---

## **SUCCESS CRITERIA**

- **Minimum**: Any config ≥79.30% (match baseline on full dataset)
- **Target**: Any config ≥80.30% (+1% improvement)
- **Stretch**: Any config ≥80.80% (+1.5% improvement)

### Current Best Result
- **Phase 1 (Layer-wise)**: 83.00% on 100 samples (0.0% improvement)
- **Note**: Baseline on 100 samples is 83.0% vs 79.30% on full dataset
- **Conclusion**: Sample subset appears easier than full dataset

---

## **NEXT ACTIONS** (Priority Order)

1. **Fix Phase 2 bugs** - Debug why head selection experiments fail
2. **Re-launch Phase 2** - Run HS-1, HS-2, HS-3 with fixes
3. **Launch Phase 1+2 Combined** - If Phase 2 shows promise
4. **Full evaluation** - If any config ≥80.30% on 100 samples, test on 3000 samples

---

## **FILES TO REFERENCE**

### Status & Documentation
- `SRF_EXPERIMENT_STATUS.md` (this file) - Current experiment status
- `PHASE2_AND_COMBINED_READY.md` - Phase 2 & combined strategy details
- `POPE_BASELINE_COMPARISON.md` - Correct baselines vs papers
- `SRF_TARGET_OBJECTIVES.md` - Success criteria and targets

### Implementation Files
- `my_analysis/llava_attn_patch.py` - Layer-wise implementation (LLaVA models)
- `srf/query_classifier.py` - Query classification for head selection
- `srf/eval.py` - CLI arguments and experiment runner
- `srf/srf.py` - Core SRF implementation

### Scripts
- `launch_combined_phase1_phase2.sh` - Combined strategy launch commands
- `monitor_and_launch_phase2.sh` - Phase 2 auto-launcher with GPU monitoring
- `check_phase2_status.sh` - Status dashboard for Phase 2

---

*This file is updated as experiments progress. Use for status monitoring and next-step planning.*