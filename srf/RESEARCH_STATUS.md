# SRF Research Status

> Update this file after every experiment run and commit it.
> This is the live source of truth for results and open tasks — replaces vault reads on remote servers.
> 
> **📊 Comprehensive Results:**
> - `ALL_RESULTS_TABLE.md` - **Table format** (Architecture | Dataset | Accuracy | Settings) ✅ **RECOMMENDED**
> - `ALL_RESULTS.md` - Detailed markdown with per-category breakdowns
> - `ALL_RESULTS.tsv` - Spreadsheet-ready format

---

## Current Best Results

### Qwen2.5-VL-3B (model: Qwen/Qwen2.5-VL-3B-Instruct)

| Method | POPE avg (9000) | MMVP pair | MME score | VLM Bias |
|--------|-----------------|-----------|-----------|----------|
| Baseline | 84.13% | 26.67% | 69.50% | 19.00% |
| **SRF** | 84.18% (+0.05%) | 26.67% (0.00%) | 69.50% (0.00%) | **19.68% (+0.68%)** |
| **SRF-E** | — | — | — | broken* |

*VLM Bias SRF-E broken: contrastive pass suppresses format tokens (`{`). Use SRF base only.

**Key Finding**: VLM Bias shows consistent small gains (+0.68%) while POPE/MME/MMVP show zero improvement.

### LLaVA-1.5-7B (model: llava-hf/llava-1.5-7b-hf)

| Method | POPE adv | MMVP pair | MME score | VLM Bias |
|--------|----------|-----------|-----------|----------|
| Baseline | 80.00% | — | 656.67 (Δ=0%) | — |
| **SRF** | ❌ autoresearch in progress | — | ❌ in progress (seq sweep) | — |

**Note:** LLaVA MME baseline (ClearSight params) showed ZERO improvement. Current sweep tests POPE-optimized params.

### Qwen2.5-VL-3B (model: Qwen/Qwen2.5-VL-3B-Instruct) — Latest Results (May 2026)

| Method | POPE avg (9000) | MMVP pair | MME score | VLM Bias |
|--------|-----------------|-----------|-----------|----------|
| Baseline | 84.13% | 26.67% | 69.50% | 19.00% |
| **SRF** | 84.18% (+0.05%) | 26.67% (0.00%) | 69.50% (0.00%) | **19.68% (+0.68%)** |
| **SRF-E** | — | — | — | broken* |

*VLM Bias SRF-E broken: contrastive pass suppresses format tokens (`{`). Use SRF base only.

**Key Finding**: VLM Bias shows consistent small gains (+0.68%) while POPE/MME/MMVP show zero improvement.

---

## Open Tasks (priority order)

1. **✅ COMPLETED: Full Qwen2.5-VL evaluation** — All datasets tested
   - POPE: 9000 samples (3 splits) → +0.05% (no improvement)
   - VLM Bias: ~300 samples → +0.68% (small improvement)
   - MMVP: 300 samples → 0.00% (no improvement)
   - MME: 2374 samples → 0.00% (no improvement)

2. **🔬 ACTIVE: Understand VLM Bias improvement** — Why does VLM Bias respond?
   - Analyze per-category results (just implemented)
   - Compare dataset characteristics vs POPE/MME/MMVP
   - Identify what makes VLM Bias different

3. **🔬 ACTIVE: Parameter sweep analysis** — Alpha has no effect
   - Tested: α ∈ {0.5, 1.0, 2.0, 4.0, 6.0, 8.0}
   - Finding: No correlation with accuracy
   - **Hypothesis**: Mechanism needs fundamental redesign

4. **⏳ NEXT: Test alternative mechanisms**
   - Post-softmax redistribution (see Next_steps.md)
   - Layer-specific modulation
   - Multi-scale saliency (CLIP + attention rollout)

---

## Recent Runs

### 2026-05-02 — Full Qwen2.5-VL Evaluation Complete
- **Command**: `python srf/eval.py --method srf --model Qwen/Qwen2.5-VL-3B-Instruct --datasets pope vlmbias`
- **Results**:
  - POPE (all 3 splits, 9000 samples): 84.13% → 84.18% (+0.05%) ❌
  - VLM Bias (full dataset, ~300 samples): 19.00% → 19.68% (+0.68%) ✅
- **Key Finding**: Alpha parameter has NO effect on accuracy
- **Implementation**: Per-category SRF tracking added for VLM Bias
- **Files**: `srf_exp_runs/results/qwen25_pope_full/`, `srf_exp_runs/results/qwen25_vlmbias_full/`

### 2026-05-01 — Parameter Sweep on POPE
- **Command**: Alpha sweep [0.5, 1.0, 2.0, 4.0, 6.0, 8.0] on 100 POPE samples
- **Results**: No correlation between alpha and accuracy
- **Conclusion**: Pre-softmax additive boost mechanism needs redesign

<!-- Add entries here after each experiment. Format:
### YYYY-MM-DD — description
- Command: ...
- Results: ...
- Notes: ...
-->

### 2026-04-28 — POPE Autoresearch Complete
- **Qwen-VL-Chat:** 81.00% accuracy (+1.0% over baseline)
  - Best config: layers 8-14, α=1.5, clip_top_k=0.20, head_top_k=0.15-0.20, suppress_thresh=0.24-0.248
  - 50 experiments sweep → found early-mid fusion (8-14) better than late (10-14)
  - Gentle boost (α=1.5) outperformed stronger boosts
  - Absence-aware strategy critical (suppress when CLIP max_sim < 0.24)
- **LLaVA:** Baseline 80.00%, autoresearch ongoing
- **Files:** `auto_research/pope_qwen/sweep_results/top10.tsv`, `auto_research/STATUS.md`

### 2026-04-28 — MME Preliminary Results (4/16 complete) ⚠️
- **LLaVA (3/8):** Δ=0.00% across ALL configs tested so far
  - Exp 1 (baseline, 10-14, α=2.0): 1650 → 1650 (Δ=0.00%)
  - Exp 2 (early fusion, 8-14, α=2.0): 1650 → 1650 (Δ=0.00%)
  - Exp 3 (wide fusion, 8-16, α=2.0): 1650 → 1650 (Δ=0.00%)
- **Qwen (1/8):** Δ=-3.16% (WORSE performance)
  - Exp 1 (POPE best, 8-14, α=1.5): 1948 → 1873 (Δ=-3.16%)
  - **Anomaly:** Baseline 1948 (82.06%) unusually high vs typical 1650 (69.50%)
- **Status:** 5 LLaVA + 7 Qwen experiments still running
- **Concern:** SRF may not work for MME task, or evaluation has bug

### 2026-04-27 — MME Full Evaluation (Qwen stopped, LLaVA complete)
- **Qwen-VL-Chat:** Stopped at ~50 min (layers 10-14, α=2.0)
- **LLaVA:** Complete but NO IMPROVEMENT — 656.67 total (SRF) vs 656.67 (baseline) = Δ=0.00%
  - Paper-style: Existence=190.00, Count=146.67, Position=146.67, Color=173.33
  - All 14 categories: Identical performance
  - **Conclusion:** ClearSight baseline parameters don't transfer to LLaVA MME
- **Files:** `auto_research/LLAVA_MME_FULL.log`

### 2026-04-25 — Smoke test MME (10 samples)
- Command: inline test script
- Results: base=9/10, SRF=10/10 (SRF fixed one wrong answer)
- Notes: End-to-end works. Need full run.

---

## Known Issues

- SRF-E + VLM Bias: contrastive pass suppresses `{` token → format broken. Use SRF base only.
- **LLaVA MME:** ClearSight baseline params (layers 10-14, α=2.0) show ZERO improvement → need POPE-optimized params
- Qwen-7B and LLaVA arch params not tuned — proportional starting points only.
- `HF_HOME` path is machine-specific — set via env var, not hardcoded.
