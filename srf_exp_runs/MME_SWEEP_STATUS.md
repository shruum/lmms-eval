# MME Parameter Sweep Status

**Started:** 2026-04-27 (after stopping Qwen-VL-Chat MME full run)
**GPUs:** 0 (Qwen), 1 (LLaVA)
**Total experiments:** 16 (8 per model)

---

## Background

Previous MME evaluation showed **NO IMPROVEMENT** with SRF:
- **LLaVA:** 656.67 total (SRF) vs 656.67 (baseline) - Δ=0.00%
- **Qwen-VL-Chat:** Evaluation stopped at ~50 min

Key issue: Using parameters from ClearSight paper baseline, not optimized for our models.

---

## Strategy: Apply POPE Learnings to MME

### Key Insights from POPE Autoresearch:

**Qwen-VL-Chat:**
- **Best layers:** 8-14 (early-mid fusion zone)
- **Best alpha:** 1.5-2.5 (gentle boost)
- **CLIP top-k:** 15-20%
- **Head top-k:** 15-20%
- **Critical:** Absence-aware strategy (clip_suppress_thresh=0.24, alpha=5.0)

**LLaVA:**
- **Different architecture:** 32 layers vs Qwen's 28
- **POPE idea3 worked:** Late fusion (layers 13-16), alpha=3.0, more heads (50%)
- **Needs stronger boost:** alpha 2.0-3.0 vs Qwen's 1.5-2.5

---

## Qwen-VL-Chat Sweep (GPU 0, PID 655717)

| Exp | Method | Layers | Alpha | Beta | CLIP top-k | Head top-k | Description |
|-----|--------|--------|-------|------|------------|------------|-------------|
| 1 | SRF | 8-14 | 1.5 | - | 20% | 15% | POPE best config |
| 2 | SRF | 8-14 | 2.0 | - | 20% | 15% | Stronger boost |
| 3 | SRF | 8-14 | 2.5 | - | 20% | 15% | Aggressive boost |
| 4 | SRF | 6-16 | 2.0 | - | 20% | 15% | Wider fusion zone |
| 5 | SRF | 8-14 | 2.0 | - | 15% | 15% | More focused CLIP |
| 6 | SRF | 8-14 | 2.0 | - | 20% | 20% | More heads |
| 7 | SRF-E | 8-14 | 2.0 | 1.5 | 20% | 15% | Moderate contrastive |
| 8 | SRF-E | 8-14 | 2.0 | 2.0 | 20% | 15% | Strong contrastive |

**Absence-aware parameters (all SRF experiments):**
- clip_suppress_thresh: 0.24
- clip_suppress_alpha: 5.0

**Expected results:**
- Baseline (paper): ~610-636 total
- Target: **>660** (improvement over ClearSight baseline)

---

## LLaVA Sweep (GPU 1, PID 655744)

| Exp | Method | Layers | Alpha | Beta | CLIP top-k | Head top-k | Description |
|-----|--------|--------|-------|------|------------|------------|-------------|
| 1 | SRF | 10-14 | 2.0 | - | 20% | 20% | Original baseline (NO improvement) |
| 2 | SRF | 8-14 | 2.0 | - | 20% | 20% | Earlier fusion (Qwen-inspired) |
| 3 | SRF | 8-16 | 2.0 | - | 20% | 20% | Wider fusion zone |
| 4 | SRF | 13-16 | 3.0 | - | 25% | 50% | Late fusion (POPE idea3) |
| 5 | SRF | 10-14 | 3.0 | - | 20% | 20% | Stronger boost |
| 6 | SRF | 8-14 | 2.0 | - | 15% | 20% | More focused CLIP |
| 7 | SRF-E | 10-14 | 2.0 | 1.5 | 20% | 20% | Moderate contrastive |
| 8 | SRF-E | 13-16 | 3.0 | 1.5 | 25% | 50% | Strong contrastive + late fusion |

**Expected results:**
- Baseline (previous run): 656.67 total (NO improvement)
- Target: **>670** (significant improvement needed)

---

## Monitoring

**Check progress:**
```bash
# GPU utilization
watch -n 5 nvidia-smi

# Qwen sweep log
tail -f srf_exp_runs/qwen_sweep.log

# LLaVA sweep log
tail -f srf_exp_runs/llava_sweep.log

# Individual experiment logs
tail -f results/qwen_mme_sweep/exp01_pope_best.log
tail -f results/llava_mme_sweep/exp01_baseline.log
```

**Estimated time:** ~8-12 hours total (30-60 min per experiment × 16 experiments)

---

## Success Criteria

**Qwen-VL-Chat:**
- Any config >640 total = improvement over paper baseline
- Target: >660 (competitive with VAF)

**LLaVA:**
- Any config >660 total = improvement over current baseline
- Target: >670 (significant gain)

**Key metric:** Paper-style reporting (Existence + Count + Position + Color)
