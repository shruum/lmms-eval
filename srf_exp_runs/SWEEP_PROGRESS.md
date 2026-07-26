# MME Parameter Sweeps - Current Status

**Started:** 2025-04-27 23:54
**Status:** IN PROGRESS
**GPUs:** 0 (Qwen-VL-Chat), 1 (LLaVA)

---

## Current Status

**LLaVA Sweep (GPU 1, PID 655744):** ✅ Running
- Currently on: Experiment 1 (baseline: layers 10-14, alpha=2.0)
- Running time: ~3 minutes

**Qwen Sweep (GPU 0, PID 657754):** ⚠️ Partially Running
- Experiments 1-2: Failed (CUDA OOM - old processes not cleared)
- Currently on: Experiment 3 (alpha=2.5, layers 8-14)
- Need to re-run: Experiments 1-2 after current batch completes

---

## Experimental Design

### Why This Sweep?

Previous MME evaluation with ClearSight paper parameters showed **ZERO IMPROVEMENT**:
- LLaVA: 656.67 total (SRF) vs 656.67 (baseline) = Δ=0.00%
- Using layers 10-14, alpha=2.0 (paper baseline)

**Hypothesis:** POPE-optimized parameters will transfer to MME and show gains.

### Key Parameters from POPE:

**Qwen-VL-Chat:**
- Best layers: 8-14 (early-mid fusion zone)
- Best alpha: 1.5-2.5 (gentle boost)
- CLIP top-k: 15-20%
- Head top-k: 15-20%

**LLaVA:**
- Different architecture (32 layers vs Qwen's 28)
- POPE idea3: Late fusion (13-16), alpha=3.0, more heads (50%)
- Needs stronger boost than Qwen

---

## Monitoring Commands

```bash
# Real-time monitor
./srf_exp_runs/monitor_mme_sweeps.sh

# Check logs
tail -f srf_exp_runs/qwen_sweep.log
tail -f srf_exp_runs/llava_sweep.log

# GPU usage
watch -n 5 nvidia-smi

# Completed experiments
ls -1 results/qwen_mme_sweep/*.log | wc -l
ls -1 results/llava_mme_sweep/*.log | wc -l
```

---

## Success Criteria

**Qwen-VL-Chat:**
- Paper baseline: ~610-636 total
- Target: **>660** (improvement over ClearSight)

**LLaVA:**
- Current baseline: 656.67 total (no improvement)
- Target: **>670** (significant gain needed)

---

## Next Steps After Sweep

1. **Analyze results** - Find best configuration for each model
2. **Compare with ClearSight baseline** - Paper-style reporting (Existence + Count + Position + Color)
3. **If successful:** Run full evaluation with best parameters
4. **If unsuccessful:** Investigate why MME doesn't benefit from POPE-optimized params
