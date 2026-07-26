# SRF AutoResearch Loop

**Inspired by:** https://github.com/karpathy/autoresearch

AI agents autonomously run SRF research experiments on LLaVA-7B + POPE while you sleep.

---

## Quick Start

### 1. Run Baseline
```bash
cd /home/anna2/shruthi/lmms-eval
export CUDA_VISIBLE_DEVICES=1,2
python srf/eval.py \
    --method srf \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --output results/baseline_llava7b/
```

### 2. Give Claude Code These Instructions

```
Hi! Please read srf/autoresearch_loop/program.md and help me run the SRF autoresearch loop.

We're testing improvements on LLaVA-7B + POPE adversarial (n=100 samples).
- GPU: 1 or 2
- Target: >85% accuracy (baseline ~84%)

Please run 3-5 experiments from the options in program.md and report results.
```

---

## Files

- **program.md** - Instructions for AI agent (read this!)
- **prepare.py** - Fixed utilities (DO NOT MODIFY)
- **autoresearch_llava.py** - Script to run experiments
- **README_AUTORESEARCH.md** - Detailed guide

---

## Quick Tests (Command Line)

```bash
# 5×5 CLIP grid
python srf/eval.py --method srf --model llava-hf/llava-1.5-7b-hf \
    --datasets pope --pope_splits adversarial --n_pope 100 \
    --clip_coarse_grid 5 --output results/test_5x5/

# Stronger boost (matches working mmvp-srf)
python srf/eval.py --method srf --model llava-hf/llava-1.5-7b-hf \
    --datasets pope --pope_splits adversarial --n_pope 100 \
    --alpha 4.0 --eps 0.2 --output results/test_strong/

# Disable absence-aware
python srf/eval.py --method srf --model llava-hf/llava-1.5-7b-hf \
    --datasets pope --pope_splits adversarial --n_pope 100 \
    --clip_suppress_thresh 0.0 --output results/test_no_absence/
```

---

## Expected Timeline

- **Baseline:** 10 minutes
- **Each variation:** 10 minutes
- **Full night (8 hours):** ~48 experiments
- **Target:** +1-3% improvement over baseline
