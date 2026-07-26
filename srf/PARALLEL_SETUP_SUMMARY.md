# Parallel Experiment Framework - Complete Setup

## Quick Start

```bash
cd /home/anna2/shruthi/lmms-eval
conda activate mllm

# One command to launch everything on 8 GPUs
bash srf/run_all_parallel.sh
```

**That's it!** This will:
1. Find 100 hard POPE samples (if not already done)
2. Launch hard POPE sweep on GPUs 0-3 (193 configs)
3. Launch layer-specific modulation on GPUs 4-7 (244 configs)
4. Monitor both experiments automatically

---

## What Was Built

### Core Framework

1. **`srf/find_hard_pope_samples.py`**
   - Finds samples where baseline fails
   - Saves 100 hard samples to JSON
   - Single GPU, ~10 minutes

2. **`srf/sweep_hard_pope_parallel.py`**
   - Tests 193 SRF configurations on hard samples
   - Runs on 4 GPUs (GPUs 0-3)
   - ~2-3 hours

3. **`srf/sweep_layer_specific_parallel.py`**
   - Tests 244 layer-specific modulation configs
   - Runs on 4 GPUs (GPUs 4-7)
   - ~2-3 hours

4. **`srf/run_all_parallel.sh`**
   - One-command launcher for both experiments
   - Handles hard sample finding automatically
   - Monitors and logs everything

### Documentation

5. **`srf/HARD_POPE_README.md`**
   - Detailed guide for hard sample framework
   - Usage examples and troubleshooting

6. **`srf/PARALLEL_EXPERIMENTS_GUIDE.md`**
   - Complete guide for 8-GPU parallel experiments
   - Monitoring, analysis, and troubleshooting

7. **`srf/srf_layer_specific.py`**
   - Implementation of layer-specific modulation
   - Three-zone: early-gentle, mid-strong, late-suppress

---

## Experiment Details

### Experiment 1: Hard POPE Sweep (GPUs 0-3)

**Goal:** Find ANY SRF configuration that improves over baseline

**Configurations tested:**
- 1 baseline
- 48 SRF (4 layer ranges × 3 head % × 4 alphas)
- 96 SRF with suppression variants
- 48 post-softmax configs

**Parameters:**
- Layer ranges: 5-10, 8-15, 10-15, 15-25
- Head percentages: 0.3, 0.5, 0.8
- Alphas: 0.15, 0.5, 1.0, 2.0
- Text suppression: beta=0.1
- Sys suppression: beta=0.1

### Experiment 2: Layer-Specific Modulation (GPUs 4-7)

**Goal:** Test three-zone layer-specific approach

**Configurations tested:**
- 1 baseline
- 243 layer-specific configs

**Three zones:**
- Early (0 to early_end): Visual detection boost
- Mid (early_end to mid_end): Saliency-guided fusion
- Late (mid_end to end): Language prior suppression

**Parameters:**
- Early ends: 5, 7, 10
- Mid ends: 15, 17, 20
- Alpha early: 0.3, 0.5, 1.0
- Alpha mid: 1.0, 2.0, 4.0
- Beta late: 0.05, 0.1, 0.2

---

## Design Principles Followed

✅ **No core files modified**
- `srf/srf.py`, `srf/eval.py`, `my_analysis/qwen_attn_patch.py` untouched
- All new code in separate wrapper files

✅ **No hardcoded parameters**
- All hyperparameters via CLI arguments
- Easy to test different combinations

✅ **Reproducible**
- Hard sample indices saved
- Same samples tested across all configs
- Fair comparison guaranteed

✅ **Scalable**
- Parallel GPU utilization
- Easy to add more configs
- Minimal code duplication

---

## Expected Outcomes

### Best Case
- One or more configs show >5% improvement over baseline
- Clear pattern emerges in optimal parameters
- Post-softmax or layer-specific shows promise
- Proceed to full dataset testing

### Likely Case (Current Trend)
- All configs show 0-2% improvement (noise level)
- No clear pattern in parameters
- Confirms current approach needs rethinking
- Move to alternative approaches (3-8 from Next_steps.md)

### Worst Case
- Some configs show negative improvement
- Post-softmax or layer-specific makes things worse
- Confirms fundamental issue with approach
- Consider more radical alternatives

---

## Monitoring

### Watch GPUs
```bash
watch -n 1 nvidia-smi
```

### Watch logs
```bash
# Hard POPE sweep
tail -f logs/hard_pope.log

# Layer-specific sweep
tail -f logs/layer_specific.log

# Individual GPU logs
tail -f results/sweep_hard_pope_parallel/gpu_0.log
tail -f results/sweep_layer_specific/gpu_0.log
```

### Check progress
```bash
# Count completed configs
ls results/sweep_hard_pope_parallel/gpu_*_results.json 2>/dev/null | wc -l
ls results/sweep_layer_specific/gpu_*_results.json 2>/dev/null | wc -l
```

---

## Results Analysis

After completion (~2-3 hours):

```bash
# View top results
python -c "
import json

# Hard POPE results
with open('results/sweep_hard_pope_parallel/aggregated_results.json') as f:
    hp = json.load(f)

# Layer-specific results
with open('results/sweep_layer_specific/aggregated_results.json') as f:
    ls = json.load(f)

print('='*60)
print('HARD POPE TOP 10')
print('='*60)
for i, e in enumerate(hp['summary'][:10]):
    cfg = e['config']
    print(f\"{i+1}. {cfg['name']}\")
    print(f\"   Acc: {e['accuracy']:.4f} ({e['correct']}/{e['total']})\")
    print(f\"   Method: {cfg['method']}\")
    if cfg.get('layer_start'):
        print(f\"   Layers: {cfg['layer_start']}-{cfg['layer_end']}, heads={cfg['head_top_k_pct']}, α={cfg['alpha']}\")
    print()

print('\\n' + '='*60)
print('LAYER-SPECIFIC TOP 10')
print('='*60)
for i, e in enumerate(ls['summary'][:10]):
    cfg = e['config']
    print(f\"{i+1}. {cfg['name']}\")
    print(f\"   Acc: {e['accuracy']:.4f} ({e['correct']}/{e['total']})\")
    if cfg['method'] == 'layer_specific':
        print(f\"   Early: 0-{cfg['layer_early_end']} (α={cfg['alpha_early']})\")
        print(f\"   Mid: {cfg['layer_early_end']}-{cfg['layer_mid_end']} (α={cfg['alpha_mid']})\")
        print(f\"   Late: {cfg['layer_mid_end']}-end (β={cfg['beta_late']})\")
    print()
"
```

---

## Next Steps After Results

### If Improvement Found (>5%)
1. Verify on full POPE dataset (9000 samples)
2. Test on MMVP and MME
3. Check saliency images for qualitative validation
4. Document findings in `srf/RESEARCH_STATUS.md`

### If No Improvement (0-2%)
1. Review saliency images (are they correct?)
2. Check if boosting is applied in right place
3. Consider alternative approaches from Next_steps.md:
   - Gradient-Based Intervention (Idea 3)
   - Causal Intervention (Idea 4)
   - Attention Entropy Regularization (Idea 5)
   - Token-Level Contrastive Enhancement (Idea 6)
   - Multi-Scale Saliency (Idea 8)

### If Negative Improvement
1. Check for bugs in implementation
2. Verify suppression isn't too aggressive
3. Try gentler parameters (lower alpha/beta)
4. Consider that current approach may be fundamentally flawed

---

## Files Created

```
srf/
├── find_hard_pope_samples.py           # Find hard samples
├── sweep_hard_pope_parallel.py         # Parallel sweep (GPUs 0-3)
├── sweep_layer_specific_parallel.py    # Parallel layer-specific (GPUs 4-7)
├── srf_layer_specific.py               # Layer-specific implementation
├── run_all_parallel.sh                 # One-command launcher
├── HARD_POPE_README.md                 # Hard sample guide
├── PARALLEL_EXPERIMENTS_GUIDE.md       # Parallel experiments guide
└── PARALLEL_SETUP_SUMMARY.md           # This file
```

---

## Summary

**Total experiments:** 437 configurations on 100 hard samples
**Total time:** ~2-3 hours on 8 GPUs
**Success criteria:** Any config with >5% improvement over baseline

**Ready to run:** `bash srf/run_all_parallel.sh`

**Framework follows all project rules:**
- ✅ No core file modifications
- ✅ All CLI parameters (no hardcoding)
- ✅ Wrapper scripts for experiments
- ✅ Reproducible and systematic

Let's find out if ANY configuration works! 🚀
