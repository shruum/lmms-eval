# SRF Parameter Sweep - POPE (MS-COCO) - LLaVA-1.5-7B

## Goal
Find best SRF configuration for POPE benchmark (MS-COCO only).

## Dataset
- **POPE (MS-COCO)**: 9000 samples total
  - Random: 3000 samples
  - Popular: 3000 samples
  - Adversarial: 3000 samples

## Model
- LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`)

## Parameter Grid

### Parameters to Sweep:
- **alpha** (boost strength): 2.0, 4.0, 6.0, 8.0
- **eps** (absence threshold): 0.0, 0.2, 0.3
- **clip_coarse_grid**: 5, 7, 9
- **clip_top_k_pct**: 0.3, 0.5, 0.7
- **clip_suppress_thresh**: 0.0 (disable absence-aware)

### Total Combinations:
- 4 × 3 × 3 × 3 × 1 = **108 configurations**
- Per category: 108 experiments
- Total: 108 × 3 categories = **324 experiments**

## Best Configurations from Previous Tests:
1. **higher_topk**: α=6.0, eps=0.3, grid=7, top_k=0.5 → +0.10% avg
2. **strong_boost**: α=4.0, eps=0.2, grid=7 → +0.10% avg
3. **no_absence_aware**: α=6.0, eps=0.3, grid=7, thresh=0.0 → +0.10% avg

## Baseline (No SRF):
- Random: 87.27%
- Popular: 85.47%
- Adversarial: 82.93%
- **Average: 85.22%**

## Target:
Find configuration that achieves **>86% average** (significant improvement over baseline).

## Execution:
```bash
# Full sweep (324 experiments, ~8-12 hours)
python srf/comprehensive_pope_sweep.py --gpu 1

# Quick test (subset of configs)
python srf/comprehensive_pope_sweep.py --quick --gpu 1

# Single category
python srf/comprehensive_pope_sweep.py --category random --gpu 1
```

## Output:
- Results: `results/pope_comprehensive_sweep/`
- Summary: `results/pope_comprehensive_sweep/sweep_summary_*.json`

## Next Steps:
1. Find best config per category
2. Find best overall config (average across categories)
3. Create final table for paper
