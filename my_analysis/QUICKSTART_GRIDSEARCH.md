# LLaVA SRF Grid Search - Quick Start Guide

## Overview

This system performs automated hyperparameter search for LLaVA SRF (Semantic Re-Focus) method, testing different combinations of:
- **Layer ranges**: Where to apply attention intervention
- **Boost/suppress ratios**: Intervention strength
- **Saliency methods**: CLIP vs DINO for important patch detection
- **Grid sizes and top-k percentages**: Saliency computation parameters

## Files Created

1. **`grid_search_llava.py`** - Main grid search script
2. **`dino_salience.py`** - DINO-based saliency detection
3. **`run_grid_search.sh`** - Launcher for multi-GPU execution
4. **`analyze_grid_search.py`** - Results analysis and recommendations

## Quick Start

### Option 1: Quick Test (3 experiments, 1 GPU)

```bash
cd ~/shruthi/lmms-eval/my_analysis

# Quick test with CLIP only
python grid_search_llava.py --gpu 0 --quick_test --output_dir quick_test_results

# Analyze results
python analyze_grid_search.py --output_dir quick_test_results
```

### Option 2: Full Grid Search (50+ experiments, 2 GPUs)

```bash
cd ~/shruthi/lmms-eval/my_analysis

# Launch on 2 GPUs (100+ experiments total)
bash run_grid_search.sh

# Monitor progress
tail -f grid_search_results/gpu_0.log
tail -f grid_search_results/gpu_1.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Option 3: Custom Search

```bash
# Use specific GPUs
python grid_search_llava.py --gpu 0 --total_gpus 4 --max_experiments 100

# Save recommended config automatically
python analyze_grid_search.py --output_dir grid_search_results --save_config
```

## What Gets Tested

### Layer Ranges (4 combinations)
- Early-mid: `8-12` (more visual features)
- Mid: `10-16` (ClearSight finding)
- Mid-late: `12-18` (more reasoning)
- Wide: `8-16` (comprehensive)

### Boost/Suppress Ratios (5 combinations)
- Gentle: `1.5/3.0` (less intervention)
- Current: `2.0/5.0` (hurts LLaVA performance)
- Suppress-only: `1.0/5.0` (no boost, just suppress)
- Aggressive: `3.0/7.0` (strong intervention)
- Moderate: `2.5/4.0` (balanced)

### Saliency Methods (5 combinations)
**CLIP-based:**
1. Standard: `7×7 grid, top-30%, soft saliency`
2. Fine: `5×5 grid, top-20%, soft saliency`
3. Hard: `6×6 grid, top-40%, hard mask`

**DINO-based:**
4. Standard: `8×8 grid, top-30%, soft saliency`
5. Fine: `6×6 grid, top-20%, soft saliency`

### Head Selection (3 combinations)
- Top-15% heads (most selective)
- Top-20% heads (current default)
- Top-25% heads (more inclusive)

**Total experiments:** 4 × 5 × 5 × 3 = **300 combinations**

## Expected Results

After running, you'll get:

1. **Per-GPU result files**: `results_gpu*_final.json`
2. **Analysis output**: Top 10 configurations, parameter effects
3. **Recommended config**: `recommended_config.json`
4. **Run script**: `run_recommended_llava.sh`

## Interpreting Results

### Key Metrics
- **Improvement**: `SRF accuracy - Baseline accuracy` (higher is better)
- **Baseline accuracy**: Model performance without SRF
- **SRF accuracy**: Model performance with SRF

### What to Look For
1. **Positive improvement**: SRF helps LLaVA (goal: >0)
2. **Saliency method**: CLIP vs DINO performance
3. **Layer range**: Which layers work best for LLaVA
4. **Intervention strength**: Optimal boost/suppress ratios

## Running Recommended Configuration

After analysis, run the best configuration:

```bash
cd ~/shruthi/lmms-eval/my_analysis

# Run with recommended parameters
bash grid_search_results/run_recommended_llava.sh

# Or manually using the recommended config
python pope_srf_eval.py \
    --arch llava \
    --model llava-hf/llava-1.5-7b-hf \
    --n_samples 500 \
    --mode both \
    --output pope_full_results/llava_optimized.json
```

## Customization

### Modify Search Space

Edit `grid_search_llava.py`:

```python
SEARCH_SPACE = {
    "layer_ranges": [
        (8, 12),   # Add your own ranges
        (10, 16),
    ],
    "boost_suppress_pairs": [
        (1.5, 3.0),  # Test different ratios
        (2.0, 5.0),
    ],
    # ... customize other parameters
}
```

### Add New Saliency Methods

Create new saliency module (e.g., `sam_salience.py`) and add to `SEARCH_SPACE["saliency_configs"]`.

## Troubleshooting

### Out of Memory
- Reduce `n_samples` in `MODEL_CONFIG`
- Use smaller models first
- Run on GPU with more VRAM

### Slow Progress
- Reduce `max_experiments`
- Use `--quick_test` for validation
- Run on more GPUs

### DINO Installation Issues
```bash
pip install transformers torch
```

## Next Steps

1. **Run quick test** first to validate setup
2. **Launch full search** on multiple GPUs
3. **Analyze results** to find best configuration
4. **Run recommended config** on full POPE dataset
5. **Compare Qwen vs LLaVA** optimal parameters

## Expected Timeline

- **Quick test**: ~30 minutes
- **Full search (100 exp)**: ~5-10 hours on 2 GPUs
- **Per experiment**: ~3-5 minutes (100 samples)
