# Hard POPE Sample Testing

## Overview

Systematic testing framework for evaluating SRF configurations on hard POPE samples (where baseline fails).

## Step 1: Find Hard Samples

Identify ~100 POPE samples where baseline performs poorly:

```bash
cd /home/anna2/shruthi/lmms-eval

conda activate mllm
export HF_HOME=/path/to/hf_cache
export CUDA_VISIBLE_DEVICES=0

python srf/find_hard_pope_samples.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --pope_splits adversarial \
    --n_samples 100 \
    --output srf/hard_samples_pope.json
```

**Parameters:**
- `--model`: Model to use (default: Qwen/Qwen2.5-VL-3B-Instruct)
- `--pope_splits`: Which POPE splits to scan (default: all three)
- `--n_samples`: Number of hard samples to find (default: 100)
- `--output`: Path to save hard samples JSON

**Output:** `srf/hard_samples_pope.json` with sample indices and metadata.

---

## Step 2: Run Configuration Sweep

Test all configurations on hard samples:

```bash
python srf/sweep_hard_pope.py \
    --hard_samples srf/hard_samples_pope.json \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --output results/sweep_hard_pope/ \
    --layer_ranges 5-10 8-15 10-15 15-25 \
    --head_pcts 0.3 0.5 0.8 \
    --alphas 0.15 0.5 1.0 2.0 \
    --with_text_suppression \
    --with_sys_suppression \
    --include_post_softmax
```

**Parameters:**

### Core
- `--hard_samples`: Path to hard samples JSON (required)
- `--model`: Model to use
- `--output`: Output directory for results

### Layer Ranges
- `--layer_ranges`: Layer ranges to test (format: start-end)
  - Default: `5-10 8-15 10-15 15-25`
  - Tests: early (5-10), mid (8-15, 10-15), late (15-25)

### Head Percentages
- `--head_pcts`: Head top-k percentages
  - Default: `0.3 0.5 0.8`
  - Tests: 30%, 50%, 80% of vision-aware heads

### Alpha Values
- `--alphas`: Attention boost magnitudes
  - Default: `0.15 0.5 1.0 2.0`
  - From VAF-like (0.15) to strong boost (2.0)

### Suppression Options
- `--with_text_suppression`: Include text token suppression (text_beta=0.1)
- `--with_sys_suppression`: Include system prompt suppression (sys_beta=0.1)

### Post-Softmax Variant
- `--include_post_softmax`: Include post-softmax redistribution variant

**Total Configurations:**
- Baseline: 1
- SRF (no suppression): 4 layer_ranges × 3 head_pcts × 4 alphas = 48
- SRF (with text suppression): +48
- SRF (with sys suppression): +48
- Post-softmax (if enabled): +48

**Example:**
- Basic sweep: 1 + 48 = 49 configs
- Full sweep (all flags): 1 + 48×3 + 48 = 193 configs

---

## Step 3: Analyze Results

Results saved to `results/sweep_hard_pope/sweep_results.json`:

```json
{
  "args": { ... },
  "hard_samples_info": { ... },
  "results": {
    "baseline": {
      "correct": 40,
      "total": 100,
      "predictions": [...]
    },
    "srf_l8-15_h0.5_a1.0": {
      "correct": 45,
      "total": 100,
      "predictions": [...]
    },
    ...
  },
  "summary": [
    {
      "config": { ... },
      "accuracy": 0.4500,
      "correct": 45,
      "total": 100
    },
    ...
  ]
}
```

**Console output:**
- Full results table (all configs)
- Top 10 configurations ranked by accuracy
- Detailed parameter breakdown for top performers

---

## Quick Test (Small Sample)

To test the pipeline on a small sample first:

```bash
# Find 10 hard samples
python srf/find_hard_pope_samples.py \
    --n_samples 10 \
    --output srf/hard_samples_pope_test.json

# Quick sweep (minimal configs)
python srf/sweep_hard_pope.py \
    --hard_samples srf/hard_samples_pope_test.json \
    --layer_ranges 8-15 \
    --head_pcts 0.5 \
    --alphas 0.5 1.0 \
    --output results/sweep_test/
```

---

## Design Principles

Following project guidelines:

1. **Core files unchanged:** `srf/srf.py`, `srf/eval.py`, `my_analysis/qwen_attn_patch.py` remain stable
2. **No hardcoded parameters:** All hyperparameters via CLI arguments
3. **Wrapper scripts:** Created separate scripts for finding samples and sweeping
4. **Reproducible:** Hard sample indices saved for consistent comparison

---

## Troubleshooting

### Out of Memory
- Reduce `--n_samples` in find_hard_pope_samples.py
- Test with fewer configs first (remove some layer ranges or alphas)

### Slow Execution
- Start with small sample (10-20) to verify pipeline
- Use fewer layer ranges initially
- Consider testing on adversarial split only first

### No Improvement
- Expected given current 0.00% delta issue
- Focus on post-softmax variant (most promising from literature)
- Check saliency images are correct

---

## Next Steps

1. **Run baseline hard sample identification** (finds ~100 samples)
2. **Run quick sweep** (10 samples, few configs) to verify
3. **Run full sweep** on all hard samples
4. **Analyze top configs** and determine if any configuration shows promise
5. **If promising config found**, test on full POPE dataset

**Key Question:** Does ANY configuration show >5% improvement over baseline on hard samples?
