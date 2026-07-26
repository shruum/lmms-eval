# Parallel Experiments on 8 GPUs

## Overview

Run two experiments simultaneously on 8 GPUs:
- **GPUs 0-3**: Hard POPE sample sweep (current SRF configurations)
- **GPUs 4-7**: Layer-Specific Modulation sweep (Idea 2 from Next_steps.md)

---

## Prerequisites

1. **Find hard POPE samples first** (one-time):
```bash
cd /home/anna2/shruthi/lmms-eval
conda activate mllm

python srf/find_hard_pope_samples.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --pope_splits adversarial \
    --n_samples 100 \
    --output srf/hard_samples_pope.json
```

---

## Experiment 1: Hard POPE Sweep (GPUs 0-3)

Tests SRF configurations on samples where baseline fails.

**Configurations tested:**
- 1 baseline
- 48 SRF configs (4 layer ranges × 3 head % × 4 alphas)
- 96 SRF with suppression (text + sys)
- 48 post-softmax configs
- **Total: 193 configurations on 100 hard samples**

### Launch on GPUs 0-3:

```bash
python srf/sweep_hard_pope_parallel.py \
    --hard_samples srf/hard_samples_pope.json \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --output results/sweep_hard_pope_parallel/ \
    --num_gpus 4 \
    --layer_ranges 5-10 8-15 10-15 15-25 \
    --head_pcts 0.3 0.5 0.8 \
    --alphas 0.15 0.5 1.0 2.0 \
    --with_text_suppression \
    --with_sys_suppression \
    --include_post_softmax
```

**Estimated time:** ~2-3 hours on 4 GPUs (193 configs / 4 GPUs)

**Output:** `results/sweep_hard_pope_parallel/aggregated_results.json`

---

## Experiment 2: Layer-Specific Modulation (GPUs 4-7)

Tests three-zone layer-specific modulation (Idea 2 from Next_steps.md).

**Three zones:**
- **Early** (0 to early_end): Gentle visual detection boost
- **Mid** (early_end to mid_end): Saliency-guided fusion
- **Late** (mid_end to end): Suppress language priors

**Configurations tested:**
- 1 baseline
- 243 layer-specific configs (3 early_ends × 3 mid_ends × 3 α_early × 3 α_mid × 3 β_late)
- **Total: 244 configurations on 100 hard samples**

### Launch on GPUs 4-7:

```bash
python srf/sweep_layer_specific_parallel.py \
    --hard_samples srf/hard_samples_pope.json \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --output results/sweep_layer_specific/ \
    --num_gpus 4 \
    --early_ends 5 7 10 \
    --mid_ends 15 17 20 \
    --alpha_early 0.3 0.5 1.0 \
    --alpha_mid 1.0 2.0 4.0 \
    --beta_late 0.05 0.1 0.2
```

**Estimated time:** ~2-3 hours on 4 GPUs (244 configs / 4 GPUs)

**Output:** `results/sweep_layer_specific/aggregated_results.json`

---

## Launch Both Experiments Simultaneously

### Option 1: Separate terminals

Open two terminals and run each experiment:

**Terminal 1:**
```bash
cd /home/anna2/shruthi/lmms-eval
conda activate mllm

# Hard POPE sweep on GPUs 0-3
python srf/sweep_hard_pope_parallel.py \
    --hard_samples srf/hard_samples_pope.json \
    --output results/sweep_hard_pope_parallel/ \
    --num_gpus 4 \
    --layer_ranges 5-10 8-15 10-15 15-25 \
    --head_pcts 0.3 0.5 0.8 \
    --alphas 0.15 0.5 1.0 2.0 \
    --with_text_suppression \
    --with_sys_suppression \
    --include_post_softmax
```

**Terminal 2:**
```bash
cd /home/anna2/shruthi/lmms-eval
conda activate mllm

# Layer-specific modulation on GPUs 4-7
python srf/sweep_layer_specific_parallel.py \
    --hard_samples srf/hard_samples_pope.json \
    --output results/sweep_layer_specific/ \
    --num_gpus 4 \
    --early_ends 5 7 10 \
    --mid_ends 15 17 20 \
    --alpha_early 0.3 0.5 1.0 \
    --alpha_mid 1.0 2.0 4.0 \
    --beta_late 0.05 0.1 0.2
```

### Option 2: Background with nohup

```bash
# Terminal 1
nohup python srf/sweep_hard_pope_parallel.py \
    --hard_samples srf/hard_samples_pope.json \
    --output results/sweep_hard_pope_parallel/ \
    --num_gpus 4 \
    --layer_ranges 5-10 8-15 10-15 15-25 \
    --head_pcts 0.3 0.5 0.8 \
    --alphas 0.15 0.5 1.0 2.0 \
    --with_text_suppression \
    --with_sys_suppression \
    --include_post_softmax \
    > logs/hard_pope.log 2>&1 &

# Terminal 2
nohup python srf/sweep_layer_specific_parallel.py \
    --hard_samples srf/hard_samples_pope.json \
    --output results/sweep_layer_specific/ \
    --num_gpus 4 \
    --early_ends 5 7 10 \
    --mid_ends 15 17 20 \
    --alpha_early 0.3 0.5 1.0 \
    --alpha_mid 1.0 2.0 4.0 \
    --beta_late 0.05 0.1 0.2 \
    > logs/layer_specific.log 2>&1 &

# Watch progress
tail -f logs/hard_pope.log
tail -f logs/layer_specific.log
```

---

## Monitoring Progress

### Check GPU utilization:
```bash
watch -n 1 nvidia-smi
```

### Check individual GPU logs:
```bash
# Hard POPE sweep
tail -f results/sweep_hard_pope_parallel/gpu_0.log
tail -f results/sweep_hard_pope_parallel/gpu_1.log
tail -f results/sweep_hard_pope_parallel/gpu_2.log
tail -f results/sweep_hard_pope_parallel/gpu_3.log

# Layer-specific sweep
tail -f results/sweep_layer_specific/gpu_0.log
tail -f results/sweep_layer_specific/gpu_1.log
tail -f results/sweep_layer_specific/gpu_2.log
tail -f results/sweep_layer_specific/gpu_3.log
```

### Check completion:
```bash
# Count completed configs
ls results/sweep_hard_pope_parallel/gpu_*_results.json | wc -l
ls results/sweep_layer_specific/gpu_*_results.json | wc -l
```

---

## Expected Results

### Experiment 1: Hard POPE Sweep

**Key questions:**
1. Does ANY SRF configuration improve over baseline on hard samples?
2. Is post-softmax more stable than pre-softmax?
3. Do text/system suppression help?

**Success criteria:**
- Any config with >5% accuracy gain over baseline
- Post-softmax showing consistent improvement
- Clear pattern in optimal parameters

### Experiment 2: Layer-Specific Modulation

**Key questions:**
1. Does treating layers differently help?
2. Which zone boundaries work best?
3. Is early-gentle / mid-strong / late-suppress better than uniform?

**Success criteria:**
- Layer-specific outperforming uniform SRF
- Optimal zone boundaries identified
- Clear pattern in alpha/beta values

---

## Combined Analysis

After both experiments complete:

```bash
# Compare top configs from both approaches
python -c "
import json

# Load hard POPE results
with open('results/sweep_hard_pope_parallel/aggregated_results.json') as f:
    hard_pope = json.load(f)

# Load layer-specific results
with open('results/sweep_layer_specific/aggregated_results.json') as f:
    layer_spec = json.load(f)

print('HARD POPE TOP 5:')
for i, e in enumerate(hard_pope['summary'][:5]):
    print(f\"  #{i+1}: {e['config']['name']} - acc={e['accuracy']:.4f}\")

print('\nLAYER-SPECIFIC TOP 5:')
for i, e in enumerate(layer_spec['summary'][:5]):
    print(f\"  #{i+1}: {e['config']['name']} - acc={e['accuracy']:.4f}\")
"
```

---

## Troubleshooting

### GPUs not detected
```bash
# Check GPU status
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.device_count())"
```

### Out of memory
- Reduce hard samples (`--n_samples 50` instead of 100)
- Test with fewer configs first
- Use smaller model (Qwen2.5-VL-3B is already small)

### Slow execution
- Check GPU utilization (`nvidia-smi`)
- Verify all GPUs are being used
- Check individual GPU logs for errors

### One experiment finishes before the other
- Normal if different config counts
- Finished GPUs will be idle (that's OK)
- Results are valid as soon as each completes

---

## Next Steps After Experiments

1. **Analyze top configs** from each approach
2. **Compare** layer-specific vs uniform SRF
3. **Test best config** on full POPE dataset
4. **If promising**, test on MMVP and MME
5. **Report findings** in `srf/RESEARCH_STATUS.md`

---

## Summary

**Total configurations:** 193 (hard POPE) + 244 (layer-specific) = **437 configs**

**Total GPU-hours:** ~4-6 hours on 8 GPUs

**Output:** Two comprehensive result files comparing:
- Baseline vs SRF (pre/post-softmax, with/without suppression)
- Baseline vs Layer-Specific Modulation (various zone boundaries)

**Goal:** Find ANY configuration that shows >5% improvement over baseline on hard samples.
