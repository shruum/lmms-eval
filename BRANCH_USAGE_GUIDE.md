# SRF Hallucination Mitigation - Branch Usage Guide

## Quick Start

This branch contains the **SRF (Spatial Reasoning Focus)** hallucination mitigation system for Vision-Language Models.

### Environment Setup
```bash
# Conda environment
conda activate mllm

# Set HuggingFace cache (optional, for faster model loading)
export HF_HOME=/path/to/hf_cache

# Repository location  
cd /home/anna2/shruthi/lmms-eval
```

### GPU Setup
```bash
# Check available GPUs
nvidia-smi

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
```

---

## Dataset Download & Setup

### POPE Dataset (Main Evaluation)
```bash
# Download POPE dataset
mkdir -p /home/anna2/shruthi/dataset/POPE_images
cd /home/anna2/shruthi/dataset/POPE_images

# Download from official repo
git clone https://github.com/AoiDragon/POPE.git temp_pope
mv temp_pope/coco/*.json ./
mv temp_pope/gqa/*.json ./
mv temp_pope/a-okvqa/*.json ./

# Download images (if not already present)
# COCO images: http://images.cocodataset.org/zips/val2014.zip
# GQA images: https://storage.googleapis.com/gqa/images.zip
# A-OKVQA images: https://storage.googleapis.com/okvqa/visual-genome/images.zip

# Organize images
mkdir -p images/val2014 images/gqa images/aokvqa
# Unzip respective datasets into these folders
```

### Other Datasets
```bash
# MMVP: Contact authors or check official repository
# MME: Download from official MME repository  
# VLM Bias: Check official evaluation setup
```

---

## Core Files & Usage

### 1. Main SRF Evaluation Script
```bash
# Basic usage
python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope \
  --pope_splits adversarial \
  --output results/test_run/

# Full POPE evaluation
python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope \
  --pope_splits random popular adversarial \
  --output results/pope_full/

# SRF-E (evidence-amplified)
python srf/eval.py \
  --method srfe \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --datasets mmvp \
  --beta 0.5 1.0 2.0 \
  --output results/srfe_sweep/
```

### 2. Hyperparameter Sweeps
```bash
# Run focused SRF sweep
bash run_focused_srf_sweep.sh

# Run comprehensive POPE sweep
bash run_srf_sweep.sh

# Launch diagnostic round (ultra-low alpha)
bash launch_diagnostic_round.sh

# Launch GQA comprehensive sweep  
bash launch_gqa_comprehensive_sweep.sh
```

### 3. Monitoring & Status
```bash
# Check detailed sweep status
bash check_sweep_detailed.sh

# Monitor GPU usage
watch -n 5 nvidia-smi

# Check running processes
ps aux | grep "eval.py" | wc -l

# Tail experiment logs
tail -f results/srf_focused_sweep/coco_adversarial_config11/run.log
```

---

## Important Configuration Files

### Core SRF Files
- **`srf/CONTEXT.md`** - Algorithm documentation, CLI reference, hyperparameters
- **`srf/config.py`** - Central configuration (models, datasets, hyperparameters)
- **`srf/eval.py`** - Main evaluation script
- **`srf/srf.py`** - SRF base implementation
- **`srf/srf_e.py`** - SRF-E (evidence-amplified) implementation

### Analysis Files
- **`my_analysis/qwen_attn_patch.py`** - Attention patching engine
- **`srf/saliency/clip_salience.py`** - CLIP-based saliency computation

### Documentation & Status
- **`SRF_TARGET_OBJECTIVES.md`** - Current goals and success criteria
- **`SRF_EXPERIMENT_STATUS.md`** - Latest experiment results
- **`POPE_BASELINE_COMPARISON.md`** - Baseline vs paper comparisons
- **`skills/vlm-proj-context.md`** - Comprehensive project context

---

## Key Parameters

### Model Parameters (in `srf/config.py`)
- **`layer_start`, `layer_end`**: Cross-modal fusion layer range
- **`head_top_k_pct`**: Fraction of vision-aware heads (default 0.20)
- **`clip_coarse_grid`**: CLIP patch grid (6 for LLaVA, 7 for Qwen)

### Dataset Parameters
- **`alpha`**: Boost factor for salient tokens (1.0-4.0 tested)
- **`eps`**: Suppression factor for background (0.1-0.2 tested)
- **`phase`**: "both" or "generation" for when to apply SRF

### Evaluation Parameters
- **`do_sample`**: Use sampling vs greedy decoding (recommended: True)
- **`temperature`**: Sampling temperature (recommended: 0.7)
- **`top_p`**: Nucleus sampling parameter (recommended: 0.9)

---

## Current Status & Results

### Latest Results (May 2026)
- **COCO Adversarial**: Best 78.77% vs baseline 79.30% (-0.53% gap)
- **GQA Adversarial**: Best 69.43% vs baseline 75.50% (-6.07% gap)
- **Total experiments**: 31+ configurations tested
- **Status**: ❌ SRF failed to beat baseline on LLaVA-1.5-7B

### Key Findings
1. Lower alpha works better (α=1.0 > α=2.0 > α=4.0)
2. Best layers: 10-18 for LLaVA-1.5-7B
3. Ultra-low alpha (0.1-0.5) showed promise but still below baseline
4. GQA performance catastrophically worse (-6% gap)

---

## Troubleshooting

### Common Issues
1. **GPU Memory**: Reduce batch size or use smaller models
2. **Dataset Paths**: Check `--pope_image_dir` points to correct location
3. **Model Loading**: Ensure `HF_HOME` is set if using cached models
4. **Baseline Discrepancies**: Ensure sampling decoding (`do_sample=True`)

### Debug Commands
```bash
# Test CLIP saliency
python -c "from srf.saliency.clip_salience import compute_clip_saliency; print('CLIP OK')"

# Test attention patching
python -c "from my_analysis.qwen_attn_patch import patch_model; print('Patching OK')"

# Quick sanity test
bash test_pope_sanity.sh
```

---

## Files to Commit for Remote Use

### Essential Files
```bash
# Core SRF implementation
srf/CONTEXT.md
srf/config.py
srf/eval.py
srf/srf.py
srf/srf_e.py
srf/eval_datasets.py

# Analysis & Tools
my_analysis/qwen_attn_patch.py
srf/saliency/clip_salience.py

# Documentation & Results
SRF_TARGET_OBJECTIVES.md
SRF_EXPERIMENT_STATUS.md  
POPE_BASELINE_COMPARISON.md
final_results_analysis.py

# Scripts & Monitoring
run_focused_srf_sweep.sh
check_sweep_detailed.sh
launch_gqa_comprehensive_sweep.sh
launch_diagnostic_round.sh

# Project Context
skills/vlm-proj-context.md
```

### Optional but Useful
```bash
# Additional documentation
SRF_DEBUG_CONTEXT.md
SRF_SWEEP_PROGRESS.md
POPE_COMPLETE_RESULTS_TABLE.md

# Additional scripts  
run_pope_*.sh
check_srf_*.sh
```

---

## Quick Reference

| Task | Command |
|------|---------|
| **Quick Test** | `python srf/eval.py --method srf --model llava-hf/llava-1.5-7b-hf --datasets pope --pope_splits adversarial --n_pope 10` |
| **Full POPE** | `bash run_pope_all_sampling_srf.sh` |
| **Status Check** | `bash check_sweep_detailed.sh` |
| **Monitor GPU** | `watch -n 5 nvidia-smi` |
| **Results Analysis** | `python final_results_analysis.py` |

---

*Last updated: May 2026 - Branch `srf-remote`*
*For detailed context, run: `/vlm-proj-context` or read `skills/vlm-proj-context.md`*