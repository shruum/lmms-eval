---
name: vlm-proj-context
description: This skill should be used when the user asks to "load vlm project context", "load hallucination mitigation context", "/vlm-proj-context", or discusses VLM hallucination mitigation, SRF (Spatial Reasoning Focus), CLIP-guided saliency steering, or the lmms-eval SRF project. Provides comprehensive project context for VLM hallucination research.
version: 1.0.0
---

# VLM Hallucination Mitigation Research — Project Context

## Research Goal

**Mitigate hallucination in Vision-Language Models (VLMs) by steering attention toward semantically relevant image regions using inference-time strategies.**

**Core Problem**: VLMs frequently hallucinate objects that don't exist in images (e.g., answering "yes" to "Is there a cat?" when no cat is present).

**Approach**: CLIP-guided spatial reasoning steering (SRF) to boost attention to relevant regions and suppress background.

---

## Step 1 — Locate the Repository

```bash
cd /home/anna2/shruthi/lmms-eval
```

Call this `{REPO}`. All paths below are relative to `{REPO}`.

---

## Step 2 — Read Core Context Files (Read in Parallel)

| File | Purpose |
|------|---------|
| `{REPO}/srf/CONTEXT.md` | Algorithm, file map, hyperparameters, CLI reference |
| `{REPO}/auto_research/Lit.md` | Literature review: foveation, token pruning, negative prompting |
| `{REPO}/srf/config.py` | Central configuration (models, datasets, hyperparameters) |
| `{REPO}/srf/RESEARCH_STATUS.md` | Current results, open tasks, experiment logs |

---

## Step 3 — Understand the Research Context

### Problem Statement

VLMs suffer from **object hallucination**: generating text that describes objects not present in the image. This is particularly problematic for:
- **Absent objects** (most common): "Is there a cat?" → "yes" (when no cat exists)
- **Spatial reasoning**: Failing to focus on relevant regions for answering
- **Language prior**: Answering based on question wording rather than visual evidence

### Solution: SRF (Spatial Reasoning Focus)

**Core Mechanism**:
1. Extract query nouns from questions (e.g., "cat" from "Is there a cat?")
2. Compute CLIP-based saliency maps to identify relevant image patches
3. During inference, in cross-modal fusion layers (8-15):
   - **Boost** attention to salient image tokens by `α` (default 4.0)
   - **Suppress** background tokens by `ε` (default 0.2)
4. This steers the VLM to look at the right regions when answering

**Variants**:
- **SRF (base)**: Single-pass attention steering
- **SRF-E (evidence-amplified)**: Two-pass contrastive (with image vs. zeroed image), amplifies visual evidence

### Key Innovations

1. **CLIP cross-modal guidance**: Uses external CLIP model to identify relevant regions based on query
2. **Query-conditioned selection**: Different questions focus on different regions
3. **Training-free**: Pure inference-time intervention, no model weight updates
4. **Layer-specific**: Targets middle fusion layers where cross-modal reasoning happens
5. **Noun-based**: Extracts relevant concepts from questions to guide saliency

### Related Work (from Literature Review)

| Method | Technique | Key Idea |
|--------|-----------|----------|
| **LLMind** (CVPR 2026) | Möbius warp + SPSA optimization | Foveation via adaptive sampling, but requires ground-truth |
| **Foveated Reasoner** | Autoregressive foveation | Triggers high-res crops during decoding, but requires training |
| **ADSC** | Attention-driven self-compression | LLM prunes its own vision tokens based on attention |
| **Multimodal Unlearning** | Negative prompts | Steer away from biased associations at inference time |
| **AIR** (LookCarefully) | OT-guided patch selection | Prunes non-salient patches, reinforces salient ones |
| **VAF** (ClearSight) | Boost visual attention in middle layers | Similar to SRF but without CLIP guidance |

**SRF Differentiators**:
- CLIP-based external guidance (vs. internal attention only)
- Noun-based query conditioning (vs. fixed saliency)
- Two-pass contrastive amplification (SRF-E)
- Focus on absence detection (hallucination mitigation)

---

## Step 4 — Current Implementation Status

### 🚨 CRITICAL ISSUE (April 2026)

**SRF gets exactly 0.00% delta across ALL datasets**:
- POPE: +0.05% (3 categories, 9000 samples)
- MME: 0.00% (14 categories, 2374 samples)
- MMVP: 0.00% (150 pairs, 300 images)

### Debug Findings (What Works)
- ✅ Saliency masks are NOT zero (mean 0.31-0.73, 31-73% of tokens selected)
- ✅ Saliency images look mostly correct (CLIP focuses on right regions)
- ✅ Code is working (enh_para set, masks applied, logits differ)
- ❌ But accuracy delta = 0.00% (different logits → same decisions)

### What We've Tested
| Configuration | Result |
|--------------|--------|
| Pre-softmax additive (α=2.0) | 0.00% Δ |
| Post-softmax redistribute | 0.00% Δ |
| VAF-like (α=0.15, layers 10-15, heads 50%) | 0.00% Δ |
| All layer/head/alpha combinations | Same accuracy |

### Root Cause Mystery
Different logits → Same accuracy suggests boosting doesn't affect final decision. Either:
1. Boosting is applied outside cross-modal fusion zone
2. Boost strength is insufficient or misdirected
3. Fundamental problem with approach (not just configuration)

### Next Step: Large-Scale Testing
Test on **100+ POPE samples** (where baseline does worse) with:
- Pre/post-softmax methods
- Layer ranges: 5-10, 8-15, 10-15, 15-25
- Head percentages: 0.3, 0.5, 0.8
- Alpha values: 0.15, 0.5, 1.0, 2.0
- With/without suppression (sys_beta, text_beta)
- All via CLI arguments (NOT config file changes)

### Models Previously Tested

| Model | Status | Notes |
|-------|--------|-------|
| **Qwen2.5-VL-3B** | ✅ Tuned | `layer_start=8, layer_end=15` — main model |
| **Qwen2.5-VL-7B** | ⚠️ Not tuned | Proportional scaling: `start=9, end=17` |
| **LLaVA-1.5-7B** | ⚠️ Not tuned | `start=8, end=20` |

### Datasets

| Dataset | N | Task | Metric | Status |
|---------|---|------|--------|--------|
| **MMVP** | 150 pairs (300 img) | A/B choice | Pair accuracy | 0.00% Δ ❌ |
| **POPE** | 9000 (3×3000) | Yes/No | Question accuracy | +0.05% Δ ❌ |
| **MME** | 2374 (14 cats) | Yes/No | Score, pair acc | 0.00% Δ ❌ |
| **VLM Bias** | ~300 (7 cats) | Generation | Exact match | ~5% (noise) |

### Hyperparameter System

Three-tier config in `srf/config.py`:

**Priority**: `CLI args → SRF_ARCH_PARAMS → SRF_DATASET_PARAMS → SRF_DEFAULTS`

```python
SRF_DEFAULTS          # Shared: sys_beta=0.10, calib_n=20, bias_mode, prob_floor
SRF_DATASET_PARAMS    # Per-dataset (arch-agnostic)
SRF_ARCH_PARAMS       # Per-model (scales with depth)
```

**Key Parameters**:
- `layer_start`, `layer_end`: Fusion zone boundaries (model-specific)
- `head_top_k_pct`: Fraction of vision-aware heads (default 0.20)
- `alpha`: Boost factor for salient tokens (default 4.0)
- `eps`: Suppression factor for background (default 0.2)
- `clip_coarse_grid`: CLIP patch grid (7 for Qwen/448px, 6 for LLaVA/336px)
- `clip_top_k_pct`: Fraction of image tokens boosted (default 0.30)

---

## Step 5 — Quick Start Commands

### Environment Setup

```bash
# Conda environment
conda activate mllm
export HF_HOME=/path/to/hf_cache
export CUDA_VISIBLE_DEVICES=0

# Repository location
cd /home/anna2/shruthi/lmms-eval
```

### Evaluation Commands

```bash
# Quick test (POPE, adversarial split)
python srf/eval.py \
  --method srf \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --datasets pope \
  --pope_splits adversarial \
  --n_pope 10 \
  --output results/test_run/

# Full evaluation (all datasets)
python srf/eval.py \
  --method srf \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --datasets mmvp pope mme vlmbias \
  --output results/srf_full/

# SRF-E (evidence-amplified, with beta sweep)
python srf/eval.py \
  --method srfe \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --datasets mmvp \
  --beta 0.5 1.0 2.0 \
  --output results/srfe_sweep/
```

### Code Architecture

```
srf/
├── config.py              # SINGLE SOURCE OF TRUTH for all hyperparams
├── eval.py                # Unified evaluation CLI
├── srf.py                 # SRF base implementation
├── srf_e.py               # SRF-E (two-pass contrastive)
├── eval_datasets.py       # Dataset loaders
├── noun_extract.py        # CLIP query noun extraction
└── saliency/
    ├── clip_salience.py   # CLIP patch saliency (cross-modal encoder)
    └── hssa_salience.py   # Hidden-state saliency (experimental)

my_analysis/
├── qwen_attn_patch.py     # Attention patching engine (shared by srf/)
│                           # Contains: patch_model, identify_visual_heads,
│                           #          update_sample, _STATE{}
└── autoresearch*/         # Completed experiment loops (reference only)
```

### Current Branch

```bash
git branch --show-current
```

Expected: `autoresearch/mmvp-srf` or similar SRF-focused branch.

---

## Step 7 — Debug Context Files

When investigating the 0.00% delta issue, read these:

| File | Purpose |
|------|---------|
| `{REPO}/SRF_DEBUG_CONTEXT.md` | Debug investigation details, root causes, test results |
| `{REPO}/Next_steps.md` | Literature review, 8 alternative approaches, experimental plan |
| `{REPO}/srf/config.py` | See `SRF_DATASET_PARAMS`, `SRF_ARCH_PARAMS` for current config |
| `{REPO}/my_analysis/qwen_attn_patch.py` | Attention patching (verify boosting is applied) |

### Key Debug Commands

```bash
# Quick debug test (2 samples, check saliency)
python srf/eval.py \
  --method srf \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --datasets mmvp \
  --n_mmvp 2 \
  --output /tmp/mmvp_debug/

# Check saliency images
ls results/saliency_images/

# Run with specific config (via CLI, not config edits)
python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope \
  --n_pope 100 \
  --layer_start 10 \
  --layer_end 15 \
  --head_top_k_pct 0.5 \
  --alpha 0.15 \
  --bias_mode global_redistribute \
  --output results/test_config/
```

---

## Step 6 — Key Design Principles

Based on literature review, the approach incorporates:

| Principle | Source | Application |
|-----------|--------|-------------|
| **Absence-aware processing** | Multimodal Unlearning | Different strategies for present vs. absent objects |
| **Query-conditioned selection** | Foveated Reasoner | Different tokens boosted per question |
| **Negative prompting** | PhysVid | Potential: steer away from biased patterns |
| **Training-free optimization** | LLMind (CSF) | Inference-time only, no gradient updates |
| **Adaptive thresholds** | AdaptVis (ICML 2025) | Layer-gated interventions |
| **Token selectivity** | AIR (ICLR 2026) | Boost salient, suppress non-salient |

---

## Step 7 — Future Directions (from Lit.md)

1. **CLIP-Guided Negative Prompting**: Different strategies for present vs absent
2. **Contrastive Saliency**: Subtract background distractor saliency from query saliency
3. **Möbius Warp + Attention**: Zoom into salient regions before VLM processing
4. **Layer-Adaptive Thresholds**: Per-layer absence detection calibration
5. **Ensemble Saliency**: Combine CLIP (external) + attention rollout (internal)

---

## Important Notes

- **🔴 ACTIVE BUG**: 0.00% delta across all datasets - see `SRF_DEBUG_CONTEXT.md`
- **Don't edit config.py** for testing - use CLI arguments instead
- **Save saliency images** for qualitative analysis (in `results/saliency_images/`)
- **Test on samples where baseline does WORSE** (harder cases)
- **Compare multiple configurations fairly** on the same sample set
- **All results** should be logged in `srf/RESEARCH_STATUS.md`
- **Debug output**: Check `[DEBUG SRF]` and `[DEBUG method_get_logits]` prints
- **Temp file cleanup**: Check `/tmp` for leaked Qwen temp PNG files

---

*Last updated: 2026-04-30 — Active debugging of 0.00% delta issue.*
