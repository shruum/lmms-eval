# SRF Project — Code Reference

> Stable reference: algorithm, file map, hyperparams, CLI.
> Only changes when the code structure changes. Commit alongside code.

---

## File Map

```
srf/
  config.py           ← SINGLE SOURCE OF TRUTH for all hyperparams
  eval.py             ← unified eval CLI
  srf.py              ← SRF base method
  srf_e.py            ← SRF-E (two-pass contrastive)
  eval_datasets.py    ← dataset loaders
  noun_extract.py     ← CLIP query noun extraction
  saliency/
    clip_salience.py  ← CLIP patch saliency (cross-modal encoder)
    hssa_salience.py  ← hidden-state saliency

my_analysis/
  qwen_attn_patch.py  ← attention patching engine (shared by srf/)
                         patch_model / identify_visual_heads / update_sample / _STATE{}
  autoresearch*/      ← completed loops (reference only, do not modify)
```

---

## Hyperparameter System

Three-tier config in `srf/config.py`.

**Priority: CLI args → SRF_ARCH_PARAMS → SRF_DATASET_PARAMS → SRF_DEFAULTS**

```python
SRF_DEFAULTS          # shared: sys_beta=0.10, calib_n=20, bias_mode, prob_floor, …

SRF_DATASET_PARAMS    # per-dataset (arch-agnostic):
  mmvp / pope / mme:  phase=both,       alpha=4.0, eps=0.2
  vlmbias:            phase=generation, alpha=8.0, eps=0.5

SRF_ARCH_PARAMS       # per-model (scale with depth — re-tune for new models):
  layer_start         # first layer of cross-modal fusion zone  (~8/28 * n_layers)
  layer_end           # last layer                              (~15/28 * n_layers)
  dataset_layer_end   # per-dataset fine-tune: {"mmvp": 15, "pope": 15, …}
  head_top_k_pct      # fraction of heads selected as vision-aware (default 0.20)
  clip_coarse_grid    # CLIP patch grid (7 for Qwen/448px, 6 for LLaVA/336px)
  clip_top_k_pct      # fraction of image tokens boosted (default 0.30)
  clip_fallback_thresh # below this CLIP sim → uniform boost (default 0.20)
```

Supported models:
```
Qwen/Qwen2.5-VL-3B-Instruct   TUNED   layer_start=8, layer_end=15
Qwen/Qwen2.5-VL-7B-Instruct   NOT TUNED (proportional: start=9, end=17)
llava-hf/llava-1.5-7b-hf      NOT TUNED (start=8, end=20; image_token=None → model.config)
```

---

## Algorithm

### SRF (base)
```
1. setup(model, processor, calib_dataset)
   - Detect model_id from model.config._name_or_path → look up SRF_ARCH_PARAMS
   - Calibrate: run calib_n samples, identify top head_top_k_pct vision-aware heads
   - Patch model with qwen_attn_patch

2. reset_for_dataset(dataset, *, phase, alpha, eps, layer_start, layer_end,
                      head_top_k_pct, clip_coarse_grid, clip_top_k_pct, clip_fallback_thresh)
   - Merge arch + dataset params; apply any CLI overrides (None = use config)
   - Re-calibrate heads, sync patch state
   - Sets noun extraction mode (mme/hallusionbench → "pope" mode)

3. prepare_sample(inp, img_start, img_end, image, question, model, processor)
   - Extract noun from question (extract_clip_noun)
   - Compute CLIP saliency → top-k image token mask
   - Push mask + params into patch._STATE

4. model(**inp)   [patched forward]
   - In layers [layer_start, layer_end], for vision-aware heads:
     boost attention logits for salient image tokens by alpha
     suppress background image tokens by eps

5. cleanup()      reset per-sample patch state
```

### SRF-E (evidence-amplified)
```
Two forward passes:
  logits_full  = model(**inp_with_image)
  logits_noval = model(**inp_image_zeroed)   # pixel_values set to zeros
  logits_final = logits_full + β * (logits_full - logits_noval)

Amplifies visual evidence, suppresses language prior.
Best β: 2.0 for MMVP. Sweep [0.5, 1.0, 2.0] for new datasets.
BROKEN for VLM Bias (contrastive suppresses { token). Use SRF base there.
```

---

## CLI Reference

```bash
python srf/eval.py \
  --method      srf | srfe \
  --model       Qwen/Qwen2.5-VL-3B-Instruct \
  --datasets    mmvp pope vlmbias mme \
  --beta        2.0                           # SRF-E only; list = sweep
  --pope_splits adversarial popular random    # default: all 3
  --n_pope      -1                            # per-split cap (-1 = all)
  --output      results/run_name/

  # Arch overrides (None = use config)
  --layer_start 8  --layer_end 15  --head_top_k_pct 0.20
  --clip_coarse_grid 7  --clip_top_k_pct 0.30  --clip_fallback_thresh 0.20

  # Dataset overrides
  --alpha 4.0  --eps 0.2  --phase both
```

---

## Datasets

| Dataset | N | Task | Metric |
|---------|---|------|--------|
| MMVP | 150 pairs (300 img) | A/B choice | pair acc (both correct) |
| POPE | 9000 (3×3000) | Yes/No | question acc |
| MME | 2374 (14 cats) | Yes/No | score (sum correct), pair acc, perception/cognition |
| VLM Bias | ~300 (7 cats) | generation | exact match after `{}` extraction |

---

## Environment

```bash
# Conda env: mllm
export HF_HOME=/path/to/hf_cache    # machine-specific
export CUDA_VISIBLE_DEVICES=0
cd /path/to/lmms-eval               # repo root

# Import smoke test (no GPU)
python -c "import sys; sys.path.insert(0,'srf'); import config; print(config.DEFAULT_MODEL)"

# Full eval
conda run -n mllm python srf/eval.py --method srf --datasets pope --pope_splits adversarial
```

---

## Git

```
Branch: autoresearch/mmvp-srf   ← all SRF work
Main:   main                    ← upstream lmms-eval (do not touch)

Files to push for remote eval:
  srf/config.py  srf/eval.py  srf/srf.py  srf/srf_e.py
  srf/eval_datasets.py  srf/noun_extract.py
  srf/saliency/clip_salience.py  srf/saliency/hssa_salience.py
  my_analysis/qwen_attn_patch.py
  srf/CONTEXT.md  srf/RESEARCH_STATUS.md
  skills/load-mllm/SKILL.md
---

## Related Work

| Method | Paper | Technique | Key Datasets |
|--------|-------|-----------|--------------|
| **VAF** (ClearSight) | arXiv:2503.13107 | Boost visual attention in middle layers (α=0.15), suppress prompts (β=0.1) | POPE, MME, NoCaps, ScienceQA |
| **AIR** (LookCarefully) | arXiv:2602.24041 | OT-guided patch selection + FFN reinforcement (layers 24-32) | CHAIR, POPE, MME, MMBench, LLaVA-Bench |
| **SRF** (ours) | — | CLIP-guided saliency + attention boost/suppress (layers 8-15) | MMVP, POPE, MME, VLM Bias |

**SRF differentiators:** CLIP cross-modal guidance, noun-based patch selection, two-pass contrastive (SRF-E)

---

### Comparison Gaps

**Missing datasets:** CHAIR, MMBench (both VAF/AIR use these)

**Missing models:** LLaVA-1.5-13B, Qwen2.5-VL-7B, GLM-4V-9B

**Model versions tested:**
- VAF/AIR papers: Old Qwen-VL-7B (v1, 2023)
- Our SRF: Qwen2.5-VL-3B (newer, 2024) ← Novel contribution

**Our unique coverage:** Qwen2.5-VL-3B (not tested in VAF/AIR), MMVP, VLM Bias
```
