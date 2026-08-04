# SRF Project — Code Reference

> Stable reference: algorithm, file map, hyperparams, CLI.
> Only changes when the code structure changes. Commit alongside code.

---

## File Map

```
srf/
  config.py              ← SINGLE SOURCE OF TRUTH for all hyperparams
  eval.py                ← unified eval CLI (all datasets + all methods)
                            --method srf | srfe | vaf | vcd | vhr | ilvad | baseline
  eval_pope_val.py       ← quick 60-sample POPE val set (fast iteration)
  eval_mme.py            ← MME-specific eval
  eval_hallusionbench.py ← HallusionBench-specific eval
  eval_ablation.py       ← ablation sweep runner
  srf.py                 ← SRF base method; exposes last_clip_result dict
  srf_e.py               ← SRF-E (two-pass contrastive; Pass 2 = zeroed pixel_values)
  vaf.py                 ← VAF/ClearSight baseline (additive logit boost, all heads)
  vcd.py                 ← VCD baseline (diffusion-noisy contrastive decoding)
  vhr.py                 ← VHR baseline (per-sample o_proj head reinforcement)
  ilvad.py               ← ILVAD baseline (inter-layer attention discrepancy, chaining softmax)
  baseline.py            ← no-op baseline (vanilla model)
  test_srffovea_mmvp.py  ← SRF-Fovea sweep: pre-encoder spatial blur via CLIP saliency
                            fovea-only / srffovea configs; sigma sweep [20, 30, 50, 100]
  eval_datasets.py       ← dataset loaders
  noun_extract.py        ← CLIP query noun extraction (pope / mmbench / vlind modes)
  saliency/
    clip_salience.py     ← CLIP patch saliency; compute_clip_salience_full_gate_v3
    hssa_salience.py     ← hidden-state saliency
    lta_salience.py      ← last-token attention saliency
    eval_presence.py     ← saliency gate accuracy on 60-sample val set
  calibration/
    sweep_heads_layers.py ← layer/head sweep on val set

my_analysis/
  qwen_attn_patch.py     ← attention patching engine (shared by srf/)
                            patch_model / identify_visual_heads / update_sample / _STATE{}
                            bias modes: additive_logit | prob_interp | prob_scale |
                                        attn_floor | global_redistribute | budget_shift
                            ⚠️ CORE CODE — do NOT modify; wrap externally
  autoresearch*/         ← completed loops (reference only, do not modify)
```

---

## Hyperparameter System

Three-tier config in `srf/config.py`.

**Priority: CLI args → SRF_ARCH_PARAMS → SRF_DATASET_PARAMS → SRF_DEFAULTS**

```python
SRF_DEFAULTS          # shared: sys_beta=0.30, calib_n=20, bias_mode="additive_logit", …

SRF_DATASET_PARAMS    # per-dataset (arch-agnostic):
  pope / mme / hallusionbench / mmbench:  phase=generation, alpha=4.0, eps=0.2
  mmvp:               phase=generation, alpha=4.0, eps=0.2
  vlmbias:            phase=generation, alpha=8.0, eps=0.5

SRF_ARCH_PARAMS       # per-model (scale with depth — re-tune for new models):
  layer_start         # first layer of cross-modal fusion zone
  layer_end           # last layer
  dataset_layer_end   # per-dataset fine-tune: {"mmvp": 15, "pope": 12, …}
  head_top_k_pct      # fraction of heads selected as vision-aware (default 0.20)
  clip_coarse_grid    # CLIP patch grid (7 for Qwen/448px, 6 for LLaVA/336px)
  clip_top_k_pct      # fraction of image tokens boosted (default 0.30)
  clip_fallback_thresh # presence threshold for CLIP gate (v3: 0.21)
```

Supported models:
```
Qwen/Qwen2.5-VL-3B-Instruct   TUNED   layer_start=6, layer_end=12, sys_beta=0.30
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

2. reset_for_dataset(dataset, *, phase, alpha, eps, neg_absent_alpha,
                      layer_start, layer_end, head_top_k_pct,
                      clip_coarse_grid, clip_top_k_pct, clip_fallback_thresh,
                      saliency_mode, bias_mode, vr_target, vr_k, ...)
   - Merge arch + dataset params; apply any CLI overrides (None = use config)
   - Re-calibrate heads if head_top_k_pct changed; sync patch state
   - Sets noun extraction mode (mme/hallusionbench → "pope" mode)
   - last_clip_result dict populated by prepare_sample (used by B4 retry)

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

### SRF-Fovea (pre-encoder spatial blur)
```
File: srf/test_srffovea_mmvp.py
Algorithm (per sample):
  1. Compute CLIP saliency → weight map W ∈ [0,1] per image patch
  2. Gaussian-blur original image: blurred = GaussianBlur(image, σ)
  3. Foveated image = W * original + (1−W) * blurred
     — salient patches stay sharp; peripheral patches are blurred
  4. Feed foveated image to ViT (instead of original)
  5. Optionally also apply SRF attention boost (srffovea config)

Best σ: 20. MMVP pair_acc: baseline=40%, srffovea_20=43.33% (+3.33pp)
POPE: neutral at σ=20 (95.0%); σ≥50 hurts.
```

### SRF-E (evidence-amplified)
```
File: srf/srf_e.py
Two forward passes:
  logits_full  = model(**inp_with_image)       # SRF active
  logits_noval = model(**inp_image_zeroed)     # pixel_values = zeros
  logits_final = logits_full + γ * (logits_full - logits_noval)

Best γ: 3.0 (MMVP=49.33%, POPE=87.70%).
BROKEN for VLMBias/VLind multi-token generation: zeroed ViT input corrupts generation.
Next: replace zeros with blurred image → may fix collapse while preserving contrastive signal.
```

### Comparison Baselines (all wired into eval.py --method)
```
baseline.py   — vanilla model, no intervention
vaf.py        — VAF/ClearSight: additive logit boost on image tokens (all heads, layers 6-14)
vcd.py        — VCD: diffusion-noisy (t=500) contrastive: logits = (1+α)*orig − α*noisy
vhr.py        — VHR: per-sample o_proj head reinforcement via text-contrast VHD
                      aug_heads selected by VHD > median; aug_ratio=2.0; target layers {1}∪{last14}
ilvad.py      — ILVAD: inter-layer attention discrepancy; chaining softmax wrapper (reads
                       patch._STATE read-only; does NOT modify qwen_attn_patch.py)

All accept same interface: setup / reset_for_dataset / prepare_sample(**kwargs) /
get_contrastive_logits(**kwargs) / generate_contrastive(**kwargs) / cleanup
```

---

## CLI Reference

```bash
# ── Full eval (all datasets, GPU, ~75 min for POPE) ──────────────────────────
python srf/eval.py \
  --method      srf | srfe \
  --model       Qwen/Qwen2.5-VL-3B-Instruct \
  --datasets    mmvp pope vlmbias mme \
  --beta        2.0                           # SRF-E only; list = sweep
  --pope_splits adversarial popular random    # default: all 3
  --n_pope      -1                            # per-split cap (-1 = all)
  --output      results/run_name/

  # Arch overrides (None = use config default)
  --layer_start 6  --layer_end 12  --head_top_k_pct 0.20
  --clip_coarse_grid 7  --clip_top_k_pct 0.30

  # Saliency mode (default: clip_full_gate_v3 — best, do not change without sweep)
  --saliency_mode clip_full_gate_v3   # best | clip (basic) | hssa | lta | srf2

  # Dataset overrides
  --alpha 4.0  --eps 0.2  --phase generation

  # Boosting method (default: additive_logit — best)
  --bias_mode additive_logit          # best | budget_shift (B3) | prob_interp | prob_scale
  --neg_absent_alpha 0.0              # absent suppression: 0=off (default), try 2.0
  --vr_target 0.15  --vr_k 3.0       # B1 visual reliance; vr_target=0 disables (default)

# ── Quick 60-sample val set (~3 min, use for all sweeps) ─────────────────────
python srf/eval_pope_val.py \
  --srf                               # enable SRF (omit for baseline)
  --srf_mode clip_full_gate_v3        # saliency mode override
  --alpha 4.0 --neg_absent_alpha 0.0
  --bias_mode additive_logit          # boosting method (see below)
  --vr_target 0.15 --vr_k 3.0        # B1: visual reliance compensation
  --b4 --b4_threshold 0.25 --b4_multiplier 2.0  # B4: two-pass retry
  --out results/pope_val_srf.json
```

### Boosting method quick reference

| Flag | Method | Val acc | Notes |
|------|--------|---------|-------|
| *(default)* | `additive_logit` | 0.900 | Best; boosts logits before softmax |
| `--bias_mode budget_shift` | B3 | 0.867 | No gain; image budget not bottleneck |
| `--vr_target 0.15` | B1 | 0.900 | Same ceiling; adaptive alpha in-loop |
| `--b4` | B4 | 0.900 | Two-pass retry; same failures as B1 |

Val set ceiling = **0.900**. Remaining 6 FNs are model capacity limits, not saliency failures.

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
  srf/docs/CONTEXT.md  srf/docs/RESEARCH_STATUS.md  srf/docs/EXPERIMENTS.md
  skills/load-mllm/SKILL.md
```
