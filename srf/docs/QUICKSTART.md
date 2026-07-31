# SRF Quickstart — Run on a New Machine

> Single self-contained page. Read `config.py` first for hyperparams, then come back here for run commands.

---

## Environment

```bash
conda activate mllm
cd /volumes2/mllm/lmms-eval
export HF_HOME=/volumes2/hugging_face_cache
```

---

## Complete Method Description

What SRF actually does — every intervention component with actual values.

### 1. Head Calibration (once per model load)

20 random calibration samples → forward pass in baseline mode → capture attention weights from text-query positions to image-key positions → select top 20% heads by mean image-attention score → store as `head_mask (bool, n_heads)`.

### 2. Noun Extraction (per sample)

Question → object noun via `noun_extract.py::extract_clip_noun()`.

Examples:
- "Is there a chair in the image?" → `"chair"`
- "Is the butterfly's wings open?" → `"butterfly"`
- "How many logos are on this image?" → `"logos"`

### 3. Bad-Noun Gate (per sample)

If the extracted noun is in the hardcoded stop-word set `_BAD_NOUNS` (e.g. "does", "this") or `len(noun) <= 2` → skip SRF for this sample, fall back to baseline.

### 4. CLIP Presence Detection — `clip_full_gate_v3` (per sample)

**Text encoding — 5-template ensemble:**
```
templates = [
    "{noun}",
    "a photo of a {noun}",
    "a photo of the {noun}",
    "there is a {noun} in this photo",
    "an image containing a {noun}",
]
noun_feat = F.normalize(mean([CLIP_text(t) for t in templates]), dim=-1)
```

**Full-image similarity:**
```
full_img_sim = cosine(CLIP(image), noun_feat)
```

**Patch similarities:**
- Crop patches at 3×3, 5×5, 7×7 grids
- `patch_max_sim = max cosine-sim across all patches and all grid scales`

**OR gate:**
```
object_present = (full_img_sim >= 0.21) OR (patch_max_sim >= 0.27)
```
Rationale: small or partially-visible objects may score below 0.21 on the full image but still exceed 0.27 on the best-matching patch.

**Spatial saliency:**
- Bilinear upsample each grid's patch-sim map to `(grid_h, grid_w)`
- Elementwise max across scales → min-max normalize → `saliency (grid_h * grid_w,)`

### 5. Attention Intervention (per forward-pass layer)

Fires in `_patched_softmax` for layers `[layer_start, layer_end]`, vision-aware heads only, pre-softmax (additive logit bias):

**System-prompt suppression** (always):
```
input[..., :sys_end] -= 0.30    # sys_beta = 0.30
```

**If object present** (saliency available):
```
bias[i] = alpha * sal[i] - eps * (1 - sal[i])
# salient tokens:    +alpha  (attend more)
# background tokens: -eps    (attend less)
input[..., img_tokens] += bias
```
Default: `alpha=2.0`, `eps=0.2`

**If object absent** (gate failed):
```
input[..., img_tokens] -= neg_absent_alpha
```
Default: `neg_absent_alpha=2.0` (POPE); `0.0` for most other datasets.

### 6. Phase Gate

Controls which forward-pass steps the intervention fires at:

| phase | Fires when | Used for |
|-------|-----------|---------|
| `"both"` | every step (prefill + decode) | POPE, MMVP, VLIND |
| `"generation"` | decode steps only (`q_len == 1`) | VLMBias, MME |

> **Fix (2026-07-26):** POPE was previously set to `phase="generation"`, which was a no-op during prefill evaluation — the intervention never fired. Changed to `phase="both"`.

### 7. SRF-E — Two-Pass Contrastive (optional)

```
Pass 1: method="srf",      full image    → logits_full  (model forward with SRF hooks)
Pass 2: method="baseline", zeroed pixels → logits_noval (no hooks, zeroed pixel_values)

logits_final = logits_full + gamma * (logits_full - logits_noval)
```
Default: `gamma=3.0`. Use `--method srfe` to enable. **Broken for VLMBias** (suppresses `{` format token) — use `--method srf` there.

---

## Per-Dataset Config (exact values)

| Dataset | Method | phase | alpha | eps | neg_absent_alpha | layer_start | layer_end | gamma |
|---------|--------|-------|-------|-----|-----------------|-------------|-----------|-------|
| POPE / RePOPE | SRF-E | both | 2.0 | 0.2 | 2.0 | 8 | 12 | 3.0 |
| MMVP | SRF-E | both | 2.0 | 0.2 | 0.0 | 8 | 16 | 3.0 |
| VLMBias | SRF | generation | 8.0 | 0.5 | 0.0 | 8 | 14 | — |
| MME | SRF | generation | 2.0 | 0.2 | 0.0 | 8 | 16 | — |
| VLIND | SRF-E | both | 2.0 | 0.2 | 0.0 | 8 | 16 | 3.0 |

**Shared across all datasets:** `layer_start=8`, `head_top_k_pct=0.20`, `sys_beta=0.30`, `clip_fallback_thresh=0.21`, `patch_thresh=0.27`

> **VLIND note:** SRF base is neutral (+0.33pp). VLIND requires global visual amplification (counterfactual relationships, anachronistic objects) — spatial attention boosting doesn't help. SRF-E's contrastive pass (image vs blank) is what drives the +5.3pp gain. Use `--method srfe` for VLIND.

---

## Run Commands

```bash
# IMPORTANT: use 'source activate mllm' NOT 'conda run -n mllm'
# conda run breaks the --n flag (ambiguous with --name)

cd /volumes2/mllm/lmms-eval
source activate mllm

# Full POPE eval (9000 samples, ~75 min) — SRF-E best: 87.70% (+1.33pp)
python srf/eval.py --method srfe --datasets pope --output results/srf_pope/ --gamma 3.0

# Full RePOPE eval (adversarial, 2684 samples)
python srf/eval.py --method srfe --datasets pope \
  --pope_file data/repope/adversarial.json --output results/srf_repope_adv/ --gamma 3.0

# MMVP (150 pairs) — SRF-E best: 49.33% (+9.33pp)
python srf/eval.py --method srfe --datasets mmvp --output results/srf_mmvp/ --gamma 3.0

# VLM Bias — use srf NOT srfe (SRF-E suppresses { format token in multi-token answers)
python srf/eval.py --method srf --datasets vlmbias --output results/srf_vlmbias/

# VLIND-Bench (302 samples) — SRF-E best: 52.32% (+5.3pp vs 47.02% baseline)
# SRF base is neutral (+0.33pp) — SRF-E is the correct method for VLIND
python srf/eval.py --method srfe --datasets vlind --output results/srf_vlind/ --gamma 3.0

# VLIND hyperparam sweep (21 configs, ~n samples)
python srf/sweep_vlind.py --method srfe          # full 302 samples
python srf/sweep_vlind.py --method srfe --n 50   # quick 50-sample test

# Baseline only
python srf/eval.py --method baseline --datasets pope --output results/baseline_pope/

# 100-sample diagnostic on RePOPE (baseline vs SRF vs SRF-gen, CLIP gate analysis)
python srf/diag_20sample.py \
  --repope_dir data/repope --splits adversarial --n 100 --out_dir results/diag_100sample
```

---

## File Map

```
srf/
  config.py              <- SINGLE SOURCE OF TRUTH — all hyperparams, read this first
  eval.py                <- unified eval CLI (pope, mmvp, vlmbias, mme, vlind)
  srf.py                 <- SRF base: setup(), reset_for_dataset(), prepare_sample(), cleanup()
  srf_e.py               <- SRF-E: two-pass contrastive (gamma=3.0)
  baseline.py            <- baseline eval (no intervention)
  vaf.py                 <- VAF/ClearSight baseline
  vcd.py                 <- VCD baseline
  sweep_vlind.py         <- VLIND hyperparam sweep (calls run_vlindbench, loads model once)
  diag_20sample.py       <- diagnostic script: per-sample CLIP gate + SRF comparison
  trace_single.py        <- single-sample full audit (best for debugging on new machine)
  noun_extract.py        <- extract object noun from question
  eval_pope_val.py       <- 60-sample val set for fast iteration
  visualize_saliency_pope.py    <- POPE heatmap grid (present vs absent, all 3 splits)
  visualize_pope_fn.py          <- POPE failure analysis heatmaps (false negatives)
  visualize_saliency_mme_hb.py  <- MME + HallusionBench heatmaps
  saliency/
    clip_salience.py     <- CLIP saliency; compute_clip_salience_full_gate_v3 (main function)
    eval_presence.py     <- CLIP gate accuracy sweep on val set
  docs/
    QUICKSTART.md        <- this file — run on new machine
    METHOD_WALKTHROUGH.md <- full code-level walkthrough
    RESEARCH_STATUS.md   <- results, open tasks, run commands
    RePOPE.md            <- RePOPE-specific findings

my_analysis/
  qwen_attn_patch.py     <- attention patching engine (patch_model, _patched_softmax, _STATE)
```

---

## Visualization & Debugging Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `srf/trace_single.py` | **Best for verifying on a new machine** — full per-sample audit: saliency map, token selection, boost values, head mask, logit deltas | stdout or `srf/docs/TRACE.md` |
| `srf/visualize_saliency_pope.py` | POPE heatmap grid — image + overlay, present vs absent, all 3 splits | `results/saliency_vis_pope/` |
| `srf/visualize_pope_fn.py` | POPE failure analysis — heatmaps for false negative cases | `results/saliency_vis_pope/failures/` |
| `srf/visualize_saliency_mme_hb.py` | MME + HallusionBench heatmaps | `results/saliency_vis_datasets/` |
| `srf/diag_20sample.py` | Per-sample baseline vs SRF accuracy + CLIP gate stats | `results/diag_100sample/` |

```bash
# Run on a single sample to verify SRF is working (saliency, boosts, logits)
source activate mllm && python srf/trace_single.py

# Save trace to file
source activate mllm && python srf/trace_single.py --out srf/docs/TRACE.md

# POPE saliency heatmaps (20 samples per split per label)
source activate mllm && python srf/visualize_saliency_pope.py --n 20
```

---

## What NOT to Run

| Command / Setting | Why broken |
|---|---|
| `conda run -n mllm python ... --n 20` | `--n` conflicts with conda's `--name` flag |
| `--method srfe` on VLMBias | SRF-E suppresses the `{` format token in multi-token answers |
| `attn_implementation=sdpa` | SRF hooks require eager attention; sdpa bypasses them |

Always load the model with `attn_implementation="eager"`.

---

## Known Issues

- **SRF-E broken for VLMBias** (multi-token answers) — use `--method srf`, not `srfe`
- **Qwen2.5-VL-7B params not tuned** — values are proportional starting points only, not swept
- **`patch_max_sim` threshold (0.27)** tuned on 20-sample RePOPE val set — validate on larger set before reporting
