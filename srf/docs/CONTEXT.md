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
  eval_ablation.py       ← ablation sweep runner (2x2x2 random-control grid)
  ablation_components.py ← cumulative component ablation (paper table); 2 anchors x 7 rows
                            calls eval.run_mmvp — does NOT reimplement the eval loop
  head_calibration.py    ← vision-responsive head selection, 11 modes.
                            SHIPPED: ratio_topk (VTAR, top-k per layer, k fixed)
                            ratio_auto (VTAR, k DERIVED per dataset by Otsu)
                            see MODES and VTAR_MODES at the top of the file
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
  param_sensitivity.py   ← one-at-a-time hyperparameter sweeps (--anchor current)
  profile_cost.py        ← ms/sample and FLOPs, split by stage
  visualize_components.py← ONE model's 4-panel component row (paper figure)
  make_paper_figure.py   ← stitches two rows into images/srf_components_combined.png
                            see SRF_details.md section 11 for which script made
                            which figure, and the settings both rows must share
  significance.py        ← CPU. paired bootstrap + McNemar over saved records
  tune_pope_gate.py      ← CPU. CLIP gate accuracy/precision on POPE
  audit_nouns_vlmbias.py ← CPU. extracted nouns and maps per VLMBias topic
  find_vis_sample.py     ← CPU. rank samples by relevance-map concentration
  make_sensitivity_table.py ← CPU. regenerate the appendix table from JSON
  noun_extract.py        ← CLIP query noun extraction (pope / mmbench / vlind modes)
  saliency/
    clip_salience.py     ← CLIP patch saliency; compute_clip_salience_full_gate_v3
                            kwargs: scale_combine (max|prod|min|mean, how the 3
                            crop scales merge) and gate_logic (or|and, how the
                            two presence thresholds combine)
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
                            ⚠️ CORE CODE — do NOT modify; wrap externally.
                            ONE exception, 2026-09-17: optional _STATE["head_weight"]
                            for soft per-head weighting (could not be done externally,
                            head_mask is a boolean index). See Code Change Log.
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

## Saliency modes: `clip` vs `clip_full_gate_v3`  (READ BEFORE CHANGING)

Both compute the SAME underlying map (CLIP patch-noun similarity, min-max
normalised to [0,1]). They differ only in the **presence gate**, which controls
alpha — not the map itself.

| | basic `clip` | `clip_full_gate_v3` |
|---|---|---|
| Patch crops | 1 scale, `coarse_n`=7 (49) | 3 scales (3,5,7) = 83 |
| Presence signal | `max_sim` (best **patch**-noun sim) | `full_img_sim` (**whole-image**-noun sim), or `patch_max_sim >= 0.27` |
| Threshold | `clip_fallback_thresh` (0.20) | `_FULL_IMG_THRESH_V3` (0.20) |
| alpha scaling | `alpha * min(max_sim/thresh, 1)` | `alpha * min(full_img_sim/thresh, 1)` |
| Gated "absent" | mask=None, **value = alpha** (full uniform boost) | mask=None, **value = -neg_absent_alpha** (0 for mmvp → no boost) |

Consequence on MMVP: `max_sim` ~ 0.25-0.35 but `full_img_sim` ~ 0.10-0.18
(whole-image similarity is diluted by background). Under v3, surviving samples
get alpha scaled to roughly half, and gated-out samples get NO boost, whereas
basic `clip` falls back to a full-alpha uniform boost.

**Which to use, and why it is dataset-dependent:**
- POPE **contains genuinely absent objects — that is the task**, so a presence
  gate is essential. v3 wins there (val acc 0.900, FPR 0.000).
- MMVP **contains no absent objects** — every question asks about an attribute of
  a present object, so the gate can only produce false negatives. Basic `clip`
  wins there (+4.00pp pair, see the component-ablation table below).

NOTE: `_FULL_IMG_THRESH_V3` is **0.20** in `clip_salience.py`. Earlier revisions
of this file and the `config.py` docstring said 0.21 — that was stale.

---

## Head calibration  (shipped behaviour vs the paper)

**The shipped selection is layer-agnostic; the paper specifies per-layer.**

The paper writes `H*_l = TopK_h( E_c[rho_{l,h}] )`, indexed by layer AND head,
and gates the logit modification on `h in H*_l`. But
`qwen_attn_patch.identify_visual_heads` accumulates a single `(n_heads,)` score
vector across EVERY decoder layer's softmax call (the `_calibrate_heads` block),
divides by `count = n_layers * n_samples`, and stores ONE mask in
`_STATE["head_mask"]` that is reused in every layer.

Implication: a head that is strongly visual inside the fusion zone [8,16] has
that signal diluted by its own weak scores across the ~20 layers that do not
mediate fusion — i.e. the method averages away exactly the layer structure that
the paper's attention-heatmap figure presents as its motivation.

`srf/head_calibration.py` tests alternatives without touching core code:

| Mode | rho(l,h) | Notes |
|---|---|---|
| `global` | shipped layer-agnostic mean image attention | reference; must reproduce 43.33% |
| `per_layer` (S1) | mean text->image attention at layer l | implements the paper's equation as written |
| `saliency` (S3) | Pearson corr(head's image-token attention profile, CLIP saliency map) at layer l | selects semantically aligned heads rather than structurally attending ones |

**Measured on MMVP (2026-09-16), published anchor, only head selection varied:**

| Mode | Pair | dref | Img | Fusion-zone overlap vs shipped mask |
|---|---|---|---|---|
| `global` (reference) | 43.33 | — | 69.67 | — |
| `per_layer` (S1) | 44.00 | +0.67 | 70.33 | 0.222 |
| `saliency` (S3) | **46.00** | **+2.67** | **71.33** | 0.167 |

The `global` control reproduced 43.33/69.67 exactly, so the deltas are attributable
to head selection alone. In layers 9, 10 and 12 the S3 overlap with the shipped mask
is 0.0 — the global mask is not a blurred version of the right answer, it selects
different heads outright in much of the fusion zone.

**Cost:** unchanged at inference — still 1 CLIP pass + 1 LLM forward, ZERO extra LLM
passes. The per-layer mask costs 36 dict assignments per forward (the pre-hook) and
576 bytes instead of 16. S3 adds 20 CLIP passes to the ONE-TIME calibration only.

**Caveats before relying on this:** S3 calibrated on only 14/20 samples (the v3 gate
discarded 6); 46.00 vs 43.33 is 69 vs 65 pairs out of 150; single calibration seed
(seed=0, n=20); MMVP only — POPE / VLMBias / MME / MMHal-Bench NOT re-run, and the
per-layer hooks have never been exercised on LLaVA.

Two mechanisms worth knowing:
- Per-layer masks are applied by registering a **forward pre-hook on each
  `layer.self_attn`** that swaps `_STATE["head_mask"]`. The patch re-reads
  head_mask on every softmax call, so no change to `qwen_attn_patch.py` is
  needed. Same wrap-externally pattern as `vhr.py`.
- Per-(layer,head) attention is captured via the patch's existing
  `_capture`/`_captured` state plus a per-layer post-hook — NOT
  `output_attentions=True`, which would hold ~880 MB across 28 layers at MMVP
  resolution instead of ~31 MB for one layer.

`srf._build_calib_inputs(..., return_meta=True)` (added 2026-09-16) additionally
returns `{"image", "question"}` per calibration sample, which S3 needs to compute
the saliency map. Populated for the `pope` and `mmvp` branches only; raises
ValueError for other datasets rather than returning empty metadata. The question
string matches what `eval.run_pope` / `eval.run_mmvp` pass to `prepare_sample`,
so noun extraction is identical to eval.

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
| VLM Bias | **2784** (7 cats) | generation | exact match after `{}` extraction |

> ⚠️ CORRECTED 2026-09-16: this table previously said "~300". VLMBias is **2784**
> samples — `eval.py --n_vlmbias_per_cat` defaults to 0, which means ALL rows in all
> 7 categories, and that is what every published VLMBias number used. Budget ~70 min
> per pass at ~1.5 s/sample (20-token generation), not the ~5 min the old figure
> implied. The paper's experimental-setup section gives no N for VLMBias, so it is
> not affected.

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

## Code Change Log

What changed and where to read about it. Detail lives in
`SRF_details.md` (repo root), which is the current record for this project.

| Date | File | Change | Detail |
|---|---|---|---|
| 2026-09-20 | — | **LLaVA handoff written.** `eval.py::load_model` is Qwen-only and MMHal-Bench has no loader, both block cluster runs | **SRF_details 8** |
| 2026-09-20 | — | **current method settled.** VTAR score, top-20% heads per layer, layers 6-31. Not promoted to config defaults | **SRF_details 7** |
| 2026-09-18 | `srf/head_calibration.py`, `srf/eval.py`, `srf/srf.py` | head-selection modes `vtar_thresh`, `vtar_ratio`, `ratio_topk`, random-map control, two passthrough bug fixes | SRF_details 6 |
| 2026-09-17 | `srf/eval.py` | `--head_mode` and friends exposed on the main eval path, dataset runs refactored into a loop | SRF_details 3.5 |
| 2026-09-17 | `srf/head_calibration.py` | VTAR selection modes `vtar_layers`, `vtar_joint`, `vtar_soft` | SRF_details 6.2 |
| 2026-09-17 | `srf/srf.py` | `clip_full_gate_v3_paper`, the paper's relevance equations | SRF_details 1.4 |
| 2026-09-17 | `srf/config.py` | `phase` unification tested and reverted | SRF_details 5.5 |
| 2026-09-17 | `my_analysis/qwen_attn_patch.py` | **the one exception to do-not-modify.** Optional `_STATE["head_weight"]` for soft per-head weighting. Additive, default None, verified byte-identical when unset | SRF_details code log |
| 2026-09-16 | `srf/srf.py` | `_build_calib_inputs(..., return_meta=True)` | SRF_details 1.5 |

### Outstanding bug — `srf/config.py` `n_layers` is wrong (NOT yet fixed)

| Model | config.py says | actual |
|---|---|---|
| Qwen2.5-VL-3B | 28 | **36** |
| Qwen2.5-VL-7B | 32 | **28** (and 28 heads, not 16) |

Shipped inference is unaffected, because `layer_start`/`layer_end` are explicit
values and never derived from `n_layers`. `srf/vhr.py` is also unaffected, it
reads the live model config. Affected are `ablation_components.py` "all layers"
rows, `eval_ablation._make_random_layer_range`, the 7B scaling comments, and
`trace_single.py` diagnostics. See SRF_details section 5.

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
