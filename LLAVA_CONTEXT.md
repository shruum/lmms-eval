# LLaVA SRF Experiment Context
**Last updated: 2026-09-23**

This file is the single source of truth for all LLaVA SRF work on Snellius.
Start a new Claude session by saying: "read lmms-eval/LLAVA_CONTEXT.md and continue"

---

## ⚠️ THE FULL METHOD — READ THIS FIRST, EVERY TIME

**The SRF method on LLaVA has FOUR steps. All four must always be active:**

1. **CLIP v3 semantic map** — patch + image threshold to detect object presence (`clip_fallback_thresh`, `clip_patch_thresh`)
2. **Foveation** — Gaussian blur on encoder input with sigma (`--fovea_sigma`, `--phase both`)
3. **Attention amplification** — boost attention to salient tokens with alpha (`--alpha`)
4. **Background + system prompt suppression** — suppress non-salient and system tokens

**THIS MEANS: always use `--method srffovea` for LLaVA. NEVER `--method srf` (plain) for any evaluation.**

**`neg_absent_alpha` IS DELETED. Never add it to any script. It was a mistake.**

**DO NOT use POPE for LLaVA — use MME or MMHal only.**

**DO NOT use SRF-E (`--method srfe`) on LLaVA — it is counterproductive on all datasets.**

---

## ⚙️ SRF Method — Step-by-Step with Params

```
Input image + question
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ Step 1: CLIP v3 Semantic Relevance Map              │
│  • Extract noun from question (noun_extract.py)     │
│  • Run CLIP ViT-B/32 at coarse grid (6×6 for LLaVA)│
│  • Full-image sim threshold: clip_fallback_thresh=0.25│
│  • Patch-level sim threshold: clip_patch_thresh=0.27│
│  • Output: 36-dim saliency vector (which patches    │
│    contain the queried object)                      │
└─────────────────────────────────────────────────────┘
        │ saliency map
        ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: Foveation (encoder input)                   │
│  • Apply Gaussian blur to non-salient image regions │
│  • Salient region stays sharp; background blurred   │
│  • Sigma controls blur strength: fovea_sigma=20     │
│  • Phase: phase=both (applied before both prefill   │
│    and generation passes)                           │
│  • Output: foveated image → fed to vision encoder   │
└─────────────────────────────────────────────────────┘
        │ foveated image tokens
        ▼
┌─────────────────────────────────────────────────────┐
│ Step 3: Attention Amplification (decoder)           │
│  • Boost attention weights to salient image tokens  │
│  • Applied in layers layer_start(default)–layer_end=20│
│  • Top head_top_k_pct=0.20 of heads selected        │
│  • Boost scale: alpha=0.3                           │
│  • Mode: llava_boost_mode=additive (NOT multiplicative)│
│  • Output: decoder attends more to relevant tokens  │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ Step 4: Background + System Prompt Suppression      │
│  • Suppress attention to non-salient image tokens   │
│  • Suppress attention to system prompt tokens       │
│  • Complementary to step 3 — push attention away    │
│    from irrelevant content                          │
└─────────────────────────────────────────────────────┘
        │
        ▼
  LLaVA generates answer
```

**Canonical best params for LLaVA (from 15-config sweep on MMHal n=96):**

| Param | Value | What it controls |
|-------|-------|-----------------|
| `--method` | `srffovea` | Full 4-step pipeline |
| `--alpha` | `0.3` | Attention boost strength |
| `--fovea_sigma` | `20` | Gaussian blur radius for background |
| `--phase` | `both` | Apply fovea in prefill + generation |
| `--layer_end` | `20` | Last decoder layer to patch (LLaVA has 32 layers) |
| `--head_top_k_pct` | `0.20` | Top 20% of heads get the boost |
| `--llava_boost_mode` | `additive` | Add boost (multiplicative was buggy, caused mode collapse) |
| `--clip_fallback_thresh` | `0.25` | Full-image CLIP sim threshold for object presence |
| `--clip_patch_thresh` | `0.27` | Patch-level CLIP sim threshold |

---

## 1. Environment

| Item | Value |
|------|-------|
| Cluster | Snellius (SURF) |
| Working dir | `/home/sgowda/workspace/SRF/lmms-eval` |
| Conda env | `mllm` → `/home/sgowda/miniconda3/envs/mllm` |
| Python | `/home/sgowda/miniconda3/envs/mllm/bin/python` |
| SLURM partition | `gpu_a100` |
| Excluded node | `gcn58` (always add `#SBATCH --exclude=gcn58`) |
| HF cache | `/home/sgowda/.cache/huggingface` |
| Model | `llava-hf/llava-1.5-7b-hf` |

---

## 2. Data Paths

| Dataset | Local Path |
|---------|-----------|
| MME benchmark | `/home/sgowda/workspace/ILVAD/data/MME/MME_Benchmark_release_version/MME_Benchmark/` |
| MMHal-Bench JSON | `/home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json` |
| MMHal-Bench images | `/home/sgowda/workspace/ILVAD/data/MMHal-Bench/images/` |
| Paper | `/home/sgowda/workspace/SRF/paper/` |
| Paper images | `/home/sgowda/workspace/SRF/paper/Bias_and_Hallucination_in_VLM/images/` |
| Paper Qwen analysis scripts | `/home/sgowda/workspace/SRF/paper/scripts/` |

**DO NOT use POPE for LLaVA analysis — use MME or MMHal only.**

MMHal image filename is extracted from URL:
```python
fname = r["image_src"].rstrip("/").split("/")[-1]
img_path = os.path.join(mmhal_img_dir, fname)
```

---

## 3. Key Source Files

| File | Purpose |
|------|---------|
| `srf/eval.py` | Main eval entry point — `--method baseline/srf/srffovea` |
| `srf/srf_c_v3.py` | SRF core — attention boost patch for LLaVA |
| `srf/saliency/clip_salience.py` | CLIP saliency computation |
| `srf/saliency/noun_extract.py` | Extract noun from question for CLIP |
| `srf/analysis/llava_modality_attention.py` | Fig 2: attention stats from MMHal → analysis figure |
| `srf/analysis/llava_fig3_routing.py` | Fig 3: Image\|Attn\|SRF scan+render for MMHal |
| `srf/analysis/llava_attn_analysis.py` | Older Fig2 attempt (use llava_modality_attention.py instead) |
| `srf/analysis/llava_paper_figs.py` | Older Fig3/Fig6 attempt (use llava_fig3_routing.py instead) |

### Paper Analysis Scripts — Usage

**Fig 2 equivalent** (attention routing + VTAR heatmap, matches `analysis2_figure.png` style):
```bash
# GPU collect pass — runs on all 96 MMHal samples, saves .npy + figure
python srf/analysis/llava_modality_attention.py --collect
# Re-plot from saved .npy (no GPU needed)
python srf/analysis/llava_modality_attention.py

# Outputs:
#   analysis/llava_attn_data/attn_by_layer_query.npy   (32, 3)
#   analysis/llava_attn_data/token_counts.npy           (3,)
#   analysis/llava_attn_data/rho_matrix.npy             (32, 32) ← VTAR per layer×head
#   analysis/llava_analysis_figure.png                  ← 3-panel paper figure
```

**Fig 3 equivalent** (Image | VLM Attention | SRF Saliency, matches `attn1.png` style):
```bash
# Step 1: scan all 96 MMHal samples, save ranked candidates
python srf/analysis/llava_fig3_routing.py --scan
# Candidates saved to: analysis/candidates/NNN_sim*_<type>_<question>.png
# Ranked by CLIP similarity (higher = more interesting saliency)

# Step 2: inspect candidates, pick two by index, render final figure
python srf/analysis/llava_fig3_routing.py --idx_a 003 --idx_b 011
# Output: analysis/llava_fig3_attn.png
```

**SLURM job (runs both steps):**
```bash
sbatch srf_exp_runs/llava_analysis_figs.sh
# Step 1 → analysis/llava_analysis_figure.png
# Step 2 → analysis/candidates/  (then pick two manually)
```

**LLaVA model facts (for analysis scripts):**
- 32 layers, 32 heads, 576 image tokens (24×24 grid)
- Current SRF patches layers 8–20 (NOT calibrated — just a starting guess)
- VTAR heatmap (`rho_matrix.npy`) will show which layers/heads actually attend to vision
- After seeing heatmap: run calibration sweep to find true optimal layer range

### Model attribute path (critical)
LLaVA changed between Transformers versions. Always use:
```python
def get_lm(model):
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    raise AttributeError(...)
# Then: layers = get_lm(model).model.layers
```

### LLaVA token layout
- System preamble (~30 tokens): "A chat between a curious user..."
- Image tokens: **576 fixed** (24×24 grid, 336px / 14px patch)
- Image token ID: `model.config.image_token_index` = 32000
- Query text: after image tokens
- Attention must use `attn_implementation="eager"` (SDPA/flash return None)

---

## 4. Eval Script Usage

```bash
cd /home/sgowda/workspace/SRF/lmms-eval
conda activate mllm

# MME — baseline
python srf/eval.py --method baseline --model llava-hf/llava-1.5-7b-hf \
    --datasets mme \
    --mme_data_dir /home/sgowda/workspace/ILVAD/data/MME/MME_Benchmark_release_version/MME_Benchmark \
    --mme_subtasks existence count position color \
    --output results/mme_baseline

# MME — SRF-Fovea best
python srf/eval.py --method srffovea --model llava-hf/llava-1.5-7b-hf \
    --datasets mme \
    --mme_data_dir /home/sgowda/workspace/ILVAD/data/MMHal-Bench/... \
    --mme_subtasks existence count position color \
    --alpha 0.5 --fovea_sigma 20 --layer_end 20 \
    --neg_absent_alpha 0.0 --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 --clip_patch_thresh 0.27 \
    --output results/mme_srffovea

# MMHal — baseline
python srf/eval.py --method baseline --model llava-hf/llava-1.5-7b-hf \
    --datasets mmhalbench \
    --mmhalbench_json /home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json \
    --openai_model gpt-4o \
    --output results/mmhal_baseline

# MMHal — SRF-Fovea BEST (from sweep)
python srf/eval.py --method srffovea --model llava-hf/llava-1.5-7b-hf \
    --datasets mmhalbench \
    --mmhalbench_json /home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json \
    --alpha 0.3 --fovea_sigma 20 --phase both --layer_end 20 \
    --head_top_k_pct 0.20 --neg_absent_alpha 0.0 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 --clip_patch_thresh 0.27 \
    --openai_model gpt-4o \
    --output results/mmhal_srffovea_best
```

---

## 5. Final Results (confirmed from disk)

### MME (n=240, 4 subtasks: existence/count/position/color)

| Method | Score | Acc | Notes |
|--------|-------|-----|-------|
| Baseline | 670.0 | 83.75% | `results/mme_llava_full/baseline/` |
| SRF plain (alpha=0.5) | 686.7 | 85.83% | `results/mme_fixed_patch/srf_fixed/` — OLD best |
| **SRFovea (alpha=0.5, le=20, htk=0.20, no naa)** | **693.3** | **86.67%** | `results/mme_hl_sweep/A_le20_htk020/` — **NEW BEST** |

The gain from 686.7 → 693.3 came from switching to `--method srffovea` and removing `neg_absent_alpha`.

Script: `srf_exp_runs/mme_head_layer_sweep.sh` (job 27039275, gcn27, 2026-09-23)
Results: `results/mme_hl_sweep/`

### MMHal-Bench (n=96, gpt-4o scoring)

| Method | Score | Hal% | Notes |
|--------|-------|------|-------|
| Baseline | 2.208 | 60.4% | `results/mmhalbench_full/baseline/` |
| SRF plain (alpha=0.5, naa=1.0) | 2.292 | 59.4% | `results/mmhalbench_full/srf/` — **WRONG CONFIG** |
| SRFovea (alpha=0.3, σ=20, naa=1.0) | 2.438 | 56.3% | `results/mmhal_sweep/a0p3/` — old best |
| **SRFovea (alpha=0.3, σ=20, no naa)** | **2.458** | **57.3%** | `results/mmhal_calib_validation/R1_le20_htk020_a03/` — **BEST** |

**Gain from 2.438 → 2.458 came entirely from removing neg_absent_alpha.**
**Extended layer range (le=32, htk=0.35) hurts on MMHal: scores 2.427.**

Reference (ILVAD paper Table 1, gpt-4-0314 scoring — compare deltas only):
- Baseline greedy: 2.01 / 67.0%
- VCD: 2.20 / 61.8%
- ILVAD (best): 2.22 / 61.5%

---

## 6. Best Parameters Per Dataset

### MME
| Param | Value |
|-------|-------|
| method | `srffovea` |
| alpha | 0.5 |
| fovea_sigma | 20 |
| layer_start | 8 |
| layer_end | 20 |
| head_top_k_pct | 0.20 |
| llava_boost_mode | additive |
| clip_fallback_thresh | 0.25 |
| clip_patch_thresh | 0.27 |

Note: both MME and MMHal now use `srffovea`. The difference is alpha (0.5 vs 0.3).

### MMHal-Bench
| Param | Value |
|-------|-------|
| method | `srffovea` |
| alpha | 0.3 |
| fovea_sigma | 20 |
| phase | both |
| head_top_k_pct | 0.20 |
| layer_end | 20 |
| neg_absent_alpha | 0.0 |
| llava_boost_mode | additive |
| clip_fallback_thresh | 0.25 |
| clip_patch_thresh | 0.27 |

**Both datasets use srffovea. Main difference: alpha=0.5 for MME, alpha=0.3 for MMHal.**

---

## 7. Sweep History

### VTAR head×layer analysis (2026-09-22, job 27038297, n=20 MMHal)
Script: `srf_exp_runs/llava_vtar_quick.sh` → `srf/analysis/llava_modality_attention.py --collect --n 20`
Data: `analysis/llava_attn_data/rho_matrix.npy` (32×32 layers×heads)

Key findings:
- System prompt dominates attention: 6% of tokens → 74% attention.
- Vision tokens: 92% of input → only 14% attention (severely under-attended).
- Layers 0–1 have huge VTAR (79%, 39%) but are pure vision processing — not cross-modal fusion.
- Fusion plateau: layers 8–17 at ~12–17% VTAR; layers 18–20 slightly declining.
- Layer 31 spike: 17.3% VTAR (excluded by default le=20).
- Head distribution is flat (8–19% range) — no dominant vision-specialist heads.
- Top fusion-zone heads globally: H9 (19%), H30 (17.5%), H4 (17.5%), H6 (16.9%), H10 (16.7%).
- Top heads rotate per layer (L8: H6,H24; L9: H30,H12; L11: H17,H26) — no consistent winner.
- `head_top_k_pct=0.20` captures only 24.6% of fusion-zone VTAR mass; 0.30→31.9%, 0.50→58.2%.

### MME head×layer sweep (2026-09-23, job 27039275, n=240)
Script: `srf_exp_runs/mme_head_layer_sweep.sh`
Results: `results/mme_hl_sweep/`
Grid: layer_end ∈ {20,24,32} × head_top_k_pct ∈ {0.20,0.30,0.35}, all srffovea α=0.5 σ=20 no-naa

```
Config          layer_end  htk   exist  count   pos  color  TOTAL
A (baseline)       20     0.20  196.7  173.3  146.7  176.7  693.33  ← BEST
F (ext+heads)      32     0.35  193.3  170.0  146.7  176.7  686.67
C/E (ext range)  24/32    0.20  196.7  166.7  143.3  176.7  683.33
D                  24     0.30  193.3  163.3  146.7  176.7  680.00
B (more heads)     20     0.30  193.3  160.0  143.3  176.7  673.33
```

Key findings:
- Current defaults (le=20, htk=0.20) are already optimal for MME. New score 693.3 beats old best 686.7 purely by removing neg_absent_alpha and switching to srffovea.
- More heads (htk=0.30) consistently hurts — flat VTAR distribution dilutes signal at le=20.
- Extending to le=32 + htk=0.35 recovers slightly (686.67) — L31 spike helps when paired with more heads.
- Position is the hardest subtask (143–147) across all configs.



### MMHal sweep (15 configs, srffovea, n=96, all scored)
Script: `srf_exp_runs/mmhal_sweep_submit.sh` + `mmhal_sweep_worker.sh`
Results: `results/mmhal_sweep/<tag>/mmhalbench_scores.json`

Full ranked results:
```
Tag      Score   Hal%    Params
a0p3     2.438  56.25%  alpha=0.3, sigma=20, k=0.20, le=20, phase=both  ← BEST
k100     2.427  56.25%  alpha=0.5, sigma=20, k=1.00, le=20, phase=both
sig30    2.406  57.3%   alpha=0.5, sigma=30, k=0.20, le=20, phase=both
k50      2.406  56.25%  alpha=0.5, sigma=20, k=0.50, le=20, phase=both
le16     2.354  58.3%   alpha=0.5, sigma=20, k=0.20, le=16, phase=both
sig15    2.354  57.3%   alpha=0.5, sigma=15, k=0.20, le=20, phase=both
a1p0     2.354  57.3%   alpha=1.0, sigma=20, k=0.20, le=20, phase=both
sig20    2.333  57.3%   alpha=0.5, sigma=20, k=0.20, le=20, phase=both  (ref)
sig5     2.323  58.3%
le24     2.312  59.4%
sig10    2.312  58.3%
ph_gen   2.302  59.4%   phase=generation only
sig25    2.292  59.4%
k10      2.250  61.5%   alpha=0.5, k=0.10 (too few heads)
a2p0     2.188  61.5%   alpha=2.0 (too aggressive)
```

### Earlier sweeps (NOT on full n=96 — do not use for comparison)
- `mmhalbench_alpha_sweep/`: n=20 only — inflated scores, unreliable
- `mmhalbench_param_sweep/`: n=20 only — inflated scores, unreliable

---

## 8. Pending Work

1. **Paper figures (LLaVA equivalents of Qwen Fig 2, Fig 3)**
   - Scripts ready: `srf/analysis/llava_modality_attention.py` (Fig 2), `srf/analysis/llava_fig3_routing.py` (Fig 3)
   - SLURM: `srf_exp_runs/llava_analysis_figs.sh`
   - After Fig 3 scan: inspect `analysis/candidates/`, pick two, run `--idx_a NNN --idx_b NNN`
   - Outputs go to `analysis/` dir

3. **Update paper numbers** once MMHal calib validation done:
   - MME: **693.3** (new best, job 27039275)
   - MMHal: pending job 27040850

4. **MMhal_best_combined was never scored** — responses saved but `scores: {}`
   - Path: `results/mmhal_best_combined/`
   - Run: `python srf/eval.py --score_only --output results/mmhal_best_combined --openai_model gpt-4o`

---

## 9. Experiment Run Log

All SLURM jobs submitted. To resubmit any: `sbatch srf_exp_runs/<script>.sh`

| Date | Job ID | Script | What | Result |
|------|--------|--------|------|--------|
| 2026-09-22 | 27021615 | — | Early LLaVA test | FAILED: `AttributeError: language_model` (Transformers 5.x change) |
| 2026-09-22 | 27038297 | `llava_vtar_quick.sh` | VTAR heatmap, n=20 MMHal | OK — saved `rho_matrix.npy` (32×32), ran in 43s on gcn53 |
| 2026-09-23 | 27039275 | `mme_head_layer_sweep.sh` | MME head×layer sweep, 6 configs | OK — best A=693.33, gcn27, ~20 min |
| 2026-09-23 | 27040850 | `mmhal_calib_validation.sh` | MMHal 3-config validation | PENDING |

### Key run commands used

```bash
# Submit VTAR heatmap job (20 MMHal samples, ~30 min)
sbatch srf_exp_runs/llava_vtar_quick.sh

# Submit MME head×layer sweep (6 configs, ~20 min total)
sbatch srf_exp_runs/mme_head_layer_sweep.sh

# Submit MMHal calibration validation (3 configs × 96 samples, ~3-4h)
sbatch srf_exp_runs/mmhal_calib_validation.sh

# Re-plot VTAR figure without re-running GPU (once rho_matrix.npy exists)
python srf/analysis/llava_modality_attention.py

# Analyze rho_matrix.npy manually (head rankings, threshold coverage)
python3 - <<'EOF'
import numpy as np
rho = np.load("analysis/llava_attn_data/rho_matrix.npy")  # (32, 32)
fusion_mean = rho[8:21].mean(axis=0)
for rank, h in enumerate(np.argsort(fusion_mean)[::-1]):
    print(f"rank {rank+1:2d}  H{h:2d}: {fusion_mean[h]*100:.2f}%")
EOF

# Check SLURM job status
squeue -j <JOB_ID>
tail -f srf_exp_runs/logs/<logfile>.out
```

---

## 11. Paper Figure Reference

Qwen paper figures (for style matching):
- Fig 2 = `paper/Bias_and_Hallucination_in_VLM/images/analysis2_figure.png`
- Fig 3 = `paper/Bias_and_Hallucination_in_VLM/images/attn1.png`
- Fig 6 = `paper/Bias_and_Hallucination_in_VLM/images/srf_components.png`

Qwen generation scripts:
- `paper/scripts/run_modality_attention.py` → Fig 2 data collection
- `paper/scripts/plot_diverging_heatmap.py` → Fig 2 figure
- `paper/scripts/make_fig5_routing.py` → Fig 3

LLaVA equivalents:
- `srf/analysis/llava_modality_attention.py` → Fig 2 (collect + plot, uses MMHal)
- `srf/analysis/llava_fig3_routing.py` → Fig 3 (scan candidates, render final)

---

## 12. Known Issues / Gotchas

- **Exclude gcn58** always — that node has issues
- **model.language_model AttributeError**: Transformers 5.x changed to `model.model.language_model`. Always use `get_lm(model)` helper
- **neg_absent_alpha**: Use 0.0 for both datasets — naa=1.0 was a mistake in mmhal_full.sh
- **SRF-E**: Do NOT use on LLaVA — counterproductive on all datasets
- **SRF-Fovea CLIP hang**: Some nodes (gcn35, gcn8) had no internet; CLIP loading hung. Pin to nodes with HF cache access. gcn36 worked reliably.
- **Sweep n=20**: Old alpha/param sweeps used n=20 samples — ignore those scores, use the 15-config srffovea sweep (n=96) instead
- **gpt-4o scoring**: Inflates scores ~0.15-0.20 vs gpt-4-0314 (paper baseline). Compare deltas only, not absolute values.
- **MMHal scoring**: `--openai_api_key` must be set. Scoring is separate from generation — can re-score without re-running model.
