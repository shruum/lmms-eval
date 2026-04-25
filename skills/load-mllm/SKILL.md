---
name: load-mllm
version: v2.0
description: Loads full project context for the MLLM/SRF NeurIPS research project. Reads all key files and produces a structured briefing. Invoke at the start of any session working on this repo. Use /load-mllm.
---

# load-mllm — Session Briefing for MLLM / SRF Research

When invoked, read the files listed below in order, then output the structured briefing defined at the end of this skill. Do NOT skip files — each one covers a distinct aspect of the project.

---

## Step 1 — Read These Files

Read all files in this sequence. Use parallel reads where possible (groups A, B, C can be read in parallel within each group).

### Group A — Research State (read first, highest priority)

| File | What it tells you |
|------|------------------|
| `/volumes1/shruum_vault/PhD/mllm/Research Status.md` | Current experiment status, best numbers, open tasks |
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/Results and Findings.md` | All experiment tables, sweep details |
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/Repo.md` | Definitive repo structure, file map, CLI reference, current results table |

### Group B — Method & Architecture (read in parallel with Group A)

| File | What it tells you |
|------|------------------|
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/Methods.md` | SRF and SRF-E algorithm details, hyperparams, verdicts |
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/VLM_Attention_Deep_Dive.md` | Why methods work architecturally |
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/NextSteps_Ideas.md` | Ideas to try next |

### Group C — Repo code (read after A and B)

| File | What it tells you |
|------|------------------|
| `/volumes2/mllm/lmms-eval/srf/config.py` | Single source of truth: all hyperparams (SRF_DEFAULTS, SRF_DATASET_PARAMS, SRF_ARCH_PARAMS) |
| `/volumes2/mllm/lmms-eval/srf/eval.py` | Unified eval CLI — method dispatch, dataset runners, all CLI args |
| `/volumes2/mllm/lmms-eval/srf/srf.py` | SRF base implementation: setup, reset_for_dataset, prepare_sample |

### Optional — read only if the task requires it

| File | When to read |
|------|-------------|
| `/volumes2/mllm/lmms-eval/srf/srf_e.py` | Modifying SRF-E (contrastive two-pass) |
| `/volumes2/mllm/lmms-eval/srf/eval_datasets.py` | Adding/modifying dataset loaders |
| `/volumes2/mllm/lmms-eval/srf/noun_extract.py` | Modifying CLIP noun extraction |
| `/volumes2/mllm/lmms-eval/srf/saliency/clip_salience.py` | Modifying CLIP saliency computation |
| `/volumes2/mllm/lmms-eval/my_analysis/qwen_attn_patch.py` | Modifying attention patching engine |
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/Research Plan.md` | Full paper structure and deadlines |
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/Literature.md` | Writing related work, comparing to prior papers |

---

## Step 2 — Output This Briefing

After reading, output a structured briefing in this exact format:

---

### Project Briefing: SRF / MLLM Research

**Paper:** Semantic Re-Focus (SRF) — query-conditioned visual token boosting for VLM hallucination/bias reduction
**Target:** NeurIPS 2026 | **Deadline:** 2026-05-05 | **Days remaining:** [compute from today's date]
**Primary model:** Qwen2.5-VL-3B-Instruct | **Secondary:** Qwen2.5-VL-7B (not yet ported), LLaVA-1.5-7B (not yet ported)
**Repo:** `/volumes2/mllm/lmms-eval` | **Method package:** `srf/`

---

**Current best results** (fill from Results and Findings):
- MMVP pair acc:   baseline=? | SRF=? | SRF-E=?
- POPE (all splits): baseline=? | SRF=? | SRF-E=?
- MME score:        baseline=? | SRF=? | SRF-E=?
- VLM Bias:         baseline=? | SRF=?

**What's done:**
[List implemented methods and completed experiments from Research Status]

**What's next (by priority):**
[List the top 3-5 open tasks from Research Status, in order]

**Active open questions:**
[List 2-3 unresolved questions that will affect paper direction]

**Key paths:**
```
Method package: /volumes2/mllm/lmms-eval/srf/
  config.py           ← ALL hyperparams (single source of truth)
  eval.py             ← unified eval CLI
  srf.py              ← SRF base
  srf_e.py            ← SRF-E (contrastive)
  eval_datasets.py    ← dataset loaders (POPE, MMVP, MME, VLM Bias, …)
  noun_extract.py     ← CLIP noun extraction
  saliency/           ← clip_salience.py, hssa_salience.py

Patching engine: /volumes2/mllm/lmms-eval/my_analysis/qwen_attn_patch.py
Results:         /volumes2/mllm/lmms-eval/results/
Vault:           /volumes1/shruum_vault/PhD/mllm/
HF cache:        /volumes2/hugging_face_cache
```

**Quick eval commands:**
```bash
cd /volumes2/mllm/lmms-eval

# SRF base — MMVP + POPE (all splits) + MME
conda run -n mllm python srf/eval.py \
  --method srf \
  --datasets mmvp pope mme \
  --output results/srf_3b/

# SRF-E — sweep β
conda run -n mllm python srf/eval.py \
  --method srfe \
  --datasets mmvp pope \
  --beta 0.5 1.0 2.0 \
  --output results/srfe_3b/

# Sweep hyperparams for a new model (override arch config)
conda run -n mllm python srf/eval.py \
  --method srf --datasets pope \
  --pope_splits adversarial \
  --layer_start 9 --layer_end 17 \
  --alpha 4.0 --eps 0.2

# POPE adversarial only (fast check)
conda run -n mllm python srf/eval.py \
  --method srf --datasets pope \
  --pope_splits adversarial
```

---

## Project Architecture (invariant — no need to re-read each session)

```
/volumes2/mllm/lmms-eval/
│
├── srf/                            ← METHOD PACKAGE (all paper-critical code)
│   ├── config.py                   ← SINGLE SOURCE OF TRUTH for all hyperparams
│   │     SRF_DEFAULTS{}            ← shared tunables (sys_beta, calib_n, …)
│   │     SRF_DATASET_PARAMS{}      ← per-dataset: phase, alpha, eps
│   │     SRF_ARCH_PARAMS{}         ← per-arch: layer_start/end, head_top_k_pct, CLIP params
│   │     get_arch(model_id)        ← helper; falls back to SRF_ARCH_FALLBACK
│   │
│   ├── srf.py                      ← SRF base
│   │     setup(model, processor, calib_dataset)
│   │     reset_for_dataset(dataset, *, phase, alpha, eps, layer_start, layer_end,
│   │                        head_top_k_pct, clip_coarse_grid, clip_top_k_pct,
│   │                        clip_fallback_thresh)   ← all kwargs override config
│   │     prepare_sample(inputs, img_start, img_end, image, question, model, processor)
│   │     cleanup()
│   │
│   ├── srf_e.py                    ← SRF-E (two-pass contrastive)
│   │     get_contrastive_logits(model, inp, beta, mode) → Tensor [1, vocab]
│   │     generate_contrastive(model, inp, processor, beta, mode, max_new_tokens)
│   │
│   ├── eval.py                     ← UNIFIED EVAL SCRIPT
│   │     --method  srf | srfe
│   │     --datasets mmvp pope vlmbias mme
│   │     --pope_splits adversarial popular random   (default: all 3)
│   │     --beta    (SRF-E only)
│   │     --layer_start / --layer_end / --head_top_k_pct   (arch overrides)
│   │     --alpha / --eps / --phase                         (dataset overrides)
│   │     --clip_coarse_grid / --clip_top_k_pct / --clip_fallback_thresh
│   │     --output  results/run_name/
│   │
│   ├── eval_datasets.py            ← dataset loaders
│   │     LOADERS{pope, mmvp, vlmbias, mme, mmbench, cv_bench, hallusionbench}
│   │
│   ├── noun_extract.py             ← CLIP query noun extraction
│   │     extract_clip_noun(question, mode)  mode ∈ {mmvp, pope, vlmbias}
│   │
│   └── saliency/
│       ├── clip_salience.py        ← CLIP patch saliency (cross-modal encoder)
│       └── hssa_salience.py        ← hidden-state saliency (internal)
│
├── my_analysis/                    ← ANALYSIS & LEGACY (not modified for paper)
│   ├── qwen_attn_patch.py          ← attention patching engine (shared by srf/)
│   │     patch_model / identify_visual_heads / update_sample / _STATE{}
│   ├── autoresearch/               ← POPE autoresearch loop (completed)
│   ├── autoresearch_mmvp/          ← MMVP autoresearch loop (completed)
│   └── autoresearch_vlmbias/       ← VLM Bias autoresearch loop (completed)
│
└── results/                        ← eval output (JSON per dataset + summary)

/volumes1/shruum_vault/PhD/mllm/    ← Obsidian research vault
```

## Method Summary (SRF paper)

```
Two finalised methods:

SRF (base):
  - Identify vision-aware heads via calibration (top head_top_k_pct by text→image attn)
  - Per sample: compute CLIP saliency (noun extracted from question, coarse grid)
  - During forward pass: boost attention logits for salient image tokens in those heads
  - Applied in layers [layer_start, layer_end] (cross-modal fusion zone)
  - Phase: prefill / generation / both (dataset-tuned)

SRF-E (evidence-amplified):
  - Two forward passes: full input + no-vision input (pixel_values zeroed)
  - logits_final = logits_full + β * (logits_full - logits_noval)
  - Amplifies visual evidence, suppresses language prior

Hyperparameter priority: CLI args > SRF_ARCH_PARAMS[model_id] > SRF_DATASET_PARAMS[dataset] > SRF_DEFAULTS

Supported datasets: MMVP, POPE (adversarial/popular/random), MME, VLM Bias
Supported models:   Qwen/Qwen2.5-VL-3B-Instruct (tuned)
                    Qwen/Qwen2.5-VL-7B-Instruct  (proportional starting point)
                    llava-hf/llava-1.5-7b-hf      (proportional starting point)
```

## Environment

```bash
# Conda env: mllm
export HF_HOME=/volumes2/hugging_face_cache
export CUDA_VISIBLE_DEVICES=0
cd /volumes2/mllm/lmms-eval

# Run eval
conda run -n mllm python srf/eval.py --method srf --datasets pope --pope_splits adversarial
```

## Git

```
Branch: autoresearch/mmvp-srf   ← all SRF work here
Main:   main                    ← lmms-eval upstream (do not touch)

All paper files in srf/ and my_analysis/ — untracked, safe from upstream merges.
```

---

*Invoke `/load-mllm` at the start of any session to get up to speed instantly.*
