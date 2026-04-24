---
name: load-mllm
version: v1.0
description: Loads full project context for the MLLM/SRF NeurIPS research project. Reads all key files and produces a structured briefing. Invoke at the start of any session working on /volumes2/mllm/lmms-eval or the PhD research. Use /load-mllm.
---

# load-mllm — Session Briefing for MLLM / SRF Research

When invoked, read the files listed below in order, then output the structured briefing defined at the end of this skill. Do NOT skip files — each one covers a distinct aspect of the project.

---

## Step 1 — Read These Files

Read all files in this sequence. Use parallel reads where possible (groups A, B, C can be read in parallel within each group).

### Group A — Research State (read first, highest priority)

| File | What it tells you |
|------|------------------|
| `/volumes1/shruum_vault/PhD/mllm/Research Status.md` | Current experiment status, best numbers, open tasks, 12-day schedule |
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/Research Plan.md` | Full paper structure, datasets, method algorithm, success criteria |
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/Results and Findings.md` | All experiment tables, sweep details, analysis plots |

### Group B — Method & Architecture (read in parallel with Group A)

| File | What it tells you |
|------|------------------|
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/Methods.md` | Every inference-time method: code, hyperparams, verdict |
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/VLM_Attention_Deep_Dive.md` | Why methods work architecturally (token sequence, head specialization) |
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/NextSteps_Ideas.md` | Ideas to try next (VISTA, CRG, AdaptVis, multi-resolution) |

### Group C — Repo & Data (read after A and B)

| File | What it tells you |
|------|------------------|
| `/volumes2/mllm/lmms-eval/CLAUDE.md` | Repo conventions: uv, Black, isort, commit style, error resolution |
| `/volumes2/mllm/lmms-eval/skills/lmms-eval-guide/SKILL.md` | Full lmms-eval codebase map, evaluation pipeline, registration |
| `/volumes1/shruum_vault/PhD/mllm/Research/VLMs Are Biased.md` | Dataset schema (anvo25/vlms-are-biased), experimental conditions |

### Optional — read only if the task requires it

| File | When to read |
|------|-------------|
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/Literature.md` | Writing related work, comparing to prior papers |
| `/volumes1/shruum_vault/PhD/mllm/NeurIPS plan/Datasets.md` | Adding a new dataset or checking evaluation splits |
| `/volumes2/mllm/lmms-eval/my_analysis/run_qwen_eval.py` | Modifying the eval loop or adding a new method |
| `/volumes2/mllm/lmms-eval/my_analysis/qwen_attn_patch.py` | Modifying attention patching logic |
| `/volumes2/mllm/lmms-eval/my_analysis/clip_salience.py` | Modifying SRF-CLIP |
| `/volumes2/mllm/lmms-eval/my_analysis/hssa_salience.py` | Modifying SRF-HSSA |
| `/volumes2/mllm/lmms-eval/skills/autoresearch/SKILL.md` | Running autonomous overnight experiments |

---

## Step 2 — Output This Briefing

After reading, output a structured briefing in this exact format:

---

### Project Briefing: SRF / MLLM Research

**Paper:** Semantic Re-Focus (SRF) — query-conditioned visual token boosting for VLM hallucination/bias reduction
**Target:** NeurIPS 2026 | **Deadline:** 2026-05-05 | **Days remaining:** [compute from today's date]
**Primary model:** Qwen2.5-VL-3B-Instruct | **Secondary:** LLaVA-1.5-7B (not yet ported)
**Repo:** `/volumes2/mllm/lmms-eval` | **Analysis scripts:** `my_analysis/`

---

**Current best results** (fill from Results and Findings):
- VLM Bias: [best method + score]
- POPE (adversarial): [best method + score]
- MMBench: [best method + score]
- MMVP: [status]

**What's done:**
[List implemented methods and completed experiments from Research Status]

**What's next (by priority):**
[List the top 3-5 open tasks from Research Status, in order]

**Active open questions:**
[List 2-3 unresolved questions that will affect paper direction]

**Key paths:**
```
Analysis:   /volumes2/mllm/lmms-eval/my_analysis/run_qwen_eval.py
Results:    /volumes2/mllm/lmms-eval/results/qwen_all/
Vault:      /volumes1/shruum_vault/PhD/mllm/
HF cache:   /volumes2/hugging_face_cache
```

**Quick eval command:**
```bash
cd /volumes2/mllm/lmms-eval
python my_analysis/run_qwen_eval.py \
  --dataset all --method vaf --sweep 1.5 --n_samples 50 \
  --output_dir results/qwen_all/<run_name>/
```

---

## Project Architecture (invariant — no need to re-read each session)

```
/volumes2/mllm/
├── lmms-eval/                   # lmms-eval eval framework (main repo)
│   ├── lmms_eval/               # Framework source
│   │   ├── __main__.py          # CLI: python -m lmms_eval
│   │   ├── evaluator.py         # Core eval loop
│   │   ├── api/                 # model.py, task.py, registry.py
│   │   ├── models/chat/         # Chat models (Qwen, OpenAI, vLLM...)
│   │   ├── models/simple/       # Legacy models (LLaVA-1.5, InstructBLIP...)
│   │   └── tasks/               # 230 task dirs, 1377 YAML configs
│   ├── my_analysis/             # PhD research scripts (NOT framework code)
│   │   ├── run_qwen_eval.py     # Main eval: all methods × datasets × sweeps
│   │   ├── qwen_attn_patch.py   # Attention patching (baseline, vhr, vaf, vcd, icd)
│   │   ├── clip_salience.py     # SRF-CLIP: CLIP ViT-L/14, 7×7 grid
│   │   ├── hssa_salience.py     # SRF-HSSA: layer-12 hidden state cosine sim
│   │   ├── visualize_salience.py# Side-by-side CLIP vs HSSA heatmaps
│   │   └── compare_methods.py   # Cross-method comparison plots
│   ├── results/                 # Experiment outputs
│   │   ├── qwen_all/            # Per-method result dirs (results.json, summary.csv)
│   │   ├── salience_vis/        # Heatmap visualizations
│   │   └── attention/           # Attention analysis outputs
│   ├── skills/                  # Claude Code skills
│   │   ├── load-mllm/           # This skill
│   │   ├── autoresearch/        # Autonomous overnight loop
│   │   └── lmms-eval-guide/     # Codebase navigation guide
│   └── CLAUDE.md                # Repo conventions
│
└── LLaVA/                       # LLaVA model repo (secondary)

/volumes1/shruum_vault/PhD/mllm/ # Obsidian vault (research notes)
├── MLLM Project MOC.md          # Map of content — start here
├── Research Status.md           # LIVE: current numbers + open tasks
├── NeurIPS plan/                # Paper planning notes
│   ├── Research Plan.md         # Full outline + schedule
│   ├── Methods.md               # All methods documented
│   ├── Results and Findings.md  # All experiment data
│   ├── VLM_Attention_Deep_Dive.md # Architecture deep-dive
│   ├── NextSteps_Ideas.md       # Future ideas + literature
│   ├── Literature.md            # Paper summaries
│   └── Datasets.md              # Dataset details
└── Research/
    ├── VLMs Are Biased.md       # Dataset: anvo25/vlms-are-biased
    └── Analysis Scripts.md      # Script reference

/volumes2/hugging_face_cache     # HF model + dataset cache (HF_HOME)
```

## Method Cheat Sheet (SRF paper)

```
Selectivity hierarchy:
  vision_boost   → ALL heads + ALL layers + ALL img tokens    → collateral damage
  vhr_boost      → VIS heads + ALL layers + ALL img tokens    → better, some damage
  vaf            → VIS heads + layers 8-15 + ALL img tokens   → surgical, stable
  SRF-CLIP       → VIS heads + layers 8-15 + CLIP-salient tokens  ← paper contribution
  SRF-HSSA       → VIS heads + layers 8-15 + HSSA-salient tokens  ← paper contribution

VIS heads = top-50% by text→image attention during 20-sample calibration
Layers 8–15 = cross-modal fusion zone (Qwen2.5-VL-3B, 28 layers total)
SRF-CLIP  = CLIP ViT-L/14, 7×7 grid, top-30% tokens, absence fallback at sim<0.2
SRF-HSSA  = cosine(h_img, h_txt) at decoder layer 12, top-30% tokens, single pass
```

## Environment Setup

```bash
cd /volumes2/mllm/lmms-eval
export HF_HOME=/volumes2/hugging_face_cache
export CUDA_VISIBLE_DEVICES=0

# Framework
uv sync && pre-commit install
python -m lmms_eval --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks mme --batch_size 1 --limit 8

# Research scripts (no uv needed — uses installed packages)
python my_analysis/run_qwen_eval.py --help
```

---

*Invoke `/load-mllm` at the start of any session to get up to speed instantly.*
