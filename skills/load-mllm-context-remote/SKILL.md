---
name: load-mllm-context-remote
version: v2.0
description: Loads project context for VLM hallucination mitigation research. Invoke with /load-mllm-context-remote.
---

# VLM Hallucination Mitigation Research — Session Briefing

## Research Goal

**Implement and evaluate a novel inference-time method to mitigate hallucination in Vision-Language Models by steering attention toward semantically relevant image regions.**

**Target Venue:** NeurIPS 2026 | **Deadline:** 2026-05-05

---

## Step 1 — Find the repo root

```bash
git rev-parse --show-toplevel
```

Call the result `{REPO}`. All paths below are relative to it.

---

## Step 2 — Read these files (in parallel)

| File | What it contains |
|------|-----------------|
| `{REPO}/srf/CONTEXT.md` | Algorithm, file map, hyperparams, CLI reference |
| `{REPO}/srf/RESEARCH_STATUS.md` | Current results, open tasks, recent runs |
| `{REPO}/srf/config.py` | Central config (models, datasets, hyperparameters) |

---

## Step 3 — Output this briefing

### Current Status

**Models:** Qwen-VL-Chat, LLaVA-1.5 (7B/13B)
**Datasets:** POPE, MME, VLM Bias

**Best Result (Qwen-VL-Chat on POPE):**
- Baseline: 80.00%
- SRF: 81.00% (+1.0% gain)

**Known Issues:**
- MME showing ZERO or NEGATIVE impact with SRF
- LLaVA results: Δ=0.00% across all configs
- Need to investigate why gains don't transfer

### Quick Eval

```bash
cd {REPO}
conda run -n mllm python srf/eval.py \
    --method srf \
    --model Qwen/Qwen-VL-Chat \
    --datasets pope \
    --output results/srf_run/
```

### Architecture Notes

- **Qwen-VL-Chat**: 32 layers, used in AIR/ClearSight papers (baseline comparison)
- **LLaVA-1.5**: 32 layers, most widely studied VLM
- **Key params in `srf/config.py`:**
  - `SRF_ARCH_PARAMS`: model-specific layer ranges, CLIP settings
  - `SRF_DATASET_PARAMS`: per-dataset hyperparameters
  - All values configurable via CLI args

### Code Organization

```
srf/
├── config.py              # Central config (single source of truth)
├── eval.py                # Unified CLI (all datasets, all models)
├── srf.py                 # SRF base implementation
├── srf_e.py               # SRF-E (two-pass contrastive)
├── eval_datasets.py       # Dataset loaders
├── noun_extract.py        # Query noun extraction
└── saliency/
    ├── clip_salience.py   # CLIP-based saliency
    └── hssa_salience.py   # Hidden-state saliency (experimental)

my_analysis/
├── qwen_attn_patch.py     # Attention patching engine
└── autoresearch/           # Experiment logs and results
```

### Current Branch

`{CURRENT_BRANCH}` — Check with `git branch --show-current`

---
*Update `srf/RESEARCH_STATUS.md` after each experiment run.*
