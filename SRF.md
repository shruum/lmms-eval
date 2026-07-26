# SRF Documentation

All research docs for the Semantic Re-Focus (SRF) project live in `srf/docs/`.

## Start Here

**[srf/docs/QUICKSTART.md](srf/docs/QUICKSTART.md)** — run on a new machine: environment, exact code flow, per-dataset config, run commands, file map, known issues.

## Active Docs

| File | Purpose |
|---|---|
| [srf/docs/QUICKSTART.md](srf/docs/QUICKSTART.md) | **New machine guide** — env setup, full method walkthrough, run commands, file map |
| [srf/docs/RESEARCH_STATUS.md](srf/docs/RESEARCH_STATUS.md) | **Live dashboard** — current best results, open tasks, recent runs |
| [srf/docs/EXPERIMENTS.md](srf/docs/EXPERIMENTS.md) | **Full experiment log** — all saliency / boosting experiments with results + conclusions |
| [srf/docs/METHOD_WALKTHROUGH.md](srf/docs/METHOD_WALKTHROUGH.md) | **Code-level walkthrough** — step-by-step through every function called per sample |
| [srf/docs/CONTEXT.md](srf/docs/CONTEXT.md) | **Code reference** — algorithm, file map, hyperparameter system, CLI |
| [srf/docs/RePOPE.md](srf/docs/RePOPE.md) | **RePOPE findings** — corrected-annotation POPE benchmark results |
| [srf/docs/METHODS.md](srf/docs/METHODS.md) | **Method comparison** — SRF-V1, V2, Contrastive across MMVP/VLMBias/POPE |

## Key Config (as of 2026-07-26)

```
saliency_mode  = clip_full_gate_v3
layer_start    = 8
layer_end      = dataset-specific (POPE=12, MMVP=16, VLIND=16, VLMBias=14)
head_top_k_pct = 0.20
sys_beta        = 0.30   # system-prompt suppression
phase           = "both" (POPE/MMVP/VLIND) | "generation" (VLMBias/MME)
alpha/eps       = 2.0/0.2 (most datasets) | 8.0/0.5 (VLMBias)
neg_absent_alpha= 2.0 (POPE) | 0.0 (others — needs sweep)
gamma (SRF-E)  = 3.0
```

Single source of truth for all hyperparams: **`srf/config.py`**

## Archive

Superseded or completed autoresearch programs in [`srf/docs/archive/`](srf/docs/archive/):

| File | Notes |
|---|---|
| `SRF_FINDINGS_v1.md` | Early POPE autoresearch findings (pre-v3 gate) |
| `program_pope.md` | POPE autoresearch loop program (completed) |
| `program_mmvp.md` | MMVP autoresearch loop program (completed) |
| `program_vlmbias.md` | VLMBias autoresearch loop program (completed) |
| `program_mme.md` | MME autoresearch loop program (completed) |
| `program_mme_targeted.md` | MME targeted sweep program (completed) |
