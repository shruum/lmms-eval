---
name: load-mllm
version: v5.0
description: Loads project context for the SRF/MLLM research repo. Invoke at the start of any session. Use /load-mllm.
---

# load-mllm

Read these files in parallel, then output the briefing below.

**Group A — read FIRST, this is the most current record:**

| File | What it covers |
|------|---------------|
| `SRF_details.md` (repo root) | **START HERE. SECTION 7 = the current method. SECTION 8 = the LLaVA/cluster handoff, which opens with two blockers.** Section 7 is the current method: the exact command, the pipeline stages, every mode and hyperparameter with its status, the latest ablation numbers, the scripts, and the known traps. Sections 1-6 are the history of how it got there. Supersedes the repo docs below where they disagree. |

**Group B — repo docs (older, partly superseded by SRF_details.md):**

| File | What it covers |
|------|---------------|
| `srf/docs/CONTEXT.md` | File map, saliency modes, head calibration, **Code Change Log** |
| `srf/docs/RESEARCH_STATUS.md` | Results tables, comparison baselines, open tasks |
| `srf/config.py` | Live hyperparameter values (single source of truth for defaults) |

**Group C — the paper:**

| File | What it covers |
|------|---------------|
| `/volumes2/mllm/PAPER/ICLR/PAPER.tex` | Current draft. Method section defines the components and their equations. |
| `/volumes2/mllm/PAPER/ICLR/appendix.tex` | Ablation + parameter sensitivity appendix section. NOT yet `\input` into PAPER.tex. |

Also note `PAPER/ICLR/tables/*.tex` are auto-generated from result JSONs.
Regenerate them, never hand-edit the numbers.

---

## Briefing format

Output this after reading:

---

### SRF Session Briefing

**Repo:** `/volumes2/mllm/lmms-eval` | **Env:** `source activate mllm`

**Paper story:** SRF (attn boost + foveal blur) + SRF-E (+ blur contrastive). Routing failure hypothesis: pre-encoder / in-decoder / post-decoding interventions.

**Status:** [from SRF_details.md "Where things stand" and section 7]

**Current method command:** [verbatim from SRF_details.md 7.1. Note the flags are
NOT config defaults, nothing has been promoted.]

**If running LLaVA:** [state the two blockers from SRF_details.md 8.1 and 8.2
before proposing any run. `eval.py::load_model` hardcodes the Qwen class, and
MMHal-Bench has no loader.]

**Best config:** [from RESEARCH_STATUS current best]

**Results (Qwen2.5-VL-3B):**
| Dataset | Baseline | SRF-E | Δ |
|---------|----------|-------|---|
| [fill from RESEARCH_STATUS] | | | |

**Comparison baselines (MMVP pair / VLMBias / VLind pair):**
| Method | MMVP | VLMBias | VLind |
|---|---|---|---|
| [fill from RESEARCH_STATUS comparison table] | | | |

**Open tasks (priority order):** [from SRF_details.md section 5.7, then RESEARCH_STATUS]

**Blocking issue:** [state whether McNemar/bootstrap CIs exist yet. MMVP is 150
pairs so 1 pair = 0.67pp, and most measured differences are 1-4 pairs.]

**Quick commands:**
```bash
cd /volumes2/mllm/lmms-eval

# Full eval — all methods, all datasets
source activate mllm && stdbuf -oL -eL python -u srf/eval.py \
  --method srfe --datasets mmvp vlmbias vlind --gamma 3.0 \
  2>&1 | tee /tmp/srfe_all.log

# SRF-Fovea sweep on MMVP
source activate mllm && python srf/test_srffovea_mmvp.py --sigma 20 30 50 100 \
  2>&1 | tee /tmp/srffovea_all.log

# Comparison baseline (one method)
source activate mllm && stdbuf -oL -eL python -u srf/eval.py \
  --method vhr --datasets mmvp vlmbias vlind 2>&1 | tee /tmp/vhr_all.log
```

---

*Per-component detail, exact functions and every measured number: `/volumes2/mllm/SRF_details.md`.
Architecture, CLI flags, code change log: `srf/docs/CONTEXT.md`.
Experiment history: `srf/docs/EXPERIMENTS.md`. Paper: `/volumes2/mllm/PAPER/ICLR/`.*
