---
name: load-mllm
version: v3.0
description: Loads project context for the SRF/MLLM research repo. Invoke at the start of any session. Use /load-mllm.
---

# load-mllm

Read these three files in parallel, then output the briefing below.

| File | What it covers |
|------|---------------|
| `srf/CONTEXT.md` | File map, algorithm, hyperparams, CLI reference |
| `srf/RESEARCH_STATUS.md` | Current results, open tasks, recent runs, known issues |
| `srf/config.py` | Live hyperparameter values (single source of truth) |

---

## Briefing format

Output this after reading:

---

### SRF Session Briefing

**Repo:** `/volumes2/mllm/lmms-eval` | **Branch:** `autoresearch/mmvp-srf` | **Env:** `conda run -n mllm`

**Best config:** [from RESEARCH_STATUS current best]

**Results:**
| Dataset | Baseline | SRF | Δ |
|---------|----------|-----|---|
| [fill from RESEARCH_STATUS] | | | |

**Open tasks (priority order):** [top 3 from RESEARCH_STATUS]

**Quick commands:**
```bash
# Val set (60 samples, ~3 min)
conda run -n mllm python srf/eval_pope_val.py --srf --out results/pope_val_srf.json

# Full POPE (~75 min)
conda run -n mllm python srf/eval.py --method srf --datasets pope --output results/srf_pope/
```

---

*For architecture details, CLI flags, and boosting methods — see `srf/CONTEXT.md`. For experiment history — see `srf/RESEARCH_STATUS.md`.*
