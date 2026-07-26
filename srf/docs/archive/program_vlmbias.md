# autoresearch — VLM Bias × Qwen2.5-VL SRF

Autonomous research loop to reduce prior-driven errors on the VLM Bias dataset
(anvo25/vlms-are-biased, all 7 categories) using SRF: CLIP-guided attention boosting.

**HSSA is removed.** Confirmed dead from POPE autoresearch — do not re-add.

---

## The Problem

VLMs answer VLM Bias questions using language priors (e.g. typical animal leg counts,
expected chess piece counts) rather than visually counting what is in the image.
SRF hypothesis: boosting attention to the CLIP-localised region forces the model to
look at the right part of the image before generating the answer.

Object is always present in VLM Bias images — no suppress logic, always boost.
When CLIP can't localise (max_sim < fallback_thresh) → uniform boost (= vhr behaviour).

```
Stage 1 — Saliency  : CLIP identifies which image tokens are query-relevant
Stage 2 — Biasing   : Shift attention toward those tokens in vision-aware heads
```

---

## Files

| File | Role |
|------|------|
| `vlmbias_eval.py` | **IMMUTABLE** — all 7 categories, n=15/cat (~105 total), Qwen2.5-VL-3B. Prints `VLM Bias accuracy: X.XXXX`. |
| `srf.py` | **YOUR SANDBOX** — modify `SALIENCY`, `BIAS`, and/or implementation. |
| `results.tsv` | Experiment log — untracked, never commit. |
| `program.md` | This file. |

---

## Setup (do once)

```bash
cd /volumes2/mllm/lmms-eval

git checkout -b autoresearch/vlmbias-srf

# Run baseline to confirm harness works
conda run -n mllm python my_analysis/autoresearch_vlmbias/vlmbias_eval.py > run.log 2>&1
grep "VLM Bias accuracy:" run.log

# Initialise results.tsv
echo -e "commit\taccuracy\tstatus\tdescription" > my_analysis/autoresearch_vlmbias/results.tsv
echo -e "$(git rev-parse --short HEAD)\t<acc>\tkeep\tbaseline: warmstart from POPE best settings" \
  >> my_analysis/autoresearch_vlmbias/results.tsv
```

---

## The Loop

```
1. Read git log + results.tsv — what has been tried? what pattern?
2. Pick ONE hypothesis (Stage 1 OR Stage 2 — never both in the same commit)
3. Edit srf.py
4. git add my_analysis/autoresearch_vlmbias/srf.py
   git commit -m "experiment: <brief description>"
5. conda run -n mllm python my_analysis/autoresearch_vlmbias/vlmbias_eval.py > run.log 2>&1
6. grep "VLM Bias accuracy:" run.log
7. For Stage 1 experiments: check vis/sample_*.png — is CLIP localising correctly?
8. Compare to current best:
   - IMPROVED (≥0.010) → keep commit, new baseline
   - NOT IMPROVED      → git reset --hard HEAD~1
9. Log to results.tsv:
   echo -e "$(git rev-parse --short HEAD)\t<acc>\t<keep|discard|crash>\t<desc>" \
     >> my_analysis/autoresearch_vlmbias/results.tsv
10. Repeat
```

**NEVER STOP. NEVER ask to continue. NEVER modify vlmbias_eval.py.**

---

## Metric

```bash
grep "VLM Bias accuracy:" run.log
```

**Baseline (Qwen2.5-VL-3B, no intervention):** ~0.171 (17.1%)
**Previous best (vhr_boost v2 w=8, full dataset):** 0.226 (22.6%)

Keep threshold: gain **≥ 0.010** (1.0%) — baseline is noisier than POPE, need clear signal.
Discard if gain < 0.005.

Per-category breakdown is in `last_run.json` and printed to run.log — use it to
understand which categories are driving gains/losses.

---

## Stage 1: CLIP Search Strategy

CLIP is useful for: **Chess Pieces** (board), **Logos** (logo), **Flags** (flag).
For Animals / Patterned Grid / Optical Illusion → CLIP returns low max_sim → uniform boost fallback.
That's fine — don't try to fix noun extraction for uncalibratable categories.

### Params to sweep

| Param | Current | Try |
|-------|---------|-----|
| `clip_coarse_grid` | 7 | 5, 9 |
| `clip_top_k_pct` | 0.30 | 0.15, 0.20, 0.25, 0.40 |
| `clip_use_soft` | True | False |
| `clip_fallback_thresh` | 0.20 | 0.15, 0.25 |

---

## Stage 2: Attention Biasing Search Strategy

Fix best Stage 1 first. Then sweep Stage 2 — same modes as POPE loop.

### bias_mode (sweep first)

| Mode | Key param |
|------|-----------|
| `additive_logit` | `boost_alpha` ∈ {1.5, 2.0, 3.0, 4.0} |
| `global_redistribute` | `img_scale` ∈ {1.5, 2.0, 3.0} |
| `prob_interp` | `interp_lambda` ∈ {0.3, 0.5, 0.7, 1.0} |

VLM Bias has stronger priors than POPE — try higher `boost_alpha` (3.0, 4.0) early.

### Layer range (after best mode found)

| Range | Rationale |
|-------|-----------|
| `(8, 14)` | POPE best (current) |
| `(8, 20)` | Wider — harder task may benefit from later layers |
| `(10, 20)` | Later fusion only |

### Other

- `head_top_k_pct`: 0.20 → try 0.30, 0.50
- `sys_beta`: 0.10 → try 0.15, 0.20 (stronger sys suppression may help prior-conflict)
- `background_eps`: 0.0 → try 0.10, 0.20 (suppress non-salient regions)

---

## When Stuck (5+ experiments no gain)

1. Check `last_run.json` — which categories are improving vs. hurting?
2. If Chess/Logos gain but Animals/Patterned hurt: try category-conditional boost strength
3. Try binary saliency (`clip_use_soft=False`) — sharper mask
4. Try large alpha + small top_k: e.g. alpha=4.0, top_k=0.15
5. Try `global_redistribute` with img_scale=3.0 — VLM Bias may need stronger total img attention

---

## Crash Protocol

Same as POPE loop. Classify: syntax/logic bug → `git commit --amend` + re-run.
Fundamental failure → `git reset --hard HEAD~1`, log as crash.

---

## results.tsv Format

```
commit	accuracy	status	description
a1b2c3d	0.1714	keep	baseline: POPE warmstart additive_logit alpha=2.0 layers 8-14
```
