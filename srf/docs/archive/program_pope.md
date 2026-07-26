# autoresearch — POPE × Qwen2.5-VL SRF

Autonomous research loop to reduce hallucination across all three POPE splits
(adversarial + popular + random) by improving the Semantic Re-Focus (SRF) method:
query-conditioned saliency-guided attention intervention. **Optimise for average accuracy.**

---

## The Problem

VLMs hallucinate objects on POPE adversarial because language priors override visual
evidence. Boosting all image tokens uniformly does not help (attn_salience, clip_salience
all = 83.3% = baseline). The hypothesis: **boosting only the tokens where the queried
object would be** creates a stronger localised signal that the language prior cannot
easily override.

The method has two independently tunable stages:

```
Stage 1 — Saliency   : which image tokens are query-relevant?
Stage 2 — Biasing    : how do we shift attention toward those tokens?
```

Both stages are configured in `srf.py`. The eval harness is `pope_eval_all.py` (never modify).

---

## Files

| File | Role |
|------|------|
| `pope_eval.py` | **IMMUTABLE** — loop harness (adversarial n=100, ~5 min). Use in the loop. |
| `pope_eval_all.py` | **IMMUTABLE** — full harness (all 3 splits n=100 each). Use for final validation only. |
| `pope_eval_fast.py` | **IMMUTABLE** — adversarial n=50. Kept for reference. |
| `srf.py` | **YOUR SANDBOX** — modify `SALIENCY`, `BIAS`, and/or the implementation. |
| `results.tsv` | Experiment log — untracked by git, never commit. |
| `program.md` | This file — human-written instructions. |

---

## Setup (do once before the loop)

```bash
cd /volumes2/mllm/lmms-eval

# NOTE: always use the mllm conda env for all evals
# conda run -n mllm python ...   OR   conda activate mllm && python ...

# 1. Create a fresh branch
git checkout -b autoresearch/<tag>   # e.g. autoresearch/apr23-pope-srf

# 2. Run baseline to confirm the harness works and log the starting point
conda run -n mllm python my_analysis/autoresearch/pope_eval.py > run.log 2>&1
grep "POPE accuracy:" run.log

# 3. Initialise results.tsv with the baseline row
echo -e "commit\taccuracy\tstatus\tdescription" > my_analysis/autoresearch/results.tsv
echo -e "$(git rev-parse --short HEAD)\t<acc>\tkeep\tbaseline: <description>" \
  >> my_analysis/autoresearch/results.tsv
```

---

## The Loop

LOOP FOREVER until manually interrupted:

```
1. Read git log + results.tsv — what has been tried? what pattern?
2. Pick ONE hypothesis (Stage 1 OR Stage 2 — never both in the same commit)
3. Edit srf.py
4. git add my_analysis/autoresearch/srf.py
   git commit -m "experiment: <brief description>"
5. conda run -n mllm python my_analysis/autoresearch/pope_eval.py > run.log 2>&1
6. grep "POPE accuracy:" run.log
7. If this is a Stage 1 experiment: inspect my_analysis/autoresearch/vis/sample_*.png
   to confirm the saliency map is localising the queried object correctly.
8. Compare to current best:
   - IMPROVED  → keep commit, it is the new baseline
   - NOT IMPROVED → git reset --hard HEAD~1
9. Log to results.tsv:
   echo -e "$(git rev-parse --short HEAD)\t<acc>\t<keep|discard|crash>\t<desc>" \
     >> my_analysis/autoresearch/results.tsv
   NOTE: once a promising config is found, validate with pope_eval_all.py (all 3 splits)
10. Repeat
```

**NEVER STOP once the loop has started.**
**NEVER ask if you should continue.**
**NEVER modify pope_eval.py, pope_eval_all.py, or pope_eval_fast.py.**

---

## Metric

```bash
grep "POPE accuracy:" run.log
```

**Loop harness**: `pope_eval.py` — adversarial split, n=100, ~5 min per run.
**Final validation**: `pope_eval_all.py` — all 3 splits n=100 each, after finding a strong config.

Higher is better.

**Qwen no-intervention baseline (adversarial n=100):** 0.8800 (measured).
Any SRF config below 0.8800 is actively hurting and must be discarded.

Statistical guidance (adversarial n=100):
- Gain ≥ 0.010 (1.0%) → clearly meaningful
- Gain 0.005–0.010   → real, keep it
- Gain < 0.005       → within noise, treat as discard

---

## Stage 1: Saliency Search Strategy

**CLIP is qualitatively better at localising salient regions for POPE.** Prioritise
CLIP variants and CLIP+HSSA ensemble over pure HSSA.

**Saliency visualizations** are saved automatically for the first 5 samples of every run:
```
my_analysis/autoresearch/vis/sample_01.png  …  sample_05.png
```
Each figure shows: Original | CLIP overlay | HSSA overlay | Ensemble (if applicable).
After a Stage 1 experiment, inspect these to confirm the queried object is correctly localised
before trusting the accuracy number. A saliency that is visually wrong is not worth keeping
even if accuracy ticked up slightly (it may be masking a bug).

### CLIP parameters to sweep

| Param | Current | Try |
|-------|---------|-----|
| `clip_coarse_grid` | 7 | 5, 9 (coarser vs finer regions) |
| `clip_top_k_pct` | 0.30 | 0.15, 0.20, 0.25, 0.40 |
| `clip_absence_thresh` | 0.20 | 0.15, 0.25, 0.30 |
| `clip_use_soft` | True | False (binary top-k) |

### HSSA parameters to sweep

| Param | Current | Try |
|-------|---------|-----|
| `hssa_layer` | 16 | **8, 10, 12, 14, 18, 20** ← only 16 tried, sweep is high priority |
| `hssa_top_k_pct` | 0.30 | 0.20, 0.40 |
| `hssa_use_soft` | True | False |

### Ensemble

Try `source = "clip_hssa"` with different weight ratios after finding best individual params:
- clip_weight=0.7, hssa_weight=0.3 (current default)
- clip_weight=0.5, hssa_weight=0.5
- clip_weight=0.8, hssa_weight=0.2

### Presence/absence-conditional strategy (high potential for POPE)

POPE has ~50% absent objects. For absent objects, the current fallback is uniform saliency.
Consider: when CLIP detects absence (max_sim < absence_thresh), apply a *different* strategy:
- Option A: set salience_mask = None (uniform — current behaviour)
- Option B: soft-suppress all image tokens slightly (negative uniform bias) to push "No"
- Option C: invert saliency (boost background, suppress nothing) to highlight absence

This is a structural change to the implementation section — valid to try as one experiment.

---

## Stage 2: Attention Biasing Search Strategy

Fix the best Stage 1 config before sweeping Stage 2.

### bias_mode — the core mathematical approach (sweep this first)

| Mode | What it does | Key param |
|------|-------------|-----------|
| `prob_interp` | Redistribute img attention budget by saliency; non-img attn unchanged | `interp_lambda` |
| `prob_scale` | p_i *= (1 + alpha·sal_i), renorm | `boost_alpha` |
| `attn_floor` | max(p_i, floor) for salient tokens, renorm | `prob_floor` |
| `additive_logit` | logit += alpha·sal_i pre-softmax (original) | `boost_alpha` |
| `global_redistribute` | Scale up *total* img fraction by img_scale, distribute within budget by saliency | `img_scale` |

**Recommended order**: `prob_interp` → `global_redistribute` → `prob_scale` → `attn_floor` → `additive_logit`

`prob_interp` is the most principled redistribution within the existing budget.
`global_redistribute` addresses the root cause: overall attention predominantly assigned to
text. It scales the total img fraction up (e.g. img_scale=2.0 doubles it) and distributes
within the enlarged budget by saliency. Text/sys tokens are scaled down proportionally.

### Per-mode parameter sweeps

**prob_interp**: `interp_lambda` ∈ {0.3, 0.5, 0.7, 1.0}
- λ=0 = no-op; λ=1 = fully replace img distribution with saliency-weighted target

**global_redistribute**: `img_scale` ∈ {1.5, 2.0, 3.0, 4.0}
- img_scale=1.0 = no-op (just redistributes by saliency, like prob_interp λ=1)
- img_scale=2.0 = doubles total img attention, distributes within by saliency
- Capped at 0.95 to prevent degenerate all-image attention

**prob_scale**: `boost_alpha` ∈ {0.5, 1.0, 1.5, 2.0, 3.0}

**attn_floor**: `prob_floor` ∈ {0.002, 0.005, 0.01, 0.02}

**additive_logit**: `boost_alpha` ∈ {1.0, 1.5, 2.0, 3.0, 4.0}; `background_eps` ∈ {0.0, 0.05, 0.10, 0.20}

### Layer range (sweep after finding best mode)

| Range | Rationale |
|-------|-----------|
| `(6, 13)` | Earlier fusion |
| `(8, 15)` | ClearSight default (current) |
| `(10, 18)` | Later fusion |
| `(12, 20)` | Very late — output shaping zone |

### Head selection (sweep after layer range)

| `head_top_k_pct` | Effect |
|------------------|--------|
| 0.30 | Most selective — only strongest vision heads |
| 0.50 | Balanced (current) |
| 0.70 | More inclusive |
| 0.00 | All heads |

### System suppression (sys_beta — all modes, always > 0)

- Empirically confirmed to help. Never set to 0.
- Current: 0.10. Try 0.05, 0.15, 0.20.
- Sweep sys_beta independently after finding best mode + layer + head config.

---

## Idea Generation When Stuck

If the last 5 experiments show no gain, try a qualitatively different angle:

1. **Switch source entirely** — if CLIP plateau, try hssa sweep; if hssa plateau, try ensemble
2. **Flip to binary saliency** — soft masks can be noisy; binary top-k is sharper
3. **Narrow the layer range** — try a single layer (e.g. layer_start=12, layer_end=12)
4. **Expand head selection** — try head_top_k_pct=0.70 or 0.0 (all heads)
5. **Try large alpha with small top_k** — e.g. alpha=4.0, top_k=0.15 (very focused boost)
6. **Check last_run.json** — look at which specific samples flipped right→wrong or wrong→right;
   patterns in failure cases inform the next hypothesis

---

## Simplicity Criterion

Equal accuracy + simpler config → keep. A 0.001 gain from an extra 10-line
implementation change → probably not worth it. A 0.001 gain from changing
one number → keep.

---

## results.tsv Format

Tab-separated. Never use commas inside descriptions (they break TSV).
Columns: commit | accuracy | status | description
(accuracy = adversarial n=50 fast score during search; use "full:" prefix when from pope_eval_all.py)

```
commit	accuracy	status	description
a1b2c3d	0.8800	keep	baseline: no intervention adversarial n=50
b2c3d4e	0.8600	keep	SRF start: prob_interp λ=0.5 clip-7x7 top30%
c3d4e5f	0.8800	keep	prob_interp λ=0.3 (best so far)
d4e5f6g	0.8700	discard	prob_interp λ=0.7 no gain over λ=0.3
e5f6g7h	0.0000	crash	IndexError in clip_sal
f6g7h8i	full:0.8950	keep	VALIDATION: full 3-split avg after finding best config
e5f6g7h	0.0000	crash	IndexError in clip_sal — noun extractor failed on short question
```

Status values: `keep` | `discard` | `crash`

---

## Crash Protocol

```bash
tail -30 run.log   # read the traceback
```

### Classify the failure

| Failure type | Action |
|---|---|
| Syntax / typo / missing import / wrong dict key | Fix in `srf.py`, `git commit --amend`, re-run |
| Logic bug in `srf.py` implementation (wrong tensor shape, index OOB, etc.) | Fix bug in `srf.py`, `git commit --amend`, re-run |
| Logic bug in `qwen_attn_patch.py` (implementation of a bias mode) | Fix bug in `qwen_attn_patch.py`, `git add my_analysis/qwen_attn_patch.py my_analysis/autoresearch/srf.py`, `git commit --amend`, re-run |
| Fundamental failure (wrong architecture assumption, CLIP/HSSA API changed) | `git reset --hard HEAD~1`, log as crash (acc=0.0000), move on |
| Timeout (>25 min) | Kill, treat as crash; if HSSA involved consider CLIP-only fallback |

**Bug fix rule:** A bug fix that has no effect on experiment intent is NOT a separate experiment commit.
Use `git commit --amend` to fold it into the experiment commit, then re-run cleanly.

**You may modify `qwen_attn_patch.py`** to fix bugs in bias mode implementations — this is part
of making the experiment work correctly, not a new experiment. The experiment commit message
should note if a bug was fixed: `"experiment: global_redistribute img_scale=2.0 (fix tensor dim)"`.

**You may NOT modify `pope_eval.py`** under any circumstances.

---

## Runtime Estimates

| Source | Per-sample | n=100 | Per experiment |
|--------|-----------|-------|----------------|
| CLIP only | ~1.5s | ~2.5 min | ~5 min total |
| HSSA only | ~3s | ~5 min | ~8 min total |
| clip_hssa | ~4s | ~7 min | ~10 min total |

Overnight run (8h): ~100 CLIP experiments, ~60 HSSA, ~50 ensemble.
