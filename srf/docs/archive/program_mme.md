# autoresearch — MME × Qwen2.5-VL-3B SRF

Autonomous research loop to improve Qwen2.5-VL-3B-Instruct accuracy on MME
(perception categories) by improving the Semantic Re-Focus (SRF) method:
query-conditioned saliency-guided attention intervention. **Optimise for MME score (correct/200).**

---

## The Problem

VLMs miss visual details on diverse VQA questions (existence, color, count, position, scene,
OCR, etc.) because attention to image tokens is diluted by text. SRF redirects attention toward
the query-relevant image region, giving the model a stronger visual signal.

MME is harder than POPE: questions are more diverse (not just "Is there a [object]?"), so
CLIP noun extraction is noisier. Expect the absence-aware suppress strategy to be LESS useful
than on POPE (since not all MME questions are existence questions). Start with suppress_thresh=0.0.

The method has two independently tunable stages:

```
Stage 1 — Saliency   : which image tokens are query-relevant?
Stage 2 — Biasing    : how do we shift attention toward those tokens?
```

Both stages are configured in `srf.py`. The eval harness is `mme_eval.py` (never modify).

---

## Files

| File | Role |
|------|------|
| `mme_eval.py` | **IMMUTABLE** — loop harness (perception n=200, ~4 min with CLIP). Use in the loop. |
| `mme_eval_full.py` | **IMMUTABLE** — full 2374-question eval. Use for final validation only. |
| `srf.py` | **YOUR SANDBOX** — modify `SALIENCY`, `BIAS`, and/or the implementation. |
| `results.tsv` | Experiment log — never commit. |
| `program.md` | This file — human-written instructions. |

---

## Setup (do once before the loop)

```bash
cd /volumes2/mllm/lmms-eval

# 1. Run baseline to confirm harness works and measure starting point
conda run -n mllm python my_analysis/autoresearch_mme/mme_eval.py > run.log 2>&1
grep "MME score:" run.log

# 2. Initialise results.tsv
echo -e "commit\tscore\tbase\tstatus\tdescription" \
  > my_analysis/autoresearch_mme/results.tsv
echo -e "$(git rev-parse --short HEAD)\t<score>/200\t<base>/200\tkeep\tbaseline: no intervention" \
  >> my_analysis/autoresearch_mme/results.tsv
```

---

## The Loop

LOOP FOREVER until manually interrupted:

```
1. Read git log + results.tsv — what has been tried? what pattern?
2. Pick ONE hypothesis (Stage 1 OR Stage 2 — never both in the same commit)
3. Edit srf.py
4. git add my_analysis/autoresearch_mme/srf.py
   git commit -m "mme-exp: <brief description>"
5. conda run -n mllm python my_analysis/autoresearch_mme/mme_eval.py > run.log 2>&1
6. grep "MME score:" run.log
7. If Stage 1 experiment: inspect vis/sample_*.png for saliency quality
8. Compare to current best:
   - IMPROVED  → keep commit, it is the new baseline
   - NOT IMPROVED → git reset --hard HEAD~1
9. Log to results.tsv:
   echo -e "$(git rev-parse --short HEAD)\t<srf>/200\t<base>/200\t<keep|discard|crash>\t<desc>" \
     >> my_analysis/autoresearch_mme/results.tsv
   NOTE: once best config found, validate with mme_eval_full.py
10. Repeat
```

**NEVER STOP once the loop has started.**
**NEVER ask if you should continue.**
**NEVER modify mme_eval.py or mme_eval_full.py.**

---

## Metric

```bash
grep "MME score:" run.log
```

**Loop harness**: `mme_eval.py` — perception n=200, ~4 min per run.
**Final validation**: `mme_eval_full.py` — full 2374 questions after finding strong config.

Higher correct/200 is better.

**First run will establish the baseline.** Any SRF config below that baseline is hurting and must be discarded.

Statistical guidance (n=200):
- Gain ≥ 4 samples (2.0%) → clearly meaningful
- Gain 2-3 samples (1.0-1.5%) → real, keep it
- Gain 1 sample (0.5%) → within noise; consider keeping if simple change
- 0 or negative → discard

---

## Stage 1: Saliency Search Strategy

MME note: Questions cover existence, color, count, position, scene, landmark, OCR, etc.
CLIP noun extraction works for most but is noisier than POPE. HSSA may be more robust for
non-existence questions since it uses the model's own question representation.

### Starting order
1. First establish Stage 2 baseline (additive_logit α=2.0 is default)
2. Then try CLIP top_k_pct sweep
3. Then try HSSA (may outperform CLIP for MME diversity)
4. Then try ensemble

### CLIP parameters to sweep

| Param | Current | Try |
|-------|---------|-----|
| `clip_top_k_pct` | 0.30 | 0.15, 0.20, 0.25, 0.40 |
| `clip_coarse_grid` | 7 | 5, 9 |
| `clip_use_soft` | True | False (binary top-k) |

### HSSA parameters to sweep

| Param | Current | Try |
|-------|---------|-----|
| `hssa_layer` | 8 | **12, 16, 20, 24** ← sweep is high priority |
| `hssa_top_k_pct` | 0.30 | 0.20, 0.40 |

### Absence-aware (MME-specific note)

`clip_suppress_thresh=0.0` (default) — always boost. This is correct starting point for MME
because most categories are NOT existence questions. Only enable suppress for existence category
if it specifically helps (would need category-specific logic — deferred).

---

## Stage 2: Attention Biasing Search Strategy

### bias_mode — sweep this first

| Mode | What it does | Key param |
|------|-------------|-----------|
| `additive_logit` | logit += alpha·sal_i pre-softmax | `boost_alpha` |
| `prob_interp` | Redistribute img attention budget by saliency | `interp_lambda` |
| `prob_scale` | p_i *= (1 + alpha·sal_i), renorm | `boost_alpha` |
| `attn_floor` | max(p_i, floor) for salient tokens, renorm | `prob_floor` |
| `global_redistribute` | Scale up total img fraction by img_scale, dist. by sal. | `img_scale` |

**Recommended sweep order**: `prob_interp` → `global_redistribute` → `prob_scale` → `attn_floor` → `additive_logit`

### Per-mode parameter sweeps

**prob_interp**: `interp_lambda` ∈ {0.3, 0.5, 0.7, 1.0}

**global_redistribute**: `img_scale` ∈ {1.5, 2.0, 3.0, 4.0}

**prob_scale**: `boost_alpha` ∈ {0.5, 1.0, 1.5, 2.0, 3.0}

**attn_floor**: `prob_floor` ∈ {0.002, 0.005, 0.01, 0.02}

**additive_logit**: `boost_alpha` ∈ {1.0, 1.5, 2.0, 3.0, 4.0}; `background_eps` ∈ {0.0, 0.05, 0.10}

### Layer range

| Range | Rationale |
|-------|-----------|
| `(6, 13)` | Earlier fusion |
| `(8, 15)` | ClearSight default (current) |
| `(10, 18)` | Later fusion |
| `(12, 20)` | Very late — output shaping zone |

### Head selection

| `head_top_k_pct` | Effect |
|------------------|--------|
| 0.20 | Most selective (current) |
| 0.30 | Slightly more inclusive |
| 0.50 | Balanced |
| 0.00 | All heads |

### sys_beta

Current: 0.10. Try 0.05, 0.15, 0.20.

---

## Idea Generation When Stuck

If last 5 experiments show no gain:

1. **Switch source** — if CLIP plateaus, try HSSA sweep
2. **Try binary saliency** — `clip_use_soft=False`
3. **Narrow the layer range** — e.g. `layer_start=12, layer_end=14`
4. **Check last_run.json** — which categories drive failures? Patterns inform next hypothesis
5. **Try large alpha small top_k** — e.g. `boost_alpha=4.0, clip_top_k_pct=0.15`
6. **Category analysis**: `python -c "import json; d=json.load(open('last_run.json')); [print(r['cat'],r['gt'],r['ok_srf']) for r in d['samples'] if not r['ok_srf']]"`

---

## Runtime Estimates

| Source | Per-sample | n=200 | Per experiment |
|--------|-----------|-------|----------------|
| CLIP only | ~1.5s | ~5 min | ~6 min total |
| HSSA only | ~3s | ~10 min | ~11 min total |
| clip_hssa | ~4s | ~13 min | ~14 min total |

Overnight run (8h): ~80 CLIP experiments, ~45 HSSA, ~35 ensemble.

---

## results.tsv Format

Tab-separated. Columns: commit | score | base | status | description

```
commit	score	base	status	description
a1b2c3d	160/200	160/200	keep	baseline: no intervention
b2c3d4e	165/200	160/200	keep	additive_logit alpha=2.0 clip-7x7 top30%
c3d4e5f	163/200	160/200	discard	alpha=3.0 worse than alpha=2.0
full:d5e6f7g	full:1842	full:1750	keep	VALIDATION: full 2374q after best config
```

Status values: `keep` | `discard` | `crash`

---

## Crash Protocol

```bash
tail -30 run.log   # read the traceback
```

| Failure type | Action |
|---|---|
| Syntax / typo / missing import | Fix in `srf.py`, `git commit --amend`, re-run |
| Logic bug in `srf.py` | Fix, `git commit --amend`, re-run |
| Bug in `qwen_attn_patch.py` | Fix both files, `git commit --amend`, re-run |
| Fundamental failure | `git reset --hard HEAD~1`, log crash, move on |
| Timeout (>20 min) | Kill, treat as crash |
