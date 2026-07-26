# autoresearch — MME Targeted (Object-Level) × Qwen2.5-VL-3B SRF

Improve SRF on MME Object-Level (existence+count+position+color = 240 questions).
This is the exact set ClearSight Table 3 reports. We compare directly.

**Current best**: 219/240 (SRF exp1, +1 vs baseline 218/240)
**Remaining errors**: count=8, position=8, color=4, existence=1
**Primary targets**: count and position (8 wrong each, most headroom)

Reporting format (MME standard score, ×200/60):
  baseline: 726.7/800  |  SRF_exp1: 730.0/800 (+3.3 on position)
  ClearSight (LLaVA-1.5-7B): baseline=610/800, VAF=636.7/800

---

## Files

| File | Role |
|------|------|
| `mme_eval_targeted.py` | **IMMUTABLE** harness — 240 questions, ~15 min/run |
| `srf.py` | **SANDBOX** — modify SALIENCY and BIAS only |
| `results_targeted.tsv` | Experiment log — append after each run |
| `program_targeted.md` | This file |

---

## The Loop

```
1. Read results_targeted.tsv and git log — what has been tried?
2. Pick ONE hypothesis targeting count OR position improvement
3. Edit srf.py (one stage change only)
4. git add my_analysis/autoresearch_mme/srf.py && git commit -m "mme-targeted: <desc>"
5. conda run -n mllm python my_analysis/autoresearch_mme/mme_eval_targeted.py \
     > my_analysis/autoresearch_mme/run_targeted.log 2>&1
6. Read result line: "MME targeted: srf=X/240  baseline=Y/240"
7. Read per-category: count=A/60  position=B/60  color=C/60  existence=D/60
8. IMPROVED on count or position → keep commit
   NOT IMPROVED → git reset --hard HEAD~1
9. Append to results_targeted.tsv:
   <commit>\t<total>/240\t<base>/240\t<exist>\t<count>\t<pos>\t<color>\t<keep|discard>\t<desc>
```

**NEVER STOP. NEVER ASK. NEVER modify the harness.**

---

## Metric

```bash
grep -E "MME targeted:|Per-category|color|count|exist|position" \
  my_analysis/autoresearch_mme/run_targeted.log | tail -8
```

Primary: maximize total /240.
Secondary priority: count, then position (8 wrong each).
Do NOT sacrifice color (4 wrong) or existence (1 wrong) to gain on count/position.

Statistical noise threshold (n=240):
- +3 or more → clearly real
- +2 → keep if simple change
- +1 → keep only if mechanistically justified
- 0 or negative → discard

---

## Search Strategy

### Priority 1 — HSSA for position (highest expected gain)
Position questions: "Is the pineapple to the left of the pot?"
HSSA captures the model's internal representation of the FULL question including
spatial prepositions. CLIP just matches "pineapple and pot" spatially.

Sweep:
  source="hssa"  hssa_layer ∈ {8, 12, 16, 20, 24}  hssa_top_k_pct ∈ {0.30, 0.40}

### Priority 2 — Wider top_k for count
Count questions need ALL instances visible (not just the peak region).
top_k_pct=0.50 or 0.60 broadens coverage to capture multiple object instances.

Sweep:
  source="clip"  clip_top_k_pct ∈ {0.40, 0.50, 0.60}  (all else fixed from exp1)

### Priority 3 — Layer range for position
Middle layers (8-15) are ClearSight's fusion zone but may over-homogenize.
Late layers (28-35) apply gentler, more localized intervention.
Try for position improvement without hurting count/color.

Sweep:
  layer_start, layer_end ∈ {(8,15), (12,20), (20,28), (28,35)}

### Priority 4 — CLIP+HSSA ensemble
Combine CLIP spatial precision with HSSA question-conditioned alignment.
  source="clip_hssa"  clip_weight=0.5  hssa_weight=0.5  hssa_layer=16

### Priority 5 — SRF-e contrastive (targeted categories only)
On targeted categories, objects ARE usually present → contrastive signal cleaner.
  Run separate β sweep: 0.1, 0.2, 0.3 in mme_eval_targeted.py (inline, not srf.py)

### Priority 6 — boost_alpha sweep
  boost_alpha ∈ {0.5, 1.0, 1.5, 3.0}  (exp1 used 2.0)

### Priority 7 — head selection
  head_top_k_pct ∈ {0.20 (current), 0.30, 0.50}
  NOTE: only 3 heads selected at 0.20 — may be too few. Try 0.30 first.

---

## Category Analysis (use when stuck)

```python
import json
d = json.load(open("my_analysis/autoresearch_mme/targeted_run.json"))
wrong = [r for r in d["samples"] if not r["ok_srf"]]
for r in wrong:
    print(r["cat"], r["gt"], r.get("base","?"), r.get("srf","?"))
```

Look for: which category flips help? which hurt? patterns by question type?

---

## When Stuck (>4 experiments no gain)

1. Check targeted_run.json — analyze which specific wrong answers SRF is close on
2. Try binary saliency: clip_use_soft=False
3. Increase head_top_k_pct to 0.30 (3 heads → 4-5 heads, more coverage)
4. Try no system suppression: sys_beta=0.0
5. Try global_redistribute with img_scale=2.0 on count questions

---

## Runtime

mme_eval_targeted.py: ~15-20 min per run (240 samples, 3 min model load + CLIP calibration)
Overnight (8h): ~24 experiments feasible.
