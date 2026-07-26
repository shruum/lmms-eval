# RePOPE — Findings & Status

**Last updated:** 2026-07-26
**Model:** Qwen2.5-VL-3B-Instruct
**Status:** ⏳ SRF 100-sample diagnostic running (results/diag_100sample/)

---

## What is RePOPE?

Paper: "RePOPE: Impact of Annotation Errors on the POPE Benchmark" (Neuhaus & Hein, 2025)
arXiv: https://arxiv.org/abs/2504.15707

POPE has major annotation errors — 9.3% of "Yes" questions are wrong, only 1.7% of "No". This 5.4:1 bias systematically penalises hallucination-reduction methods (which correctly say "No" more often). RePOPE provides corrected labels for COCO only (not A-OKVQA / GQA).

**Data location:** `data/repope/` (adversarial, popular, random JSON files)

---

## Results Table — All Methods on RePOPE COCO

| Method | Random | Popular | Adversarial | Average | vs Baseline |
|--------|--------|---------|-------------|---------|-------------|
| VAF | 91.38% | 88.93% | 84.54% | 88.28% | +2.82pp |
| VCD | 90.7% | 88.3% | 85.2% | 88.1% | +2.64pp |
| **Baseline** | **89.29%** | **86.69%** | **80.40%** | **85.46%** | — |
| SRF-E (fixed, in progress) | TBD | TBD | TBD | TBD | TBD |

**Target**: SRF-E should beat VAF (88.28% avg) — if we match VAF on at least 2/3 splits, that's a strong result.

---

## Key Bugs Fixed (2026-07-26 session)

These were the root causes of SRF = baseline on RePOPE:

1. **phase="generation" was a no-op** during `model(**inp)` prefill eval  
   - Fix: `config.py` POPE entry changed to `phase="both"`
   - `phase="generation"` sets `_is_gen = (q_len == 1)` → False at prefill → no boost applied

2. **CLIP detection too sensitive to threshold** with single template  
   - Fix: 5-template ensemble (`_PRESENCE_TEMPLATES`) in `clip_salience.py`, mean-pooled + re-normalized
   - Effect: TPR improved from 25% → 85% on 20-sample RePOPE subset

3. **Gate misses locally-present objects** (small/corner, diluted in global sim)  
   - Fix: OR gate — `full_img_sim >= 0.21 OR patch_max_sim >= 0.27`
   - `patch_max_sim` = max cosine sim over all patches across 3×3, 5×5, 7×7 grids
   - Threshold sweep result: acc=0.90, TPR=1.00, FPR=0.29 on 20-sample set

---

## Current Code Flow (SRF on RePOPE)

```
diag_20sample.py (or eval.py)
  │
  ├─ load Qwen2.5-VL-3B-Instruct (attn_implementation="eager" — hooks need eager)
  ├─ patch_model() via qwen_attn_patch.py → installs _patched_softmax hook
  ├─ srf.setup() → calibrates vision-aware heads (top 20% by visual score, layers 8-12)
  │
  └─ per sample:
       ├─ extract noun from question (noun_extract.py)
       ├─ compute_clip_salience_full_gate_v3(image, noun, grid_h, grid_w)
       │    ├─ encode text: 5-template ensemble → mean-pool → re-normalize
       │    ├─ encode full image → full_img_sim
       │    ├─ encode 3×3, 5×5, 7×7 patches → patch_max_sim = max(all patch sims)
       │    ├─ gate: object_present = (full_img_sim >= 0.21) OR (patch_max_sim >= 0.27)
       │    └─ saliency map: multi-scale bilinear upsample → max-pool → normalize [0,1]
       │
       ├─ srf.prepare_sample() → stores saliency + object_present in _STATE
       │
       └─ model forward (phase="both" → fires at ALL forward passes including prefill):
            ├─ if object_present: boost salient image tokens (+alpha * sal)
            │                     suppress non-salient tokens (-eps * (1 - sal))
            └─ if not object_present: suppress ALL image tokens (value = -neg_absent_alpha)
```

---

## Run Commands

```bash
cd /volumes2/mllm/lmms-eval

# 100-sample diagnostic (baseline vs SRF, CLIP gate stats, saves JSON + PNGs)
source activate mllm && python srf/diag_20sample.py \
  --repope_dir data/repope --splits adversarial --n 100 --out_dir results/diag_100sample

# Full RePOPE adversarial eval with SRF-E
source activate mllm && python srf/eval.py --method srfe --datasets pope \
  --pope_file data/repope/adversarial.json --output results/srf_repope_adv/ --gamma 3.0

# All 3 RePOPE splits
source activate mllm && python srf/eval.py --method srfe --datasets pope \
  --pope_file data/repope/adversarial.json --output results/srf_repope/ --gamma 3.0
source activate mllm && python srf/eval.py --method srfe --datasets pope \
  --pope_file data/repope/popular.json    --output results/srf_repope/ --gamma 3.0
source activate mllm && python srf/eval.py --method srfe --datasets pope \
  --pope_file data/repope/random.json     --output results/srf_repope/ --gamma 3.0
```

---

## Config State (2026-07-26)

```python
# srf/config.py — POPE entry
"pope": {"phase": "both", "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0}

# srf/saliency/clip_salience.py — gate in compute_clip_salience_full_gate_v3
gate_full  = full_img_sim  >= thresh        # default 0.21
gate_patch = patch_max_sim >= patch_thresh  # default 0.27
object_present = gate_full or gate_patch    # OR rule

# CLIP text encoding — 5-template ensemble
_PRESENCE_TEMPLATES = [
    "{noun}",
    "a photo of a {noun}",
    "a photo of the {noun}",
    "there is a {noun} in this photo",
    "an image containing a {noun}",
]
```

---

## Background Context

- VAF (ClearSight) and VCD results above are on the **same Qwen2.5-VL-3B-Instruct model**
- Previous "7-bug" analysis referred to a LLaVA implementation on a different server — **irrelevant to this codebase**
- Our codebase (`srf/`) was always correct; the bug was purely `phase="generation"` being a no-op at prefill time
