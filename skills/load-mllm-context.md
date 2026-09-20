---
name: load-mllm-context
description: Load full MLLM/SRF research context for LLaVA-POPE experiments. Use when the user says "/load-mllm-context", "load mllm context", or starts working on LLaVA hallucination mitigation, SRF on LLaVA, or POPE evaluation. Fully standalone — no need to load vlm-proj-context separately.
version: 1.0.0
---

# MLLM Research Context — LLaVA + POPE Focus

## Step 1 — Repo & Environment

```
Repo:  /home/sgowda/workspace/SRF/lmms-eval      ← call this {REPO}
Conda: mllm
HF cache: set via $HF_HOME (machine-specific)
GPU: CUDA_VISIBLE_DEVICES=0 (single GPU, LLaVA fits in ~15GB)
```

All paths below are relative to {REPO}.

---

## Step 2 — Read These Files in Parallel (on load)

| File | Purpose |
|------|---------|
| `info/CONTEXT.md` | Algorithm reference, hyperparams, CLI |
| `srf/config.py` | Central config — all arch/dataset params |
| `srf/ALL_RESULTS.md` | All experiment results across all models |
| `SRF_EXPERIMENT_STATUS.md` | Latest LLaVA sweep status |
| `srf/srf.py` | SRF base implementation |
| `my_analysis/llava_attn_patch.py` | LLaVA-specific attention patch |

---

## Step 3 — Project Identity

**Goal:** Mitigate VLM object hallucination via inference-time attention steering.
No training. Pure forward-pass intervention.

**Method (SRF):**
1. Extract query noun from question ("Is there a cat?" → "cat")
2. CLIP ViT-B/32 computes patch saliency for that noun over the image
3. During forward pass, in cross-modal fusion layers:
   - Boost attention logits for salient image tokens by α
   - If CLIP max_sim < suppress_thresh → object likely absent → suppress all image tokens
4. Two variants: SRF (single pass) and SRF-E (two-pass contrastive, β amplification)

**Current focus: LLaVA-1.5-7B + POPE. SRF works on Qwen but consistently fails on LLaVA.**

---

## Step 4 — POPE Dataset Structure

POPE = Polling-based Object Probing Evaluation.
**3 image-source datasets × 3 question splits = 9 combinations, 3000 samples each = 27,000 total.**

### The 3 datasets (image sources):
| Dataset | Images | Source |
|---------|--------|--------|
| COCO | 500 COCO val2014 images | `COCO_val2014_000000XXXXXX.jpg` |
| A-OKVQA | 500 COCO val2014 images (different from COCO's 500, 11 overlap) | same format |
| GQA | 500 GQA/Visual Genome images | `XXXXXXX.jpg` |

### The 3 splits (how absent objects are sampled):
| Split | Strategy | Difficulty |
|-------|----------|------------|
| Random | Random objects not in image | Easiest |
| Popular | Frequent objects not in image | Medium |
| Adversarial | Semantically related objects not in image | Hardest |

### Local data location (on this machine):
```
Annotation JSONs (from AoiDragon/POPE GitHub):
  COCO:    ~/pope_data/coco/coco_pope_{adversarial,popular,random}.json         (JSONL)
  A-OKVQA: ~/pope_data/aokvqa/aokvqa_pope_seem_{adversarial,popular,random}.json (JSON array)
  GQA:     ~/pope_data/gqa/gqa_pope_seem_{adversarial,popular,random}.json       (JSON array)

Images:
  COCO + A-OKVQA: ~/pope_data/images/val2014/
  GQA:            ~/pope_data/images/gqa/
```

Each JSON entry: `{"question_id": int, "image": "<filename>", "text": "Is there a X?", "label": "yes"|"no"}`

### Evaluation method (matching VCD/AIR papers):
- `model.generate()` → parse "yes"/"no" from text
- Decoding: `do_sample=True, temperature=0.7, top_p=0.9`
- Metrics: Accuracy, Precision, Recall, F1, yes_ratio (~0.5 = balanced)

### In code: use `--datasets pope_vcd` with `--pope_vcd_file` + `--pope_image_dir`
The HF `lmms-lab/POPE` dataset covers COCO only (3 splits, images embedded). Use VCD-format for all 9.

---

## Step 5 — Validated Baselines (LLaVA-1.5-7B, n=3000, sampling)

These match VCD/AIR papers within ±0.1% average:

| Dataset | Split | Our Baseline | VCD Paper | AIR Paper |
|---------|-------|-------------|-----------|-----------|
| COCO | Random | 83.43% | 83.29% | 83.70% |
| COCO | Popular | 81.23% | 81.88% | 78.20% |
| COCO | Adversarial | 79.30% | 78.96% | 75.00% |
| A-OKVQA | Random | 83.47% | 83.45% | 83.40% |
| A-OKVQA | Popular | 79.30% | 79.90% | 79.90% |
| A-OKVQA | Adversarial | 75.80% | 74.04% | 74.00% |
| GQA | Random | 82.47% | 83.73% | 83.70% |
| GQA | Popular | 76.40% | 78.17% | 78.20% |
| GQA | Adversarial | 75.50% | 75.08% | 75.10% |

**Target:** SRF must beat our baseline AND match/exceed papers. Goal: +1% over baseline.

### VAF (ClearSight) paper results on LLaVA-1.5-7B — the bar to beat:
| Split | Baseline | VAF |
|-------|----------|-----|
| COCO Random | 88.2% | 89.8% (+1.6%) |
| COCO Popular | 86.1% | 87.5% (+1.4%) |
| COCO Adversarial | 82.3% | 83.4% (+1.1%) |
| A-OKVQA Random | 87.6% | 89.4% (+1.8%) |
| A-OKVQA Popular | 81.9% | 84.2% (+2.3%) |
| A-OKVQA Adversarial | 74.3% | 77.2% (+2.9%) |
| GQA Random | 88.0% | 89.5% (+1.5%) |
| GQA Popular | 79.4% | 81.8% (+2.4%) |
| GQA Adversarial | 76.3% | 79.7% (+3.4%) |

Note: VAF baselines are higher than ours — likely greedy decoding vs our sampling. Still the target method to beat.

---

## Step 6 — All SRF Results on LLaVA-1.5-7B POPE

### Round 1: n=100, COCO only, VAF ClearSight params (enh=1.15, sup=0.95)
Small n, no sampling — not reliable, but gave initial signal:
- Random: 84% → 85% (+1.0%) ✅
- Popular: 84% → 84% (0.0%)
- Adversarial: 83% → 84% (+1.0%) ✅
- **Average: +0.67%**

### Round 2: n=3000, all 9 splits, α=0.15, layers 10–15, 50% heads, sampling
| Dataset | Split | Baseline | SRF | Δ |
|---------|-------|----------|-----|---|
| COCO | Adversarial | 77.90% | 76.90% | -1.00% ❌ |
| COCO | Popular | 83.73% | 84.03% | +0.30% ✅ |
| COCO | Random | 85.57% | 86.53% | +0.97% ✅ |
| A-OKVQA | Adversarial | 69.47% | 69.13% | -0.33% ❌ |
| A-OKVQA | Popular | 77.97% | 78.17% | +0.20% ✅ |
| A-OKVQA | Random | 83.97% | 82.57% | -1.40% ❌ |
| GQA | Adversarial | 68.17% | 68.73% | +0.57% ✅ |
| GQA | Popular | 72.93% | 72.80% | -0.13% ❌ |
| GQA | Random | 83.70% | 82.87% | -0.83% ❌ |
| **Mean** | | | | **-0.19%** ❌ |

### Round 3: COCO Adversarial sweep, n=3000, 17 configs (α=1.0–4.0), sampling
Corrected baseline: 79.30%
Top results:
- Config 11 (α=1.0, layers 10–18, 50% heads, eps=0.1): 78.77% → **-0.53%** ❌
- Config 3 (α=2.0, layers 12–18, 90% heads, eps=0.1): 78.70% → -0.60% ❌
- All 17 configs below baseline. Lower α consistently better.

### Round 4: Diagnostic — ultra-low α (0.1–0.5, VAF-like, narrow layers 12–16), 12 configs
Status: **Results pending / not yet retrieved.**

### Summary:
| Round | Coverage | Best Δ | Verdict |
|-------|----------|--------|---------|
| n=100 VAF params | COCO ×3 | +0.67% avg | Noisy, not reliable |
| n=3000 α=0.15 | All 9 | -0.19% avg | Net negative |
| n=3000 α=1–4 sweep | COCO Adv | -0.53% best | All configs fail |
| Diagnostic α=0.1–0.5 | COCO Adv | Unknown | Pending |

---

## Step 7 — Why SRF Fails on LLaVA: Working Hypotheses

1. **Patch mechanism mismatch:**
   - Qwen patch (`qwen_attn_patch.py`): pre-softmax additive logit → exponential effect
   - LLaVA patch (`llava_attn_patch.py`): post-softmax multiplicative + renorm (ClearSight style) → weaker, bounded
   - With 576 image tokens, each has ~0.001 attention weight. Multiplying by 2 barely moves logits.

2. **CLIP-LLaVA feature misalignment:**
   - LLaVA uses CLIP ViT-L/14 (336px, 24×24 = 576 patches) as its vision encoder
   - SRF saliency uses ViT-B/32 with 6×6 coarse grid → different feature space
   - Saliency map may not correspond to what LLaVA's internals actually attend to

3. **Image token count:**
   - 576 tokens is large; top-30% = 173 tokens to boost — too diffuse?
   - Qwen uses spatial merge (2×2) → fewer tokens, higher per-token impact

4. **Head calibration:**
   - `in_language_model` flag guards the patch — must verify hooks fire on correct positions
   - LLaVA expands single `<image>` placeholder to 576 tokens inside the model, not in input_ids

---

## Step 8 — Key Implementation Files

| File | Role |
|------|------|
| `srf/eval.py` | Unified eval CLI — `run_pope_vcd()` for all 9 splits |
| `srf/srf.py` | SRF base: `setup()`, `reset_for_dataset()`, `prepare_sample()`, `cleanup()` |
| `srf/saliency/clip_salience.py` | CLIP saliency computation |
| `my_analysis/llava_attn_patch.py` | LLaVA patch: post-softmax multiplicative, head calib |
| `my_analysis/qwen_attn_patch.py` | Qwen patch: pre-softmax additive, many bias modes |
| `srf/config.py` | All hyperparams — `SRF_ARCH_PARAMS["llava-hf/llava-1.5-7b-hf"]` |

LLaVA arch params in config:
```python
"llava-hf/llava-1.5-7b-hf": {
    "n_layers": 32, "spatial_merge_size": 1,
    "image_token": None,       # uses model.config.image_token_index
    "layer_start": 10, "layer_end": 15,
    "head_top_k_pct": 0.50,
    "clip_coarse_grid": 6,     # 336px images
    "clip_top_k_pct": 0.30,
    "clip_fallback_thresh": 0.20,
    "dataset_layer_end": {"mmvp": 20, "pope": 20, ...}
}
```

---

## Step 9 — Running Experiments

### Baseline (single split):
```bash
cd {REPO}
conda run -n mllm python srf/eval.py \
  --method baseline \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope_vcd \
  --pope_vcd_file ~/pope_data/coco/coco_pope_adversarial.json \
  --pope_vcd_name coco_adversarial \
  --pope_image_dir ~/pope_data/images/val2014 \
  --eval_method generation \
  --do_sample --temperature 0.7 --top_p 0.9 \
  --output results/llava_baseline/
```

### SRF (single split):
```bash
conda run -n mllm python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope_vcd \
  --pope_vcd_file ~/pope_data/coco/coco_pope_adversarial.json \
  --pope_vcd_name coco_adversarial \
  --pope_image_dir ~/pope_data/images/val2014 \
  --eval_method generation \
  --do_sample --temperature 0.7 --top_p 0.9 \
  --alpha 1.0 --layer_start 10 --layer_end 18 --head_top_k_pct 0.5 \
  --output results/llava_srf/
```

### Quick test (small n):
```bash
# Add --n_pope 50 to any command above
```

---

## Important Notes

- Always use `--do_sample --temperature 0.7 --top_p 0.9` for paper-comparable results
- Always use `--eval_method generation` for LLaVA (logits method doesn't work reliably)
- Calibration uses HF `lmms-lab/POPE` regardless of eval dataset — this is correct
- LLaVA loads slowly (~2–3 min); budget accordingly
- Debug prints are active in `srf/srf.py` — remove before production runs
- Results verified against VCD/AIR papers — baselines are correct
