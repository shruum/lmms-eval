# SRF Research Status

> Update this file after every experiment run and commit it.
> This is the live source of truth for results and open tasks — replaces vault reads on remote servers.

---

## Current Best Results (Qwen2.5-VL-3B-Instruct)

| Method | MMVP pair | POPE COCO adv | POPE full (9 combos) | MME (perception, /200) | VLM Bias |
|--------|-----------|---------------|----------------------|------------------------|----------|
| Baseline | 40.00% | 83.33% | — | 177/200 (88.5%) | 17.14% |
| **SRF** | ~44.00% | 84.33% (+1pp) | ❌ not run yet | 176/200 (-0.5%) | 22.86% |
| **SRF-E** β=2.0 | **49.33%** | ~86% (est.) | ❌ not run yet | 174/200 (-1.5%) | broken* |

*VLM Bias SRF-E broken: contrastive pass suppresses format tokens (`{`). Use SRF base only.

All numbers from autoresearch runs (Qwen 3B). Full POPE (9000 samples, all splits) and MME not yet run.

### MME key findings (autoresearch_mme, 2026-04)
- SRF cannot improve MME — best is 176/200 (-0.5% vs baseline)
- Root cause: 77% of MME categories require global context (artwork, celebrity, scene); any attention redistribution hurts
- SRF-E also fails: zero-pixel ViT produces noisy features, not a clean language prior
- Results in `my_analysis/autoresearch_mme/results.tsv`

### VLM Bias key findings (autoresearch_vlmbias, 2026-05)
- Best config: uniform boost (clip_fallback_thresh=1.0) + deep layers 20-28, alpha=8.0, eps=0.5
- Per-category gains: Logos 1→4, GameBoards 1→2, OI 8→9 (noisy)
- Hard ceiling: Animals=0, Chess=0 across ALL configs — counting/enumeration failure (GT=31 vs PRED=16); MLP not attention
- CLIP guidance irrelevant: top-10%, top-80%, uniform all give identical accuracy
- Config committed in `my_analysis/autoresearch_vlmbias/srf.py`

---

## Qwen-VL-Chat Status (ClearSight comparison baseline)

ClearSight paper (arXiv 2503.13107) uses Qwen-VL-Chat and LLaVA-1.5-7B — NOT Qwen2.5-VL.
We need these for direct comparison.

| Method | POPE adv | MME score |
|--------|----------|-----------|
| ClearSight baseline (Qwen-VL-Chat) | 88.2% | 606 |
| Our baseline | TBD | TBD |
| SRF (Qwen-VL-Chat, tuned) | TBD | TBD |

**Architecture ported** (2026-04-25):
- `qwen_attn_patch.py`: `_get_lm_module`, `_get_decoder_layers` (→ `model.transformer.h`), `_get_attn_module` (→ `layer.attn`)
- `srf/config.py`: `Qwen/Qwen-VL-Chat` entry with `n_img_tokens=256`, `layer_start=9`, `layer_end=17`
- `srf/srf.py`: Qwen-VL-Chat temp-file input path in `_build_calib_inputs` and `prepare_sample`
- `srf/eval.py`: `is_qwen_vl_chat()`, `build_model_inputs()`, `get_tokenizer()` helpers; `load_model()` dispatch
- Autoresearch scripts in `my_analysis/autoresearch_qvlchat/`

**Next steps for Qwen-VL-Chat**:
1. Download complete (in progress) → run baseline_test.py (n=20)
2. Confirm baseline ~88% → run autoresearch sweep (Phase 1: layer sweep)
3. Update config.py with tuned params, run full POPE + MME

---

## Open Tasks (priority order)

1. **Run full POPE eval** (Qwen2.5-VL-3B) — all 9 combos (3 datasets × 3 splits) with baseline + SRF
   - COCO: use lmms-lab/POPE HF (already cached) or download annotation JSONs
   - A-OKVQA + GQA: need image download — see `NeurIPS plan/pope.md` for complete instructions
   - Target: Acc + F1 for all 9 cells matching Table 2 format in ClearSight (2503.13107)
2. **Run MME** (Qwen2.5-VL-3B) — get first MME numbers
3. **Qwen-VL-Chat baseline + autoresearch** — model downloading, run sweep after
4. **Port to Qwen-7B** — arch params in config are proportional starting points, need tuning sweep
5. **Write paper sections** — method, experiments, related work

---

## Recent Runs

<!-- Add entries here after each experiment. Format:
### YYYY-MM-DD — description
- Command: ...
- Results: ...
- Notes: ...
-->

### 2026-05-03 — VLMbias diagnostic sweep (15 experiments)
- Command: `conda run -n mllm python my_analysis/autoresearch_vlmbias/sweep.py`
- Results: 22.86% (uniform+deep L20-28) vs 17.14% baseline (+5.7pp)
- Notes: CLIP guidance irrelevant (all top-k strategies identical). Animals/Chess intractable (counting failure). Deep layers (20-28) > early layers (8-14) for counting tasks.

### 2026-04-25 — Smoke test MME (10 samples)
- Command: inline test script
- Results: base=9/10, SRF=10/10 (SRF fixed one wrong answer)
- Notes: End-to-end works. Need full run.

---

## Known Issues

- SRF-E + VLM Bias: contrastive pass suppresses `{` token → format broken. Use SRF base.
- SRF-E + Qwen-VL-Chat: `_make_noval_inp` skips zeroing (no `pixel_values`) → no contrastive effect. Use SRF base for Qwen-VL-Chat.
- Qwen-7B and LLaVA arch params not tuned — proportional starting points only.
- `HF_HOME` path is machine-specific — set via env var, not hardcoded.

## Quick Commands

```bash
cd /volumes2/mllm/lmms-eval

# Qwen-VL-Chat baseline smoke test (n=20)
conda run -n mllm python my_analysis/autoresearch_qvlchat/baseline_test.py

# Qwen-VL-Chat autoresearch sweep (Phase 1: layer sweep, POPE adv n=100)
conda run -n mllm python my_analysis/autoresearch_qvlchat/sweep.py --phase 1 --n 100

# Full sweep + MME validation
conda run -n mllm python my_analysis/autoresearch_qvlchat/sweep.py --phase all --validate

# Qwen2.5-VL-3B: full POPE eval (COCO only, 3 splits, HF dataset)
conda run -n mllm python srf/eval.py --method srf --datasets pope --output results/srf_3b/
# → For A-OKVQA + GQA splits: download images first — see vault NeurIPS plan/pope.md
#   Harness: my_analysis/autoresearch_srf_v2/pope_eval_all.py (COCO×3 only currently)

# VLM Bias (with current best config in autoresearch_vlmbias/srf.py)
conda run -n mllm python my_analysis/autoresearch_vlmbias/sweep.py

# Qwen-VL-Chat: full POPE + MME eval (after tuning)
conda run -n mllm python srf/eval.py --method srf --model Qwen/Qwen-VL-Chat --datasets pope mme --output results/srf_qvlchat/
```
