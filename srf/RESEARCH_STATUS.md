# SRF Research Status

> Update this file after every experiment run and commit it.
> This is the live source of truth for results and open tasks — replaces vault reads on remote servers.

---

## Current Best Results (Qwen2.5-VL-3B-Instruct)

**Best SRF config:** `clip_full_gate_v3`, ls=6, le=12, alpha=4.0, phase=generation, sys_beta=0.30

| Dataset | Baseline | SRF | Δ | Notes |
|---|---|---|---|---|
| POPE (9000) | 86.6% | **87.5%** | +0.9pp | adv=86.4%, pop=87.6%, ran=88.6% |
| MMVP | 40.0% | **49.3%** | +9.3pp | SRF-E β=2.0 |
| VLM Bias | 17.1% | **21.9%** | +4.8pp | SRF base (SRF-E broken*) |
| MME | 2362.9 | 2361.8 | −1.1 | Neutral — bad nouns skipped |
| HallusionBench | aAcc=0.694 | aAcc=0.686 | −0.8pp | VD questions hurt by local SRF |

*VLM Bias SRF-E broken: contrastive pass suppresses `{` token.

**60-sample val set ceiling: 0.900** (TP=24/30, FP=0/30). 6 remaining FNs are model capacity limits.

### MME key findings (autoresearch_mme, 2026-04)
- SRF cannot improve MME — best is 176/200 (-0.5% vs baseline)
- Root cause: 77% of MME categories require global context (artwork, celebrity, scene); any attention redistribution hurts
- SRF-E also fails: zero-pixel ViT produces noisy features, not a clean language prior
- Results in `my_analysis/autoresearch_mme/results.tsv`

### VLM Bias key findings (autoresearch_vlmbias, 2026-05)
- so we can ident- Best config: uniform boost (clip_fallback_thresh=1.0) + deep layers 20-28, alpha=8.0, eps=0.5
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

1. **Absent suppression** — sweep `--neg_absent_alpha` on val set; check if FP=1 is fixable
   ```bash
   conda run -n mllm python srf/eval_pope_val.py --srf --neg_absent_alpha 2.0 --out results/pope_val_srf_neg2.json
   ```
2. **B2 (per-question head weighting)** — surgical head activation weighting; likely same ceiling
3. **Deploy patch_combo gate** (Exp A, T=0.29) in `clip_salience.py` once absent-suppression is done
4. **Qwen-VL-Chat baseline + autoresearch** — model downloading, run sweep after
5. **Port to Qwen-7B** — arch params in config are proportional starting points, need tuning sweep
6. **Write paper sections** — method, experiments, related work

**Done:**
- ~~Full POPE eval (9000 samples)~~ ✅ 87.5% (+0.9pp)
- ~~MME + HallusionBench~~ ✅ Neutral / slight hurt diagnosed
- ~~Head/layer calibration~~ ✅ ls=6, le=12 optimal
- ~~Stage 3 boost/suppress~~ ✅ All methods saturate at 0.900 val
- ~~B1/B3/B4 boosting experiments~~ ✅ B1/B4 hit 0.900; B3 no gain

---

## Recent Runs

<!-- Add entries here after each experiment. Format:
### YYYY-MM-DD — description
- Command: ...
- Results: ...
- Notes: ...
-->

### 2026-06-12 — Gate sweep + Boosting experiments (B1–B4, val set)

- **Experiment A (offline gate sweep):** Combined gate `full>=0.21 OR (patch>=T AND contrast>=1.40)`.
  T=0.29 recovers 1 FN, 0 new FPs. T=0.265 recovers 2 FNs, 1 new FP. Not yet deployed.
- **B3 (budget_shift):** Zero gain — image/text budget imbalance not the bottleneck.
- **B1 (visual reliance compensation):** acc 0.867→0.900, robust across all vr_target/k values.
- **B4 (two-pass retry):** acc 0.867→0.900, recovers same 3 samples as B1.
- **B1+B4 combined:** No additive benefit — identical failure modes targeted.
- Val set ceiling confirmed at 0.900. 6 remaining FNs = model capacity limit.
- New flags in `eval_pope_val.py`: `--bias_mode`, `--vr_target`, `--vr_k`, `--b4`.
- Next: absent suppression experiments (`--neg_absent_alpha`).

### 2026-06-09 — MME + HallusionBench (phase=gen + bad-noun gate fix)

- MME: 2362.9→2361.8 (neutral, −1.1pts). Broken phase=both fix eliminated −19.7pt regression.
- HallusionBench: aAcc 0.694→0.686 (−0.8pp). VD questions fundamentally hurt by local SRF.
- Visualizations: `results/saliency_vis_datasets/mme/` and `hallusionbench/`

### 2026-06-08 — Full POPE (9000 samples) + POPE failure analysis

- SRF: 87.5% (+0.9pp). FN=981 (88% of failures), FP=140 (12%).
- FN root causes: Case A (CLIP gate fails), Case B (model capacity limit).
- Failure visualizations: `results/saliency_vis_pope/failures/{split}/`

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
