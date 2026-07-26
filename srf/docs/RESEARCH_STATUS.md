# SRF Research Status

> Update this file after every experiment run and commit it.
> This is the live source of truth for results and open tasks — replaces vault reads on remote servers.

---

## Current Best Results (Qwen2.5-VL-3B-Instruct)

**Best SRF-E config:** `clip_full_gate_v3`, ls=8, le=12 (POPE) / le=16 (MMVP), alpha=2.0, phase=both, sys_beta=0.30, gamma=3.0

> ⚠️ **2026-07-26 updates**: (1) `phase` changed from `"generation"` to `"both"` for POPE in `config.py` — generation phase was a no-op during prefill eval, SRF was never active. (2) CLIP template ensemble (5 templates, mean-pool) added. (3) OR gate: `full_img_sim >= 0.21 OR patch_max_sim >= 0.27`. Full re-evaluation on RePOPE in progress.

| Dataset | Baseline | SRF-E (γ=3.0) | Δ | Notes |
|---|---|---|---|---|
| POPE adv (9000) | 86.37% | **87.70%** | +1.33pp | ls=8, le=12, α=2.0, full 3000 adv |
| MMVP | 40.0% | **49.33%** | +9.33pp | ls=8, le=16, α=2.0 |
| VLMBias | 19.04% | 19.0% (γ=0) | ≈0 | SRF-E broken for multi-token gen |
| MME | 2362.9 | 2361.8 | −1.1 | Neutral |
| HallusionBench | aAcc=0.694 | aAcc=0.686 | −0.8pp | VD questions hurt |

*VLM Bias SRF-E broken: contrastive pass suppresses `{` token. Use γ=0 (SRF base only).*

**RePOPE baselines (Qwen2.5-VL-3B, corrected annotations):**

| Split | Baseline | VAF | VCD | SRF (in progress) |
|-------|----------|-----|-----|-------------------|
| Random | 89.29% | 91.38% | 90.7% | TBD |
| Popular | 86.69% | 88.93% | 88.3% | TBD |
| Adversarial | 80.40% | 84.54% | 85.2% | TBD |

**Target**: Beat VAF (88.28% avg) and VCD (88.1% avg) on RePOPE.

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

1. **Verify SRF beats baseline on RePOPE** — 100-sample diagnostic running (results/diag_100sample/)
2. **Full RePOPE eval** — all 3 splits once diagnostic shows SRF > baseline
3. **Tune alpha** — current alpha=2.0 may not flip confident wrong predictions; check if higher needed
4. **Absent suppression sweep** — neg_absent_alpha=2.0 currently; check FP reduction
5. **MMVP with phase=both** — already configured; may improve since phase fix activates prefill boost
6. **Write paper sections** — method, experiments, related work

**Done:**
- ~~Full POPE eval (9000 samples)~~ ✅ 87.70% (+1.33pp)
- ~~MME + HallusionBench~~ ✅ Neutral / slight hurt diagnosed
- ~~Head/layer calibration~~ ✅ ls=8, le=12 optimal (POPE), le=16 (MMVP)
- ~~Phase bug fix~~ ✅ `phase="generation"` → `phase="both"` in config.py for POPE
- ~~CLIP template ensemble~~ ✅ 5 templates, mean-pooled normalized embeddings
- ~~OR gate~~ ✅ `full_img_sim >= 0.21 OR patch_max_sim >= 0.27` in clip_salience.py
- ~~OOM fix in diag script~~ ✅ `torch.cuda.empty_cache()` after each sample

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

> ⚠️ Use `source activate mllm && python ...` — NOT `conda run -n mllm` (breaks `--n` flag)

```bash
cd /volumes2/mllm/lmms-eval

# ── Diagnostic / debugging ─────────────────────────────────────────────────

# 100-sample diagnostic on RePOPE adversarial (baseline vs SRF, CLIP gate analysis)
source activate mllm && python srf/diag_20sample.py \
  --repope_dir data/repope --splits adversarial --n 100 --out_dir results/diag_100sample

# CLIP gate accuracy sweep (eval_presence on 60-sample val set)
source activate mllm && python srf/saliency/eval_presence.py

# ── Full evaluation ────────────────────────────────────────────────────────

# Full POPE (all 9000 samples) with SRF-E best config
source activate mllm && python srf/eval.py --method srfe --datasets pope \
  --output results/srf_pope/ --gamma 3.0

# Full RePOPE adversarial (2684 samples)
source activate mllm && python srf/eval.py --method srfe --datasets pope \
  --pope_file data/repope/adversarial.json --output results/srf_repope_adv/ --gamma 3.0

# MMVP with current best config
source activate mllm && python srf/eval.py --method srfe --datasets mmvp \
  --output results/srf_mmvp/ --gamma 3.0

# VLM Bias (gamma=0 required — SRF base only)
source activate mllm && python srf/eval.py --method srf --datasets vlmbias \
  --output results/srf_vlmbias/

# ── Config state (2026-07-26) ──────────────────────────────────────────────
# config.py POPE entry:  phase="both", alpha=2.0, eps=0.2, neg_absent_alpha=2.0
# clip_salience.py gate: full_img_sim >= 0.21 OR patch_max_sim >= 0.27
# CLIP text encoding:    5-template ensemble (mean-pooled, re-normalized)
```
