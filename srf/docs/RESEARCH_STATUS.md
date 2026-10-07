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
| VLIND pair_acc | 47.02% | **52.32%** (SRF-E) / ~46% (SRF) | +5.3pp / −1pp | SRF base hurts; SRF-E helps via contrastive pass |

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

## VLIND-Bench Notes (2026-07-27)

- `noun_extract.py`: `extract_vlind_nouns(statement)` → `(subject_noun, context_noun)`. subject→CLIP saliency map, context→presence gate.
- `srf.py` `prepare_sample`: accepts `noun_override=subj, gate_noun_override=ctx`
- `clip_salience.py` `compute_clip_salience_full_gate_v3`: accepts `gate_noun` param
- `eval.py` `run_vlindbench`: already calls dual-noun extraction and passes both nouns
- VLIND calibration: uses VLIND images (same domain), processor auto-caps at `DEFAULT_MAX_PIXELS`. Do NOT override this.
- Sweep: must use `eval.py`'s `run_vlindbench` — do NOT reimplement eval loop in new scripts
- VLIND eval command: `source activate mllm && python srf/eval.py --method srfe --datasets vlind --gamma 3.0`

### Why SRF base is neutral on VLIND (investigated 2026-07-27)

SRF base is flat on VLIND (47.35% vs 47.02% baseline). Root cause: **VLIND requires global visual amplification, not spatial attention boosting.**

- `extract_vlind_nouns` breaks on abstract/historical statements (extracts 'revolutionary', 'ancient', 'could' etc.) — not fixable with regex
- `existent_noun` / `non-existent_noun` fields in the dataset are noise (non_existent is random: 'lighthouse', 'bookshelf', 'mountain') — not the semantic counterpart we hoped
- VLIND's hard concepts (weight, time, size, climate) involve *relationships* between objects or *anachronistic technology* — spatial attention to a single noun region doesn't capture the counterfactual
- Tried `--vlind_data_nouns` flag using dataset fields: pair_acc dropped to 45.70% (−1.32pp vs baseline) — confirmed noise

**SRF-E at 52.32% is the correct result.** The contrastive pass (image vs blank) amplifies visual evidence globally without needing correct saliency, which is exactly what VLIND needs.

---

## LLaVA-1.5-7B Results (2026-08-06)

Model: `llava-hf/llava-1.5-7b-hf` | Eval: `decode_first_token` (logit-based yes/no)
Config: `--alpha 0.5 --layer_end 20 --neg_absent_alpha 1.0 --llava_boost_mode additive --clip_fallback_thresh 0.25 --clip_patch_thresh 0.27`
Results: `results/mme_llava_full/`

### Cross-dataset summary (SRF vs SRF-Fovea)

| Dataset | SRF | SRF-Fovea σ=20 | Winner | Why |
|---------|-----|----------------|--------|-----|
| LLaVA MMHal (score↑) | 2.29 | **2.40** | SRF-Fovea ✓ | Object-focused questions benefit from blur |
| LLaVA MME (total↑) | **670.00** | 663.33 | SRF ✓ | Count/position need global context |
| Qwen MMVP (pair acc↑) | 41.33% | **+3.33pp** | SRF-Fovea ✓ | Fine-grained visual perception benefits from focus |

SRF-Fovea beats SRF on open-ended hallucination/perception tasks. Only loses on MME count/position subtasks where blurring the background destroys the spatial evidence the model needs.

### MME (all 4 perception subtasks, n=240)

| Method | exist | count | pos | color | **TOTAL** | Δ |
|--------|------:|------:|----:|------:|----------:|---:|
| Baseline | 193.33 | 160.00 | 150.00 | 166.67 | **670.00** | — |
| SRF | 196.67 | 156.67 | 146.67 | 170.00 | **670.00** | **±0** |
| SRF-Fovea (σ=20) | 193.33 | 156.67 | 140.00 | 173.33 | **663.33** | −6.67 |
| SRF-C pixel-zero (γ=0.3) | 190.00 | 146.67 | 136.67 | 160.00 | **633.33** | −36.67 |
| SRF-C pixel-zero γ=1.0 | 190.00 | 133.33 | 136.67 | 140.00 | 600.00 | −70.00 |
| SRF-C pixel-zero γ=3.0 | 183.33 | 126.67 | 133.33 | 126.67 | 570.00 | −100.00 |
| SRF-C v2 salient-zero (γ=0.3) | 190.00 | 153.33 | 143.33 | 166.67 | **653.33** | −16.67 |
| SRF-C v2 salient-zero γ=1.0 | 190.00 | 136.67 | 140.00 | 150.00 | 616.67 | −53.33 |
| SRF-C v2 salient-zero γ=3.0 | 183.33 | 133.33 | 140.00 | 136.67 | 593.33 | −76.67 |

*ILVAD Table 2 ref: Baseline=641.66, ILVAD=686.67*

⚠️ **Eval protocol gap**: our baseline (670.00) > ILVAD baseline (641.66). We use `decode_first_token` (logit-based, always produces yes/no); ILVAD uses `model.generate` + text parsing (some outputs fail to parse → counted wrong). Direct comparison with ILVAD Table 2 requires matching their protocol.

**MME findings:**
- SRF: neutral (±0) — existence/color gains cancel count/position losses. Global-context tasks can't benefit from local attention boosting.
- SRF-Fovea: −6.67 — foveal blur destroys positional context for count/position subtasks; only color benefits (+6.67).
- SRF-C pixel-zero: catastrophically bad (−36.67 to −100). Frozen CLIP produces garbage embeddings from zeroed pixels.
- SRF-C v2 salient-zero: less bad (−16.67 at γ=0.3) but still hurts. Root cause same: upsampled pixel zeroing still corrupts CLIP.

### MMHal-Bench (n=96, GPT-scored)

Results: `results/mmhalbench_full/` and `results/mmhalbench_srfe_sweep/`

| Method | Score | Hal% | Δ Score |
|--------|------:|-----:|--------:|
| Baseline | 2.21 | 60.4% | — |
| SRF (α=0.5, naa=1.0) | 2.29 | 59.4% | +0.08 |
| SRF-Fovea σ=5 | 2.39 | 57.3% | +0.18 |
| SRF-Fovea σ=10 | 2.39 | 57.3% | +0.18 |
| SRF-Fovea σ=15 | 2.36 | 57.3% | +0.15 |
| **SRF-Fovea σ=20** | **2.40** | **56.2%** | **+0.19** |
| SRF-Fovea σ=30 | 2.35 | 56.2% | +0.14 |
| SRF-C pixel-zero γ=0.1 | 1.35 | 85.0% | −0.86 |
| SRF-C pixel-zero γ=0.2 | 1.55 | 80.0% | −0.66 |
| SRF-C pixel-zero γ=0.3 | 1.15 | 90.0% | −1.06 |
| SRF-C v2 salient-zero γ=0.3 | 1.26 | 86.5% | −0.95 |
| SRF-C v2 salient-zero γ=1.0 | 1.00 | 90.6% | −1.21 |
| SRF-C v2 salient-zero γ=3.0 | 0.76 | 89.6% | −1.45 |

**MMHal findings:**
- SRF base: small but consistent gain (+0.08 score, −1pp hal%).
- SRF-Fovea σ=20: best overall → Score=2.40, Hal%=56.2% (+0.19 vs baseline). Results at `results/mmhalbench_fovea_sweep/`.
- SRF-C (pixel-zero and salient-zero): both catastrophic. Open-ended generation compounds contrastive noise across tokens.
- **Next**: SRF-C v3 (embedding-space token zeroing) — bypasses CLIP OOD problem entirely. Pending.

---

## Open Tasks (priority order)

1. **Verify SRF beats baseline on RePOPE** — 100-sample diagnostic running (results/diag_100sample/)
2. **Full RePOPE eval** — all 3 splits once diagnostic shows SRF > baseline
3. **MMVP with phase=both** — already configured; may improve since phase fix activates prefill boost
4. **Write paper sections** — method, experiments, related work

**Done:**
- ~~Full POPE eval (9000 samples)~~ ✅ 87.70% (+1.33pp)
- ~~MME + HallusionBench~~ ✅ Neutral / slight hurt diagnosed
- ~~Head/layer calibration~~ ✅ ls=8, le=12 optimal (POPE), le=16 (MMVP)
- ~~Phase bug fix~~ ✅ `phase="generation"` → `phase="both"` in config.py for POPE
- ~~CLIP template ensemble~~ ✅ 5 templates, mean-pooled normalized embeddings
- ~~OR gate~~ ✅ `full_img_sim >= 0.21 OR patch_max_sim >= 0.27` in clip_salience.py
- ~~OOM fix in diag script~~ ✅ `torch.cuda.empty_cache()` after each sample
- ~~VLIND SRF base investigation~~ ✅ SRF base neutral (+0.33pp); SRF-E 52.32% (+5.3pp) is the result

---

## Recent Runs

<!-- Add entries here after each experiment. Format:
### YYYY-MM-DD — description
- Command: ...
- Results: ...
- Notes: ...
-->

### 2026-08-08 — SRF-C v2 (salient-zero) MME + MMHal (LLaVA-1.5-7B)

- Scripts: `srf_exp_runs/mme_llava_srfc2.sh` (job 25357165), `srf_exp_runs/mmhal_llava_srfc2.sh` (job 25357289)
- Results: `results/mme_llava_full/srfc2/`, `results/mmhalbench_srfc2_sweep/g{0.3,1.0,3.0}/`
- MME best: γ=0.3 → 653.33 (−16.67 vs baseline). Better than pixel-zero (633.33) but still hurts.
- MMHal best: γ=0.3 → Score=1.26, Hal%=86.5% (−0.95 vs baseline). Catastrophic.
- Root cause confirmed: zeroing pixels (even partially) corrupts frozen CLIP-L → garbage LLM tokens → noisy contrastive signal.
- **Fix**: SRF-C v3 — zero the 576 projected visual tokens directly in LLM embedding space, bypassing CLIP.
- Note: eval.py bug fixed (gamma gating was srfe-only, added srfc2). Bug caused earlier jobs to run with γ=0.0.

### 2026-08-06 — MMHal SRF-Fovea σ sweep (LLaVA-1.5-7B, n=96, GPT-4o scored)

- Script: `srf_exp_runs/mmhal_srffovea_sigma_sweep.sh` (job 25285666)
- Results: `results/mmhalbench_fovea_sweep/s{5,10,15,20,30}/`
- Best: **σ=20 → Score=2.40, Hal%=56.2%** (+0.19 vs baseline, +0.11 vs SRF base)
- Shape: scores plateau at σ=5–10 (2.39), dip at σ=15 (2.36), peak at σ=20 (2.40), fall at σ=30 (2.35). Hal% flat 57.3% for σ≤15, drops to 56.2% at σ≥20.
- σ=20 confirmed as optimal for LLaVA MMHal (matches Qwen MMVP best σ).

### 2026-08-06 — LLaVA MME full eval (all 4 subtasks, 3 methods)

- Scripts: `srf_exp_runs/mme_llava_baseline_srf.sh`, `mme_llava_srffovea.sh`, `mme_llava_srfc.sh`
- Results: `results/mme_llava_full/{baseline,srf,srffovea,srfc}/mme.json`
- SRF: 670.00 (=baseline, ±0). SRF-Fovea: 663.33 (−6.67). SRF-C γ=0.3: 633.33 (−36.67).
- Root cause: count/position subtasks need global context — any attention redistribution or foveal blur hurts. SRF-C contrastive pass produces noise on LLaVA at all γ.
- Eval protocol differs from ILVAD: our baseline=670 vs ILVAD=641.66 (decode_first_token vs model.generate+parse).

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
