# SRF Research Status

---

## Paper Story

**Two methods:**
- **SRF** = attention boosting (in-decoder) + foveal blur (pre-encoder). Works on ALL datasets. Zero extra LLM passes, one CLIP pass per sample.
- **SRF-E** = SRF + contrastive decoding (one extra LLM pass). Currently only valid for single-token answers (MMVP, POPE). Multi-token generation (VLMBias) collapses because zeroed `pixel_values` corrupts ViT.

**Next experiment:** Replace SRF-E Pass 2 zeros with Gaussian-blurred image → may fix VLMBias collapse → one unified method for all datasets.

**Routing failure hypothesis:** Three-stage fix — pre-encoder (fovea) → in-decoder (attn boost) → post-decoding (contrastive).

---

## Best Config (Qwen2.5-VL-3B-Instruct)

`clip_full_gate_v3`, ls=8, alpha=2.0, phase=both, sys_beta=0.30, gamma=3.0 (SRF-E only)

| Dataset | le | alpha | eps | gamma |
|---|---|---|---|---|
| MMVP | 16 | 2.0 | 0.2 | 3.0 |
| POPE | 12 | 2.0 | 0.2 | 3.0 |
| VLMBias | 14 | 8.0 | 0.5 | 0 (SRF base) |
| VLind | 12 | 2.0 | 0.2 | 3.0 |

---

## Results Table (Qwen2.5-VL-3B-Instruct, 2026-08-03)

### Full Results (2026-08-04, all methods complete)

| Method | MMVP pair | MMVP img | VLMBias | VLind q_acc | VLind pair | Log |
|---|---|---|---|---|---|---|
| Baseline | 40.0% | 67.7% | 19.0% | 59.6% | 47.0% | `/tmp/baseline_all.log` |
| VAF | 40.0% | 69.0% | 17.9% | 58.9% | 45.7% | `/tmp/vaf_all.log` |
| VCD | 37.3% | 65.7% | 9.6% | 62.3% | 47.4% | `/tmp/vcd_all.log` |
| ILVAD | 38.7% | 68.0% | 17.7% | 59.3% | 46.0% | `/tmp/ilvad_all.log` |
| VHR | 38.7% | 68.3% | 17.9% | 61.6% | 44.0% | `/tmp/vhr_all.log` |
| SRF | 41.3% | 68.7% | **19.7%** | 58.6% | 45.4% | `/tmp/srf_all.log` + `/tmp/srf_vlmbias_vlind.log` |
| SRF-Fovea (σ=20) | 43.3% | 69.7% | 19.6% | 56.5% | 42.4% | `/tmp/srffovea_eval.log` |
| **SRF-E (γ=3)** | **45.3%** | **70.7%** | 0.6% (collapsed) | **73.7%** | **52.7%** | `/tmp/srfe_all.log` |
| SRF-E (VLMBias γ=0) | — | — | **19.7%** | — | — | same as SRF |

*Animals + Chess = 0% across ALL methods on VLMBias — counting/enumeration failure, not attention.*

### VLMBias per-category

| Method | Animals | Chess | Flags | GameBoards | Logos | Optical | PatGrid |
|---|---|---|---|---|---|---|---|
| Baseline | 0% | 0% | 24.6% | 4.2% | 5.6% | 49.7% | 14.0% |
| VAF | 0% | 0% | 12.5% | 11.9% | 6.8% | 49.7% | 8.0% |
| VCD | 1.1% | 0% | 18.3% | 8.3% | 0.5% | 24.1% | 3.0% |
| ILVAD | 0% | 0% | 18.3% | 6.5% | 4.1% | 48.5% | 11.3% |
| VHR | 0.9% | 0% | 23.8% | 11.9% | 2.2% | 51.1% | 0.3% |
| SRF | 0% | 0% | 22.5% | 5.4% | 8.2% | 50.4% | 15.2% |
| SRF-Fovea | 0% | 0% | 21.7% | 2.4% | 6.5% | **52.7%** | 13.4% |

---

## Key Findings (2026-08-04)

### VLind — SRF base hurts, SRF-E helps a lot
- **Attention-routing methods all hurt VLind**: SRF −1.6pp, SRF-Fovea −4.6pp, VAF −1.3pp, VHR −3.0pp
- **Only contrastive methods help**: SRF-E +5.7pp, VCD +0.4pp
- **Root cause**: VLind tests counterfactual/relational understanding (climate, anachronistic tech, size relationships). These require holistic image comprehension, not routing to a single salient noun. Spatial attention boost to one region actively hurts.
- **Story implication**: SRF-E's contrastive pass (image vs. no-image) amplifies ALL visual evidence globally — exactly what VLind needs. SRF's spatial routing is the wrong intervention for relational tasks.

### SRF-E vs SRF gap — two different failure modes
- **MMVP** (+4pp): SRF partially helps (spatial routing correct object); SRF-E adds contrastive amplification on top
- **VLind** (+7.3pp pair): SRF routing hurts; SRF-E's global amplification fixes a different failure mode
- **Gap is not just "more boost"** — they address orthogonal problems. SRF = WHERE to look. SRF-E = HOW MUCH the image matters vs. language prior.

### SRF-E MMVP discrepancy (45.33% now vs 49.33% previously)
- Previous 49.33% came from autoresearch sweep (`autoresearch_mmvp_v2`) with possibly different alpha/eps per dataset
- Current eval.py run uses config defaults: ls=8, le=16, alpha=2.0, phase=both
- ⚠️ Need to verify: check if sweep used a different alpha or layer config. Run `eval.py --method srfe --datasets mmvp --layer_end 16 --alpha 2.0 --gamma 3.0` and compare.

### SRF-Fovea — good on MMVP, backfires on VLind
- MMVP: +3.3pp pair (pre-encoder blur helps fine-grained visual discrimination)
- VLind: −4.6pp pair (fovea over-focuses on a single region, hurts relational comprehension)
- VLMBias: neutral (≈baseline)
- **Use case**: SRF-Fovea is complementary to SRF for object-centric tasks; avoid for relational tasks

## Open Tasks

0. **Head calibration — MMVP positive, VLMBias NULL.** S3 gives +2.67pp on MMVP
   but +0.07pp on VLMBias (n=2784). Does NOT currently support a method change.
   Next: seed-robustness on MMVP to decide whether the MMVP gain is real at all.
0a. **PAPER.tex Table 1 VLMBias average is wrong** (20.9 should be 19.65). Confirmed
   by the control pass above. Fix independently of everything else.
0c. **Efficiency table (time/FLOPs)** — no timing infrastructure exists in `srf/`.
   Needed: ms/sample, peak memory, extra LLM passes, one-time calibration cost, and
   the per-sample CLIP overhead reported separately (the current `+pass` column
   undersells SRF, which pays a CLIP pass that VAF does not). Note v3 runs 83 patch
   crops vs basic `clip`'s 49.
0d. **`config.py` n_layers bug** — 3B is 36 (not 28), 7B is 28 (not 32). See
   CONTEXT.md "Code Change Log". Shipped inference and VHR unaffected; the ablation's
   "all layers" rows were [0,27] and are mislabelled.
0b. **Leave-one-out ablation from full SRF** — needed to disentangle the component
   interaction flagged in the ablation caveat above. NOT run.
1. **Investigate SRF-E MMVP 45.33% vs previous 49.33%** — verify config, possibly re-run sweep
2. **SRF-E with blurred Pass 2** — fix VLMBias collapse; if resolved → one unified method
3. **Write paper** — routing failure framing: SRF fixes spatial routing (MMVP/VLMBias), SRF-E fixes global evidence (VLind). Two complementary failure modes.
4. **Second model: Qwen2.5-VL-7B** — same codebase, minimal porting.

---

## Component Ablation — MMVP / Qwen2.5-VL-3B (2026-09-16)

Script: `srf/ablation_components.py` (calls `eval.run_mmvp`; 14 passes, one model load)
Log: `/tmp/ablation_components_mmvp.log`
JSON: `results/ablation/ablation_components_mmvp_qwen3b.json`

Cumulative build-up, alpha held fixed within each anchor so the table isolates
*targeting* rather than boost strength.

**Anchor `published`** — v3 gate, alpha=2.0, layers [8,16], eps=0.2, sys_beta=0.30, sigma=20
(the configuration behind the paper's 43.3%)

| Row | Pair | dprev | dbase | Img |
|---|---|---|---|---|
| A baseline | 40.00 | | | 67.67 |
| B uniform attention boost | 42.00 | +2.00 | +2.00 | 69.67 |
| C + semantic relevance mask | 39.33 | **-2.67** | -0.67 | 68.33 |
| D + head calibration | 40.00 | +0.67 | 0.00 | 68.00 |
| E + fusion-layer targeting | 41.33 | +1.33 | +1.33 | 69.00 |
| F + suppression (SRF decoder-only) | 41.33 | 0.00 | +1.33 | 68.67 |
| G + semantic foveation (full SRF) | **43.33** | +2.00 | +3.33 | 69.67 |

**Anchor `mmvp_tuned`** — basic `clip` gate, alpha=4.0, layers [8,15], same eps/sys_beta/sigma

| Row | Pair | dprev | dbase | Img |
|---|---|---|---|---|
| A baseline | 40.00 | | | 67.67 |
| B uniform attention boost | 42.00 | +2.00 | +2.00 | 68.33 |
| C + semantic relevance mask | **43.33** | +1.33 | +3.33 | **70.67** |
| D + head calibration | 39.33 | **-4.00** | -0.67 | 67.67 |
| E + fusion-layer targeting | 40.67 | +1.33 | +0.67 | 68.33 |
| F + suppression | 40.67 | 0.00 | +0.67 | 68.33 |
| G + semantic foveation (full SRF) | 40.67 | 0.00 | +0.67 | 68.33 |

### Validity checks (all passed)
- Baseline identical under both anchors (40.00) -> no state leaking between variants
- `reusing visual heads` on every non-warm-up row -> head selection constant within an anchor
- Three exact reproductions of known numbers: baseline 40.00/67.67, SRF decoder-only
  41.33 (= `EXPERIMENTS.md` srf 0.4133), full SRF 43.33/69.67 (= `srffovea_20.0`)
- `mmvp_tuned` rows E/F/G share the same aggregate but have DIFFERENT per-60-image
  trajectories -> genuine coincidence, not a silent no-op

### Findings
1. **Uniform boosting captures +2.00pp of the +3.33pp headline** under both anchors —
   no semantic mask, no head selection, no layer targeting.
2. **Suppression (eps, sys_beta) contributes exactly 0.00pp** under both anchors on MMVP.
3. **Head calibration is the weakest component**: +0.67pp (published), -4.00pp (mmvp_tuned).
   Prime suspect: selection is layer-agnostic (see CONTEXT.md "Head calibration").
4. **The semantic mask flips sign with the gate**: -2.67pp under v3, +1.33pp under basic
   `clip`. Same component, opposite sign — the gate, not the mask, is responsible.
5. **Best single config in the run is `mmvp_tuned` row C: 43.33 / 70.67** — semantic mask
   with ALL heads and ALL layers. Ties published pair acc and beats its img acc, with a
   strictly simpler method. NOTE: off the tuned manifold in two ways at once, and it
   conflicts with `MMVP.md`'s head sweep (which found htk=0.20 best at tuned layers);
   a leave-one-out is needed before drawing conclusions.

### Caveat on the cumulative design
Row D measures "head calibration GIVEN all 28 layers", not head calibration in
isolation, and rows C->E show the selectivity axes interact. A leave-one-out from
full SRF would measure each component at the real operating point and could give
head calibration a different number. Leave-one-out is NOT yet run.

### Paper inconsistencies found (NOT yet fixed in PAPER.tex)
- Table 1 mixes variants across columns: MMVP 43.3/69.7 is `srffovea` (sigma=20), but the
  VLMBias per-category figures (Optical 50.4, Logos 8.2, PatGrid 15.2) are SRF base.
  `srffovea` gives 52.7/6.5/13.4 there.
- Table 1 baseline VLMBias Avg is 18.1; this file's results table says 19.0. SRF's gain is
  +2.8pp under one and +0.7pp under the other.
- `MMVP.md` best-params table lists `sys_beta=0.10`, contradicted by its own logs
  (`mmvp_le15_sysbeta0.1_generate.log` = 39.33% vs 42.00% at 0.30). The 0.10 figure
  appears to come from the old autoresearch harness, not `eval.py`. Use 0.30.

---

## Head Calibration — MMVP / Qwen2.5-VL-3B (2026-09-16)

Script: `srf/head_calibration.py` | Log: `/tmp/head_calibration_mmvp.log`
JSON: `results/ablation/head_calibration_mmvp_qwen3b.json`

Published anchor, full SRF + foveation, **only head selection varies**:

| Mode | Pair | dref | Img |
|---|---|---|---|
| `global` — shipped layer-agnostic mask (reference) | 43.33 | — | 69.67 |
| `per_layer` (S1) | 44.00 | +0.67 | 70.33 |
| `saliency` (S3) | **46.00** | **+2.67** | **71.33** |

`global` reproduced 43.33/69.67 exactly → deltas attributable to head selection alone.

**MMVP standing after this run:**

| | Pair | Extra LLM passes |
|---|---|---|
| Baseline | 40.00 | — |
| Published SRF | 43.33 | 0 |
| SRF-E (contrastive) | 45.33 | +1 |
| **SRF + S3 head calibration** | **46.00** | **0** |

Root cause: head selection was layer-agnostic while the paper specifies per-layer.
See CONTEXT.md "Head calibration" and METHOD_WALKTHROUGH.md Step 0c.

**S3 exceeds the paper's stated method.** The paper's rho is query-agnostic mean
attention; S3 ranks heads by correlation with the per-sample CLIP saliency map, i.e.
query-conditioned head selection. This needs a rewritten equation, not a correction.

**NOT validated — shipped default unchanged:** S3 calibrated on 14/20 samples (v3 gate
discarded 6); 46.00 vs 43.33 = 69 vs 65 pairs of 150; single seed; MMVP only.

---

## Head Calibration on VLMBias — NULL RESULT (2026-09-17)

Script: `srf/head_calibration.py --dataset vlmbias` | Log: `/tmp/head_calibration_vlmbias.log`
JSON: `results/ablation/head_calibration_vlmbias_qwen3b.json`

Config from config.py defaults for vlmbias: v3 gate, alpha=8.0, eps=0.5,
layers [8,14], phase=generation, sys_beta=0.30, fovea OFF (SRF base), 2784 samples.

| Mode | Acc | dref | Overlap [8,14] |
|---|---|---|---|
| `global` (shipped) | 19.65 | — | — |
| S2 contrastive (real - blank) | 19.36 | **-0.29** | 0.229 |
| S3 saliency-aligned | 19.72 | **+0.07** | 0.186 |

Per-category (%):

| Mode | Animals | Chess | Flags | GameBoards | Logos | Optical | PatGrid |
|---|---|---|---|---|---|---|---|
| global | 0.0 | 0.0 | 22.5 | 5.4 | 8.2 | 50.4 | 15.2 |
| contrast | 0.0 | 0.0 | 25.0 | 2.4 | 7.0 | 49.9 | 15.2 |
| saliency | 0.0 | 0.0 | 23.8 | 1.8 | 9.2 | 50.4 | 15.5 |

### S3's MMVP gain does NOT transfer
+2.67pp on MMVP became +0.07pp on VLMBias, i.e. noise at n=2784. S2 is slightly
negative. Head selection genuinely changed in both cases (overlap 0.19-0.23), so
this is not a failure to apply the intervention — different heads were boosted and
the answer barely moved.

Two facts make the null MORE credible than the MMVP positive:
- S3's VLMBias calibration was CLEANER (19/20 samples survived the gate, vs 14/20
  on MMVP). The dataset where S3 had better calibration is the one where it did
  nothing.
- n=2784 vs MMVP's 150 pairs. +2.67pp on MMVP is 4 pairs; +/-0.07pp on 2784 is far
  better resolved.

**Honest reading: the MMVP +2.67pp may itself be small-n noise.** A seed-robustness
check on MMVP is required before any head-calibration claim enters the paper. The
shipped default remains unchanged.

Animals and Chess are 0.0% under every mode, consistent with the existing finding
that these are counting failures untouched by attention routing.

### CONFIRMED: PAPER.tex Table 1 VLMBias average is WRONG
The `global` control closed at **19.65%**, matching this file's 19.7% for SRF base.
PAPER.tex Table 1 reports **20.9%** for that cell. Against the 19.0% baseline the
real gain is **+0.65pp, not +2.8pp**. The per-category figures in Table 1 (Optical
50.4, Logos 8.2, PatGrid 15.2) reproduce exactly, so only the average is wrong.

---

## Quick Commands

> ⚠️ Use `source activate mllm && python ...` — NOT `conda run -n mllm`

```bash
cd /volumes2/mllm/lmms-eval

# SRF-E (best, single-token tasks)
source activate mllm && stdbuf -oL -eL python -u srf/eval.py \
  --method srfe --datasets mmvp vlmbias vlind --gamma 3.0 2>&1 | tee /tmp/srfe_all.log

# SRF-Fovea sweep
source activate mllm && python srf/test_srffovea_mmvp.py --sigma 20 30 50 100 \
  2>&1 | tee /tmp/srffovea_all.log

# Any comparison baseline
source activate mllm && stdbuf -oL -eL python -u srf/eval.py \
  --method vhr --datasets mmvp vlmbias vlind 2>&1 | tee /tmp/vhr_all.log

# VLIND only
source activate mllm && python srf/eval.py --method srfe --datasets vlind --gamma 3.0
```

---

## Known Issues

- SRF-E collapses on VLMBias/VLind at any γ>0: zeroed pixel_values corrupts ViT multi-token generation
- Animals/Chess in VLMBias = 0% across ALL methods (counting failure, not attention)
- `qwen_attn_patch.py` is core SRF — never modify; all baselines (vhr, ilvad) wrap around it
- `HF_HOME` path is machine-specific — set via env var
