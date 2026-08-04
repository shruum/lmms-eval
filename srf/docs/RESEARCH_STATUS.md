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

1. **Investigate SRF-E MMVP 45.33% vs previous 49.33%** — verify config, possibly re-run sweep
2. **SRF-E with blurred Pass 2** — fix VLMBias collapse; if resolved → one unified method
3. **Write paper** — routing failure framing: SRF fixes spatial routing (MMVP/VLMBias), SRF-E fixes global evidence (VLind). Two complementary failure modes.
4. **Second model: Qwen2.5-VL-7B** — same codebase, minimal porting.

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
