# Branch diff: `autoresearch/mmvp-srf` vs `srf-llava`

Written 2026-09-20. Companion to `SRF_METHOD_CARD.md` (what the method is) and
`SRF_details.md` (full history).

Both branches fork from **`570dcb15`** (2026-07-31, "vlind: dual-noun extraction").

| Branch | Tip | Commits since fork | Theme |
|---|---|---|---|
| `autoresearch/mmvp-srf` | `32dc8b26` 2026-09-20 | 2 | Qwen2.5-VL-3B. VTAR head selection, ablations, paper tables |
| `srf-llava` | `ff0eb05d` 2026-09-20 | 2 | LLaVA-1.5-7B support, MMHal-Bench, cluster run scripts |

Neither has been merged into the other. No commit is shared beyond the fork point.

---

## 1. Files added on only one side

**Only on `autoresearch/mmvp-srf`**

| File | What |
|---|---|
| `srf/head_calibration.py` | **the important one.** VTAR scoring, 10 head-selection modes, per-layer mask hooks |
| `srf/ablation_components.py` | cumulative component ablation, reuses `eval.run_mmvp` |
| `srf/param_sensitivity.py` | one-at-a-time hyperparameter sweeps |
| `SRF_details.md` | full method + experiment record |
| `srf/ilvad.py`, `srf/vhr.py` | extra baselines |

**Only on `srf-llava`**

| File | What |
|---|---|
| `my_analysis/llava_attn_patch.py` | **the important one.** LLaVA attention patch |
| `srf/score_mmhalbench.py` | GPT-4 judge scoring for MMHal-Bench |
| `srf/srf_c_v2.py`, `srf/srf_c_v3.py` | contrastive SRF variants (`--method srfc2/srfc3`) |
| `srf/saliency/save_llava_heatmaps.py`, `save_mmhal_heatmaps.py` | visualisation |
| `snellius.md`, `skills/load-mllm-context.md` | cluster notes |
| `srf_exp_runs/*.sh` (15 scripts) | SLURM / sweep drivers for LLaVA, MMHal, POPE, RePOPE |

## 2. Files modified on both sides = the conflict set

| File | mmvp-srf | srf-llava | Severity |
|---|---|---|---|
| `srf/eval.py` | +163, adds `--head_mode` family and `_install_head_mode()` | +468, adds LLaVA dispatch, `run_mmhalbench`, `--fovea_sigma` | **high** |
| `srf/srf.py` | +126, `random` saliency, `clip_full_gate_v3_paper`, calib metadata | +262, LLaVA branches in `prepare_sample` / `_build_calib_inputs` | **high** |
| `srf/config.py` | +24, phase-revert comments, MME warning | +17, `mmhalbench` entries, `clip_patch_thresh`, LLaVA arch block | low, disjoint regions |
| `srf/srf_fovea.py` | added independently | added independently | **add/add**, see below |

Inside `eval.py` both sides touch `parse_args`, `main`, `run_pope`,
`run_vlindbench`.
Inside `srf.py` both touch `setup`, `prepare_sample`, `_make_saliency`,
`_build_calib_inputs`.

### `srf_fovea.py` is an add/add conflict, and the LLaVA copy wins

Same algorithm, same `SIGMA = 20.0`, same
`W*I + (1-W)*GaussianBlur(I, sigma)`. The LLaVA copy is a **functional
superset**: it adds `_model_type()` architecture dispatch and reads
`SIGMA` as an override target so `eval.py` can set it from `--fovea_sigma`.

**Resolution: take the `srf-llava` copy wholesale.** `param_sensitivity.py`
assigns `srf_fovea_mod.SIGMA` directly, which still works, so nothing on the
MMVP side breaks.

## 3. Flags that exist on only one side

```
only on mmvp-srf : --head_mode  --head_calib_dataset  --n_layers_sel
                   --kappa  --vtar_thresh  --save_records

only on srf-llava: --fovea_sigma  --clip_patch_thresh  --llava_boost_mode
                   --mme_data_dir  --mme_subtasks
                   --mmhalbench_json  --mmhalbench_n  --openai_model
```

Note `--save_records` is on the MMVP side but almost certainly originated in the
LLaVA line. Expect it to appear as a near-duplicate hunk during the merge.
**Resolve duplicates by keeping one copy, not by concatenating.**

## 4. Capability matrix

| | mmvp-srf | srf-llava |
|---|---|---|
| Qwen2.5-VL-3B / 7B | yes | yes |
| **LLaVA-1.5-7B** | **no** (`load_model` hardcodes `Qwen2_5_VLForConditionalGeneration`) | **yes**, dispatches on `AutoConfig.model_type` |
| Datasets | mmvp, pope, vlmbias, mme, vlind | same **+ mmhalbench** |
| Methods | srf, srfe, srffovea, vcd, vaf, baseline | same **+ srfc2, srfc3** |
| **VTAR head selection (`ratio_topk`)** | **yes** | **no**, legacy pooled `global` only |
| Soft head weights (`_STATE["head_weight"]`) | yes, in `qwen_attn_patch` | no |
| Random-map control | yes (`saliency_mode="random"`) | no |
| Ablation / sensitivity harnesses | yes | no |
| LLaVA arch block in `config.py` | no | yes, see below |

The `srf-llava` `config.py` already carries a tuned LLaVA entry:

```python
"llava-hf/llava-1.5-7b-hf": {
    "n_layers": 32, "spatial_merge_size": 1,
    "image_token": None,            # uses model.config.image_token_index
    "layer_start": 8, "layer_end": 20,
    "head_top_k_pct": 0.20,
    "clip_coarse_grid": 6,          # 336px input, not Qwen's 448px
    "saliency_mode": "clip_full_gate_v3",
}
```

---

## 5. Two ways forward

### Option A (recommended) — cherry-pick onto a branch off `srf-llava`

Avoids the `eval.py` and `srf.py` conflicts entirely. `head_calibration.py` is a
new file, so it just lands.

```bash
git checkout srf-llava
git checkout -b srf-llava-vtar
git checkout origin/autoresearch/mmvp-srf -- srf/head_calibration.py
```

Then hand-port two things into `srf-llava`'s `eval.py`:

1. the `--head_mode / --head_top_k_pct / --n_layers_sel / --kappa /
   --vtar_thresh` arguments in `parse_args`
2. the `_install_head_mode()` function and its call site in `main`

and make one edit inside `head_calibration.py`:

> line 132, `import qwen_attn_patch as patch`. On LLaVA this must be the
> LLaVA patch. `eval.py::load_model` already sets a module-level `patch`
> global per architecture, so take it from there instead of importing
> directly.

`_get_decoder_layers` in `qwen_attn_patch` is **already architecture-agnostic**
and handles `LlamaForCausalLM.model.layers`, so it can stay as the layer
accessor even under LLaVA.

`_STATE["head_weight"]` does **not** need porting. Only `vtar_soft` uses it, and
the shipped rule `ratio_topk` uses the boolean `head_mask`, which
`llava_attn_patch` already has.

### Option B — full merge

```bash
git checkout srf-llava
git merge origin/autoresearch/mmvp-srf
```

Expect conflicts in `eval.py`, `srf.py`, `config.py` and an add/add on
`srf_fovea.py`. Resolution policy:

| File | Take |
|---|---|
| `srf_fovea.py` | `srf-llava` version wholesale |
| `config.py` | both, the edits are in disjoint regions |
| `eval.py` | `srf-llava` for `load_model`, `run_mmhalbench`, encoding helpers. `mmvp-srf` for the `--head_mode` family and `_install_head_mode` |
| `srf.py` | `srf-llava` for anything under an arch branch. `mmvp-srf` for `saliency_mode="random"` and `clip_full_gate_v3_paper` |

Do Option A first. Merge later once LLaVA numbers exist.

---

## 6. Running the current method on LLaVA

Target configuration, from `SRF_METHOD_CARD.md`:
`tau=0.20`, `sigma=20`, `lambda_sem=2.0`, `lambda_bg=0.2`, `lambda_sys=0.30`,
VTAR top-k per layer over a mid-band.

### k and the band do not transfer

| | Qwen2.5-VL-3B | LLaVA-1.5-7B |
|---|---|---|
| layers x heads | 36 x 16 = 576 slots | 32 x 32 = 1024 slots |
| band used | [6, 31] = 26 of 36 layers (72%) | [6, 31] would be 26 of 32 (81%) |
| `k = 0.20` gives | 3 of 16 heads | 6.4 of 32 heads |
| resulting slots | **78** | **166** |

Two defensible ports, and they disagree:

- **fraction-matched**: `k=0.20`, band = middle 75% = layers **4-27** -> ~166 slots
- **slot-matched**: `k=0.10`, band **4-27** -> ~78 slots

On Qwen, widening the slot budget hurt at **every** setting measured
(`SRF_details.md` 5.8, 6.3). So run slot-matched first:

```bash
python -u srf/eval.py --model llava-hf/llava-1.5-7b-hf \
  --method srffovea --datasets pope \
  --head_mode ratio_topk --head_top_k_pct 0.1 \
  --layer_start 4 --layer_end 27 \
  --output results/llava_vtar_k10/ 2>&1 | tee /tmp/llava_vtar_k10.log
```

then `--head_top_k_pct 0.2` as the comparison. Do **not** assume 0.2 transfers.

### Sanity checks before trusting any LLaVA number

1. Baseline reproduces the published LLaVA-1.5-7B POPE number
2. `get_img_range` finds the image span via `model.config.image_token_index`,
   not the Qwen `<|image_pad|>` string
3. `clip_coarse_grid = 6` is in effect (336px), not 7
4. **Run the random-map control.** If random matches semantic, the relevance map
   is not reaching the model, and the run is measuring nothing
