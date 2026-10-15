# SRF method card

One page. What the method is, what each component costs, and how to run it on
LLaVA from the `srf-llava` branch. Long-form history lives in `SRF_details.md`.

Status 2026-09-20. Validated on Qwen2.5-VL-3B / MMVP only.

---

## 1. The four components

| # | Component | Where it acts | What it does | Params | Value (Qwen-3B) |
|---|---|---|---|---|---|
| 1 | **Semantic relevance map** (`clip_full_gate_v3`) | offline, before the model | Extract the query noun. Score every image patch by CLIP similarity at 3 crop grids (3x3, 5x5, 7x7), take the elementwise max. Soft presence gate `c = min(full_img_sim / tau, 1)` scales the whole map | `tau` | **0.20** |
| 2 | **Semantic foveation** | encoder input, before ViT | `I_SRF = M*I + (1-M)*Blur(I)`. Gaussian blur on everything the map calls irrelevant. Physically removes distractor detail | `sigma` (std dev, px) | **20** |
| 3 | **Head + layer selection** | offline, 20 unlabelled samples | Fix layers to a mid-band. Score each head by **VTAR** = attention mass on image keys / total attention mass. Keep the top fraction **within each layer** | `[l_s, l_e]`, `head_top_k_pct` | **[6, 31]**, **0.20** -> 3 of 16 heads/layer, 78 slots |
| 4 | **Attention re-weighting** | decoder, selected slots only | Pre-softmax logit shift. `Z_ij + lambda_sem*s_j - lambda_bg*(1-s_j)` on image keys, `-lambda_sys` on system-prompt keys | `lambda_sem`, `lambda_bg`, `lambda_sys` | **2.0**, **0.2**, **0.30** |

Component 1 feeds both 2 and 4. It is an input, not an intervention, so it is
ablated by **replacement** with a random map, never by removal.

## 2. What each component is worth (MMVP pair acc, 150 pairs, 1 pair = 0.67pp)

| Configuration | Pair | Delta |
|---|---|---|
| baseline | 40.00 | |
| + `lambda_sem` | 41.33 | +2 pairs |
| + `lambda_bg` | 40.67 | -1 pair |
| + `lambda_sys` | 42.67 | +3 pairs |
| **+ foveation = full SRF** | **45.33** | **+4 pairs** |
| random map, full config | 40.67 | inert |

Semantic 45.33 vs random 40.67 is a 7-pair gap. The map buys 1 pair at the
decoder and 6 through the encoder. Foveation is the single largest component.

## 3. Hyperparameters, all of them

| Symbol | Flag | Value | Sensitivity |
|---|---|---|---|
| `tau` | `--clip_fallback_thresh` | 0.20 | largest, monotonic. Best value is the one that disables the gate on MMVP |
| `sigma` | `srf_fovea.SIGMA` | 20 | non-monotonic. 10 is worse than 0 |
| `lambda_sem` | `--alpha` | 2.0 | flat 0.5-2, collapses above 4 |
| `lambda_bg` | `--eps` | 0.2 | within noise, costs 1 pair alone |
| `lambda_sys` | `--sys_beta` | 0.30 | within noise |
| `[l_s, l_e]` | `--layer_start/end` | 6, 31 | middle-75% argument, NOT swept |
| `head_top_k_pct` | `--head_top_k_pct` | 0.20 | flat 0.1-0.2, worse above |

Dead / no-op, ignore them: `clip_top_k_pct`, `clip_coarse_grid` (fixed under
v3), `interp_lambda`, `prob_floor`, `img_scale`, `text_beta`, `vr_target`,
`vr_k`, `clip_soft_gate` (mathematically a no-op, verified).

## 4. The command (Qwen, current best)

```bash
python -u srf/eval.py --method srffovea --datasets mmvp \
  --head_mode ratio_topk --head_top_k_pct 0.2 \
  --layer_start 6 --layer_end 31 \
  --output results/srf_current/ 2>&1 | tee /tmp/srf_current.log
```

-> MMVP 45.33 pair / 70.00 img. Reproduced five times.
**These are not the `config.py` defaults.** Nothing was promoted, deliberately,
because none of it is validated off MMVP.

---

## 5. Branch split

Fork point `570dcb15` (2026-07-31).

| | `autoresearch/mmvp-srf` | `srf-llava` (Snellius) |
|---|---|---|
| Model | Qwen2.5-VL-3B only (`load_model` hardcodes the Qwen class) | **Qwen + LLaVA**, dispatches on `config.model_type`, uses `my_analysis/llava_attn_patch.py` |
| Datasets | mmvp, pope, vlmbias, mme, vlind | same **+ mmhalbench** (`run_mmhalbench`, `score_mmhalbench.py`) |
| Methods | srf, srfe, srffovea, vcd, vaf, baseline | same **+ srfc2, srfc3** |
| Head selection | **`--head_mode ratio_topk` + `head_calibration.py`** (VTAR, 10 modes) | **absent.** Only the legacy `global` pooled top-20% |
| Foveation | `srf/srf_fovea.py` | `srf/srf_fovea.py` (independent copy) |
| Ablation tooling | `ablation_components.py`, `param_sensitivity.py`, `saliency_mode="random"` | absent |
| Soft head weights | `qwen_attn_patch._STATE["head_weight"]` | absent in `llava_attn_patch` |

Both sides edited `srf/config.py`, `srf/eval.py`, `srf/srf.py`. In `eval.py` the
overlap is `parse_args`, `main`, `run_pope`, `run_vlindbench`.

## 6. To run the current method on LLaVA

`srf-llava` can load LLaVA but does not have the head selection that produces
45.33. Three things must move across.

1. **`srf/head_calibration.py`** -> copy as-is, then change line 132
   `import qwen_attn_patch as patch` to take the patch module from
   `eval.py`'s global (which `load_model` already sets per architecture).
   `_get_decoder_layers` in `qwen_attn_patch` is already arch-agnostic and
   handles `LlamaForCausalLM.model.layers`, so it can stay the layer accessor.
2. **`eval.py`**: the `--head_mode / --head_top_k_pct / --n_layers_sel /
   --kappa / --vtar_thresh` flags and `_install_head_mode()`.
3. **`llava_attn_patch.py`**: add the optional `_STATE["head_weight"]` float
   vector, mirroring the `qwen_attn_patch` change. **Only needed for
   `vtar_soft`.** `ratio_topk` uses the boolean `head_mask`, which already
   exists, so this step can be skipped for the shipped rule.

### k does not transfer

LLaVA-1.5-7B is 32 layers x 32 heads = 1024 slots. Qwen-3B is 36 x 16 = 576.
`head_top_k_pct = 0.20` gives 3 of 16 heads on Qwen but **6.4 of 32** on LLaVA,
and the mid-band `[6, 31]` is 26 of 36 layers on Qwen but 26 of 32 on LLaVA,
a much wider band.

Two ways to carry the setting over, and they disagree:

- **match the fraction** (`k=0.20`, band = middle 75% -> layers 4-27):
  ~166 slots, roughly 2x the Qwen slot count
- **match the slot count** (~78 slots over 24 layers -> `k ≈ 0.10`)

Widening the budget hurt on Qwen at every setting measured, so **match the slot
count first** (`--head_top_k_pct 0.1 --layer_start 4 --layer_end 27`) and run
`0.2` as the comparison. Do not assume 0.2 transfers.

### Before trusting any LLaVA number

- Confirm the baseline reproduces the published LLaVA-1.5-7B POPE / MMVP number
- Confirm the image token range is found correctly (LLaVA uses
  `model.config.image_token_index`, not `<|image_pad|>`)
- Confirm CLIP patch grid matches the 336px input, not Qwen's 448px
- Run the random-map control. If random matches semantic, the map is not
  reaching the model
