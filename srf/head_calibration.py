#!/usr/bin/env python3
"""
head_calibration.py — alternative vision-responsive head selection for SRF.

WHY THIS EXISTS
---------------
The paper specifies head selection PER LAYER:

    H*_l = TopK_h( E_c[ rho_{l,h}^{(c)} ] )         (method section, eq. for H*)

but `qwen_attn_patch.identify_visual_heads` computes ONE (n_heads,) score vector
by accumulating mean image attention over EVERY decoder layer's softmax call
(qwen_attn_patch.py, the `_calibrate_heads` block) and dividing by
count = n_layers * n_samples. The resulting single mask is reused in every layer.
srf/docs/MMVP.md states this outright: "CLIP calibration is layer-agnostic —
head selection scores across all layers regardless of layer_end."

So the shipped method averages away the layer structure that the paper's own
attention-heatmap figure presents as the motivation for selective intervention.
A head that is strongly visual in the fusion zone [8,16] has that signal diluted
by its own weak scores in the ~20 layers that do not mediate fusion.

The 2026-09-16 component ablation (srf/ablation_components.py) found head
calibration to be the weakest component on MMVP: +0.67pp under the published
anchor, and -4.00pp under the MMVP-tuned anchor. This module tests whether a
better head-selection signal changes that.

MODES
-----
  global    reference — the shipped layer-agnostic mask from
            patch.identify_visual_heads. Reproduces the published result and
            acts as the control for the other modes.

  per_layer S1 — rho(l,h) = mean attention from text-query positions to image
            tokens, computed separately for each layer l. Implements the paper's
            equation as written. Same signal as `global`, resolved per layer.

  contrast  S2 — rho(l,h) = mean image attention with the real image MINUS mean
            image attention with pixel_values zeroed. Isolates heads whose
            attention actually DEPENDS on visual content, rather than heads that
            attend to image positions structurally. This is the improvement named
            in srf/docs/MMVP.md ("contrastive calibration ... would more cleanly
            separate the 3 genuinely semantic heads from structural ones").
            Costs 2 forward passes per calibration sample instead of 1.
            Caveat: zeroed pixel_values is an out-of-distribution ViT input (the
            same ablation SRF-E's Pass 2 uses). We only read attention
            differences, not generation quality, so OOD-ness matters less here
            than it does for SRF-E decoding — but it is not a clean "blank scene".

  saliency  S3 — rho(l,h) = Pearson correlation between head h's attention
            profile over image tokens (at layer l) and the CLIP semantic
            relevance map for that sample. Selects heads whose attention is
            already aligned with query-relevant regions, rather than heads that
            merely attend to image positions structurally. This addresses the
            weakness noted in srf/docs/MMVP.md: mean attention cannot separate
            genuinely semantic heads from structurally attending ones.

DESIGN CONSTRAINTS HONOURED
---------------------------
1. `my_analysis/qwen_attn_patch.py` is core code and is NOT modified. Per-layer
   masks are applied by registering our own forward pre-hook on each
   `layer.self_attn` that swaps `patch._STATE["head_mask"]` before that layer
   computes attention. The patch re-reads head_mask on every softmax call, so
   this is sufficient. Same wrap-externally pattern as srf/vhr.py.

2. Attention is captured via the patch's existing `_capture` / `_captured`
   mechanism plus a per-layer post-hook, NOT `output_attentions=True`. Holding
   all 28 layers' attention at once costs ~880 MB at MMVP resolution; this holds
   one layer at a time (~31 MB).

3. Calibration samples come from `srf._build_calib_inputs` (same n and seed as
   the real method), saliency from `srf.prepare_sample`, and evaluation from
   `eval.run_mmvp` via `ablation_components._SRFShim`. No eval loop, saliency
   function, or noun extractor is reimplemented here.

PARAMETERS
----------
Anchors are DERIVED FROM config.py per dataset (`_dataset_anchor`), not hardcoded,
so every run uses the same parameters that produced the published result:

  mmvp     saliency v3, alpha=2.0, eps=0.2, layers=[8,16], phase=both,
           sys_beta=0.30, htk=0.20, fovea sigma=20 (SRF-Fovea)
           reference to beat: pair=43.33%  img=69.67%
  vlmbias  saliency v3, alpha=8.0, eps=0.5, layers=[8,14], phase=generation,
           sys_beta=0.30, htk=0.20, fovea OFF (SRF base — Table 1's VLMBias
           figures are SRF base; SRF-Fovea was ~neutral there, 19.6 vs 19.7)

Only head selection varies within a dataset. The `global` mode is the in-run
control and must reproduce the shipped number.

Usage
-----
  cd /volumes2/mllm/lmms-eval

  source activate mllm && stdbuf -oL -eL python -u srf/head_calibration.py \\
    --output results/ablation/ 2>&1 | tee /tmp/head_calibration_mmvp.log

  # VLMBias, contrastive + saliency-aligned
  source activate mllm && stdbuf -oL -eL python -u srf/head_calibration.py \\
    --dataset vlmbias --modes global contrast saliency --output results/ablation/ \\
    2>&1 | tee /tmp/head_calibration_vlmbias.log

  # single mode
  python srf/head_calibration.py --modes global per_layer
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
from typing import Dict, List, Optional

_SRF_DIR      = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR / "saliency"))
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import config as CFG

os.environ.setdefault("HF_HOME", CFG.HF_HOME)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch

import ablation_components as abl_comp
import eval as eval_mod
import qwen_attn_patch as patch
import srf as srf_mod

MODES = ["global", "per_layer", "contrast", "saliency", "vtar_layers", "vtar_joint", "vtar_soft", "vtar_thresh", "vtar_ratio", "ratio_topk"]

MODE_LABELS = {
    "global":    "global mask (shipped, layer-agnostic)  [reference]",
    "per_layer": "per-layer mean image attention  [S1]",
    "contrast":  "per-layer contrastive (real - blank)  [S2]",
    "saliency":  "per-layer saliency-aligned  [S3]",
    "vtar_layers": "VTAR: top-N layers, then top-k heads in each  [S4]",
    "vtar_joint":  "VTAR: top-k over (layer,head) pairs  [S5]",
    "vtar_soft":   "mid-band layers + SOFT per-head VTAR  [S6]",
    "vtar_thresh": "per-layer threshold  rho > mu_l + kappa*sigma_l  [S7]",
    "vtar_ratio":  "absolute threshold on the vision attention ratio  VTAR > T  [S8]",
    "ratio_topk":  "VTAR score, top-k heads per layer  [S9]",
}

# Modes returning float per-head weights instead of boolean masks. These keep the
# architecturally motivated mid-band layer interval and weight heads continuously:
#   w[l,h] = clip( rho[l,h] / (kappa * mean_h rho[l]), 0, 1 )
# Each head is scored RELATIVE to its own layer's mean, so kappa is dimensionless
# and transfers across layers and models, unlike an absolute VTAR threshold.
SOFT_MODES = {"vtar_soft"}

# Modes whose per-layer masks do ALL the layer scoping, so the shim must open the
# layer range to the full decoder depth and let the masks select. Layers not
# selected get an all-False mask, which makes the patch a no-op there.
FULL_DEPTH_MODES = {"vtar_layers", "vtar_joint"}

# Per-dataset evaluation wiring. `fovea` follows what each dataset's published
# number actually used: MMVP = SRF-Fovea (sigma=20), VLMBias = SRF base.
DATASETS = {
    "mmvp":    {"run": "run_mmvp",    "variant": "srf_full",    "fovea": True},
    "vlmbias": {"run": "run_vlmbias", "variant": "srf_decoder", "fovea": False},
}


PRIMARY_LABEL = {"mmvp": "pair", "vlmbias": "acc"}
SECOND_LABEL  = {"mmvp": "img",  "vlmbias": None}

# Known shipped numbers the `global` control must reproduce. A mismatch means the
# shipped path changed and the run is untrustworthy.
#   mmvp    43.33 = srffovea_20 (docs/EXPERIMENTS.md), also PAPER.tex Table 1
#   vlmbias 19.7  = SRF base (docs/RESEARCH_STATUS.md results table).
#           NOTE PAPER.tex Table 1 reports 20.9 for this cell — a known
#           inconsistency, see RESEARCH_STATUS.md "Paper inconsistencies found".
EXPECTED_REF = {"mmvp": 0.4333, "vlmbias": 0.197}


def _dataset_anchor(model_id: str, dataset: str) -> dict:
    """
    Build the anchor from config.py so it always matches the shipped defaults for
    this dataset (the parameters behind the published number). Nothing hardcoded.
    """
    arch = CFG.get_arch(model_id)
    dp   = CFG.SRF_DATASET_PARAMS[dataset]
    return {
        "saliency_mode": arch["saliency_mode"],
        "alpha":         dp["alpha"],
        "eps":           dp["eps"],
        "phase":         dp["phase"],
        "layer_start":   arch["layer_start"],
        "layer_end":     arch["dataset_layer_end"].get(dataset, arch["layer_end"]),
        "sys_beta":      CFG.SRF_DEFAULTS["sys_beta"],
        "note":          f"config.py defaults for {dataset}",
    }


# ---------------------------------------------------------------------------
# Per-(layer, head) score capture
# ---------------------------------------------------------------------------

def _text_to_image_attention(attn: torch.Tensor, s: int, e: int) -> Optional[torch.Tensor]:
    """
    Slice a captured attention tensor down to text-query -> image-key entries.

    attn : (n_heads, q_len, kv_len), as stored in patch._STATE["_captured"]
    Returns (n_heads, n_text_q, n_img), or None when the slice is empty.

    Mirrors the convention in qwen_attn_patch's `_calibrate_heads` block: score
    from TEXT query positions only, so we measure how much each head pulls from
    image tokens rather than image tokens attending to themselves.
    """
    if attn is None or attn.dim() != 3:
        return None
    text_q_start = e + 1
    q_len        = attn.shape[1]
    if text_q_start >= q_len:
        return None
    return attn[:, text_q_start:, s : e + 1]


def _vision_attention_ratio(attn: torch.Tensor, s: int, e: int) -> Optional[torch.Tensor]:
    """
    Vision attention ratio (VTAR) per head, the quantity plotted in the
    attention-allocation figure of the paper.

        VTAR_h = mean over text-query positions of
                 ( sum of attention to image keys / sum of attention to all keys )

    This is a FRACTION in [0,1], unlike the score used by
    qwen_attn_patch.identify_visual_heads, which averages attention per image
    token and therefore scales as 1/n_img. Because VTAR is a ratio it is
    interpretable ("this head sends 30 percent of its attention to the image"),
    it can be thresholded with an absolute value, and it transfers across models
    with different numbers of image tokens or heads.

    attn : (n_heads, q_len, kv_len) as stored in patch._STATE["_captured"]
    """
    if attn is None or attn.dim() != 3:
        return None
    text_q_start = e + 1
    if text_q_start >= attn.shape[1]:
        return None
    rows       = attn[:, text_q_start:, :].float()          # (n_heads, n_text_q, kv)
    vision_sum = rows[:, :, s : e + 1].sum(dim=-1)          # (n_heads, n_text_q)
    total_sum  = rows.sum(dim=-1).clamp(min=1e-9)
    return (vision_sum / total_sum).mean(dim=1)             # (n_heads,)


def _budget_matched_weights(sc: torch.Tensor, k: float) -> torch.Tensor:
    """
    Solve  w = clip(sc / z, 0, 1)  for the per-layer scale z such that
    sum_h w == k, i.e. the SOFT weights spend exactly the same head budget as a
    hard top-k mask. This removes the free multiplier kappa entirely and makes
    "soft vs hard selection" a comparison at matched budget rather than one
    confounded with how many heads get boosted.

    sum_h w is monotonically decreasing in z, so a bisection is exact enough.
    """
    sc = sc.float().clamp(min=0.0)
    n  = sc.numel()
    if k >= n or float(sc.max()) <= 0.0:
        return torch.full_like(sc, min(1.0, k / max(n, 1)))
    f  = lambda z: float((sc / z).clamp(0.0, 1.0).sum())
    lo, hi = 1e-8, float(sc.max()) * n          # f(lo) ~ n >= k ; f(hi) <= 1 <= k
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > k:
            lo = mid
        else:
            hi = mid
    return (sc / (0.5 * (lo + hi))).clamp(0.0, 1.0)


def _pearson_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Row-wise Pearson correlation between each row of x (n, d) and vector y (d,).
    Returns (n,). Rows with zero variance yield 0.0.
    """
    xc = x - x.mean(dim=1, keepdim=True)
    yc = y - y.mean()
    num = (xc * yc).sum(dim=1)
    den = xc.norm(dim=1) * yc.norm()
    out = torch.zeros_like(num)
    ok  = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def calibrate_per_layer_heads(
    model,
    processor,
    mode: str,
    dataset: str = "mmvp",
    n: Optional[int] = None,
    seed: Optional[int] = None,
    top_k_pct: Optional[float] = None,
    n_layers_sel: int = 9,
    kappa: float = 2.0,
    vtar_thresh: float = 0.20,
    return_scores: bool = False,
) -> Dict[int, torch.Tensor]:
    """
    Compute a per-layer vision-responsive head mask.

    Returns {layer_idx: bool tensor (n_heads,)} selecting the top `top_k_pct`
    heads in each layer. Defaults for n / seed / top_k_pct come from config so
    calibration matches the shipped method exactly.
    """
    if mode not in ("per_layer", "contrast", "saliency", "vtar_layers",
                    "vtar_joint", "vtar_soft", "vtar_thresh", "vtar_ratio",
                    "ratio_topk"):
        raise ValueError(f"calibrate_per_layer_heads: unsupported mode {mode!r}")

    n         = CFG.SRF_DEFAULTS["calib_n"]    if n         is None else n
    seed      = CFG.SRF_DEFAULTS["calib_seed"] if seed      is None else seed
    top_k_pct = (CFG.get_arch(getattr(model.config, "_name_or_path", ""))["head_top_k_pct"]
                 if top_k_pct is None else top_k_pct)

    need_meta = (mode == "saliency")
    built = srf_mod._build_calib_inputs(dataset, n=n, seed=seed, return_meta=need_meta)
    if need_meta:
        calib_inputs, img_ranges, calib_meta = built
    else:
        calib_inputs, img_ranges = built
        calib_meta = [None] * len(calib_inputs)

    layers   = patch._get_decoder_layers(model)
    n_layers = len(layers)

    # Accumulators: layer -> (n_heads,) running sum, plus a per-layer sample count
    # (the saliency mode skips samples whose CLIP gate returns no mask).
    # `contrast` keeps a second set for the blank-image pass and subtracts.
    acc:   Dict[int, torch.Tensor] = {}
    acc_b: Dict[int, torch.Tensor] = {}
    count: Dict[int, int]          = {i: 0 for i in range(n_layers)}

    # State captured per sample, read by the post-hooks.
    # pass_id: "real" -> acc, "blank" -> acc_b (contrast mode only)
    ctx: dict = {"s": None, "e": None, "saliency": None, "active": False,
                 "pass_id": "real"}

    def _make_post_hook(layer_idx: int):
        def _hook(module, args, output):
            if not ctx["active"]:
                return
            attn = patch._STATE.get("_captured")
            if mode in ("vtar_ratio", "ratio_topk"):
                score = _vision_attention_ratio(attn, ctx["s"], ctx["e"])
                if score is None:
                    return
                prev = acc.get(layer_idx)
                acc[layer_idx] = score if prev is None else prev + score
                count[layer_idx] += 1
                return
            sel  = _text_to_image_attention(attn, ctx["s"], ctx["e"])
            if sel is None:
                return
            if mode in ("per_layer", "contrast", "vtar_layers", "vtar_joint",
                        "vtar_soft", "vtar_thresh"):
                score = sel.mean(dim=(1, 2))                    # (n_heads,)
            else:
                sal = ctx["saliency"]
                if sal is None:
                    return
                profile = sel.mean(dim=1)                        # (n_heads, n_img)
                if profile.shape[1] != sal.shape[0]:
                    return
                score = _pearson_rows(profile.float(), sal.float())
            target = acc_b if ctx["pass_id"] == "blank" else acc
            prev   = target.get(layer_idx)
            target[layer_idx] = score if prev is None else prev + score
            if ctx["pass_id"] == "real":
                count[layer_idx] += 1
        return _hook

    handles = [layers[i].self_attn.register_forward_hook(_make_post_hook(i))
               for i in range(n_layers)]

    saved_capture = patch._STATE.get("_capture", False)
    saved_method  = patch._STATE.get("method")
    n_used        = 0
    try:
        patch._STATE["_capture"] = True
        for inp, (s, e), meta in zip(calib_inputs, img_ranges, calib_meta):
            ctx["s"], ctx["e"] = s, e
            ctx["saliency"]    = None

            if mode == "saliency":
                # Real saliency path: srf.prepare_sample runs the configured
                # saliency_mode (noun extraction, CLIP gate, bad-noun fallback)
                # and leaves the map in patch._STATE["salience_mask"].
                srf_mod.prepare_sample(inp, s, e, meta["image"], meta["question"],
                                        model, processor)
                sal = patch._STATE.get("salience_mask")
                srf_mod.cleanup()
                if sal is None:
                    continue          # gated absent / bad noun — no target to align to
                ctx["saliency"] = sal.detach().cpu()

            # Calibrate on the unmodified model, as identify_visual_heads does.
            patch._STATE["method"] = "baseline"
            ctx["pass_id"] = "real"
            ctx["active"]  = True
            with torch.inference_mode():
                model(**inp)
            ctx["active"] = False

            if mode == "contrast":
                # Second pass with the visual channel removed. Zeroing
                # pixel_values is the same ablation SRF-E's Pass 2 uses and is
                # shape-safe (no re-processing, so the token grid cannot shift).
                blank = dict(inp)
                blank["pixel_values"] = torch.zeros_like(inp["pixel_values"])
                ctx["pass_id"] = "blank"
                ctx["active"]  = True
                with torch.inference_mode():
                    model(**blank)
                ctx["active"]  = False
                ctx["pass_id"] = "real"
                del blank

            n_used += 1
    finally:
        ctx["active"] = False
        for h in handles:
            h.remove()
        patch._STATE["_capture"]  = saved_capture
        patch._STATE["_captured"] = None
        patch._STATE["method"]    = saved_method

    if not acc:
        raise RuntimeError(
            f"head calibration mode={mode!r} captured no attention. Check that the "
            f"model uses eager attention and that calibration samples were built.")

    if mode == "contrast" and not acc_b:
        raise RuntimeError(
            "contrast mode captured no blank-pass attention — the second forward "
            "did not reach the decoder softmax.")

    # Per-(layer, head) score matrix rho[l] -> (n_heads,)
    rho: Dict[int, torch.Tensor] = {}
    for layer_idx, total in acc.items():
        c = max(1, count[layer_idx])
        sc = total / c
        if mode == "contrast":
            # rho = attn(real) - attn(blank): keep heads whose attention to image
            # positions actually depends on visual content.
            sc = sc - (acc_b.get(layer_idx, torch.zeros_like(total)) / c)
        rho[layer_idx] = sc

    n_heads = len(next(iter(rho.values())))
    k       = max(1, round(n_heads * top_k_pct))

    def _topk_mask(sc: torch.Tensor) -> torch.Tensor:
        return sc >= sc.topk(k).values[-1]

    masks: Dict[int, torch.Tensor] = {}

    if mode == "vtar_joint":
        # Top-(n_layers * k) slots over ALL (layer, head) pairs. Unconstrained
        # optimum: layers may contribute anywhere from 0 to n_heads slots.
        budget = n_layers_sel * k
        flat   = sorted(((float(rho[l][h]), l, h) for l in rho for h in range(n_heads)),
                        reverse=True)[:budget]
        for l in rho:
            masks[l] = torch.zeros(n_heads, dtype=torch.bool)
        for _, l, h in flat:
            masks[l][h] = True
        chosen = sorted({l for _, l, _ in flat})
        print(f"  [headcal] joint top-{budget} slots over {len(rho)*n_heads} pairs; "
              f"{len(chosen)} layers used")
        for l in chosen:
            hs = masks[l].nonzero().flatten().tolist()
            print(f"      L{l:2d}  heads {hs}  "
                  f"rho={[round(float(rho[l][h]), 4) for h in hs]}")

    elif mode == "ratio_topk":
        # Score by the vision attention ratio (theoretically the right quantity)
        # but select with top-k per layer rather than an absolute threshold.
        # Within a layer the ratio and the per-token mean rank heads identically,
        # because they differ only by the constant image-token count, so this
        # isolates the SELECTION RULE from the SCORE.
        for l in rho:
            masks[l] = _topk_mask(rho[l])
        print(f"  [headcal] VTAR score, top-{k} per layer, "
              f"{sum(int(m.sum()) for m in masks.values())} slots")

    elif mode == "vtar_ratio":
        # Absolute threshold on the vision attention ratio. A head is selected
        # when it sends at least `vtar_thresh` of its attention to image tokens.
        # Layers where no head clears the bar select nothing, so the layer set is
        # produced by the same rule rather than configured separately.
        for l in rho:
            masks[l] = rho[l] > vtar_thresh
        _n   = {l: int(masks[l].sum()) for l in sorted(masks)}
        _tot = sum(_n.values())
        _live = [l for l in _n if _n[l] > 0]
        print(f"  [headcal] VTAR > {vtar_thresh}: {_tot} slots in {len(_live)} layers "
              f"(of {len(_n)}). VTAR range "
              f"{min(float(v.min()) for v in rho.values()):.3f} to "
              f"{max(float(v.max()) for v in rho.values()):.3f}")
        if _tot == 0:
            print("  [headcal] WARNING: no head clears the threshold, "
                  "the intervention will be a no-op. Lower vtar_thresh.")

    elif mode == "vtar_thresh":
        # Per-layer relative threshold. A head is selected when its routing score
        # stands out against the OTHER HEADS IN ITS OWN LAYER, so the number of
        # selected heads varies with the layer instead of being asserted. kappa is
        # dimensionless, which is what lets one value transfer across layers and
        # models. Layers where no head stands out select none, and the layer gate
        # [layer_start, layer_end] does the rest of the scoping.
        for l in rho:
            sc = rho[l].float()
            masks[l] = sc > (sc.mean() + kappa * sc.std())
        _n = {l: int(masks[l].sum()) for l in sorted(masks)}
        _tot = sum(_n.values())
        print(f"  [headcal] threshold kappa={kappa}: {_tot} slots total, "
              f"heads/layer min={min(_n.values())} mean={_tot/len(_n):.2f} "
              f"max={max(_n.values())}  (fixed top-k would give {k})")

    elif mode == "vtar_soft":
        # Soft per-head weights inside the (unchanged) mid-band layer interval.
        # No head is excluded outright; weak heads simply get a small weight.
        # kappa <= 0 (default) = BUDGET MATCHED: solve the per-layer scale so
        # sum_h w == k, so this is a fair soft-vs-hard comparison with no extra
        # parameter. kappa > 0 reproduces the earlier relative-to-layer-mean
        # variant, whose budget was ~7.5 heads/layer and therefore confounded.
        for l in rho:
            sc = rho[l].float()
            if kappa > 0:
                mean  = float(sc.mean())
                denom = kappa * mean if mean > 1e-12 else 1e-12
                masks[l] = (sc / denom).clamp(0.0, 1.0)
            else:
                masks[l] = _budget_matched_weights(sc, float(k))
        _eff = {l: float(masks[l].sum()) for l in sorted(masks)}
        _tag = f"kappa={kappa}" if kappa > 0 else f"budget-matched to k={k}"
        print(f"  [headcal] soft VTAR weights, {_tag}: "
              f"effective heads/layer (sum of w) "
              f"min={min(_eff.values()):.2f} mean={sum(_eff.values())/len(_eff):.2f} "
              f"max={max(_eff.values()):.2f}   (hard top-k budget was {k})")
        for l in sorted(masks):
            top = masks[l].topk(min(5, n_heads))
            print(f"      L{l:2d}  sum_w={_eff[l]:.2f}  top heads "
                  f"{top.indices.tolist()} w={[round(float(v),2) for v in top.values]}")

    elif mode == "vtar_layers":
        # Score each layer by the mean rho of the k heads we would actually boost
        # there, NOT by the mean over all heads. A layer with one very strong
        # visual head and 15 weak ones is a good target; mean-ranking hides that.
        layer_score = {l: float(rho[l].topk(k).values.mean()) for l in rho}
        chosen = sorted(sorted(layer_score, key=layer_score.get,
                               reverse=True)[:n_layers_sel])
        for l in rho:
            masks[l] = (_topk_mask(rho[l]) if l in chosen
                        else torch.zeros(n_heads, dtype=torch.bool))
        print(f"  [headcal] top-{n_layers_sel} layers by top-{k} head VTAR: {chosen}")
        for l in chosen:
            hs = masks[l].nonzero().flatten().tolist()
            print(f"      L{l:2d}  heads {hs}  "
                  f"rho={[round(float(rho[l][h]), 4) for h in hs]}  "
                  f"layer_score={layer_score[l]:.4f}")

    else:
        for l in rho:
            masks[l] = _topk_mask(rho[l])

    _tot  = int(sum(int(m.sum()) for m in masks.values()))
    _used = sum(1 for m in masks.values() if bool(m.any()))
    if return_scores:
        return masks, rho
    print(f"  [headcal] mode={mode}  samples_used={n_used}/{len(calib_inputs)}  "
          f"layers_with_heads={_used}/{len(masks)}  total_slots={_tot}")
    return masks


# ---------------------------------------------------------------------------
# Per-layer mask application (no core-code changes)
# ---------------------------------------------------------------------------

def install_per_layer_masks(model, masks: Dict[int, torch.Tensor]) -> List:
    """
    Register a forward pre-hook on each decoder layer's self_attn that swaps
    patch._STATE["head_mask"] to that layer's mask.

    The patch reads head_mask fresh inside every softmax call, and these hooks
    fire before the layer computes attention, so they take precedence over the
    global mask written by srf.prepare_sample / srf.cleanup.

    Returns hook handles — pass to remove_hooks() when done.
    """
    layers  = patch._get_decoder_layers(model)
    handles = []
    for i in range(len(layers)):
        m = masks.get(i)
        if m is None:
            continue

        def _make(mask: torch.Tensor):
            def _pre(module, args):
                patch._STATE["head_mask"] = mask
            return _pre

        handles.append(layers[i].self_attn.register_forward_pre_hook(_make(m.clone())))
    return handles


def install_per_layer_weights(model, weights: Dict[int, torch.Tensor]) -> List:
    """
    Like install_per_layer_masks but installs FLOAT per-head weights into
    patch._STATE["head_weight"], which the patch uses to scale the whole bias row
    per head. Requires the additive head_weight support in qwen_attn_patch.py.
    """
    layers  = patch._get_decoder_layers(model)
    handles = []
    for i in range(len(layers)):
        w = weights.get(i)
        if w is None:
            continue

        def _make(wt: torch.Tensor):
            def _pre(module, args):
                patch._STATE["head_weight"] = wt
            return _pre

        handles.append(layers[i].self_attn.register_forward_pre_hook(
            _make(w.detach().float().clone())))
    return handles


def remove_hooks(handles: List) -> None:
    for h in handles:
        h.remove()
    # Always clear the soft-weight state so it cannot leak into the next mode.
    patch._STATE["head_weight"] = None


# ---------------------------------------------------------------------------
# Overlap diagnostics
# ---------------------------------------------------------------------------

def mask_overlap(per_layer: Dict[int, torch.Tensor], global_mask: torch.Tensor,
                 layer_start: int, layer_end: int) -> dict:
    """
    Jaccard overlap between each per-layer mask and the shipped global mask,
    reported over the fusion zone and over all layers. Overlap ~1.0 inside the
    fusion zone means per-layer selection cannot change the result.
    """
    def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
        inter = (a & b).sum().item()
        union = (a | b).sum().item()
        return inter / union if union else 1.0

    per = {i: _jaccard(m, global_mask) for i, m in sorted(per_layer.items())}
    fus = [v for i, v in per.items() if layer_start <= i <= layer_end]
    return {
        "per_layer":       per,
        "fusion_mean":     sum(fus) / len(fus) if fus else float("nan"),
        "all_layer_mean":  sum(per.values()) / len(per) if per else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate alternative head-calibration signals for SRF on MMVP.")
    p.add_argument("--model", default=CFG.DEFAULT_MODEL)
    p.add_argument("--dataset", default="mmvp", choices=list(DATASETS),
                   help="Evaluation dataset (default mmvp).")
    p.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    p.add_argument("--calib_dataset", default=None,
                   choices=["mmvp", "pope", "vlmbias"],
                   help="Dataset for head calibration (default: same as --dataset).")
    p.add_argument("--n_vlmbias_per_cat", type=int, default=0,
                   help="VLMBias: samples per category (0 = all).")
    p.add_argument("--n_layers_sel", type=int, default=9,
                   help="vtar_layers/vtar_joint: number of layers (or layer-equivalents "
                        "of budget) to select. Default 9 = width of the shipped [8,16].")
    p.add_argument("--vtar_thresh", type=float, default=0.20,
                   help="head_mode vtar_ratio: keep heads whose vision attention "
                        "ratio exceeds this. 0.20 means the head sends at least "
                        "20 percent of its attention to image tokens.")
    p.add_argument("--kappa", type=float, default=0.0,
                   help="vtar_soft: 0 (default) = budget-matched, solve z so sum_h w == k "
                        "(no free parameter, fair vs hard top-k). >0 = legacy variant, "
                        "w = clip(rho/(kappa*mean_h rho), 0, 1), budget NOT matched.")
    p.add_argument("--sigma", type=float, default=abl_comp.FOVEA_SIGMA)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset   = args.dataset
    dcfg      = DATASETS[dataset]
    calib_ds  = args.calib_dataset or dataset
    anchor    = _dataset_anchor(args.model, dataset)
    run_fn    = getattr(eval_mod, dcfg["run"])
    variant   = abl_comp.VARIANT_BY_KEY[dcfg["variant"]]
    sigma     = args.sigma if dcfg["fovea"] else 0.0

    model, processor = eval_mod.load_model(args.model)
    img_token_id     = processor.tokenizer.convert_tokens_to_ids(CFG.IMAGE_TOKEN)
    device           = next(model.parameters()).device
    n_layers         = CFG.get_arch(args.model)["n_layers"]

    # Patch + shipped calibration (also the `global` reference mask).
    srf_mod.setup(model, processor, calib_dataset=calib_ds)

    base_args = abl_comp.build_eval_args(args.model)
    eargs     = abl_comp.anchor_args(base_args, anchor, suppress=True)
    eargs.datasets = [dataset]
    eargs.n_vlmbias_per_cat = args.n_vlmbias_per_cat

    # Warm-up reset so the global mask corresponds to this anchor's layer_end.
    srf_mod.reset_for_dataset(dataset=dataset, **eval_mod._reset_overrides(eargs))
    global_mask = patch._STATE["head_mask"].clone()
    print(f"\n  global mask: {int(global_mask.sum())}/{len(global_mask)} heads")

    results: Dict[str, dict] = {}
    rows: List[tuple] = []

    for mode in args.modes:
        print("\n" + "=" * 84)
        print(f"HEAD CALIBRATION — {MODE_LABELS[mode]}")
        print("=" * 84)

        masks   = None
        overlap = None
        if mode != "global":
            masks   = calibrate_per_layer_heads(model, processor, mode,
                                                 dataset=calib_ds,
                                                 n_layers_sel=args.n_layers_sel,
                                                 kappa=args.kappa,
                                                 vtar_thresh=args.vtar_thresh)
            _ov_in  = ({l: (m > 0.5) for l, m in masks.items()}
                       if mode in SOFT_MODES else masks)
            overlap = mask_overlap(_ov_in, global_mask,
                                    anchor["layer_start"], anchor["layer_end"])
            print(f"  overlap vs global mask — fusion zone "
                  f"[{anchor['layer_start']},{anchor['layer_end']}]: "
                  f"{overlap['fusion_mean']:.3f}   all layers: "
                  f"{overlap['all_layer_mean']:.3f}")

        # VTAR modes scope layers via the masks themselves, so open the shim's
        # layer range to full depth; all-False masks make unselected layers no-ops.
        v = dict(variant)
        if mode in FULL_DEPTH_MODES:
            v["layers"] = "all"
        shim    = abl_comp._SRFShim(v, global_mask, n_layers, anchor, sigma)
        if masks and mode in SOFT_MODES:
            handles = install_per_layer_weights(model, masks)
        elif masks:
            handles = install_per_layer_masks(model, masks)
        else:
            handles = []
        try:
            res = run_fn(shim, model, processor, img_token_id, device, eargs)
        finally:
            remove_hooks(handles)

        if dataset == "mmvp":
            primary, secondary = res["method_pair"][0.0], res["method_img"][0.0]
            per_cat = None
        else:
            primary, secondary = res["method"][0.0], None
            per_cat = {c: v["acc"] for c, v in res["per_category"][0.0].items()}

        results[mode] = {
            "label":        MODE_LABELS[mode],
            "selected_heads": (
                {int(l): [round(float(v), 4) for v in m] for l, m in masks.items()}
                if masks and mode in SOFT_MODES else
                {int(l): m.nonzero().flatten().tolist()
                 for l, m in masks.items() if bool(m.any())} if masks else None),
            "primary":      primary,
            "secondary":    secondary,
            "per_category": per_cat,
            "overlap":      overlap,
        }
        rows.append((mode, MODE_LABELS[mode], primary, secondary, per_cat))
        sec = f"  {SECOND_LABEL[dataset]}={secondary*100:.2f}%" if secondary is not None else ""
        print(f"  → {PRIMARY_LABEL[dataset]}={primary*100:.2f}%{sec}")

    # ── Summary ────────────────────────────────────────────────────────────────
    ref = results.get("global", {}).get("primary")
    print("\n" + "=" * 96)
    print(f"HEAD CALIBRATION SUMMARY — {dataset.upper()}  ({anchor['note']})")
    print(f"  saliency_mode={anchor['saliency_mode']}  alpha={anchor['alpha']}  "
          f"layers=[{anchor['layer_start']},{anchor['layer_end']}]  "
          f"eps={anchor['eps']}  phase={anchor['phase']}  "
          f"sys_beta={anchor['sys_beta']}  "
          f"fovea={'sigma=' + str(sigma) if dcfg['fovea'] else 'off'}  "
          f"calib={calib_ds}")
    print("=" * 96)
    hdr2 = SECOND_LABEL[dataset] or ""
    print(f"{'Mode':<48} {PRIMARY_LABEL[dataset]:>8} {'Δref':>7} {hdr2:>8}")
    print("-" * 96)
    for mode, label, primary, secondary, _ in rows:
        d   = "" if ref is None or mode == "global" else f"{(primary-ref)*100:+.2f}"
        sec = f"{secondary*100:.2f}" if secondary is not None else ""
        print(f"{label:<48} {primary*100:>8.2f} {d:>7} {sec:>8}")

    if any(r[4] for r in rows):
        cats = sorted({c for r in rows if r[4] for c in r[4]})
        print(f"\n  Per-category accuracy (%):")
        print("  " + f"{'Mode':<34}" + "".join(f"{c[:11]:>13}" for c in cats))
        for mode, label, _, _, per_cat in rows:
            if not per_cat:
                continue
            print("  " + f"{mode:<34}" + "".join(f"{per_cat[c]*100:>13.1f}" for c in cats))

    expect = EXPECTED_REF.get(dataset)
    if ref is not None and expect is not None and abs(ref - expect) > 1e-3:
        print(f"\n  WARNING: the `global` reference gave {ref*100:.2f}%, not the "
              f"expected {expect*100:.2f}%. Something changed in the shipped path — "
              f"treat the other rows as untrustworthy until this is explained.")

    if args.output:
        out_dir = pathlib.Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"head_calibration_{dataset}_qwen3b.json"
        out_path.write_text(json.dumps({
            "model":   args.model,
            "dataset": dataset,
            "calib_dataset": calib_ds,
            "anchor":  anchor,
            "sigma":   sigma,
            "fovea":   dcfg["fovea"],
            "results": results,
        }, indent=2, default=str))
        print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
