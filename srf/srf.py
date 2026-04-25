#!/usr/bin/env python3
"""
SRF (Semantic Re-Focus) base method — clean standalone.

Query-driven saliency re-focus: CLIP-guided attention logit boost to
query-relevant image tokens in vision-aware middle-layer heads.

Hyperparameters are resolved in priority order:
  1. Explicit CLI overrides passed to reset_for_dataset()
  2. Per-arch defaults from config.SRF_ARCH_PARAMS[model_id]
  3. Per-dataset defaults from config.SRF_DATASET_PARAMS[dataset]
  4. Shared defaults from config.SRF_DEFAULTS

Call reset_for_dataset() when switching datasets or sweeping hyperparams.

Public interface:
    setup(model, processor, calib_dataset="pope")
    reset_for_dataset(dataset, phase, alpha, eps,
                      layer_start, layer_end, head_top_k_pct,
                      clip_coarse_grid, clip_top_k_pct, clip_fallback_thresh)
    prepare_sample(inputs, img_start, img_end, image, question, model, processor)
    cleanup()
"""
from __future__ import annotations

import pathlib
import random
import re
import sys

_SRF_DIR      = pathlib.Path(__file__).parent           # srf/
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"         # my_analysis/
sys.path.insert(0, str(_SRF_DIR / "saliency"))           # clip_salience, hssa_salience
sys.path.insert(0, str(_SRF_DIR))                         # config, noun_extract, srf
sys.path.insert(0, str(_ANALYSIS_DIR))                    # qwen_attn_patch

import torch
import qwen_attn_patch as patch
import clip_salience as clip_sal
from noun_extract import extract_clip_noun
import config as CFG


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

BIAS     : dict = {}
SALIENCY : dict = {}

_model      = None
_processor  = None
_spatial    = 2
_noun_mode  = "pope"
_model_id   = ""        # set in setup(); used to look up SRF_ARCH_PARAMS


# ---------------------------------------------------------------------------
# Config helpers — build BIAS and SALIENCY dicts from config + overrides
# ---------------------------------------------------------------------------

def _get_arch() -> dict:
    """Return arch params for the currently loaded model."""
    return CFG.get_arch(_model_id)


def _make_bias(dataset: str, overrides: dict) -> dict:
    """
    Build the BIAS config dict for a given dataset.

    Sources (highest priority last wins):
      arch["layer_start/end/head_top_k_pct"]
      arch["dataset_layer_end"][dataset]
      dataset_params["phase/alpha/eps"]
      overrides (CLI args)
    """
    arch  = _get_arch()
    dp    = CFG.SRF_DATASET_PARAMS.get(dataset, CFG.SRF_DATASET_PARAMS["pope"])
    d     = CFG.SRF_DEFAULTS

    # layer_end: arch default → dataset-specific fine-tune
    layer_end = arch["dataset_layer_end"].get(dataset, arch["layer_end"])

    b = {
        "layer_start":      arch["layer_start"],
        "layer_end":        layer_end,
        "head_top_k_pct":   arch["head_top_k_pct"],
        "sys_beta":         d["sys_beta"],
        "text_beta":        d["text_beta"],
        "text_layer_start": d["text_layer_start"],
        "text_layer_end":   d["text_layer_end"],
        "bias_mode":        d["bias_mode"],
        "boost_alpha":      dp["alpha"],
        "background_eps":   dp["eps"],
        "interp_lambda":    d["interp_lambda"],
        "prob_floor":       d["prob_floor"],
        "img_scale":        d["img_scale"],
        "srf_apply_phase":  dp["phase"],
    }

    # Apply CLI overrides
    if overrides.get("phase")     is not None: b["srf_apply_phase"] = overrides["phase"]
    if overrides.get("alpha")     is not None: b["boost_alpha"]     = overrides["alpha"]
    if overrides.get("eps")       is not None: b["background_eps"]  = overrides["eps"]
    if overrides.get("layer_start") is not None: b["layer_start"]   = overrides["layer_start"]
    if overrides.get("layer_end") is not None: b["layer_end"]       = overrides["layer_end"]
    if overrides.get("head_top_k_pct") is not None: b["head_top_k_pct"] = overrides["head_top_k_pct"]

    return b


def _make_saliency(overrides: dict) -> dict:
    """
    Build the SALIENCY config dict.
    Sources: arch defaults → CLI overrides.
    """
    arch = _get_arch()
    s = {
        "clip_coarse_grid":     arch["clip_coarse_grid"],
        "clip_top_k_pct":       arch["clip_top_k_pct"],
        "clip_use_soft":        True,   # always soft — hard mask hurts boundary tokens
        "clip_fallback_thresh": arch["clip_fallback_thresh"],
    }
    if overrides.get("clip_coarse_grid")     is not None: s["clip_coarse_grid"]     = overrides["clip_coarse_grid"]
    if overrides.get("clip_top_k_pct")       is not None: s["clip_top_k_pct"]       = overrides["clip_top_k_pct"]
    if overrides.get("clip_fallback_thresh") is not None: s["clip_fallback_thresh"] = overrides["clip_fallback_thresh"]
    return s


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _build_calib_inputs(dataset: str, n: int, seed: int):
    """Build (inputs, img_ranges) for vision-aware head calibration."""
    from qwen_vl_utils import process_vision_info as pvi
    from datasets import load_dataset as hf_load

    device = next(_model.parameters()).device
    arch   = _get_arch()
    # LLaVA-style: image_token is None → use model.config.image_token_index
    if arch["image_token"] is not None:
        img_id = _processor.tokenizer.convert_tokens_to_ids(arch["image_token"])
    else:
        img_id = _model.config.image_token_index

    rng = random.Random(seed)
    calib_inputs, img_ranges = [], []

    if dataset == "pope":
        ds   = hf_load("lmms-lab/POPE", split="test")
        rows = list(ds)   # all splits — calibration doesn't need to match eval split
        rng.shuffle(rows)
        for r in rows[:n]:
            q    = str(r["question"]).strip() + "\nAnswer with Yes or No only."
            img  = r["image"].convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text",  "text":  q}]}]
            text = _processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
            vis, _ = pvi(msgs)
            inp = _processor(text=[text], images=vis, return_tensors="pt",
                             padding=True).to(device)
            ids = inp["input_ids"][0].tolist()
            s = ids.index(img_id)
            e = len(ids) - 1 - ids[::-1].index(img_id)
            calib_inputs.append(inp); img_ranges.append((s, e))

    elif dataset == "mmvp":
        import pandas as pd
        df     = pd.read_csv(CFG.MMVP_CSV)
        img_ds = hf_load("MMVP/MMVP", split="train")
        lex_sorted = sorted(range(1, 301), key=str)
        csv_to_hf  = {c: h for h, c in enumerate(lex_sorted)}
        idxs = list(range(1, 301)); rng.shuffle(idxs)
        for csv_1idx in idxs[:n]:
            row  = df.iloc[csv_1idx - 1]
            img  = img_ds[csv_to_hf[csv_1idx]]["image"].convert("RGB")
            opts = re.findall(r'\(([ab])\)\s*([^(]+)', str(row["Options"]), re.IGNORECASE)
            opt_text = "\n".join(f"{m[0].upper()}. {m[1].strip()}" for m in opts)
            prompt   = (f"{row['Question']}\n{opt_text}\n"
                        "Answer with the option's letter directly.")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text",  "text":  prompt}]}]
            text = _processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
            vis, _ = pvi(msgs)
            inp = _processor(text=[text], images=vis, return_tensors="pt",
                             padding=True).to(device)
            ids = inp["input_ids"][0].tolist()
            s = ids.index(img_id)
            e = len(ids) - 1 - ids[::-1].index(img_id)
            calib_inputs.append(inp); img_ranges.append((s, e))

    elif dataset == "vlmbias":
        ds   = hf_load("anvo25/vlms-are-biased", split="main")
        rows = list(ds); rng.shuffle(rows)
        for r in rows[:n]:
            img  = r["image"].convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text",  "text":  r["prompt"]}]}]
            text = _processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
            vis, _ = pvi(msgs)
            inp = _processor(text=[text], images=vis, return_tensors="pt",
                             padding=True).to(device)
            ids = inp["input_ids"][0].tolist()
            s = ids.index(img_id)
            e = len(ids) - 1 - ids[::-1].index(img_id)
            calib_inputs.append(inp); img_ranges.append((s, e))

    elif dataset in ("mme", "hallusionbench"):
        # MME / HallusionBench — Yes/No questions, use same format as POPE calibration
        ds   = hf_load("lmms-lab/MME", split="test")
        rows = list(ds); rng.shuffle(rows)
        for r in rows[:n]:
            q    = str(r.get("question", "")).strip()
            img  = r["image"].convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text",  "text":  q}]}]
            text = _processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
            vis, _ = pvi(msgs)
            inp = _processor(text=[text], images=vis, return_tensors="pt",
                             padding=True).to(device)
            ids = inp["input_ids"][0].tolist()
            s = ids.index(img_id)
            e = len(ids) - 1 - ids[::-1].index(img_id)
            calib_inputs.append(inp); img_ranges.append((s, e))

    else:
        raise ValueError(f"Unknown calib dataset: {dataset!r}. "
                         f"Supported: pope, mmvp, vlmbias, mme, hallusionbench")

    return calib_inputs, img_ranges


def _sync_patch_state() -> None:
    """Push current BIAS values into the shared patch state dict."""
    patch._STATE["vaf_layer_start"]      = BIAS["layer_start"]
    patch._STATE["vaf_layer_end"]        = BIAS["layer_end"]
    patch._STATE["vaf_beta"]             = BIAS["sys_beta"]
    patch._STATE["srf_background_eps"]   = BIAS["background_eps"]
    patch._STATE["srf_bias_mode"]        = BIAS["bias_mode"]
    patch._STATE["srf_interp_lambda"]    = BIAS["interp_lambda"]
    patch._STATE["srf_prob_floor"]       = BIAS["prob_floor"]
    patch._STATE["srf_img_scale"]        = BIAS["img_scale"]
    patch._STATE["srf_apply_phase"]      = BIAS["srf_apply_phase"]
    patch._STATE["srf_text_beta"]        = BIAS["text_beta"]
    patch._STATE["srf_text_layer_start"] = BIAS["text_layer_start"]
    patch._STATE["srf_text_layer_end"]   = BIAS["text_layer_end"]
    patch._STATE["srf_layer_alphas"]     = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup(model, processor, calib_dataset: str = "pope") -> None:
    """
    One-time setup: identify vision-aware heads and patch the model.
    Call once after loading the model. Detects architecture automatically.
    """
    global _model, _processor, _spatial, _model_id, BIAS, SALIENCY
    _model, _processor = model, processor
    _model_id  = getattr(model.config, "_name_or_path", "")
    _spatial   = getattr(model.config.vision_config, "spatial_merge_size",
                         CFG.get_arch(_model_id)["spatial_merge_size"])

    arch = _get_arch()
    if _model_id not in CFG.SRF_ARCH_PARAMS:
        print(f"  [SRF] WARNING: model_id {_model_id!r} not in SRF_ARCH_PARAMS — "
              f"using fallback arch params. Add it to config.py after tuning.")

    # Initialise BIAS/SALIENCY with dataset defaults (no overrides yet)
    BIAS     = _make_bias(calib_dataset, overrides={})
    SALIENCY = _make_saliency(overrides={})

    n    = CFG.SRF_DEFAULTS["calib_n"]
    seed = CFG.SRF_DEFAULTS["calib_seed"]
    print(f"  [SRF] Calibrating on {calib_dataset} (n={n}, seed={seed})…")
    print(f"  [SRF] arch: {_model_id or 'unknown'}  "
          f"layer_start={arch['layer_start']}  layer_end={arch['layer_end']}  "
          f"head_top_k={arch['head_top_k_pct']}  "
          f"clip_grid={arch['clip_coarse_grid']}  clip_topk={arch['clip_top_k_pct']}")

    calib_inputs, img_ranges = _build_calib_inputs(calib_dataset, n=n, seed=seed)
    patch.identify_visual_heads(model, calib_inputs, img_ranges, BIAS["head_top_k_pct"])
    n_sel = int(patch._STATE["head_mask"].sum().item())
    print(f"  [SRF] {n_sel} vision-aware heads (top {BIAS['head_top_k_pct']*100:.0f}%)")
    del calib_inputs, img_ranges
    torch.cuda.empty_cache()

    patch.patch_model(model, "vaf", max(float(BIAS["boost_alpha"]), 1e-6))
    _sync_patch_state()


def reset_for_dataset(
    dataset: str = "pope",
    *,
    # dataset-specific tunables
    phase:    str   | None = None,
    alpha:    float | None = None,
    eps:      float | None = None,
    # arch/layer tunables (scale with model depth)
    layer_start:    int   | None = None,
    layer_end:      int   | None = None,
    head_top_k_pct: float | None = None,
    # CLIP saliency tunables
    clip_coarse_grid:     int   | None = None,
    clip_top_k_pct:       float | None = None,
    clip_fallback_thresh: float | None = None,
) -> None:
    """
    Switch to a new dataset or apply a new hyperparameter configuration.

    All keyword args are optional overrides. When None, the value is loaded
    from SRF_ARCH_PARAMS[model_id] and SRF_DATASET_PARAMS[dataset].

    head_top_k_pct triggers re-calibration (expensive); all other params are cheap.

    Example — sweep layer_end:
        for le in [12, 14, 15, 17]:
            srf.reset_for_dataset("pope", layer_end=le)
            run_eval(...)
    """
    global _noun_mode, BIAS, SALIENCY

    # Map datasets to noun-extraction mode (mme/hallusionbench use POPE-style Yes/No questions)
    _NOUN_MODE_MAP = {"mme": "pope", "hallusionbench": "pope"}
    _noun_mode = _NOUN_MODE_MAP.get(dataset, dataset)
    overrides  = {
        "phase":                phase,
        "alpha":                alpha,
        "eps":                  eps,
        "layer_start":          layer_start,
        "layer_end":            layer_end,
        "head_top_k_pct":       head_top_k_pct,
        "clip_coarse_grid":     clip_coarse_grid,
        "clip_top_k_pct":       clip_top_k_pct,
        "clip_fallback_thresh": clip_fallback_thresh,
    }

    BIAS     = _make_bias(dataset, overrides)
    SALIENCY = _make_saliency(overrides)

    print(f"\n  [SRF] reset → dataset={dataset}  "
          f"phase={BIAS['srf_apply_phase']}  alpha={BIAS['boost_alpha']}  "
          f"eps={BIAS['background_eps']}  "
          f"layers=[{BIAS['layer_start']},{BIAS['layer_end']}]  "
          f"head_topk={BIAS['head_top_k_pct']}  "
          f"clip_grid={SALIENCY['clip_coarse_grid']}  "
          f"clip_topk={SALIENCY['clip_top_k_pct']}")

    n    = CFG.SRF_DEFAULTS["calib_n"]
    seed = CFG.SRF_DEFAULTS["calib_seed"]
    calib_inputs, img_ranges = _build_calib_inputs(dataset, n=n, seed=seed)
    patch.identify_visual_heads(_model, calib_inputs, img_ranges, BIAS["head_top_k_pct"])
    del calib_inputs, img_ranges
    torch.cuda.empty_cache()
    _sync_patch_state()


def prepare_sample(inputs, img_start: int, img_end: int,
                   image, question: str, model, processor) -> None:
    """
    Per-sample setup: compute CLIP saliency and configure patch state.
    Must be called before every model forward pass.
    """
    patch.update_sample(img_start, img_end)
    patch._STATE["value"]             = BIAS["boost_alpha"]
    patch._STATE["method"]            = "srf"
    patch._STATE["srf_bias_mode"]     = BIAS["bias_mode"]
    patch._STATE["srf_interp_lambda"] = BIAS["interp_lambda"]
    patch._STATE["srf_prob_floor"]    = BIAS["prob_floor"]
    patch._STATE["srf_img_scale"]     = BIAS["img_scale"]
    patch._STATE["srf_text_beta"]     = BIAS["text_beta"]

    grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)
    noun   = extract_clip_noun(question, mode=_noun_mode)
    result = clip_sal.compute_clip_salience(
        image, noun, grid_h, grid_w,
        top_k_pct=SALIENCY["clip_top_k_pct"],
        coarse_n=SALIENCY["clip_coarse_grid"],
    )
    if result.max_sim < SALIENCY["clip_fallback_thresh"]:
        patch._STATE["salience_mask"] = None
    else:
        patch._STATE["salience_mask"] = (
            result.saliency if SALIENCY["clip_use_soft"] else result.mask
        )


def cleanup() -> None:
    """Reset per-sample patch state after inference."""
    patch._STATE["salience_mask"] = None
    patch._STATE["method"]        = "srf"
