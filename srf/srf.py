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
import hssa_salience as hssa_sal
import lta_salience as lta_sal
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

# Last CLIP result — set by prepare_sample, read by callers for B4 retry logic.
# Fields: object_present (bool), full_img_sim (float), saliency (tensor|None)
last_clip_result: dict = {}  # empty if saliency_mode has no CLIP gate or noun was bad


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
        "boost_alpha":       dp["alpha"],
        "background_eps":    dp["eps"],
        "neg_absent_alpha":  dp.get("neg_absent_alpha", 0.0),
        "interp_lambda":     d["interp_lambda"],
        "prob_floor":        d["prob_floor"],
        "img_scale":         d["img_scale"],
        "srf_apply_phase":   dp["phase"],
    }

    # Apply CLI overrides
    if overrides.get("phase")            is not None: b["srf_apply_phase"]   = overrides["phase"]
    if overrides.get("alpha")            is not None: b["boost_alpha"]        = overrides["alpha"]
    if overrides.get("eps")             is not None: b["background_eps"]     = overrides["eps"]
    if overrides.get("neg_absent_alpha") is not None: b["neg_absent_alpha"]  = overrides["neg_absent_alpha"]
    if overrides.get("layer_start")     is not None: b["layer_start"]        = overrides["layer_start"]
    if overrides.get("layer_end")       is not None: b["layer_end"]          = overrides["layer_end"]
    if overrides.get("head_top_k_pct")  is not None: b["head_top_k_pct"]    = overrides["head_top_k_pct"]
    if overrides.get("bias_mode")        is not None: b["bias_mode"]          = overrides["bias_mode"]
    if overrides.get("interp_lambda")   is not None: b["interp_lambda"]      = overrides["interp_lambda"]
    if overrides.get("vr_target")       is not None: b["vr_target"]          = overrides["vr_target"]
    if overrides.get("vr_k")            is not None: b["vr_k"]               = overrides["vr_k"]

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
        "clip_model":           arch.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL),
        "saliency_mode":        arch.get("saliency_mode", "clip"),
        "hssa_layer_idx":       arch.get("hssa_layer_idx", 12),
        "clip_saliency_method": arch.get("clip_saliency_method", "clip_patch"),
        "srf2_clip_weight":     arch.get("srf2_clip_weight", 0.7),
        "srf2_hssa_weight":     arch.get("srf2_hssa_weight", 0.3),
        "lta_layer_idx":        arch.get("lta_layer_idx", -1),
        "lta_weight":           arch.get("lta_weight", 0.6),
        "clip_weight":          arch.get("clip_weight", 0.4),
    }
    if overrides.get("clip_coarse_grid")     is not None: s["clip_coarse_grid"]     = overrides["clip_coarse_grid"]
    if overrides.get("clip_top_k_pct")       is not None: s["clip_top_k_pct"]       = overrides["clip_top_k_pct"]
    if overrides.get("clip_fallback_thresh") is not None: s["clip_fallback_thresh"] = overrides["clip_fallback_thresh"]
    if overrides.get("clip_model")           is not None: s["clip_model"]           = overrides["clip_model"]
    if overrides.get("saliency_mode")        is not None: s["saliency_mode"]        = overrides["saliency_mode"]
    if overrides.get("hssa_layer_idx")       is not None: s["hssa_layer_idx"]       = overrides["hssa_layer_idx"]
    if overrides.get("clip_saliency_method") is not None: s["clip_saliency_method"] = overrides["clip_saliency_method"]
    if overrides.get("srf2_clip_weight")     is not None: s["srf2_clip_weight"]     = overrides["srf2_clip_weight"]
    if overrides.get("srf2_hssa_weight")     is not None: s["srf2_hssa_weight"]     = overrides["srf2_hssa_weight"]
    if overrides.get("lta_layer_idx")        is not None: s["lta_layer_idx"]        = overrides["lta_layer_idx"]
    if overrides.get("lta_weight")           is not None: s["lta_weight"]           = overrides["lta_weight"]
    if overrides.get("clip_weight")          is not None: s["clip_weight"]          = overrides["clip_weight"]
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

    elif dataset in ("mmbench", "hallusionbench"):
        # MMBench / HallusionBench — use MMBench validation images for calibration
        ds   = hf_load("HuggingFaceM4/MMBench", split="validation")
        rows = list(ds); rng.shuffle(rows)
        for r in rows[:n]:
            hint = str(r.get("hint", "")).strip()
            q    = str(r["question"]).strip()
            opts = "\n".join(
                f"{letter}. {r[letter]}"
                for letter in ("A", "B", "C", "D")
                if r.get(letter) and str(r[letter]).strip() not in ("None", "")
            )
            prompt = (f"{hint}\n{q}\n{opts}" if hint and hint != "None"
                      else f"{q}\n{opts}") + "\nAnswer with A, B, C, or D only."
            img  = r["image"].convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text",  "text": prompt}]}]
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
                         f"Supported: pope, mmvp, vlmbias, mme, hallusionbench, mmbench")

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
    patch._STATE["srf_vr_target"]        = BIAS.get("vr_target", 0.0)
    patch._STATE["srf_vr_k"]            = BIAS.get("vr_k", 3.0)


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
    phase:             str   | None = None,
    alpha:             float | None = None,
    eps:               float | None = None,
    neg_absent_alpha:  float | None = None,
    # arch/layer tunables (scale with model depth)
    layer_start:    int   | None = None,
    layer_end:      int   | None = None,
    head_top_k_pct: float | None = None,
    # CLIP/SigLIP saliency tunables
    clip_coarse_grid:     int   | None = None,
    clip_top_k_pct:       float | None = None,
    clip_fallback_thresh: float | None = None,
    clip_model:           str   | None = None,
    saliency_mode:        str   | None = None,
    hssa_layer_idx:       int   | None = None,
    # GradCAM / SRF2 method selection
    clip_saliency_method: str   | None = None,
    srf2_clip_weight:     float | None = None,
    srf2_hssa_weight:     float | None = None,
    # LTA / clip_lta params
    lta_layer_idx:        int   | None = None,
    lta_weight:           float | None = None,
    clip_weight:          float | None = None,
    # Bias mode + related
    bias_mode:            str   | None = None,
    interp_lambda:        float | None = None,
    # B1: visual reliance compensation
    vr_target:            float | None = None,
    vr_k:                 float | None = None,
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

    # Map datasets to noun-extraction mode
    _NOUN_MODE_MAP = {
        "mme":            "pope",        # Yes/No existence questions
        "hallusionbench": "mmbench",     # Visual reasoning questions
        "mmbench":        "mmbench",     # MCQ questions
    }
    _noun_mode = _NOUN_MODE_MAP.get(dataset, dataset)
    overrides  = {
        "phase":                phase,
        "alpha":                alpha,
        "eps":                  eps,
        "neg_absent_alpha":     neg_absent_alpha,
        "layer_start":          layer_start,
        "layer_end":            layer_end,
        "head_top_k_pct":       head_top_k_pct,
        "clip_coarse_grid":     clip_coarse_grid,
        "clip_top_k_pct":       clip_top_k_pct,
        "clip_fallback_thresh": clip_fallback_thresh,
        "clip_model":           clip_model,
        "saliency_mode":        saliency_mode,
        "hssa_layer_idx":       hssa_layer_idx,
        "clip_saliency_method": clip_saliency_method,
        "srf2_clip_weight":     srf2_clip_weight,
        "srf2_hssa_weight":     srf2_hssa_weight,
        "lta_layer_idx":        lta_layer_idx,
        "lta_weight":           lta_weight,
        "clip_weight":          clip_weight,
        "bias_mode":            bias_mode,
        "interp_lambda":        interp_lambda,
        "vr_target":            vr_target,
        "vr_k":                 vr_k,
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
    Per-sample setup: compute saliency and configure patch state.
    Saliency source is controlled by SALIENCY["saliency_mode"]:
      "clip" — CLIP ViT-B/32 patch similarity (default, no extra forward pass)
      "hssa" — Qwen hidden-state cosine similarity (extra forward pass, better alignment)
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

    sal_mode   = SALIENCY.get("saliency_mode", "clip")
    sal_method = SALIENCY.get("clip_saliency_method", "clip_patch")

    # ── Bad-noun gate ─────────────────────────────────────────────────────────
    # If noun extraction returns a function/stop word (e.g. "this", "does",
    # "according", "only"), CLIP saliency is meaningless and actively harmful.
    # Fall back to a no-op (method=baseline) for this sample.
    _BAD_NOUNS = {
        "this", "that", "does", "only", "answer", "appropriate", "according",
        "object", "image", "picture", "taken", "from", "which", "some", "with",
        "right", "left", "true", "false", "sequence", "above", "below",
        "correct", "incorrect", "whether", "what", "where", "when", "have",
        "there", "their", "more", "also", "both", "each", "than",
    }
    global last_clip_result
    last_clip_result = {}   # reset each sample
    _test_noun = extract_clip_noun(question, mode=_noun_mode)
    if _test_noun in _BAD_NOUNS or len(_test_noun) <= 2:
        patch._STATE["method"] = "baseline"
        return

    if sal_mode == "hssa":
        # Hidden-State Semantic Alignment: cosine similarity between image token
        # hidden states and question-word hidden states at a middle decoder layer.
        # Run the HSSA forward pass in baseline mode to avoid circular dependency.
        patch._STATE["method"] = "baseline"
        hssa_result = hssa_sal.compute_hssa_salience(
            model, inputs, img_start, img_end,
            layer_idx=SALIENCY.get("hssa_layer_idx", 12),
            top_k_pct=SALIENCY["clip_top_k_pct"],
        )
        patch._STATE["method"]        = "srf"
        patch._STATE["salience_mask"] = hssa_result.saliency

    elif sal_mode == "lta":
        # Last-Token Attention: attention from the last input token → image tokens.
        # Run a baseline forward to avoid circular dependency with SRF hooks.
        patch._STATE["method"] = "baseline"
        lta_result = lta_sal.compute_lta_salience(
            model, inputs, img_start, img_end,
            layer_idx=SALIENCY.get("lta_layer_idx", -1),
            head_mask=patch._STATE.get("head_mask"),
            top_k_pct=SALIENCY["clip_top_k_pct"],
        )
        patch._STATE["method"]        = "srf"
        patch._STATE["salience_mask"] = lta_result.saliency

    elif sal_mode == "clip_lta":
        # Combine LTA (spatial localization) with CLIP (presence gate).
        # saliency = w_lta * lta_map + w_clip * clip_map ; gate = CLIP max_sim
        patch._STATE["method"] = "baseline"
        lta_result = lta_sal.compute_lta_salience(
            model, inputs, img_start, img_end,
            layer_idx=SALIENCY.get("lta_layer_idx", -1),
            head_mask=patch._STATE.get("head_mask"),
            top_k_pct=SALIENCY["clip_top_k_pct"],
        )
        patch._STATE["method"] = "srf"

        grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)
        noun   = extract_clip_noun(question, mode=_noun_mode)
        clip_result = clip_sal.compute_clip_salience(
            image, noun, grid_h, grid_w,
            top_k_pct=SALIENCY["clip_top_k_pct"],
            coarse_n=SALIENCY["clip_coarse_grid"],
            clip_model_name=SALIENCY.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL),
        )
        w_l = SALIENCY.get("lta_weight", 0.6)
        w_c = SALIENCY.get("clip_weight", 0.4)
        combined = w_l * lta_result.saliency + w_c * clip_result.saliency
        c_min, c_max = combined.min(), combined.max()
        combined = (combined - c_min) / (c_max - c_min + 1e-8)

        # CLIP max_sim gates presence/absence (LTA has no absence signal)
        object_present = clip_result.max_sim >= SALIENCY["clip_fallback_thresh"]
        patch._STATE["salience_mask"] = combined if object_present else None

    elif sal_mode == "clip_full_gate":
        # Multi-scale CLIP with hard presence gate (3-signal combined).
        # When object is absent → uniform 0.5; when present → spatial localization.
        grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)
        noun   = extract_clip_noun(question, mode=_noun_mode)
        result = clip_sal.compute_clip_salience_multiscale_full_gate(
            image, noun, grid_h, grid_w,
            top_k_pct=SALIENCY["clip_top_k_pct"],
            clip_model_name=SALIENCY.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL),
        )
        if result.object_present:
            clip_conf = min(result.full_img_sim / clip_sal._FULL_IMG_ABSENCE_THRESH.get(
                SALIENCY.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL), 0.24), 1.0)
            patch._STATE["value"]         = BIAS["boost_alpha"] * clip_conf
            patch._STATE["salience_mask"] = result.saliency
        else:
            patch._STATE["salience_mask"] = None
            patch._STATE["value"]         = -BIAS.get("neg_absent_alpha", 0.0)

    elif sal_mode == "clip_full_gate_v3_ramp":
        # Per-layer alpha ramp: Gaussian peak at the midpoint of [layer_start, layer_end].
        # Layers near the middle get full boost_alpha; edge layers get less.
        # sigma = range/4 so ±2σ spans the full layer range.
        import math as _math
        grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)
        noun   = extract_clip_noun(question, mode=_noun_mode)
        result = clip_sal.compute_clip_salience_full_gate_v3(
            image, noun, grid_h, grid_w,
            top_k_pct=SALIENCY["clip_top_k_pct"],
            clip_model_name=SALIENCY.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL),
            backup="none",
        )
        if result.object_present:
            ls   = BIAS["layer_start"]
            le   = BIAS["layer_end"]
            mid  = (ls + le) / 2.0
            sig  = max((le - ls) / 4.0, 0.5)
            base = BIAS["boost_alpha"]
            layer_alphas = {
                l: base * _math.exp(-0.5 * ((l - mid) / sig) ** 2)
                for l in range(ls, le + 1)
            }
            patch._STATE["value"]            = base
            patch._STATE["salience_mask"]    = result.saliency
            patch._STATE["srf_layer_alphas"] = layer_alphas
        else:
            patch._STATE["salience_mask"]    = None
            patch._STATE["value"]            = -BIAS.get("neg_absent_alpha", 0.0)
            patch._STATE["srf_layer_alphas"] = None

    elif sal_mode in ("clip_full_gate_v3", "clip_full_gate_v3_iou",
                      "clip_full_gate_v3_adaptive"):
        # v3: tuned threshold (0.21), no gate_patch fallback, single GPU block.
        # clip_full_gate_v3          → gate_full only, capped confidence (alpha * min(conf,1))
        # clip_full_gate_v3_iou      → + cross_scale_iou backup
        # clip_full_gate_v3_adaptive → uncapped confidence (alpha * conf, no ceiling)
        #                              strongly-present objects get proportionally higher boost
        backup = "cross_scale" if sal_mode == "clip_full_gate_v3_iou" else "none"
        grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)
        noun   = extract_clip_noun(question, mode=_noun_mode)
        result = clip_sal.compute_clip_salience_full_gate_v3(
            image, noun, grid_h, grid_w,
            top_k_pct=SALIENCY["clip_top_k_pct"],
            clip_model_name=SALIENCY.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL),
            backup=backup,
        )
        last_clip_result["object_present"] = result.object_present
        last_clip_result["full_img_sim"]   = result.full_img_sim
        last_clip_result["saliency"]       = result.saliency
        if result.object_present:
            raw_conf  = result.full_img_sim / clip_sal._FULL_IMG_THRESH_V3
            clip_conf = raw_conf if sal_mode == "clip_full_gate_v3_adaptive" else min(raw_conf, 1.0)
            patch._STATE["value"]         = BIAS["boost_alpha"] * clip_conf
            patch._STATE["salience_mask"] = result.saliency
        else:
            patch._STATE["salience_mask"] = None
            patch._STATE["value"]         = -BIAS.get("neg_absent_alpha", 0.0)

    elif sal_mode == "clip_full_gate_v3_dynhead":
        # Experiment 3D — Dynamic per-sample head selection.
        # One extra baseline forward pass identifies which heads are most visually
        # active for THIS specific image+question pair, rather than relying on a
        # global calibration-set average.
        # Cost: ~1× extra forward pass vs. clip_full_gate_v3.
        patch.identify_visual_heads(
            model, [inputs], [(img_start, img_end)],
            BIAS["head_top_k_pct"],
        )
        # identify_visual_heads sets method="baseline" internally; restore
        patch._STATE["method"] = "srf"

        grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)
        noun   = extract_clip_noun(question, mode=_noun_mode)
        result = clip_sal.compute_clip_salience_full_gate_v3(
            image, noun, grid_h, grid_w,
            top_k_pct=SALIENCY["clip_top_k_pct"],
            clip_model_name=SALIENCY.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL),
            backup="none",
        )
        if result.object_present:
            raw_conf  = result.full_img_sim / clip_sal._FULL_IMG_THRESH_V3
            clip_conf = min(raw_conf, 1.0)
            patch._STATE["value"]         = BIAS["boost_alpha"] * clip_conf
            patch._STATE["salience_mask"] = result.saliency
        else:
            patch._STATE["salience_mask"] = None
            patch._STATE["value"]         = -BIAS.get("neg_absent_alpha", 0.0)

    elif sal_mode == "clip_full_gate_v3_dynlayer":
        # Experiment 3E — Dynamic per-sample layer selection.
        # One extra baseline forward pass measures per-layer visual attention,
        # then selects the contiguous range covering layers with above-average
        # attention for this specific sample.
        # Cost: ~1× extra forward pass vs. clip_full_gate_v3.
        patch._STATE["method"]                = "baseline"
        patch._STATE["_calibrate_layer_attn"] = True
        patch._STATE["_calib_layer_acc"]      = {}
        patch._STATE["_calib_layer_count"]    = 0
        patch.update_sample(img_start, img_end)
        with torch.inference_mode():
            model(**inputs)
        patch._STATE["_calibrate_layer_attn"] = False
        patch._STATE["method"]                = "srf"

        layer_scores = patch._STATE["_calib_layer_acc"]
        if layer_scores:
            mean_score   = sum(layer_scores.values()) / len(layer_scores)
            active       = [l for l, s in layer_scores.items() if s > mean_score]
            # Clamp to global calibrated range
            active       = [l for l in active
                            if BIAS["layer_start"] <= l <= BIAS["layer_end"]]
            if active:
                dyn_ls = min(active)
                dyn_le = max(active)
            else:
                dyn_ls = BIAS["layer_start"]
                dyn_le = BIAS["layer_end"]
        else:
            dyn_ls = BIAS["layer_start"]
            dyn_le = BIAS["layer_end"]

        patch._STATE["vaf_layer_start"] = dyn_ls
        patch._STATE["vaf_layer_end"]   = dyn_le

        grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)
        noun   = extract_clip_noun(question, mode=_noun_mode)
        result = clip_sal.compute_clip_salience_full_gate_v3(
            image, noun, grid_h, grid_w,
            top_k_pct=SALIENCY["clip_top_k_pct"],
            clip_model_name=SALIENCY.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL),
            backup="none",
        )
        if result.object_present:
            raw_conf  = result.full_img_sim / clip_sal._FULL_IMG_THRESH_V3
            clip_conf = min(raw_conf, 1.0)
            patch._STATE["value"]         = BIAS["boost_alpha"] * clip_conf
            patch._STATE["salience_mask"] = result.saliency
        else:
            patch._STATE["salience_mask"] = None
            patch._STATE["value"]         = -BIAS.get("neg_absent_alpha", 0.0)

    elif sal_mode == "clip_soft_gate":
        # Multi-scale CLIP with soft presence gate.
        # Blends spatial and uniform by w = min(full_img_sim / thresh, 1).
        grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)
        noun   = extract_clip_noun(question, mode=_noun_mode)
        result = clip_sal.compute_clip_salience_soft_gate(
            image, noun, grid_h, grid_w,
            top_k_pct=SALIENCY["clip_top_k_pct"],
            clip_model_name=SALIENCY.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL),
        )
        patch._STATE["salience_mask"] = result.saliency

    elif sal_method == "clip_gradcam":
        # GradCAM: full image → CLIP → cosine sim → backprop → spatial map
        grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)
        noun   = extract_clip_noun(question, mode=_noun_mode)
        result = clip_sal.compute_clip_gradcam(
            image, noun, grid_h, grid_w,
            top_k_pct=SALIENCY["clip_top_k_pct"],
            clip_model_name=SALIENCY.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL),
        )
        patch._STATE["salience_mask"] = result.saliency if result.object_present else None

    elif sal_method == "srf2":
        # SRF2: 0.7 * GradCAM saliency + 0.3 * HSSA saliency
        grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)
        noun   = extract_clip_noun(question, mode=_noun_mode)
        gradcam = clip_sal.compute_clip_gradcam(
            image, noun, grid_h, grid_w,
            top_k_pct=SALIENCY["clip_top_k_pct"],
            clip_model_name=SALIENCY.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL),
        )
        # HSSA in baseline mode to avoid circular dependency
        patch._STATE["method"] = "baseline"
        hssa_result = hssa_sal.compute_hssa_salience(
            model, inputs, img_start, img_end,
            layer_idx=SALIENCY.get("hssa_layer_idx", 12),
            top_k_pct=SALIENCY["clip_top_k_pct"],
        )
        patch._STATE["method"] = "srf"
        w_c      = SALIENCY.get("srf2_clip_weight", 0.7)
        w_h      = SALIENCY.get("srf2_hssa_weight", 0.3)
        combined = w_c * gradcam.saliency + w_h * hssa_result.saliency
        c_min, c_max = combined.min(), combined.max()
        combined = (combined - c_min) / (c_max - c_min + 1e-8)
        patch._STATE["salience_mask"] = combined if gradcam.object_present else None

    else:
        # Default: CLIP/SigLIP patch similarity (clip_patch, backward-compatible)
        grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)
        noun   = extract_clip_noun(question, mode=_noun_mode)
        result = clip_sal.compute_clip_salience(
            image, noun, grid_h, grid_w,
            top_k_pct=SALIENCY["clip_top_k_pct"],
            coarse_n=SALIENCY["clip_coarse_grid"],
            clip_model_name=SALIENCY.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL),
        )
        if result.max_sim < SALIENCY["clip_fallback_thresh"]:
            patch._STATE["salience_mask"] = None
            patch._STATE["value"]         = -BIAS.get("neg_absent_alpha", 0.0)
        else:
            clip_conf = min(result.max_sim / SALIENCY["clip_fallback_thresh"], 1.0)
            patch._STATE["value"]         = BIAS["boost_alpha"] * clip_conf
            patch._STATE["salience_mask"] = (
                result.saliency if SALIENCY["clip_use_soft"] else result.mask
            )


def cleanup() -> None:
    """Reset per-sample patch state after inference."""
    patch._STATE["salience_mask"]   = None
    patch._STATE["method"]          = "srf"
    # Restore layer range in case dynlayer mode modified it per-sample
    if BIAS:
        patch._STATE["vaf_layer_start"] = BIAS["layer_start"]
        patch._STATE["vaf_layer_end"]   = BIAS["layer_end"]
