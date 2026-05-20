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

# Dynamically import the correct patch module (will be set by eval.py)
patch = None

import clip_salience as clip_sal
from noun_extract import extract_clip_noun


def _apply_chat_template(processor, msgs: list, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
    """Apply chat template, handling both processor and tokenizer."""
    # For Qwen-VL, use format_qwen_msgs
    if hasattr(processor, 'from_list_format') and callable(processor.from_list_format):
        # Import format_qwen_msgs from eval.py
        import sys
        import os
        eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
        from eval import format_qwen_msgs
        text, _ = format_qwen_msgs(msgs, processor, _model)
        return text

    # For LLaVA with old transformers (no chat_template), use manual format
    if hasattr(processor, 'tokenizer'):
        tok = processor.tokenizer
        # Check if chat_template is None or missing (transformers < 4.40)
        if not hasattr(tok, 'chat_template') or tok.chat_template is None:
            # LLaVA-1.5 Vicuna format with intro
            user_content = []
            for msg in msgs:
                if msg["role"] == "user":
                    for item in msg["content"]:
                        if item["type"] == "image":
                            user_content.append("<image>")
                        elif item["type"] == "text":
                            user_content.append(item["text"])
            user_text = " ".join(user_content)
            prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {user_text} ASSISTANT:"
            return prompt

    # Standard chat template
    if hasattr(processor, 'apply_chat_template') and callable(processor.apply_chat_template):
        return processor.apply_chat_template(msgs, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
    elif hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'apply_chat_template'):
        return processor.tokenizer.apply_chat_template(msgs, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
    else:
        raise AttributeError(f"No apply_chat_template method found on processor {type(processor)}")
import config as CFG

# Global counter for saliency image saving (only save first N samples for debugging)
_SALIENCY_SAVE_COUNTER = 0
_MAX_SALIENCY_SAVES = 20


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
        "layer_start":           arch["layer_start"],
        "layer_end":             layer_end,
        "head_top_k_pct":        arch["head_top_k_pct"],
        "sys_beta":              d["sys_beta"],
        "text_beta":             d["text_beta"],
        "text_layer_start":      d["text_layer_start"],
        "text_layer_end":        d["text_layer_end"],
        "bias_mode":             d["bias_mode"],
        "boost_alpha":           dp["alpha"],
        "background_eps":        dp["eps"],
        "clip_suppress_thresh":  dp.get("clip_suppress_thresh", 0.0),
        "clip_suppress_alpha":   dp.get("clip_suppress_alpha", 5.0),
        "interp_lambda":         d["interp_lambda"],
        "prob_floor":            d["prob_floor"],
        "img_scale":             d["img_scale"],
        "srf_apply_phase":       dp["phase"],
    }

    # Apply CLI overrides
    if overrides.get("phase")                is not None: b["srf_apply_phase"]         = overrides["phase"]
    if overrides.get("alpha")                is not None: b["boost_alpha"]             = overrides["alpha"]
    if overrides.get("eps")                  is not None: b["background_eps"]          = overrides["eps"]
    if overrides.get("clip_suppress_thresh") is not None: b["clip_suppress_thresh"]    = overrides["clip_suppress_thresh"]
    if overrides.get("clip_suppress_alpha")  is not None: b["clip_suppress_alpha"]     = overrides["clip_suppress_alpha"]
    if overrides.get("layer_start")          is not None: b["layer_start"]             = overrides["layer_start"]
    if overrides.get("layer_end")            is not None: b["layer_end"]               = overrides["layer_end"]
    if overrides.get("head_top_k_pct")       is not None: b["head_top_k_pct"]          = overrides["head_top_k_pct"]

    return b


def _make_saliency(overrides: dict) -> dict:
    """
    Build the SALIENCY config dict.
    Sources: arch defaults → CLI overrides.
    """
    arch = _get_arch()
    s = {
        "clip_coarse_grid":       arch["clip_coarse_grid"],
        "clip_top_k_pct":         arch["clip_top_k_pct"],
        "clip_use_soft":          True,   # always soft — hard mask hurts boundary tokens
        "clip_fallback_thresh":   arch["clip_fallback_thresh"],
        "clip_upsample_to_tokens": False,  # disabled by default
    }
    if overrides.get("clip_coarse_grid")       is not None: s["clip_coarse_grid"]       = overrides["clip_coarse_grid"]
    if overrides.get("clip_top_k_pct")         is not None: s["clip_top_k_pct"]         = overrides["clip_top_k_pct"]
    if overrides.get("clip_fallback_thresh")   is not None: s["clip_fallback_thresh"]   = overrides["clip_fallback_thresh"]
    if overrides.get("clip_upsample_to_tokens") is not None: s["clip_upsample_to_tokens"] = overrides["clip_upsample_to_tokens"]
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

    # Import get_token_id helper from eval.py for consistent token ID handling
    import sys
    import os
    eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from eval import get_token_id

    # LLaVA-style: image_token is None → use model.config.image_token_index
    try:
        img_id = get_token_id(_processor, arch["image_token"] or "<|image_pad|>")
    except (ValueError, AttributeError):
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
            # Qwen-VL has no padding token - handle separately
            if hasattr(_processor, 'from_list_format') and callable(_processor.from_list_format):
                # Import format_qwen_msgs and replicate eval.py approach
                import sys
                import os
                eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if eval_dir not in sys.path:
                    sys.path.insert(0, eval_dir)
                from eval import format_qwen_msgs
                text, _ = format_qwen_msgs(msgs, _processor, _model)
                inp = _processor(text=text, return_tensors="pt", padding=False).to(device)
            else:
                text = _apply_chat_template(_processor, msgs, tokenize=False,
                                            add_generation_prompt=True)
                vis, _ = pvi(msgs)
                inp = _processor(text=[text], images=vis, return_tensors="pt",
                                 padding=True).to(device)
            ids = inp["input_ids"][0].tolist()
            # For Qwen-VL, image tokens might not be explicit in tokenized output
            # Use fallback: image region is after first few tokens
            if img_id in ids:
                s = ids.index(img_id)
                e = len(ids) - 1 - ids[::-1].index(img_id)
            else:
                # Qwen-VL: estimate image region (tokens 1-256 typically)
                s = 1
                e = min(256, len(ids) - 1)
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
            # Qwen-VL has no padding token - handle separately
            if hasattr(_processor, 'from_list_format') and callable(_processor.from_list_format):
                # Import format_qwen_msgs and replicate eval.py approach
                import sys
                import os
                eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if eval_dir not in sys.path:
                    sys.path.insert(0, eval_dir)
                from eval import format_qwen_msgs
                text, _ = format_qwen_msgs(msgs, _processor, _model)
                inp = _processor(text=text, return_tensors="pt", padding=False).to(device)
            else:
                text = _apply_chat_template(_processor, msgs, tokenize=False,
                                            add_generation_prompt=True)
                vis, _ = pvi(msgs)
                inp = _processor(text=[text], images=vis, return_tensors="pt",
                                 padding=True).to(device)
            ids = inp["input_ids"][0].tolist()
            # For Qwen-VL, image tokens might not be explicit
            if img_id in ids:
                s = ids.index(img_id)
                e = len(ids) - 1 - ids[::-1].index(img_id)
            else:
                s = 1
                e = min(256, len(ids) - 1)
            calib_inputs.append(inp); img_ranges.append((s, e))

    elif dataset == "vlmbias":
        ds   = hf_load("anvo25/vlms-are-biased", split="main")
        rows = list(ds); rng.shuffle(rows)
        for r in rows[:n]:
            img  = r["image"].convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text",  "text":  r["prompt"]}]}]
            # Qwen-VL has no padding token - handle separately
            if hasattr(_processor, 'from_list_format') and callable(_processor.from_list_format):
                # Import format_qwen_msgs and replicate eval.py approach
                import sys
                import os
                eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if eval_dir not in sys.path:
                    sys.path.insert(0, eval_dir)
                from eval import format_qwen_msgs
                text, _ = format_qwen_msgs(msgs, _processor, _model)
                inp = _processor(text=text, return_tensors="pt", padding=False).to(device)
            else:
                text = _apply_chat_template(_processor, msgs, tokenize=False,
                                            add_generation_prompt=True)
                vis, _ = pvi(msgs)
                inp = _processor(text=[text], images=vis, return_tensors="pt",
                                 padding=True).to(device)
            ids = inp["input_ids"][0].tolist()
            # For Qwen-VL, image tokens might not be explicit in tokenized output
            if img_id in ids:
                s = ids.index(img_id)
                e = len(ids) - 1 - ids[::-1].index(img_id)
            else:
                # Qwen-VL fallback: estimate image region
                s = 1
                e = min(256, len(ids) - 1)
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
            # Qwen-VL has no padding token - handle separately
            if hasattr(_processor, 'from_list_format') and callable(_processor.from_list_format):
                # Import format_qwen_msgs and replicate eval.py approach
                import sys
                import os
                eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if eval_dir not in sys.path:
                    sys.path.insert(0, eval_dir)
                from eval import format_qwen_msgs
                text, _ = format_qwen_msgs(msgs, _processor, _model)
                inp = _processor(text=text, return_tensors="pt", padding=False).to(device)
            else:
                text = _apply_chat_template(_processor, msgs, tokenize=False,
                                            add_generation_prompt=True)
                vis, _ = pvi(msgs)
                inp = _processor(text=[text], images=vis, return_tensors="pt",
                                 padding=True).to(device)
            ids = inp["input_ids"][0].tolist()
            # For Qwen-VL, image tokens might not be explicit in tokenized output
            if img_id in ids:
                s = ids.index(img_id)
                e = len(ids) - 1 - ids[::-1].index(img_id)
            else:
                # Qwen-VL fallback: estimate image region
                s = 1
                e = min(256, len(ids) - 1)
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
    # Handle models without vision_config (like Qwen-VL v1)
    arch = CFG.get_arch(_model_id)
    _spatial = getattr(model.config, "spatial_merge_size", arch.get("spatial_merge_size", 2))

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

    patch.patch_model(model, "srf", max(float(BIAS["boost_alpha"]), 1e-6))
    _sync_patch_state()


def reset_for_dataset(
    dataset: str = "pope",
    *,
    # dataset-specific tunables
    phase:    str   | None = None,
    alpha:    float | None = None,
    eps:      float | None = None,
    # text-token suppression
    text_beta:        float | None = None,
    text_layer_start: int   | None = None,
    text_layer_end:   int   | None = None,
    # arch/layer tunables (scale with model depth)
    layer_start:    int   | None = None,
    layer_end:      int   | None = None,
    head_top_k_pct: float | None = None,
    # CLIP saliency tunables
    clip_coarse_grid:     int   | None = None,
    clip_top_k_pct:       float | None = None,
    clip_fallback_thresh: float | None = None,
    # Absence-aware parameters
    clip_suppress_thresh:  float | None = None,
    clip_suppress_alpha:   float | None = None,
    # Upsampling option
    clip_upsample_to_tokens: bool | None = None,
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
        "text_beta":            text_beta,
        "text_layer_start":     text_layer_start,
        "text_layer_end":       text_layer_end,
        "layer_start":          layer_start,
        "layer_end":            layer_end,
        "head_top_k_pct":       head_top_k_pct,
        "clip_coarse_grid":     clip_coarse_grid,
        "clip_top_k_pct":       clip_top_k_pct,
        "clip_fallback_thresh": clip_fallback_thresh,
        "clip_suppress_thresh": clip_suppress_thresh,
        "clip_suppress_alpha":  clip_suppress_alpha,
        "clip_upsample_to_tokens": clip_upsample_to_tokens,
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

    Absence-aware strategy (key innovation):
      - If max_sim < clip_suppress_thresh → object likely absent → SUPPRESS all image tokens
      - If max_sim >= clip_suppress_thresh → object likely present → BOOST salient tokens
    """
    patch.update_sample(img_start, img_end)
    patch._STATE["method"]            = "srf"
    patch._STATE["srf_bias_mode"]     = BIAS["bias_mode"]
    patch._STATE["srf_interp_lambda"] = BIAS["interp_lambda"]
    patch._STATE["srf_prob_floor"]    = BIAS["prob_floor"]
    patch._STATE["srf_img_scale"]     = BIAS["img_scale"]
    patch._STATE["srf_text_beta"]     = BIAS["text_beta"]

    # Detect model type for grid dimensions
    model_type = "llava" if "llava" in _model_id.lower() else "qwen"
    grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial, model_type)
    noun   = extract_clip_noun(question, mode=_noun_mode)

    # Optional upsampling to match actual image token count
    target_n_tokens = None
    if SALIENCY.get("clip_upsample_to_tokens", False):
        target_n_tokens = img_end - img_start + 1

    result = clip_sal.compute_clip_salience(
        image, noun, grid_h, grid_w,
        top_k_pct=SALIENCY["clip_top_k_pct"],
        coarse_n=SALIENCY["clip_coarse_grid"],
        target_n_tokens=target_n_tokens,  # Pass token count for upsampling
    )

    # ========== DEBUG ==========
    print(f"[DEBUG SRF] max_sim: {result.max_sim:.4f}", flush=True)
    print(f"[DEBUG SRF] fallback_thresh: {SALIENCY['clip_fallback_thresh']:.4f}", flush=True)
    print(f"[DEBUG SRF] clip_top_k_pct: {SALIENCY['clip_top_k_pct']}", flush=True)
    print(f"[DEBUG SRF] clip_coarse_grid: {SALIENCY['clip_coarse_grid']}", flush=True)
    print(f"[DEBUG SRF] noun: {noun}", flush=True)
    # ========== END DEBUG ==========

    # Set saliency mask
    if result.max_sim < SALIENCY["clip_fallback_thresh"]:
        print(f"[DEBUG SRF] ❌ max_sim < thresh → salience_mask = None", flush=True)
        patch._STATE["salience_mask"] = None
    else:
        print(f"[DEBUG SRF] ✅ max_sim >= thresh → salience_mask set", flush=True)
        patch._STATE["salience_mask"] = (
            result.saliency if SALIENCY["clip_use_soft"] else result.mask
        )
        if patch._STATE["salience_mask"] is not None:
            mask = patch._STATE["salience_mask"]
            if torch.is_tensor(mask):
                print(f"[DEBUG SRF] mask: shape={mask.shape}, sum={mask.sum().item():.2f}, mean={mask.mean().item():.4f}", flush=True)
            else:
                print(f"[DEBUG SRF] mask: type={type(mask)}", flush=True)

            # ========== SAVE SALIENCY IMAGE FOR VISUAL INSPECTION ==========
            global _SALIENCY_SAVE_COUNTER, _MAX_SALIENCY_SAVES
            if _SALIENCY_SAVE_COUNTER < _MAX_SALIENCY_SAVES:
                try:
                    import os
                    import uuid
                    from PIL import Image
                    import numpy as np

                    # Create output directory (organize by dataset)
                    # Detect dataset from question content or use default
                    output_dir = "results/saliency_images_pope"  # Will change dynamically below
                    # Check if question contains counting keywords (VLM-Bias style)
                    question_lower = question.lower() if question else ""
                    if any(word in question_lower for word in ["how many", "count", "number of", "much"]):
                        output_dir = "results/saliency_images_vlmbias"
                    elif "butterfly" in question_lower or "arrow" in question_lower or "direction" in question_lower:
                        output_dir = "results/saliency_images_mmvp"
                    else:
                        output_dir = "results/saliency_images_pope"

                    os.makedirs(output_dir, exist_ok=True)

                    # Generate unique filename with counter
                    unique_id = uuid.uuid4().hex[:8]
                    filename = f"{output_dir}/saliency_{_SALIENCY_SAVE_COUNTER+1}_{noun.replace(' ', '_')}.png"

                    # Get original image size
                    orig_w, orig_h = image.size

                    # Convert mask to numpy
                    if torch.is_tensor(mask):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = np.array(mask)

                    # Normalize mask to 0-255
                    if mask_np.max() > 0:
                        mask_normalized = ((mask_np - mask_np.min()) / (mask_np.max() - mask_np.min()) * 255).astype(np.uint8)
                    else:
                        mask_normalized = np.zeros_like(mask_np, dtype=np.uint8)

                    # Create heatmap (same size as original image)
                    if len(mask_np.shape) == 1:
                        grid_size = int(np.sqrt(len(mask_np)))
                        if grid_size * grid_size == len(mask_np):
                            mask_2d = mask_normalized.reshape(grid_size, grid_size)
                        else:
                            grid_h = int(np.sqrt(len(mask_np) * (orig_h / orig_w)))
                            grid_w = len(mask_np) // grid_h
                            mask_2d = mask_normalized[:grid_h * grid_w].reshape(grid_h, grid_w)
                    else:
                        mask_2d = mask_normalized

                    # Resize to original image size
                    heatmap = Image.fromarray(mask_2d).resize((orig_w, orig_h), Image.NEAREST)

                    # Create overlay: blend original image with heatmap
                    original_rgb = image.convert("RGB")
                    heatmap_rgb = heatmap.convert("RGB")

                    # Use red channel for heatmap
                    heatmap_colored = Image.new("RGB", (orig_w, orig_h), (0, 0, 0))
                    for y in range(orig_h):
                        for x in range(orig_w):
                            intensity = heatmap.getpixel((x, y)) / 255.0
                            if intensity < 0.33:
                                r = int(intensity * 3 * 255)
                                g = 0
                                b = 0
                            elif intensity < 0.66:
                                r = 255
                                g = int((intensity - 0.33) * 3 * 255)
                                b = 0
                            else:
                                r = 255
                                g = 255
                                b = int((intensity - 0.66) * 3 * 255)
                            heatmap_colored.putpixel((x, y), (r, g, b))

                    # Blend: 50% original + 50% heatmap
                    overlay = Image.blend(original_rgb, heatmap_colored, 0.5)

                    # Save side-by-side comparison
                    comparison = Image.new("RGB", (orig_w * 3, orig_h))
                    comparison.paste(original_rgb, (0, 0))
                    comparison.paste(heatmap_rgb, (orig_w, 0))
                    comparison.paste(overlay, (orig_w * 2, 0))
                    comparison.save(filename)

                    _SALIENCY_SAVE_COUNTER += 1
                    print(f"[DEBUG SRF] 📸 Saved saliency visualization ({_SALIENCY_SAVE_COUNTER}/{_MAX_SALIENCY_SAVES}): {filename}", flush=True)
                except Exception as e:
                    print(f"[DEBUG SRF] ⚠️  Failed to save saliency image: {e}", flush=True)
            # ========== END SALIENCY SAVE ==========

    # Absence-aware conditional boost/suppress (key innovation)
    # When object likely absent → use LOW enh_para to suppress all image tokens
    # When object likely present → use HIGH enh_para to boost salient tokens
    suppress_thresh = BIAS.get("clip_suppress_thresh", 0.0)
    suppress_alpha  = BIAS.get("clip_suppress_alpha", 5.0)

    if suppress_thresh > 0.0:
        if result.max_sim < suppress_thresh:
            # Object likely absent → strong suppression (multiplier << 1.0)
            # Use 1.0 / (1.0 + suppress_alpha) to get a small positive value
            patch._STATE["enh_para"] = 1.0 / (1.0 + abs(suppress_alpha))
            print(f"[DEBUG SRF] Absence-aware: SUPPRESSING (enh_para={patch._STATE['enh_para']:.4f})", flush=True)
        else:
            # Object likely present → gentle boost
            patch._STATE["enh_para"] = abs(BIAS["boost_alpha"])
            print(f"[DEBUG SRF] Absence-aware: BOOSTING (enh_para={patch._STATE['enh_para']:.4f})", flush=True)
    else:
        # Absence-aware disabled → always boost
        patch._STATE["enh_para"] = BIAS["boost_alpha"]
        print(f"[DEBUG SRF] enh_para: {patch._STATE['enh_para']:.4f}", flush=True)


def cleanup() -> None:
    """Reset per-sample patch state after inference."""
    patch._STATE["salience_mask"] = None
    patch._STATE["method"]        = "srf"
