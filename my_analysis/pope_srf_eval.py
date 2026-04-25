#!/usr/bin/env python3
"""
POPE full evaluation with SRF (Semantic Re-Focus) — best settings from autoresearch.

Runs all three POPE splits (adversarial / popular / random) × N samples each.
Supports Qwen2.5-VL and LLaVA-1.5 architectures.

Best SRF settings (empirically determined, 100+ autoresearch experiments):
  - Stage 1: CLIP ViT-B/32, 7×7 coarse grid, soft top-30% saliency
  - Stage 1: Absence-aware: suppress when CLIP max_sim < thresh, boost when ≥ thresh
  - Stage 2: additive_logit, layers 8-14 (Qwen) / 8-15 (LLaVA), top-20% vision heads
  - Stage 2: boost_alpha=2.0, suppress_alpha=5.0, prefill-only, background_eps=0.0

Usage
-----
# Qwen2.5-VL-3B (default), full 500 per split
python pope_srf_eval.py --arch qwen

# Qwen2.5-VL-7B
python pope_srf_eval.py --arch qwen --model Qwen/Qwen2.5-VL-7B-Instruct

# LLaVA-1.5-7B
python pope_srf_eval.py --arch llava

# LLaVA-1.5-13B
python pope_srf_eval.py --arch llava --model llava-hf/llava-1.5-13b-hf

# Quick sanity check (50 per split)
python pope_srf_eval.py --arch qwen --n_samples 50

# Baseline only (no SRF)
python pope_srf_eval.py --arch qwen --mode baseline

# Both baseline and SRF
python pope_srf_eval.py --arch qwen --mode both
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import sys
import time
from typing import List, Tuple

import torch

os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

ANALYSIS_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR / "autoresearch"))

import qwen_attn_patch as patch
from srf.saliency import clip_salience as clip_sal

# ---------------------------------------------------------------------------
# Architecture defaults
# ---------------------------------------------------------------------------

ARCH_DEFAULTS = {
    "qwen": {
        "model_id":          "Qwen/Qwen2.5-VL-3B-Instruct",
        "layer_start":        8,
        "layer_end":          14,   # empirically optimal for Qwen2.5-VL-3B
        "spatial_merge":      2,    # Qwen's spatial_merge_size
    },
    "llava": {
        "model_id":          "llava-hf/llava-1.5-7b-hf",
        "layer_start":        8,
        "layer_end":          15,   # ClearSight (CVPR25) finding for LLaVA-1.5
        "spatial_merge":      1,    # LLaVA has no spatial merging
    },
}

# SRF hyperparameters — fixed globally (from autoresearch)
SRF = {
    "clip_coarse_grid":    7,
    "clip_top_k_pct":      0.30,
    "clip_use_soft":       True,
    "boost_alpha":         2.0,    # present-object boost (wide plateau 1.5–5.0)
    "suppress_alpha":      5.0,    # absent-object suppress (narrow optimum)
    "head_top_k_pct":      0.20,   # auto-calibrated per model
    "sys_beta":            0.10,   # system-prompt suppression
    "background_eps":      0.0,    # must be 0.0
    "n_calib_samples":     20,     # for both head mask and suppress threshold
    "calib_seed":          0,      # separate from eval seed so calib ≠ test
}

EVAL_SEED      = 42
SPLITS         = ["adversarial", "popular", "random"]
MAX_NEW_TOKENS = 10


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def extract_noun(question: str) -> str:
    """Extract queried noun from POPE-style question."""
    for pat in [
        r"is there an?\s+([\w\s]+?)\s+in\s+the\s+image",
        r"do you see an?\s+([\w\s]+?)\s+in\s+the\s+image",
        r"is there an?\s+([\w\s]+?)[\?\.!\n]",
    ]:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            return m.group(1).lower().strip()
    return question.split("\n")[0].split("?")[0].strip()


def pope_f1(results: List[dict]) -> dict:
    """Compute POPE accuracy, precision, recall, F1."""
    tp = sum(1 for r in results if r["gt"] == "yes" and r["pred"] == "yes")
    fp = sum(1 for r in results if r["gt"] == "no"  and r["pred"] == "yes")
    tn = sum(1 for r in results if r["gt"] == "no"  and r["pred"] == "no")
    fn = sum(1 for r in results if r["gt"] == "yes" and r["pred"] == "no")
    n  = len(results)

    acc  = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    yes_rate = (tp + fp) / n if n else 0.0

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "yes_rate": yes_rate, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_pope_split(split: str, n: int, seed: int = EVAL_SEED) -> List[dict]:
    from datasets import load_dataset as hf_load
    print(f"  Loading POPE {split} (n={n}, seed={seed})…")
    ds   = hf_load("lmms-lab/POPE", split="test")
    rows = [r for r in ds
            if str(r.get("category", r.get("type", ""))).strip().lower() == split]
    rng  = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    samples = []
    for r in rows:
        gt = "yes" if str(r.get("answer", "")).strip().lower() == "yes" else "no"
        q  = str(r.get("question", "")).strip() + "\nAnswer with Yes or No only."
        samples.append({"image": r["image"].convert("RGB"), "question": q, "gt": gt})
    print(f"  → {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_qwen(model_id: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print(f"  Loading {model_id} (bfloat16, eager)…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def load_llava(model_id: str):
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    print(f"  Loading {model_id} (float16, eager)…")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16,
        device_map="auto", attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


# ---------------------------------------------------------------------------
# Image-token range
# ---------------------------------------------------------------------------

def get_img_range_qwen(input_ids: torch.Tensor, img_token_id: int) -> Tuple[int, int]:
    ids   = input_ids[0].tolist()
    start = next(i for i, t in enumerate(ids) if t == img_token_id)
    end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return start, end


def get_img_range_llava(inputs: dict, model) -> Tuple[int, int]:
    """LLaVA has 1 placeholder token that expands to n_img_tokens inside the LM."""
    vis_cfg  = model.config.vision_config
    n_img    = (vis_cfg.image_size // vis_cfg.patch_size) ** 2   # e.g. 576
    ids      = inputs["input_ids"][0].tolist()
    ph_idx   = model.config.image_token_index                    # 32000
    pos      = ids.index(ph_idx)
    return pos, pos + n_img - 1


# ---------------------------------------------------------------------------
# Grid dimensions for CLIP saliency
# ---------------------------------------------------------------------------

def get_grid_dims_qwen(inputs: dict, spatial_merge: int) -> Tuple[int, int]:
    thw    = inputs["image_grid_thw"][0]
    grid_h = int(thw[1].item()) // spatial_merge
    grid_w = int(thw[2].item()) // spatial_merge
    return grid_h, grid_w


def get_grid_dims_llava(model) -> Tuple[int, int]:
    vis_cfg = model.config.vision_config
    n_side  = vis_cfg.image_size // vis_cfg.patch_size   # 24 for ViT-L/14-336
    return n_side, n_side


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def build_inputs_qwen(sample: dict, model, processor) -> dict:
    from qwen_vl_utils import process_vision_info
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": sample["image"]},
        {"type": "text",  "text":  sample["question"]},
    ]}]
    text      = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    vis, _    = process_vision_info(msgs)
    inputs    = processor(text=[text], images=vis, return_tensors="pt", padding=True)
    return inputs.to(next(model.parameters()).device)


def build_inputs_llava(sample: dict, model, processor) -> dict:
    conv = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": sample["question"]},
    ]}]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(images=sample["image"], text=prompt, return_tensors="pt")
    return inputs.to(next(model.parameters()).device)


# ---------------------------------------------------------------------------
# Suppress threshold calibration
# ---------------------------------------------------------------------------

def calibrate_suppress_thresh(
    samples_calib: List[dict],
    arch: str,
    model,
    processor,
    build_inputs_fn,
    get_grid_fn,
    coarse_n: int = 7,
) -> float:
    """
    Auto-calibrate the CLIP absence threshold from labeled calibration samples.
    Threshold = midpoint of mean(max_sim|present) and mean(max_sim|absent).
    For always-present datasets, returns 0.0 (disable suppression).
    """
    sims_present, sims_absent = [], []
    print(f"  Calibrating suppress threshold on {len(samples_calib)} samples…")

    for s in samples_calib:
        inputs  = build_inputs_fn(s, model, processor)
        grid_h, grid_w = get_grid_fn(inputs) if arch == "qwen" else get_grid_fn()
        noun    = extract_noun(s["question"])
        result  = clip_sal.compute_clip_salience(
            s["image"], noun, grid_h, grid_w,
            top_k_pct=SRF["clip_top_k_pct"], coarse_n=coarse_n,
        )
        if s["gt"] == "yes":
            sims_present.append(result.max_sim)
        else:
            sims_absent.append(result.max_sim)

    if not sims_present or not sims_absent:
        print("  WARNING: calibration lacks one class — using default 0.248")
        return 0.248

    mean_p = sum(sims_present) / len(sims_present)
    mean_a = sum(sims_absent)  / len(sims_absent)
    thresh = (mean_p + mean_a) / 2.0
    print(f"  present max_sim mean={mean_p:.4f}  absent mean={mean_a:.4f}  → thresh={thresh:.4f}")
    return thresh


# ---------------------------------------------------------------------------
# Head calibration
# ---------------------------------------------------------------------------

def calibrate_heads(
    samples_calib: List[dict],
    arch: str,
    model,
    processor,
    build_inputs_fn,
    get_img_range_fn,
) -> None:
    """Calibrate vision-aware head mask via identify_visual_heads()."""
    print(f"  Calibrating vision-aware heads on {len(samples_calib)} samples…")
    calib_inputs, img_ranges = [], []
    for s in samples_calib:
        inp = build_inputs_fn(s, model, processor)
        if arch == "qwen":
            img_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            s_pos, e_pos = get_img_range_fn(inp["input_ids"], img_token_id)
        else:
            s_pos, e_pos = get_img_range_fn(inp, model)
        calib_inputs.append(inp)
        img_ranges.append((s_pos, e_pos))

    patch.identify_visual_heads(model, calib_inputs, img_ranges, SRF["head_top_k_pct"])
    n_sel = int(patch._STATE["head_mask"].sum().item())
    print(f"  {n_sel} vision heads selected (top {SRF['head_top_k_pct']*100:.0f}%)")


# ---------------------------------------------------------------------------
# Single-sample SRF inference
# ---------------------------------------------------------------------------

def run_sample_srf(
    sample: dict,
    arch: str,
    model,
    processor,
    build_inputs_fn,
    get_img_range_fn,
    get_grid_fn,
    layer_start: int,
    layer_end: int,
    spatial_merge: int,
    suppress_thresh: float,
) -> str:
    """Run one sample with SRF intervention. Returns predicted token ('yes'/'no')."""
    inputs = build_inputs_fn(sample, model, processor)

    if arch == "qwen":
        img_token_id      = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        img_start, img_end = get_img_range_fn(inputs["input_ids"], img_token_id)
        grid_h, grid_w    = get_grid_fn(inputs)
    else:
        img_start, img_end = get_img_range_fn(inputs, model)
        grid_h, grid_w    = get_grid_fn()

    # Stage 1: CLIP saliency
    noun   = extract_noun(sample["question"])
    result = clip_sal.compute_clip_salience(
        sample["image"], noun, grid_h, grid_w,
        top_k_pct=SRF["clip_top_k_pct"],
        coarse_n=SRF["clip_coarse_grid"],
    )
    sal = result.saliency if SRF["clip_use_soft"] else result.mask

    # Stage 1: Absence-aware direction
    if suppress_thresh > 0.0 and result.max_sim < suppress_thresh:
        alpha = -abs(SRF["suppress_alpha"])   # suppress: absent object
        sal   = torch.ones_like(sal)          # uniform suppress (absent mask)
    else:
        alpha = abs(SRF["boost_alpha"])        # boost: present object

    # Stage 2: Arm patch
    patch.update_sample(img_start, img_end)
    patch._STATE["salience_mask"]     = sal
    patch._STATE["value"]             = alpha
    patch._STATE["method"]            = "srf"
    patch._STATE["srf_bias_mode"]     = "additive_logit"
    patch._STATE["srf_background_eps"] = SRF["background_eps"]
    patch._STATE["srf_apply_phase"]   = "prefill"
    patch._STATE["vaf_layer_start"]   = layer_start
    patch._STATE["vaf_layer_end"]     = layer_end
    patch._STATE["vaf_beta"]          = SRF["sys_beta"]

    with torch.inference_mode():
        gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        if arch == "llava":
            gen_kwargs.update(temperature=None, top_p=None, num_beams=1)
        out_ids = model.generate(**inputs, **gen_kwargs)

    # Reset state
    patch._STATE["salience_mask"] = None
    patch._STATE["method"]        = "vaf"

    pred_str = processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip().lower()

    return "yes" if pred_str.startswith("yes") else "no"


def run_sample_baseline(
    sample: dict,
    arch: str,
    model,
    processor,
    build_inputs_fn,
) -> str:
    patch._STATE["method"] = "baseline"
    inputs = build_inputs_fn(sample, model, processor)

    with torch.inference_mode():
        gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        if arch == "llava":
            gen_kwargs.update(temperature=None, top_p=None, num_beams=1)
        out_ids = model.generate(**inputs, **gen_kwargs)

    pred_str = processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip().lower()

    return "yes" if pred_str.startswith("yes") else "no"


# ---------------------------------------------------------------------------
# Full eval loop
# ---------------------------------------------------------------------------

def eval_split(
    split: str,
    n_samples: int,
    mode: str,          # "baseline" | "srf" | "both"
    arch: str,
    model,
    processor,
    build_inputs_fn,
    get_img_range_fn,
    get_grid_fn,
    layer_start: int,
    layer_end: int,
    spatial_merge: int,
    suppress_thresh: float,
) -> dict:
    samples = load_pope_split(split, n_samples)
    results = {"baseline": [], "srf": []}
    t0 = time.time()

    for i, s in enumerate(samples):
        if mode in ("baseline", "both"):
            pred_b = run_sample_baseline(s, arch, model, processor, build_inputs_fn)
            results["baseline"].append({"gt": s["gt"], "pred": pred_b,
                                         "correct": pred_b == s["gt"]})

        if mode in ("srf", "both"):
            pred_s = run_sample_srf(
                s, arch, model, processor,
                build_inputs_fn, get_img_range_fn, get_grid_fn,
                layer_start, layer_end, spatial_merge, suppress_thresh,
            )
            results["srf"].append({"gt": s["gt"], "pred": pred_s,
                                    "correct": pred_s == s["gt"]})

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (n_samples - i - 1)
            parts   = []
            if results["baseline"]:
                acc_b = sum(r["correct"] for r in results["baseline"]) / len(results["baseline"])
                parts.append(f"baseline={acc_b:.4f}")
            if results["srf"]:
                acc_s = sum(r["correct"] for r in results["srf"]) / len(results["srf"])
                parts.append(f"srf={acc_s:.4f}")
            print(f"  [{split} {i+1:3d}/{n_samples}]  {' | '.join(parts)}  ETA={eta/60:.1f}min")

    metrics = {}
    for m_name, recs in results.items():
        if recs:
            metrics[m_name] = pope_f1(recs)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="POPE SRF full evaluation")
    parser.add_argument("--arch",      choices=["qwen", "llava"], default="qwen")
    parser.add_argument("--model",     type=str, default=None,
                        help="Override default model ID for the chosen arch")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Samples per split (max 500; default=500 = full POPE)")
    parser.add_argument("--mode",      choices=["baseline", "srf", "both"], default="both",
                        help="Which methods to run")
    parser.add_argument("--output",    type=str, default=None,
                        help="Path for JSON results (default: pope_srf_results_<arch>.json)")
    args = parser.parse_args()

    arch     = args.arch
    cfg      = ARCH_DEFAULTS[arch]
    model_id = args.model or cfg["model_id"]
    out_path = args.output or f"pope_srf_results_{arch}.json"

    print(f"\n{'='*60}")
    print(f"  POPE SRF Evaluation")
    print(f"  arch={arch}  model={model_id}")
    print(f"  mode={args.mode}  n_per_split={args.n_samples}")
    print(f"  SRF: layers {cfg['layer_start']}-{cfg['layer_end']}, "
          f"head_top_k={SRF['head_top_k_pct']}, "
          f"boost={SRF['boost_alpha']}, suppress={SRF['suppress_alpha']}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if arch == "qwen":
        model, processor = load_qwen(model_id)
    else:
        model, processor = load_llava(model_id)

    # ------------------------------------------------------------------
    # Architecture-specific function handles
    # ------------------------------------------------------------------
    if arch == "qwen":
        img_token_id   = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        spatial_merge  = getattr(model.config.vision_config, "spatial_merge_size", 2)

        def build_inputs(s, m, p):    return build_inputs_qwen(s, m, p)
        def get_img_range(inp, *_):   return get_img_range_qwen(inp, img_token_id)
        def get_grid(inp=None):
            if inp is not None:
                return get_grid_dims_qwen(inp, spatial_merge)
            return (15, 15)  # fallback
    else:
        spatial_merge = 1

        def build_inputs(s, m, p):    return build_inputs_llava(s, m, p)
        def get_img_range(inp, m):    return get_img_range_llava(inp, m)
        def get_grid(inp=None):       return get_grid_dims_llava(model)

    layer_start = cfg["layer_start"]
    layer_end   = cfg["layer_end"]

    # ------------------------------------------------------------------
    # One-time calibration (head mask + suppress threshold)
    # ------------------------------------------------------------------
    if args.mode in ("srf", "both"):
        print("\n── Calibration ──────────────────────────────────────────────")
        # Load calibration samples from adversarial split (balanced yes/no)
        calib_all   = load_pope_split("adversarial", 40, seed=SRF["calib_seed"])
        calib_yes   = [s for s in calib_all if s["gt"] == "yes"][:SRF["n_calib_samples"]//2]
        calib_no    = [s for s in calib_all if s["gt"] == "no"] [:SRF["n_calib_samples"]//2]
        calib_samps = calib_yes + calib_no
        random.Random(SRF["calib_seed"]).shuffle(calib_samps)

        # Register patch (needed for head calibration)
        patch.patch_model(model, "vaf", SRF["boost_alpha"])
        patch._STATE["vaf_layer_start"]    = layer_start
        patch._STATE["vaf_layer_end"]      = layer_end
        patch._STATE["vaf_beta"]           = SRF["sys_beta"]
        patch._STATE["srf_background_eps"] = SRF["background_eps"]
        patch._STATE["srf_bias_mode"]      = "additive_logit"
        patch._STATE["srf_apply_phase"]    = "prefill"

        # Head calibration
        if arch == "qwen":
            calibrate_heads(calib_samps, arch, model, processor,
                            build_inputs, get_img_range_qwen)
        else:
            calibrate_heads(calib_samps, arch, model, processor,
                            build_inputs, get_img_range_llava)

        # Suppress threshold calibration
        if arch == "qwen":
            def _get_grid_calib(inp):
                return get_grid_dims_qwen(inp, spatial_merge)
            suppress_thresh = calibrate_suppress_thresh(
                calib_samps, arch, model, processor,
                build_inputs_qwen,
                _get_grid_calib,
                coarse_n=SRF["clip_coarse_grid"],
            )
        else:
            def _get_grid_calib_llava():
                return get_grid_dims_llava(model)
            suppress_thresh = calibrate_suppress_thresh(
                calib_samps, "llava", model, processor,
                build_inputs_llava,
                _get_grid_calib_llava,
                coarse_n=SRF["clip_coarse_grid"],
            )

        print(f"  suppress_thresh = {suppress_thresh:.4f}")
    else:
        suppress_thresh = 0.248   # unused in baseline-only mode
        patch.patch_model(model, "baseline", 1.0)

    # ------------------------------------------------------------------
    # Evaluate all splits
    # ------------------------------------------------------------------
    all_metrics = {}
    all_results = {}

    for split in SPLITS:
        print(f"\n── {split.upper()} ──────────────────────────────────────────")
        metrics = eval_split(
            split, args.n_samples, args.mode,
            arch, model, processor,
            build_inputs,
            (get_img_range_qwen if arch == "qwen" else get_img_range_llava),
            (lambda inp: get_grid_dims_qwen(inp, spatial_merge)) if arch == "qwen" else (lambda: get_grid_dims_llava(model)),
            layer_start, layer_end, spatial_merge, suppress_thresh,
        )
        all_metrics[split] = metrics

        for m_name, m_vals in metrics.items():
            print(f"  {split:12s} [{m_name:8s}]  "
                  f"acc={m_vals['accuracy']:.4f}  "
                  f"f1={m_vals['f1']:.4f}  "
                  f"yes_rate={m_vals['yes_rate']:.3f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS  ({arch} | {model_id})")
    print(f"  n_per_split={args.n_samples}  suppress_thresh={suppress_thresh:.4f}")
    print(f"{'='*60}")

    for method in ["baseline", "srf"]:
        split_accs = []
        split_f1s  = []
        print(f"\n  [{method.upper()}]")
        print(f"  {'split':12s}  {'acc':>6}  {'f1':>6}  {'prec':>6}  {'rec':>6}  {'yes%':>6}")
        print(f"  {'-'*55}")
        for split in SPLITS:
            m = all_metrics.get(split, {}).get(method)
            if m:
                split_accs.append(m["accuracy"])
                split_f1s.append(m["f1"])
                print(f"  {split:12s}  "
                      f"{m['accuracy']:>6.4f}  {m['f1']:>6.4f}  "
                      f"{m['precision']:>6.4f}  {m['recall']:>6.4f}  "
                      f"{m['yes_rate']:>6.3f}")

        if split_accs:
            print(f"  {'AVERAGE':12s}  "
                  f"{sum(split_accs)/len(split_accs):>6.4f}  "
                  f"{sum(split_f1s)/len(split_f1s):>6.4f}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    output = {
        "model":            model_id,
        "arch":             arch,
        "n_per_split":      args.n_samples,
        "suppress_thresh":  suppress_thresh,
        "srf_config":       {**SRF, "layer_start": layer_start, "layer_end": layer_end},
        "metrics":          all_metrics,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
