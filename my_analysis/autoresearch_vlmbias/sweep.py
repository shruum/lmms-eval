#!/usr/bin/env python3
"""
VLM Bias autoresearch sweep — AGENT MODIFIES EXPERIMENT LIST.

Loads Qwen2.5-VL-3B-Instruct once, runs all experiments, saves results.
Per-category breakdown is printed for every experiment.

Key diagnostic:
  --diagnose  Run per-sample logit shift analysis: for each sample, records
              logit of GT token and predicted token for baseline vs each exp.
              Tells us whether SRF is pushing in the correct direction.

Usage:
    cd /volumes2/mllm/lmms-eval
    conda run -n mllm env \\
        PYTHONPATH="/volumes2/mllm/lmms-eval/srf:/volumes2/mllm/lmms-eval/srf/saliency" \\
        python my_analysis/autoresearch_vlmbias/sweep.py 2>&1 | tee /tmp/vlmbias_sweep.log
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import re
import sys
import time
from copy import deepcopy
from typing import Any

os.environ.setdefault("HF_HOME", "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

_SWEEP_DIR   = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SWEEP_DIR.parent
sys.path.insert(0, str(_SWEEP_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

MODEL_ID       = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_TOKEN    = "<|image_pad|>"
N_PER_CATEGORY = 15
SEED           = 42
MAX_NEW_TOKENS = 20

CATEGORIES = [
    "Animals", "Chess Pieces", "Flags",
    "Game Boards", "Logos", "Optical Illusion", "Patterned Grid",
]

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# Each experiment is a dict with keys:
#   name, desc, SALIENCY (optional override), BIAS (optional override),
#   no_srf (bool, run without any SRF)
#   uniform_saliency (bool, skip CLIP — use uniform boost over all image tokens)

_SALIENCY_DEFAULT = {
    "clip_coarse_grid":     7,
    "clip_top_k_pct":       0.30,
    "clip_use_soft":        True,
    "clip_fallback_thresh": 0.20,
}

_BIAS_DEFAULT = {
    "layer_start":      8,
    "layer_end":        14,
    "head_top_k_pct":   0.20,
    "sys_beta":         0.10,
    "text_beta":        0.0,
    "text_layer_start": 20,
    "text_layer_end":   27,
    "bias_mode":        "additive_logit",
    "boost_alpha":      8.0,
    "background_eps":   0.5,
    "interp_lambda":    1.0,
    "prob_floor":       0.005,
    "img_scale":        1.5,
    "srf_apply_phase":  "generation",
}

EXPERIMENTS = [
    # -----------------------------------------------------------------------
    # Anchors
    # -----------------------------------------------------------------------
    {
        "name": "e00_no_srf",
        "desc": "Baseline: raw model, no SRF",
        "no_srf": True,
    },
    {
        "name": "e01_current_best",
        "desc": "Current best: alpha=8 eps=0.5 phase=generation layers 8-14",
        # defaults are the current best
    },

    # -----------------------------------------------------------------------
    # Phase sweep — does prefill matter for counting?
    # -----------------------------------------------------------------------
    {
        "name": "e02_phase_both",
        "desc": "phase=both (boost at prefill AND generation)",
        "BIAS": {"srf_apply_phase": "both"},
    },
    {
        "name": "e03_phase_prefill",
        "desc": "phase=prefill only",
        "BIAS": {"srf_apply_phase": "prefill"},
    },

    # -----------------------------------------------------------------------
    # CLIP guidance vs uniform boost — does saliency map matter?
    # -----------------------------------------------------------------------
    {
        "name": "e04_uniform_boost",
        "desc": "Uniform boost: no CLIP mask, boost ALL image tokens equally",
        "uniform_saliency": True,
    },
    {
        "name": "e05_top80pct",
        "desc": "CLIP top 80% tokens (nearly all image tokens boosted)",
        "SALIENCY": {"clip_top_k_pct": 0.80, "clip_fallback_thresh": 0.0},
    },
    {
        "name": "e06_top10pct",
        "desc": "CLIP top 10% tokens (very focused boost)",
        "SALIENCY": {"clip_top_k_pct": 0.10},
    },

    # -----------------------------------------------------------------------
    # Alpha sweep — does strength matter for counting?
    # -----------------------------------------------------------------------
    {
        "name": "e07_alpha_2",
        "desc": "alpha=2.0 (weak boost)",
        "BIAS": {"boost_alpha": 2.0},
    },
    {
        "name": "e08_alpha_16",
        "desc": "alpha=16.0 (strong boost)",
        "BIAS": {"boost_alpha": 16.0},
    },
    {
        "name": "e09_alpha_32",
        "desc": "alpha=32.0 (very strong boost)",
        "BIAS": {"boost_alpha": 32.0},
    },

    # -----------------------------------------------------------------------
    # Layer range — do deeper (reasoning) layers matter?
    # -----------------------------------------------------------------------
    {
        "name": "e10_deep_layers",
        "desc": "Deep layers 20-28 (reasoning/language-prior zone)",
        "BIAS": {"layer_start": 20, "layer_end": 28},
    },
    {
        "name": "e11_mid_layers",
        "desc": "Mid layers 14-22",
        "BIAS": {"layer_start": 14, "layer_end": 22},
    },
    {
        "name": "e12_all_layers",
        "desc": "All layers 0-35",
        "BIAS": {"layer_start": 0, "layer_end": 35},
    },

    # -----------------------------------------------------------------------
    # Combined: uniform boost + deep layers
    # (Test if bypassing CLIP and going deep helps counting)
    # -----------------------------------------------------------------------
    {
        "name": "e13_uniform_deep",
        "desc": "Uniform boost + deep layers 20-28",
        "uniform_saliency": True,
        "BIAS": {"layer_start": 20, "layer_end": 28, "boost_alpha": 8.0},
    },

    # -----------------------------------------------------------------------
    # Per-category diagnostic: stronger boost on counting categories
    # (uniform + very high alpha to see if any count correction is possible)
    # -----------------------------------------------------------------------
    {
        "name": "e14_uniform_alpha32_deep",
        "desc": "Uniform boost alpha=32 deep layers 20-28",
        "uniform_saliency": True,
        "BIAS": {"layer_start": 20, "layer_end": 28, "boost_alpha": 32.0},
    },
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_vlmbias(n_per_cat: int = N_PER_CATEGORY, seed: int = SEED) -> list[dict]:
    from datasets import load_dataset as hf_load
    from collections import defaultdict

    print(f"  [harness] Loading VLM Bias (n={n_per_cat}/cat, seed={seed})…")
    ds = hf_load("anvo25/vlms-are-biased", split="main")

    by_cat: dict = defaultdict(list)
    for r in ds:
        by_cat[r["topic"]].append(r)

    samples = []
    rng = random.Random(seed)
    for cat in CATEGORIES:
        rows = by_cat.get(cat, [])
        rng.shuffle(rows)
        for r in rows[:n_per_cat]:
            samples.append({
                "image":    r["image"].convert("RGB"),
                "question": r["prompt"],
                "gt":       _normalise(str(r["ground_truth"])),
                "category": cat,
            })
    rng.shuffle(samples)
    print(f"  [harness] {len(samples)} samples ({len(CATEGORIES)} categories)")
    return samples


def _normalise(s: str) -> str:
    return s.strip().lower().lstrip("{").rstrip("}")


def _extract_answer(text: str) -> str:
    m = re.search(r'\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()
    return text.strip().split()[0] if text.strip() else ""


def _get_img_range(input_ids, img_token_id: int):
    ids   = input_ids[0].tolist()
    start = next(i for i, t in enumerate(ids) if t == img_token_id)
    end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return start, end


# ---------------------------------------------------------------------------
# SRF setup helpers (mirrors srf.py but parameterised)
# ---------------------------------------------------------------------------

def _setup_srf(model, processor, bias: dict, saliency: dict,
               no_srf: bool, uniform_saliency: bool):
    """Patch model with SRF params; calibrate heads; return (patch, clip_sal, noun_extract)."""
    import qwen_attn_patch as patch
    import clip_salience   as clip_sal
    from noun_extract import extract_clip_noun

    if no_srf:
        import torch.nn.functional as F
        import torch
        # Unpatch if previously patched
        try:
            patch.unpatch_model()
        except Exception:
            pass
        patch._STATE["method"]        = "baseline"
        patch._STATE["salience_mask"] = None
        return patch, clip_sal, extract_clip_noun

    # Calibrate heads (always on VLMbias samples)
    _calibrate_heads(model, processor, bias, patch)

    patch.patch_model(model, "vaf", max(float(bias["boost_alpha"]), 1e-6))
    patch._STATE["vaf_layer_start"]       = bias["layer_start"]
    patch._STATE["vaf_layer_end"]         = bias["layer_end"]
    patch._STATE["vaf_beta"]              = bias["sys_beta"]
    patch._STATE["srf_background_eps"]    = bias["background_eps"]
    patch._STATE["srf_bias_mode"]         = bias["bias_mode"]
    patch._STATE["srf_interp_lambda"]     = bias["interp_lambda"]
    patch._STATE["srf_prob_floor"]        = bias["prob_floor"]
    patch._STATE["srf_img_scale"]         = bias["img_scale"]
    patch._STATE["srf_apply_phase"]       = bias["srf_apply_phase"]
    patch._STATE["srf_text_beta"]         = bias["text_beta"]
    patch._STATE["srf_text_layer_start"]  = bias["text_layer_start"]
    patch._STATE["srf_text_layer_end"]    = bias["text_layer_end"]

    return patch, clip_sal, extract_clip_noun


def _calibrate_heads(model, processor, bias: dict, patch):
    import torch
    from datasets import load_dataset as hf_load
    from qwen_vl_utils import process_vision_info as pvi

    if bias["head_top_k_pct"] <= 0.0:
        patch._STATE["head_mask"] = None
        return

    ds   = hf_load("anvo25/vlms-are-biased", split="main")
    rows = list(ds)
    random.Random(0).shuffle(rows)
    img_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    device = next(model.parameters()).device

    calib_inputs, img_ranges = [], []
    for r in rows[:20]:
        img  = r["image"].convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                              {"type": "text",  "text":  r["prompt"]}]}]
        text   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = pvi(msgs)
        inp    = processor(text=[text], images=vis, return_tensors="pt", padding=True).to(device)
        ids    = inp["input_ids"][0].tolist()
        calib_inputs.append(inp)
        img_ranges.append((ids.index(img_id), len(ids)-1-ids[::-1].index(img_id)))

    patch.identify_visual_heads(model, calib_inputs, img_ranges, bias["head_top_k_pct"])
    n_sel = int(patch._STATE["head_mask"].sum().item())
    print(f"  [srf] {n_sel} vision-aware heads (top {bias['head_top_k_pct']*100:.0f}%)")
    del calib_inputs, img_ranges
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Single experiment eval
# ---------------------------------------------------------------------------

def _eval_one(model, processor, samples, img_token_id, device, exp: dict) -> dict:
    import torch
    from qwen_vl_utils import process_vision_info

    name   = exp["name"]
    no_srf = exp.get("no_srf", False)
    unif   = exp.get("uniform_saliency", False)

    # Build effective config by merging defaults + overrides
    bias     = {**_BIAS_DEFAULT,     **exp.get("BIAS", {})}
    saliency = {**_SALIENCY_DEFAULT, **exp.get("SALIENCY", {})}

    patch, clip_sal, extract_clip_noun = _setup_srf(
        model, processor, bias, saliency, no_srf, unif
    )

    correct   = 0
    cat_stats = {c: {"correct": 0, "total": 0} for c in CATEGORIES}
    results   = []
    t0        = time.time()

    for i, s in enumerate(samples):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": s["image"]},
            {"type": "text",  "text":  s["question"]},
        ]}]
        text      = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_in, _ = process_vision_info(msgs)
        inputs    = processor(
            text=[text], images=img_in, return_tensors="pt", padding=True
        ).to(device)

        img_start, img_end = _get_img_range(inputs["input_ids"], img_token_id)

        if not no_srf:
            patch.update_sample(img_start, img_end)
            patch._STATE["value"]         = bias["boost_alpha"]
            patch._STATE["method"]        = "srf"
            patch._STATE["srf_bias_mode"] = bias["bias_mode"]

            if unif:
                # Uniform boost: no CLIP mask → all image tokens boosted equally
                patch._STATE["salience_mask"] = None
            else:
                import clip_salience as clip_sal_mod
                grid_h, grid_w = clip_sal_mod.get_grid_dims(
                    inputs, getattr(model.config.vision_config, "spatial_merge_size", 2)
                )
                noun   = extract_clip_noun(s["question"], mode="vlmbias")
                result = clip_sal_mod.compute_clip_salience(
                    s["image"], noun, grid_h, grid_w,
                    top_k_pct=saliency["clip_top_k_pct"],
                    coarse_n=saliency["clip_coarse_grid"],
                )
                if result.max_sim < saliency["clip_fallback_thresh"]:
                    patch._STATE["salience_mask"] = None
                else:
                    patch._STATE["salience_mask"] = (
                        result.saliency if saliency["clip_use_soft"] else result.mask
                    )

        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

        # Cleanup patch state
        if not no_srf:
            patch._STATE["salience_mask"] = None
            patch._STATE["method"]        = "vaf"

        raw  = processor.decode(
            out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        pred = _normalise(_extract_answer(raw))
        ok   = (pred == s["gt"])

        if ok:
            correct += 1
        cat_stats[s["category"]]["correct"] += int(ok)
        cat_stats[s["category"]]["total"]   += 1

        results.append({
            "gt": s["gt"], "pred": pred, "correct": ok,
            "category": s["category"], "response": raw,
        })

    n_total = len(samples)
    acc     = correct / n_total
    elapsed = int(time.time() - t0)

    return {"name": name, "acc": acc, "n": n_total,
            "cat_stats": cat_stats, "samples": results, "t": elapsed}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def _merge_bias(override: dict) -> dict:
    return {**_BIAS_DEFAULT, **override}


def run(args):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    exps = EXPERIMENTS
    if args.exp:
        exps = [e for e in exps if e["name"] in args.exp]

    results_dir = _SWEEP_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path  = results_dir / "sweep.csv"

    # Load existing results for skip_existing
    done = set()
    if args.skip_existing and csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add(row["name"])
        exps = [e for e in exps if e["name"] not in done]

    print(f"[sweep] {len(exps)} experiments to run (skip_existing={args.skip_existing})\n")

    print(f"[sweep] Loading {MODEL_ID}…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager",
    ).eval()
    processor    = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=512 * 28 * 28)
    img_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    device       = next(model.parameters()).device

    samples = load_vlmbias()

    all_rows = []

    for idx, exp in enumerate(exps):
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"[sweep] ({idx+1}/{len(exps)})  {exp['name']}  |  {exp['desc']}")
        print(sep)

        result = _eval_one(model, processor, samples, img_token_id, device, exp)

        acc = result["acc"]
        t   = result["t"]

        # Per-category printout
        print(f"\n  Per-category breakdown:")
        for cat in CATEGORIES:
            st      = result["cat_stats"][cat]
            cat_acc = st["correct"] / st["total"] if st["total"] else 0.0
            bar     = "#" * st["correct"] + "." * (st["total"] - st["correct"])
            print(f"    {cat:20s}  {cat_acc:.2f}  [{bar}]  {st['correct']}/{st['total']}")

        print(f"\nRESULT  acc={acc:.4f}  n={result['n']}  exp={exp['name']}  t={t}s")

        row = {
            "name":    exp["name"],
            "desc":    exp["desc"],
            "acc":     f"{acc:.4f}",
            "t":       t,
            **{cat: result["cat_stats"][cat]["correct"]
               for cat in CATEGORIES},
        }
        all_rows.append(row)

        # Append to CSV incrementally
        write_header = not csv_path.exists() or (idx == 0 and not args.skip_existing)
        with open(csv_path, "a", newline="") as f:
            fieldnames = ["name", "desc", "acc", "t"] + CATEGORIES
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                f.seek(0, 2)
                if f.tell() == 0:
                    w.writeheader()
            w.writerow(row)

        # Save per-experiment JSON
        json_path = results_dir / f"{exp['name']}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        torch.cuda.empty_cache()

    # Final summary table
    print(f"\n{'=' * 70}")
    print(f"SWEEP SUMMARY  ({len(all_rows)} experiments)")
    print(f"{'=' * 70}")
    hdr = f"{'Exp':<32}  {'Acc':>6}  {'Anim':>4}  {'Chess':>5}  {'Flags':>5}  {'GameB':>5}  {'Logos':>5}  {'OI':>4}  {'PGrid':>5}  {'t':>6}"
    print(hdr)
    print("-" * len(hdr))
    for row in sorted(all_rows, key=lambda r: -float(r["acc"])):
        cats = [row.get(c, "-") for c in CATEGORIES]
        print(f"{row['name']:<32}  {row['acc']:>6}  "
              + "  ".join(f"{str(c):>5}" for c in cats)
              + f"  {row['t']:>5}s")

    print(f"\n[sweep] Results written to {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp",            nargs="*", default=None,
                   help="Run only specific experiment names")
    p.add_argument("--skip_existing",  action="store_true", default=True)
    p.add_argument("--no_skip",        dest="skip_existing", action="store_false")
    args = p.parse_args()
    run(args)
