#!/usr/bin/env python3
"""
SRF Ablation Study — full 2×2×2 factorial + baseline.

Three binary factors:
  saliency : clip  | rand  (CLIP-guided vs random k tokens, same k)
  heads    : cal   | rand  (calibrated top-20% vs random mask, same cardinality)
  layers   : tuned | rand  (tuned [8,15] vs random range, same width)

Runs 9 conditions in one GPU pass (model loaded once):
  baseline                       — no intervention
  clip_cal_tuned                 — full SRF (ours)
  rand_sal_cal_tuned             — random saliency, correct heads+layers
  clip_rand_heads_tuned          — CLIP saliency, random heads, tuned layers
  clip_cal_rand_layers           — CLIP saliency, calibrated heads, random layers
  rand_sal_rand_heads_tuned      — random saliency+heads, tuned layers
  rand_sal_cal_rand_layers       — random saliency+layers, calibrated heads
  clip_rand_heads_rand_layers    — CLIP saliency, random heads+layers
  rand_sal_rand_heads_rand_layers — everything random (naive random boost)

Usage
-----
  cd /volumes2/mllm/lmms-eval

  # POPE adversarial, all samples
  conda run -n mllm python srf/eval_ablation.py \\
      --dataset pope --pope_split adversarial

  # VLM Bias, all samples
  conda run -n mllm python srf/eval_ablation.py --dataset vlmbias

  # Select specific variants + save results
  conda run -n mllm python srf/eval_ablation.py \\
      --dataset pope --variants baseline clip_cal_tuned rand_sal_cal_tuned \\
      --output results/ablation/
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datasets import load_dataset as hf_load

import qwen_attn_patch as patch
import srf as srf_mod


# ---------------------------------------------------------------------------
# Variant definitions: each variant = (saliency, heads, layers)
# saliency : "clip" | "rand"
# heads    : "cal"  | "rand"
# layers   : "tuned"| "rand"
# ---------------------------------------------------------------------------

ALL_VARIANTS = [
    "baseline",
    "uniform_boost",                    # boost all tokens equally — null spatial hypothesis
    "clip_cal_tuned",                   # full SRF (ours)
    "rand_sal_cal_tuned",               # random saliency, correct heads+layers
    "clip_rand_heads_tuned",            # CLIP, random heads, tuned layers
    "clip_cal_rand_layers",             # CLIP, calibrated heads, random layers
    "rand_sal_rand_heads_tuned",        # random saliency+heads, tuned layers
    "rand_sal_cal_rand_layers",         # random saliency+layers, calibrated heads
    "clip_rand_heads_rand_layers",      # CLIP, random heads+layers
    "rand_sal_rand_heads_rand_layers",  # everything random (naive boost)
]

VARIANT_LABELS = {
    "baseline":                         "Baseline (no SRF)",
    "uniform_boost":                    "Uniform boost (all tokens equally)",
    "clip_cal_tuned":                   "SRF: CLIP + cal.heads + tuned layers  [OURS]",
    "rand_sal_cal_tuned":               "SRF: rand.sal + cal.heads + tuned layers",
    "clip_rand_heads_tuned":            "SRF: CLIP + rand.heads + tuned layers",
    "clip_cal_rand_layers":             "SRF: CLIP + cal.heads + rand.layers",
    "rand_sal_rand_heads_tuned":        "SRF: rand.sal + rand.heads + tuned layers",
    "rand_sal_cal_rand_layers":         "SRF: rand.sal + cal.heads + rand.layers",
    "clip_rand_heads_rand_layers":      "SRF: CLIP + rand.heads + rand.layers",
    "rand_sal_rand_heads_rand_layers":  "SRF: rand.sal + rand.heads + rand.layers (naive)",
}

# Paper-friendly 6-row subset that tells the clearest ablation story:
#   baseline → uniform_boost → naive_random → rand_sal_correct_hl → clip_correct_hl → full_SRF
PAPER_VARIANTS = [
    "baseline",
    "uniform_boost",                    # does ANY image boost help?
    "rand_sal_rand_heads_rand_layers",  # random everything — naive boost floor
    "rand_sal_cal_tuned",               # correct heads+layers, wrong saliency — isolates CLIP
    "clip_rand_heads_rand_layers",      # correct CLIP, wrong heads+layers — isolates selectivity
    "clip_cal_tuned",                   # full SRF (ours) — ceiling
]


def _variant_flags(variant: str) -> Tuple[str, str, str]:
    """Return (saliency, heads, layers) flags for a variant."""
    if variant in ("baseline", "uniform_boost"):
        return (variant, variant, variant)
    sal    = "clip"    if variant.startswith("clip") else "rand"
    heads  = "rand"    if "rand_heads" in variant     else "cal"
    layers = "tuned"   if variant.endswith("tuned")   else "rand"
    return sal, heads, layers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SRF 2x2x2 component ablation study.")
    p.add_argument("--model",       default=CFG.DEFAULT_MODEL)
    p.add_argument("--dataset",     required=True, choices=["pope", "vlmbias"])
    p.add_argument("--pope_split",  default="adversarial",
                   choices=["adversarial", "popular", "random"])
    p.add_argument("--n",           type=int, default=-1,
                   help="Number of samples (-1 = all).")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output",      default=None)
    p.add_argument("--variants",    nargs="+", default=PAPER_VARIANTS,
                   choices=ALL_VARIANTS,
                   help="Variants to run (default: paper-friendly 5-row subset).")
    p.add_argument("--all_variants", action="store_true",
                   help="Run all 9 variants (overrides --variants).")
    p.add_argument("--clip_model",  default=None,
                   help="Override CLIP/SigLIP model (e.g. google/siglip-base-patch16-224).")
    p.add_argument("--saliency_mode", default=None,
                   choices=["clip", "hssa", "lta", "clip_lta", "clip_full_gate", "clip_soft_gate"],
                   help="Saliency source override: 'clip', 'hssa', 'lta', 'clip_lta', 'clip_full_gate', or 'clip_soft_gate'.")
    p.add_argument("--clip_method", default=None,
                   choices=["clip_patch", "clip_gradcam", "srf2"],
                   help="CLIP saliency method: clip_patch (default), clip_gradcam, or srf2.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    print(f"Loading {model_id}…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id,
                                              max_pixels=CFG.DEFAULT_MAX_PIXELS)
    return model, processor


def get_img_range(input_ids: torch.Tensor, img_token_id: int) -> Tuple[int, int]:
    ids   = input_ids[0].tolist()
    start = next(i for i, t in enumerate(ids) if t == img_token_id)
    end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return start, end


# ---------------------------------------------------------------------------
# Ablation state helpers
# ---------------------------------------------------------------------------

def _make_random_saliency(n_img: int, rng: random.Random) -> torch.Tensor:
    """
    Random saliency: soft continuous map in [0,1], same format as result.saliency
    from compute_clip_salience() — which is also a min-max normalised float map
    over all n_img tokens.

    Using torch.rand gives each token an independent uniform weight with no
    spatial preference.  No CLIP is computed.  No fixed-k selection — this
    matches what SRF actually uses (clip_use_soft=True always; top_k_pct /
    result.mask are never used in the boost path).
    """
    gen = torch.Generator()
    gen.manual_seed(rng.randint(0, 2**31 - 1))
    return torch.rand(n_img, dtype=torch.float32, generator=gen)


def _make_random_head_mask(calibrated_mask: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Random bool mask with identical cardinality to the calibrated mask."""
    n_heads    = len(calibrated_mask)
    n_selected = int(calibrated_mask.sum().item())
    idx        = rng.sample(range(n_heads), n_selected)
    mask       = torch.zeros(n_heads, dtype=torch.bool)
    mask[idx]  = True
    return mask


def _make_random_layer_range(l0: int, l1: int, n_layers: int,
                              rng: random.Random) -> Tuple[int, int]:
    """Random layer range of same width as [l0, l1], positioned away from it."""
    width = l1 - l0
    valid = [s for s in range(0, n_layers - width) if abs(s - l0) > 2]
    if not valid:
        valid = [s for s in range(0, n_layers - width) if s != l0]
    new_start = rng.choice(valid)
    return new_start, new_start + width


def _setup_variant(
    variant: str,
    dataset: str,
    calibrated_mask: torch.Tensor,
    n_layers: int,
    seed: int,
    clip_model: Optional[str] = None,
    saliency_mode: Optional[str] = None,
    clip_method: Optional[str] = None,
) -> Tuple[str, str, random.Random]:
    """
    Call reset_for_dataset and apply one-time state modifications
    (random heads, random layers).  Returns (sal_mode, heads_mode, per_sample_rng).
    """
    seed_rng = random.Random(seed)

    if variant in ("baseline", "uniform_boost"):
        # uniform_boost still needs SRF state (alpha, eps, layers, heads) but
        # saliency = all-ones; set it up via reset so BIAS dict is populated.
        if variant == "uniform_boost":
            srf_mod.reset_for_dataset(dataset=dataset,
                                       clip_model=clip_model,
                                       saliency_mode=saliency_mode,
                                       clip_saliency_method=clip_method)
            patch._STATE["head_mask"] = calibrated_mask.clone()
        return variant, variant, seed_rng

    # Reset SRF to dataset defaults (restores layer range from config)
    srf_mod.reset_for_dataset(dataset=dataset,
                               clip_model=clip_model,
                               saliency_mode=saliency_mode,
                               clip_saliency_method=clip_method)

    sal_mode, heads_mode, layers_mode = _variant_flags(variant)

    # ── Heads ──────────────────────────────────────────────────────────────
    if heads_mode == "rand":
        rand_mask = _make_random_head_mask(calibrated_mask, seed_rng)
        patch._STATE["head_mask"] = rand_mask
        n = int(rand_mask.sum().item())
        print(f"    [ablation] rand_heads: {n}/{len(rand_mask)} heads randomly selected")
    else:
        patch._STATE["head_mask"] = calibrated_mask.clone()

    # ── Layers ─────────────────────────────────────────────────────────────
    if layers_mode == "rand":
        l0 = patch._STATE["vaf_layer_start"]
        l1 = patch._STATE["vaf_layer_end"]
        new_l0, new_l1 = _make_random_layer_range(l0, l1, n_layers, seed_rng)
        patch._STATE["vaf_layer_start"] = new_l0
        patch._STATE["vaf_layer_end"]   = new_l1
        print(f"    [ablation] rand_layers: [{new_l0},{new_l1}]  (tuned [{l0},{l1}])")

    return sal_mode, heads_mode, seed_rng


def _prepare_saliency(
    sal_mode: str,
    inp, s: int, e: int,
    image, question: str,
    model, processor,
    seed_rng: random.Random,
) -> None:
    """
    Set salience_mask in patch state for the current sample.
    For 'clip': calls srf_mod.prepare_sample() — runs CLIP, uses soft saliency map.
    For 'rand': torch.rand(n_img) in [0,1] — same format as clip soft map, no CLIP.
    For 'uniform': all ones — boost every token equally (null spatial hypothesis).
    """
    if sal_mode == "clip":
        srf_mod.prepare_sample(inp, s, e, image, question, model, processor)
    elif sal_mode == "uniform":
        n_img = e - s + 1
        patch.update_sample(s, e)
        patch._STATE["method"]        = "srf"
        patch._STATE["value"]         = srf_mod.BIAS["boost_alpha"]
        patch._STATE["salience_mask"] = torch.ones(n_img, dtype=torch.float32)
        patch._STATE["srf_bias_mode"]     = srf_mod.BIAS["bias_mode"]
        patch._STATE["srf_interp_lambda"] = srf_mod.BIAS["interp_lambda"]
        patch._STATE["srf_prob_floor"]    = srf_mod.BIAS["prob_floor"]
        patch._STATE["srf_img_scale"]     = srf_mod.BIAS["img_scale"]
        patch._STATE["srf_text_beta"]     = srf_mod.BIAS["text_beta"]
    else:  # rand
        # Soft random map in [0,1] — same format as clip_use_soft=True saliency.
        # No fixed-k selection: every token gets an independent uniform weight.
        # Different random map per sample (seed_rng advances each call).
        n_img    = e - s + 1
        rand_sal = _make_random_saliency(n_img, seed_rng)
        patch.update_sample(s, e)
        patch._STATE["method"]        = "srf"
        patch._STATE["value"]         = srf_mod.BIAS["boost_alpha"]
        patch._STATE["salience_mask"] = rand_sal
        patch._STATE["srf_bias_mode"]     = srf_mod.BIAS["bias_mode"]
        patch._STATE["srf_interp_lambda"] = srf_mod.BIAS["interp_lambda"]
        patch._STATE["srf_prob_floor"]    = srf_mod.BIAS["prob_floor"]
        patch._STATE["srf_img_scale"]     = srf_mod.BIAS["img_scale"]
        patch._STATE["srf_text_beta"]     = srf_mod.BIAS["text_beta"]


# ---------------------------------------------------------------------------
# POPE evaluation
# ---------------------------------------------------------------------------

def run_pope(
    rows: List[dict],
    variant: str,
    model, processor,
    img_token_id: int, device,
    calibrated_mask: torch.Tensor,
    n_layers: int,
    dataset: str, seed: int,
    clip_model: Optional[str] = None,
    saliency_mode: Optional[str] = None,
    clip_method: Optional[str] = None,
) -> Dict:
    sal_mode, heads_mode, seed_rng = _setup_variant(
        variant, dataset, calibrated_mask, n_layers, seed,
        clip_model=clip_model, saliency_mode=saliency_mode, clip_method=clip_method)

    correct   = 0
    yes_count = 0

    for i, r in enumerate(rows):
        image = r["image"].convert("RGB")
        q     = str(r["question"]).strip() + "\nAnswer with Yes or No only."
        gt    = "yes" if str(r.get("answer", "")).strip().lower() == "yes" else "no"

        msgs   = [{"role": "user", "content": [{"type": "image", "image": image},
                                                {"type": "text",  "text":  q}]}]
        text   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = process_vision_info(msgs)
        inp    = processor(text=[text], images=vis, return_tensors="pt",
                           padding=True).to(device)
        s, e   = get_img_range(inp["input_ids"], img_token_id)

        if variant == "baseline":
            patch._STATE["method"] = "baseline"
            with torch.inference_mode():
                logits = model(**inp).logits[:, -1, :].float()
        else:
            _prepare_saliency(sal_mode, inp, s, e, image, q,
                               model, processor, seed_rng)
            with torch.inference_mode():
                logits = model(**inp).logits[:, -1, :].float()
            srf_mod.cleanup()

        pred = processor.decode(logits.argmax(-1), skip_special_tokens=True).strip().lower()
        pred = "yes" if pred.startswith("yes") else "no"
        if pred == gt:
            correct += 1
        if pred == "yes":
            yes_count += 1

        if (i + 1) % 200 == 0 or (i + 1) == len(rows):
            n = i + 1
            print(f"    [{n:4d}/{len(rows)}]  acc={correct/n*100:.2f}%")

    n   = len(rows)
    return {"n": n, "accuracy": correct / n, "yes_rate": yes_count / n}


# ---------------------------------------------------------------------------
# VLM Bias evaluation
# ---------------------------------------------------------------------------

def _normalise(s: str) -> str:
    return s.strip().lower().lstrip("{").rstrip("}")


def _extract_answer(text: str) -> str:
    m = re.search(r"\{([^}]+)\}", text)
    return m.group(1).strip() if m else (text.strip().split()[0] if text.strip() else "")


def run_vlmbias(
    rows: List[dict],
    variant: str,
    model, processor,
    img_token_id: int, device,
    calibrated_mask: torch.Tensor,
    n_layers: int,
    dataset: str, seed: int,
    clip_model: Optional[str] = None,
    saliency_mode: Optional[str] = None,
    clip_method: Optional[str] = None,
) -> Dict:
    sal_mode, heads_mode, seed_rng = _setup_variant(
        variant, dataset, calibrated_mask, n_layers, seed,
        clip_model=clip_model, saliency_mode=saliency_mode, clip_method=clip_method)

    correct    = 0
    bias_count = 0

    for i, sample in enumerate(rows):
        image         = sample["image"].convert("RGB")
        question      = sample["question"]
        gt            = _normalise(str(sample["gt"]))
        expected_bias = _normalise(str(sample.get("expected_bias", "")))

        msgs   = [{"role": "user", "content": [{"type": "image", "image": image},
                                                {"type": "text",  "text":  question}]}]
        text   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = process_vision_info(msgs)
        inp    = processor(text=[text], images=vis, return_tensors="pt",
                           padding=True).to(device)
        s, e   = get_img_range(inp["input_ids"], img_token_id)

        if variant == "baseline":
            patch._STATE["method"] = "baseline"
            with torch.inference_mode():
                out_ids = model.generate(**inp, max_new_tokens=20, do_sample=False)
            raw = processor.decode(out_ids[0, inp["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
        else:
            _prepare_saliency(sal_mode, inp, s, e, image, question,
                               model, processor, seed_rng)
            with torch.inference_mode():
                out_ids = model.generate(**inp, max_new_tokens=20, do_sample=False)
            raw = processor.decode(out_ids[0, inp["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            srf_mod.cleanup()

        pred = _normalise(_extract_answer(raw))
        if pred == gt:
            correct += 1
        if expected_bias and pred == expected_bias:
            bias_count += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            n = i + 1
            print(f"    [{n:4d}/{len(rows)}]  acc={correct/n*100:.2f}%  "
                  f"bias={bias_count/n*100:.2f}%")

    n = len(rows)
    return {"n": n, "accuracy": correct / n, "bias_rate": bias_count / n}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pope_rows(split: str, n: int, seed: int) -> List[dict]:
    print(f"  Loading POPE ({split})…")
    ds   = hf_load("lmms-lab/POPE", split="test")
    rows = [r for r in ds
            if str(r.get("category", r.get("type", ""))).lower() == split.lower()]
    rng  = random.Random(seed)
    rng.shuffle(rows)
    if n > 0:
        rows = rows[:n]
    print(f"  {len(rows)} samples")
    return rows


def load_vlmbias_rows(n_per_cat: Optional[int], seed: int) -> List[dict]:
    print("  Loading VLM Bias…")
    ds     = hf_load("anvo25/vlms-are-biased", split="main")
    by_cat = defaultdict(list)
    for r in ds:
        by_cat[r["topic"]].append(r)

    rng     = random.Random(seed)
    samples = []
    for cat in CFG.VLM_BIAS_CATEGORIES:
        rows_cat = by_cat.get(cat, [])
        rng.shuffle(rows_cat)
        subset = rows_cat[:n_per_cat] if n_per_cat else rows_cat
        for r in subset:
            samples.append({
                "image":         r["image"],
                "question":      r["prompt"],
                "gt":            str(r.get("ground_truth", "")),
                "expected_bias": str(r.get("expected_bias", "")),
                "category":      cat,
            })
    rng.shuffle(samples)
    print(f"  {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args     = parse_args()
    variants = ALL_VARIANTS if args.all_variants else args.variants

    model, processor = load_model(args.model)
    img_token_id     = processor.tokenizer.convert_tokens_to_ids(CFG.IMAGE_TOKEN)
    device           = next(model.parameters()).device
    arch             = CFG.get_arch(args.model)
    n_layers         = arch["n_layers"]

    # SRF calibration (identifies vision-aware heads once)
    calib_ds = args.dataset if args.dataset in ("pope", "vlmbias") else "pope"
    srf_mod.setup(model, processor, calib_dataset=calib_ds)
    calibrated_mask = patch._STATE["head_mask"].clone()
    print(f"  Calibrated head mask: {int(calibrated_mask.sum())}/{len(calibrated_mask)} heads")
    print(f"  n_layers={n_layers}")

    # Load data once, reuse across all variants
    dataset = args.dataset
    if dataset == "pope":
        rows = load_pope_rows(args.pope_split, args.n, args.seed)
    else:
        n_per_cat = args.n if args.n > 0 else None
        rows      = load_vlmbias_rows(n_per_cat, args.seed)

    run_fn = run_pope if dataset == "pope" else run_vlmbias

    # Run each variant
    results: Dict[str, Dict] = {}
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Variant: {VARIANT_LABELS[variant]}")
        print(f"{'='*60}")

        res = run_fn(
            rows, variant, model, processor, img_token_id, device,
            calibrated_mask, n_layers, dataset, args.seed,
            clip_model=args.clip_model,
            saliency_mode=args.saliency_mode,
            clip_method=args.clip_method,
        )
        results[variant] = res

        if dataset == "pope":
            print(f"  → acc={res['accuracy']*100:.2f}%  yes_rate={res['yes_rate']*100:.2f}%")
        else:
            print(f"  → acc={res['accuracy']*100:.2f}%  bias_rate={res['bias_rate']*100:.2f}%")

    # Summary table
    srf_acc  = results.get("clip_cal_tuned", {}).get("accuracy", float("nan"))
    base_acc = results.get("baseline",       {}).get("accuracy", float("nan"))

    print("\n" + "="*80)
    print("ABLATION SUMMARY")
    print("="*80)
    if dataset == "pope":
        print(f"{'Variant':<52} {'Acc':>7} {'Δbase':>7} {'Δsrf':>7} {'Yes%':>7}")
        print("-"*76)
        for v in variants:
            r    = results[v]
            acc  = r["accuracy"]
            db   = acc - base_acc
            ds   = acc - srf_acc if v != "clip_cal_tuned" else 0.0
            mark = " ★" if v == "clip_cal_tuned" else ("  " if v != "baseline" else "  ")
            print(f"{VARIANT_LABELS[v]:<52}{mark} "
                  f"{acc*100:>6.2f}% "
                  f"{db*100:>+6.2f}% "
                  f"{'---' if v in ('clip_cal_tuned','baseline') else f'{ds*100:>+5.2f}%':>7} "
                  f"{r['yes_rate']*100:>6.2f}%")
    else:
        print(f"{'Variant':<52} {'Acc':>7} {'Δbase':>7} {'Δsrf':>7} {'Bias%':>7}")
        print("-"*76)
        for v in variants:
            r    = results[v]
            acc  = r["accuracy"]
            db   = acc - base_acc
            ds   = acc - srf_acc if v != "clip_cal_tuned" else 0.0
            mark = " ★" if v == "clip_cal_tuned" else "  "
            print(f"{VARIANT_LABELS[v]:<52}{mark} "
                  f"{acc*100:>6.2f}% "
                  f"{db*100:>+6.2f}% "
                  f"{'---' if v in ('clip_cal_tuned','baseline') else f'{ds*100:>+5.2f}%':>7} "
                  f"{r['bias_rate']*100:>6.2f}%")

    if args.output:
        out_dir = pathlib.Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag  = f"ablation_{dataset}"
        if dataset == "pope":
            tag += f"_{args.pope_split}"
        if args.clip_method and args.clip_method != "clip_patch":
            tag += f"_{args.clip_method}"
        elif args.saliency_mode and args.saliency_mode != "clip":
            tag += f"_{args.saliency_mode}"
        elif args.clip_model and "siglip" in args.clip_model.lower():
            tag += "_siglip"
        path = out_dir / f"{tag}.json"
        with open(path, "w") as f:
            json.dump({"args": vars(args), "results": results,
                       "labels": VARIANT_LABELS}, f, indent=2)
        print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
