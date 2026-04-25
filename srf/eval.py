#!/usr/bin/env python3
"""
Unified SRF evaluation — runs SRF or SRF-E on any supported dataset.

Usage:
    cd /volumes2/mllm/lmms-eval

    # SRF base — MMVP + POPE full (3000)
    conda run -n mllm python srf/eval.py \\
        --method srf \\
        --datasets mmvp pope \\
        --output results/srf_3b/

    # SRF-E — MMVP + POPE all 3 splits, sweep β
    conda run -n mllm python srf/eval.py \\
        --method srfe \\
        --datasets mmvp pope \\
        --beta 0.5 1.0 2.0 \\
        --output results/srfe_3b/

    # POPE adversarial only, sweep layer_end
    conda run -n mllm python srf/eval.py \\
        --method srf --datasets pope \\
        --pope_splits adversarial \\
        --layer_end 14 --alpha 4.0 --eps 0.2

    # SRF-E — single β, specific model
    conda run -n mllm python srf/eval.py \\
        --method srfe --beta 2.0 \\
        --model Qwen/Qwen2.5-VL-7B-Instruct \\
        --datasets mmvp pope

Method dispatch:
    srf  → srf.py   — SRF base: single forward pass, patched attention
    srfe → srf_e.py — SRF-E:   two forward passes, contrastive combination
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SRF / SRF-E evaluation")
    p.add_argument("--method",   required=True, choices=["srf", "srfe"],
                   help="srf = SRF base; srfe = SRF-E (contrastive)")
    p.add_argument("--model",    default=CFG.DEFAULT_MODEL)
    p.add_argument("--datasets", nargs="+", default=["mmvp", "pope"],
                   choices=["mmvp", "pope", "vlmbias", "mme"])
    p.add_argument("--beta",     type=float, nargs="+", default=[CFG.SRFE_DEFAULT_BETA],
                   help="Contrastive β (srfe only; ignored for srf)")

    # ── POPE ──────────────────────────────────────────────────────────────────
    p.add_argument("--pope_splits", nargs="+", default=CFG.POPE_SPLITS,
                   choices=["adversarial", "popular", "random"],
                   help="POPE splits to run (default: all three)")
    p.add_argument("--n_pope",   type=int, default=CFG.POPE_N_FULL,
                   help="POPE samples per split (-1=all, default=-1=all)")
    p.add_argument("--seed",     type=int, default=CFG.POPE_SEED)
    p.add_argument("--n_vlmbias_per_cat", type=int, default=0,
                   help="VLM Bias samples per category (0=all)")
    p.add_argument("--output",   default=None,
                   help="Directory to save JSON results")

    # ── SRF hyperparams (all optional — override arch/dataset config) ─────────
    p.add_argument("--layer_start",      type=int,   default=None,
                   help="First layer to apply SRF (overrides arch config)")
    p.add_argument("--layer_end",        type=int,   default=None,
                   help="Last layer to apply SRF (overrides arch/dataset config)")
    p.add_argument("--head_top_k_pct",   type=float, default=None,
                   help="Fraction of heads selected as vision-aware (e.g. 0.20)")
    p.add_argument("--alpha",            type=float, default=None,
                   help="Attention logit boost magnitude (overrides dataset config)")
    p.add_argument("--eps",              type=float, default=None,
                   help="Background suppression epsilon (overrides dataset config)")
    p.add_argument("--phase",            default=None,
                   choices=["prefill", "generation", "both"],
                   help="Which phase to apply SRF (overrides dataset config)")
    # ── CLIP saliency ─────────────────────────────────────────────────────────
    p.add_argument("--clip_coarse_grid", type=int,   default=None,
                   help="CLIP patch grid size (e.g. 7 for Qwen, 6 for LLaVA)")
    p.add_argument("--clip_top_k_pct",   type=float, default=None,
                   help="Fraction of image tokens boosted by CLIP saliency")
    p.add_argument("--clip_fallback_thresh", type=float, default=None,
                   help="CLIP max-sim below which uniform boost is used")

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


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_img_range(input_ids: torch.Tensor, img_token_id: int) -> tuple[int, int]:
    ids   = input_ids[0].tolist()
    start = next(i for i, t in enumerate(ids) if t == img_token_id)
    end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return start, end


def decode_first_token(logits: torch.Tensor, processor) -> str:
    return processor.decode(logits.argmax(dim=-1), skip_special_tokens=True).strip().lower()


def save_results(results: dict, output_dir: str | None, tag: str) -> None:
    if not output_dir:
        return
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Method dispatch helpers
# ---------------------------------------------------------------------------

def method_get_logits(method_mod, model, inp: dict, beta: float) -> torch.Tensor:
    """Single-token logits [1, vocab]: SRF uses one pass, SRF-E uses two."""
    if hasattr(method_mod, "get_contrastive_logits"):
        return method_mod.get_contrastive_logits(model, inp, beta=beta)
    # SRF base: patched model, single pass
    patch._STATE["method"] = "srf"
    with torch.inference_mode():
        out = model(**inp)
    return out.logits[:, -1, :].float()


def method_generate(method_mod, model, inp: dict, processor, beta: float,
                    max_new_tokens: int = 20) -> list[int]:
    """Token generation: SRF uses model.generate, SRF-E uses contrastive loop."""
    if hasattr(method_mod, "generate_contrastive"):
        return method_mod.generate_contrastive(
            model, inp, processor, beta=beta, max_new_tokens=max_new_tokens)
    # SRF base: standard generation with patched model
    patch._STATE["method"] = "srf"
    with torch.inference_mode():
        out_ids = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return out_ids[0, inp["input_ids"].shape[1]:].tolist()


# ---------------------------------------------------------------------------
# Hyperparameter override helper
# ---------------------------------------------------------------------------

def _reset_overrides(args) -> dict:
    """Collect CLI hyperparameter overrides; None means 'use config default'."""
    return dict(
        phase=args.phase,
        alpha=args.alpha,
        eps=args.eps,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        head_top_k_pct=args.head_top_k_pct,
        clip_coarse_grid=args.clip_coarse_grid,
        clip_top_k_pct=args.clip_top_k_pct,
        clip_fallback_thresh=args.clip_fallback_thresh,
    )


# ---------------------------------------------------------------------------
# POPE
# ---------------------------------------------------------------------------

def run_pope(method_mod, model, processor, img_token_id, device, args) -> dict:
    betas = args.beta if args.method == "srfe" else [0.0]

    splits_filter = {s.lower() for s in args.pope_splits}
    splits_label  = "+".join(sorted(splits_filter))

    print("\n" + "="*60)
    n_req = args.n_pope if args.n_pope > 0 else "all"
    print(f"DATASET: POPE ({splits_label}, n={n_req}/split, seed={args.seed})")
    print("="*60)

    method_mod.reset_for_dataset(dataset="pope", **_reset_overrides(args))

    ds   = hf_load("lmms-lab/POPE", split="test")
    rows = [r for r in ds
            if str(r.get("category", r.get("type", ""))).lower() in splits_filter]
    rng  = random.Random(args.seed)
    rng.shuffle(rows)
    if args.n_pope > 0:
        rows = rows[:args.n_pope]
    print(f"  Loaded {len(rows)} POPE samples ({splits_label})")

    correct_base = 0
    correct_srf  = {b: 0 for b in betas}

    for i, r in enumerate(rows):
        image = r["image"].convert("RGB")
        q     = str(r["question"]).strip() + "\nAnswer with Yes or No only."
        gt    = "yes" if str(r.get("answer", "")).strip().lower() == "yes" else "no"

        msgs  = [{"role": "user", "content": [{"type": "image", "image": image},
                                               {"type": "text",  "text":  q}]}]
        text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = process_vision_info(msgs)
        inp    = processor(text=[text], images=vis, return_tensors="pt",
                           padding=True).to(device)
        s, e   = get_img_range(inp["input_ids"], img_token_id)

        # Baseline
        patch._STATE["method"] = "baseline"
        with torch.inference_mode():
            logits_base = model(**inp).logits[:, -1, :].float()
        pred_base = "yes" if decode_first_token(logits_base, processor).startswith("yes") else "no"
        if pred_base == gt:
            correct_base += 1

        # Method
        method_mod.prepare_sample(inp, s, e, image, q, model, processor)
        for beta in betas:
            logits = method_get_logits(method_mod, model, inp, beta)
            pred = "yes" if decode_first_token(logits, processor).startswith("yes") else "no"
            if pred == gt:
                correct_srf[beta] += 1
        method_mod.cleanup()

        if (i + 1) % 500 == 0 or (i + 1) == len(rows):
            n = i + 1
            srf_str = "  ".join(f"β={b}:{correct_srf[b]/n:.4f}" for b in betas)
            print(f"  [{n:5d}/{len(rows)}] base={correct_base/n:.4f}  {srf_str}")

    n        = len(rows)
    acc_base = correct_base / n
    acc_srf  = {b: correct_srf[b] / n for b in betas}

    print(f"\nPOPE baseline:  {acc_base:.4f}  ({correct_base}/{n})")
    for b in betas:
        delta = acc_srf[b] - acc_base
        label = f"β={b}" if args.method == "srfe" else "SRF"
        print(f"  {label}: {acc_srf[b]:.4f}  Δ={delta:+.4f}")

    return {"n": n, "baseline": acc_base, "method": acc_srf}


# ---------------------------------------------------------------------------
# MMVP
# ---------------------------------------------------------------------------

def run_mmvp(method_mod, model, processor, img_token_id, device, args) -> dict:
    import pandas as pd
    betas = args.beta if args.method == "srfe" else [0.0]

    print("\n" + "="*60)
    print("DATASET: MMVP (150 pairs, 300 images, full)")
    print("="*60)

    method_mod.reset_for_dataset(dataset="mmvp", **_reset_overrides(args))

    df         = pd.read_csv(CFG.MMVP_CSV)
    img_ds     = hf_load("MMVP/MMVP", split="train")
    lex_sorted = sorted(range(1, 301), key=str)
    csv_to_hf  = {c: h for h, c in enumerate(lex_sorted)}

    a_id = processor.tokenizer.convert_tokens_to_ids("A")
    b_id = processor.tokenizer.convert_tokens_to_ids("B")

    pair_base = defaultdict(dict)
    pair_srf  = {b: defaultdict(dict) for b in betas}
    n_base_ok = 0
    n_srf_ok  = {b: 0 for b in betas}

    for csv_1idx in range(1, 301):
        row_idx  = csv_1idx - 1
        row      = df.iloc[row_idx]
        opts     = re.findall(r'\(([ab])\)\s*([^(]+)', str(row["Options"]), re.IGNORECASE)
        if not opts:
            continue
        opt_text = "\n".join(f"{m[0].upper()}. {m[1].strip()}" for m in opts)
        prompt   = f"{row['Question']}\n{opt_text}\nAnswer with the option's letter directly."
        gt_raw   = str(row["Correct Answer"]).strip().strip("()").upper()
        gt       = CFG.MMVP_GT_CORRECTIONS.get(row_idx, gt_raw)
        image    = img_ds[csv_to_hf[csv_1idx]]["image"].convert("RGB")
        pair_id  = row_idx // 2
        img_key  = "a" if row_idx % 2 == 0 else "b"

        msgs  = [{"role": "user", "content": [{"type": "image", "image": image},
                                               {"type": "text",  "text":  prompt}]}]
        text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = process_vision_info(msgs)
        inp    = processor(text=[text], images=vis, return_tensors="pt",
                           padding=True).to(device)
        s, e   = get_img_range(inp["input_ids"], img_token_id)

        # Baseline
        patch._STATE["method"] = "baseline"
        with torch.inference_mode():
            logits_base = model(**inp).logits[:, -1, :].float()
        pred_base = "A" if logits_base[0, a_id] >= logits_base[0, b_id] else "B"
        ok_base   = (pred_base == gt)
        if ok_base:
            n_base_ok += 1
        pair_base[pair_id][img_key] = ok_base

        # Method
        method_mod.prepare_sample(inp, s, e, image, row["Question"], model, processor)
        for beta in betas:
            logits = method_get_logits(method_mod, model, inp, beta)
            pred   = "A" if logits[0, a_id] >= logits[0, b_id] else "B"
            ok     = (pred == gt)
            if ok:
                n_srf_ok[beta] += 1
            pair_srf[beta][pair_id][img_key] = ok
        method_mod.cleanup()

        if csv_1idx % 60 == 0:
            bp  = [(pid, r) for pid, r in pair_base.items() if "a" in r and "b" in r]
            bpa = sum(1 for _, r in bp if r["a"] and r["b"]) / max(1, len(bp))
            srf_str = "  ".join(f"β={b}:{n_srf_ok[b]/csv_1idx:.4f}" for b in betas)
            print(f"  [{csv_1idx:3d}/300] base_pair={bpa:.4f}  {srf_str}")

    def _pair_acc(pd_):
        pairs = [(pid, r) for pid, r in pd_.items() if "a" in r and "b" in r]
        return sum(1 for _, r in pairs if r["a"] and r["b"]) / max(1, len(pairs))

    acc_base = _pair_acc(pair_base)
    acc_srf  = {b: _pair_acc(pair_srf[b]) for b in betas}
    img_base = n_base_ok / 300

    print(f"\nMMVP baseline:  pair={acc_base:.4f}  img={img_base:.4f}")
    for b in betas:
        delta = acc_srf[b] - acc_base
        label = f"β={b}" if args.method == "srfe" else "SRF"
        print(f"  {label}: pair={acc_srf[b]:.4f}  Δ={delta:+.4f}")

    return {"baseline_pair": acc_base, "baseline_img": img_base, "method_pair": acc_srf}


# ---------------------------------------------------------------------------
# VLM Bias
# ---------------------------------------------------------------------------

def run_vlmbias(method_mod, model, processor, img_token_id, device, args) -> dict:
    betas     = args.beta if args.method == "srfe" else [0.0]
    n_per_cat = args.n_vlmbias_per_cat if args.n_vlmbias_per_cat > 0 else None

    print("\n" + "="*60)
    n_label = f"n={n_per_cat}/cat" if n_per_cat else "all/cat"
    print(f"DATASET: VLM Bias (7 cats, {n_label}, seed={CFG.VLM_BIAS_SEED})")
    print("="*60)

    method_mod.reset_for_dataset(dataset="vlmbias", **_reset_overrides(args))

    def normalise(s: str) -> str:
        return s.strip().lower().lstrip("{").rstrip("}")

    def extract_answer(text: str) -> str:
        m = re.search(r'\{([^}]+)\}', text)
        return m.group(1).strip() if m else (text.strip().split()[0] if text.strip() else "")

    ds     = hf_load("anvo25/vlms-are-biased", split="main")
    by_cat = defaultdict(list)
    for r in ds:
        by_cat[r["topic"]].append(r)

    rng     = random.Random(CFG.VLM_BIAS_SEED)
    samples = []
    for cat in CFG.VLM_BIAS_CATEGORIES:
        rows_cat = by_cat.get(cat, [])
        rng.shuffle(rows_cat)
        subset = rows_cat[:n_per_cat] if n_per_cat else rows_cat
        for r in subset:
            samples.append({
                "image":    r["image"].convert("RGB"),
                "question": r["prompt"],
                "gt":       normalise(str(r["ground_truth"])),
                "category": cat,
            })
    rng.shuffle(samples)

    correct_base = 0
    correct_srf  = {b: 0 for b in betas}
    cat_base     = {c: {"correct": 0, "total": 0} for c in CFG.VLM_BIAS_CATEGORIES}

    for i, sample in enumerate(samples):
        msgs  = [{"role": "user", "content": [{"type": "image", "image": sample["image"]},
                                               {"type": "text",  "text":  sample["question"]}]}]
        text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = process_vision_info(msgs)
        inp    = processor(text=[text], images=vis, return_tensors="pt",
                           padding=True).to(device)
        s, e   = get_img_range(inp["input_ids"], img_token_id)

        # Baseline
        patch._STATE["method"] = "baseline"
        with torch.inference_mode():
            out_ids = model.generate(**inp, max_new_tokens=20, do_sample=False)
        raw_base  = processor.decode(out_ids[0, inp["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        pred_base = normalise(extract_answer(raw_base))
        ok_base   = (pred_base == sample["gt"])
        if ok_base:
            correct_base += 1
        cat_base[sample["category"]]["correct"] += int(ok_base)
        cat_base[sample["category"]]["total"]   += 1

        # Method
        method_mod.prepare_sample(inp, s, e, sample["image"], sample["question"],
                                   model, processor)
        for beta in betas:
            gen_ids = method_generate(method_mod, model, inp, processor,
                                      beta=beta, max_new_tokens=20)
            raw_c   = processor.decode(gen_ids, skip_special_tokens=True)
            pred_c  = normalise(extract_answer(raw_c))
            if pred_c == sample["gt"]:
                correct_srf[beta] += 1
        method_mod.cleanup()

        if (i + 1) % 20 == 0:
            n = i + 1
            srf_str = "  ".join(f"β={b}:{correct_srf[b]/n:.4f}" for b in betas)
            print(f"  [{n:3d}/{len(samples)}] base={correct_base/n:.4f}  {srf_str}")

    total    = len(samples)
    acc_base = correct_base / total
    acc_srf  = {b: correct_srf[b] / total for b in betas}

    print(f"\nVLM Bias baseline:  {acc_base:.4f}  ({correct_base}/{total})")
    for b in betas:
        label = f"β={b}" if args.method == "srfe" else "SRF"
        print(f"  {label}: {acc_srf[b]:.4f}  Δ={acc_srf[b]-acc_base:+.4f}")
    print("\n  Per-category (baseline):")
    for cat in CFG.VLM_BIAS_CATEGORIES:
        cb = cat_base[cat]
        print(f"    {cat:20s}: {cb['correct']}/{cb['total']}")

    return {"n": total, "baseline": acc_base, "method": acc_srf, "cat_baseline": cat_base}


# ---------------------------------------------------------------------------
# MME
# ---------------------------------------------------------------------------

# Standard MME perception / cognition split
_MME_COGNITION = {"code_reasoning", "numerical_calculation", "text_translation",
                  "commonsense_reasoning"}


def run_mme(method_mod, model, processor, img_token_id, device, args) -> dict:
    """MME evaluation — 2374 Yes/No questions across 14 categories.

    Metrics reported:
      - Per-question accuracy (correct / total)
      - Pair accuracy — both questions for the same image correct
      - Per-category score (# correct)
      - MME score = total correct questions (standard metric, published as integer sum)
      - Perception / Cognition sub-scores
    """
    betas = args.beta if args.method == "srfe" else [0.0]

    print("\n" + "="*60)
    print("DATASET: MME (2374 Yes/No, 14 categories, full)")
    print("="*60)

    method_mod.reset_for_dataset(dataset="mme", **_reset_overrides(args))

    ds = hf_load("lmms-lab/MME", split="test")
    samples = list(ds)
    print(f"  Loaded {len(samples)} MME samples")

    yes_id = processor.tokenizer.convert_tokens_to_ids("Yes")
    no_id  = processor.tokenizer.convert_tokens_to_ids("No")

    # Trackers
    n_total      = len(samples)
    correct_base = 0
    correct_srf  = {b: 0 for b in betas}

    # Per-category: {cat: {"base": 0, "srf": {b: 0}, "total": 0}}
    cat_stats: dict = defaultdict(lambda: {"base": 0, "srf": {b: 0 for b in betas}, "total": 0})

    # Pair accuracy: {pair_id: {"base": [], "srf": {b: []}}}
    pair_stats: dict = defaultdict(lambda: {"base": [], "srf": {b: [] for b in betas}})

    for i, r in enumerate(samples):
        image    = r["image"].convert("RGB")
        q        = str(r["question"]).strip()
        gt       = str(r["answer"]).strip()          # "Yes" or "No"
        gt_lower = gt.lower()
        cat      = str(r.get("category", "unknown")).strip()
        pair_id  = str(r.get("question_id", f"{cat}_{i}"))

        msgs  = [{"role": "user", "content": [{"type": "image", "image": image},
                                               {"type": "text",  "text":  q}]}]
        text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = process_vision_info(msgs)
        inp    = processor(text=[text], images=vis, return_tensors="pt",
                           padding=True).to(device)
        s, e   = get_img_range(inp["input_ids"], img_token_id)

        # ── Baseline (compare Yes vs No logit directly) ────────────────────────
        patch._STATE["method"] = "baseline"
        with torch.inference_mode():
            logits_base = model(**inp).logits[:, -1, :].float()
        pred_base = "yes" if logits_base[0, yes_id] >= logits_base[0, no_id] else "no"
        ok_base   = (pred_base == gt_lower)
        if ok_base:
            correct_base += 1
        cat_stats[cat]["base"]  += int(ok_base)
        cat_stats[cat]["total"] += 1
        pair_stats[pair_id]["base"].append(ok_base)

        # ── Method ────────────────────────────────────────────────────────────
        method_mod.prepare_sample(inp, s, e, image, q, model, processor)
        for beta in betas:
            logits = method_get_logits(method_mod, model, inp, beta)
            pred   = "yes" if logits[0, yes_id] >= logits[0, no_id] else "no"
            ok     = (pred == gt_lower)
            if ok:
                correct_srf[beta] += 1
            cat_stats[cat]["srf"][beta]  += int(ok)
            pair_stats[pair_id]["srf"][beta].append(ok)
        method_mod.cleanup()

        if (i + 1) % 500 == 0 or (i + 1) == n_total:
            n = i + 1
            srf_str = "  ".join(f"β={b}:{correct_srf[b]/n:.4f}" for b in betas)
            print(f"  [{n:4d}/{n_total}] base={correct_base/n:.4f}  {srf_str}")

    # ── Compute pair accuracy ──────────────────────────────────────────────────
    def _pair_acc(key, beta=None):
        pairs = [v for v in pair_stats.values() if len(v[key] if beta is None else v["srf"][beta]) == 2]
        if not pairs:
            return 0.0
        if beta is None:
            return sum(1 for p in pairs if all(p["base"])) / len(pairs)
        return sum(1 for p in pairs if all(p["srf"][beta])) / len(pairs)

    acc_base = correct_base / n_total
    acc_srf  = {b: correct_srf[b] / n_total for b in betas}
    pair_base = _pair_acc("base")
    pair_srf  = {b: _pair_acc("srf", b) for b in betas}

    # ── Perception / Cognition sub-scores ─────────────────────────────────────
    def _subscores(key, beta=None):
        perc, cogn = 0, 0
        for cat, s in cat_stats.items():
            v = s["base"] if beta is None else s["srf"][beta]
            if cat in _MME_COGNITION:
                cogn += v
            else:
                perc += v
        return perc, cogn

    perc_base, cogn_base = _subscores("base")
    perc_srf = {}; cogn_srf = {}
    for b in betas:
        perc_srf[b], cogn_srf[b] = _subscores("srf", b)

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\nMME baseline:  acc={acc_base:.4f}  pair={pair_base:.4f}")
    print(f"  MME score: {correct_base} / {n_total}  "
          f"(perception={perc_base}, cognition={cogn_base})")
    for b in betas:
        label = f"β={b}" if args.method == "srfe" else "SRF"
        print(f"  {label}: acc={acc_srf[b]:.4f}  pair={pair_srf[b]:.4f}  "
              f"Δ={acc_srf[b]-acc_base:+.4f}  "
              f"score={correct_srf[b]} (perc={perc_srf[b]}, cogn={cogn_srf[b]})")

    print("\n  Per-category (baseline | SRF):")
    all_cats = sorted(cat_stats.keys())
    for cat in all_cats:
        s = cat_stats[cat]
        srf_str = "  ".join(f"β={b}:{s['srf'][b]}" for b in betas)
        marker = "[C]" if cat in _MME_COGNITION else "[P]"
        print(f"    {marker} {cat:28s}: base={s['base']:3d}/{s['total']}  {srf_str}")

    return {
        "n": n_total,
        "baseline_acc":  acc_base,
        "baseline_pair": pair_base,
        "baseline_score": correct_base,
        "baseline_perception": perc_base,
        "baseline_cognition":  cogn_base,
        "method_acc":   acc_srf,
        "method_pair":  pair_srf,
        "method_score": correct_srf,
        "method_perception": perc_srf,
        "method_cognition":  cogn_srf,
        "cat_stats": {k: dict(v) for k, v in cat_stats.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Import method module (srf/ is on sys.path — import directly by filename)
    if args.method == "srf":
        import srf as method_mod
    else:
        import srf_e as method_mod

    model, processor = load_model(args.model)
    arch = CFG.get_arch(args.model)
    if arch["image_token"] is not None:
        img_token_id = processor.tokenizer.convert_tokens_to_ids(arch["image_token"])
    else:
        img_token_id = model.config.image_token_index   # LLaVA-style
    device = next(model.parameters()).device

    beta_str = f"β={args.beta}" if args.method == "srfe" else "base"
    print(f"\n[{args.method.upper()}] Setup  model={args.model}  {beta_str}")
    method_mod.setup(model, processor, calib_dataset=args.datasets[0])

    results = {}

    if "mmvp" in args.datasets:
        results["mmvp"] = run_mmvp(method_mod, model, processor, img_token_id, device, args)
        save_results(results["mmvp"], args.output, "mmvp")

    if "vlmbias" in args.datasets:
        results["vlmbias"] = run_vlmbias(method_mod, model, processor, img_token_id, device, args)
        save_results(results["vlmbias"], args.output, "vlmbias")

    if "pope" in args.datasets:
        results["pope"] = run_pope(method_mod, model, processor, img_token_id, device, args)
        save_results(results["pope"], args.output, "pope")

    if "mme" in args.datasets:
        results["mme"] = run_mme(method_mod, model, processor, img_token_id, device, args)
        save_results(results["mme"], args.output, "mme")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"FINAL SUMMARY — {args.method.upper()} vs Baseline")
    print("="*70)
    print(f"Model: {args.model}  |  method={args.method}  |  {beta_str}")

    betas = args.beta if args.method == "srfe" else [0.0]

    def _p(v): return f"{v*100:.2f}%"

    if "mmvp" in results:
        r = results["mmvp"]
        print(f"\nMMVP pair acc:  baseline={_p(r['baseline_pair'])}")
        for b in betas:
            label = f"β={b}" if args.method == "srfe" else "SRF"
            print(f"  {label}: {_p(r['method_pair'][b])}  "
                  f"Δ={_p(r['method_pair'][b]-r['baseline_pair'])}")

    if "pope" in results:
        r = results["pope"]
        print(f"\nPOPE acc (n={r['n']}):  baseline={_p(r['baseline'])}")
        for b in betas:
            label = f"β={b}" if args.method == "srfe" else "SRF"
            print(f"  {label}: {_p(r['method'][b])}  Δ={_p(r['method'][b]-r['baseline'])}")

    if "vlmbias" in results:
        r = results["vlmbias"]
        print(f"\nVLM Bias acc (n={r['n']}):  baseline={_p(r['baseline'])}")
        for b in betas:
            label = f"β={b}" if args.method == "srfe" else "SRF"
            print(f"  {label}: {_p(r['method'][b])}  Δ={_p(r['method'][b]-r['baseline'])}")

    if "mme" in results:
        r = results["mme"]
        print(f"\nMME (n={r['n']}):  baseline score={r['baseline_score']}  "
              f"acc={_p(r['baseline_acc'])}  pair={_p(r['baseline_pair'])}")
        print(f"  Perception={r['baseline_perception']}  Cognition={r['baseline_cognition']}")
        for b in betas:
            label = f"β={b}" if args.method == "srfe" else "SRF"
            print(f"  {label}: score={r['method_score'][b]}  acc={_p(r['method_acc'][b])}  "
                  f"Δ={_p(r['method_acc'][b]-r['baseline_acc'])}  "
                  f"perc={r['method_perception'][b]}  cogn={r['method_cognition'][b]}")

    if args.output:
        save_results(results, args.output, "summary")
        print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main()
