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
        --gamma 0.5 1.0 2.0 \\
        --output results/srfe_3b/

    # POPE adversarial only, sweep layer_end
    conda run -n mllm python srf/eval.py \\
        --method srf --datasets pope \\
        --pope_splits adversarial \\
        --layer_end 14 --alpha 4.0 --eps 0.2

    # SRF-E — single β, specific model
    conda run -n mllm python srf/eval.py \\
        --method srfe --gamma 2.0 \\
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
from noun_extract import extract_vlind_nouns
os.environ.setdefault("HF_HOME", CFG.HF_HOME)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datasets import load_dataset as hf_load

import qwen_attn_patch as patch  # default; overridden per-model in load_model()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SRF / SRF-E evaluation")
    p.add_argument("--method",   required=True,
                   choices=["srf", "srfe", "vcd", "vaf", "baseline"],
                   help="srf = SRF base; srfe = SRF-E; "
                        "vcd = Visual Contrastive Decoding; vaf = Visual Amplification Fusion; "
                        "baseline = no intervention")
    p.add_argument("--model",    default=CFG.DEFAULT_MODEL)
    p.add_argument("--datasets", nargs="+", default=["mmvp", "pope"],
                   choices=["mmvp", "pope", "vlmbias", "mme", "vlind", "mmhalbench"])
    p.add_argument("--gamma",       type=float, nargs="+", default=[CFG.SRFE_DEFAULT_GAMMA],
                   help="Contrastive γ (srfe only; ignored for srf)")

    # ── POPE ──────────────────────────────────────────────────────────────────
    p.add_argument("--pope_splits", nargs="+", default=CFG.POPE_SPLITS,
                   choices=["adversarial", "popular", "random"],
                   help="POPE splits to run (default: all three)")
    p.add_argument("--n_pope",   type=int, default=CFG.POPE_N_FULL,
                   help="POPE samples per split (-1=all, default=-1=all)")
    p.add_argument("--seed",     type=int, default=CFG.POPE_SEED)
    p.add_argument("--repope_dir", default=None,
                   help="Path to dir with coco_repope_*.json (corrected labels). "
                        "If set, replaces original POPE labels with RePOPE annotations "
                        "and skips ambiguous/removed samples.")
    p.add_argument("--n_vlmbias_per_cat", type=int, default=0,
                   help="VLM Bias samples per category (0=all)")
    p.add_argument("--vlind_download", action="store_true", default=False,
                   help="Download missing VLind-Bench images from HF (slow, ~3.8GB)")
    p.add_argument("--vlind_n", type=int, default=None,
                   help="Limit VLIND samples for quick sweeps (None=all 302)")
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
                   help="clip_full_gate_v3: full-image CLIP sim threshold for presence gate (default 0.20)")
    p.add_argument("--clip_patch_thresh", type=float, default=None,
                   help="clip_full_gate_v3: patch max-sim backup presence gate threshold (default 0.27)")
    p.add_argument("--saliency_mode", default=None,
                   help="Override saliency mode: clip_full_gate_v3 (default/best) | clip | hssa | lta | clip_lta | srf2")

    # ── Boosting method ───────────────────────────────────────────────────────
    p.add_argument("--neg_absent_alpha", type=float, default=None,
                   help="Suppression logit when CLIP says object absent (0=off, default)")
    p.add_argument("--llava_boost_mode", default=None,
                   choices=["multiplicative", "additive"],
                   help="LLaVA only: how image token attention is modified. "
                        "multiplicative=post-softmax multiply (default, VAF-compatible); "
                        "additive=pre-softmax logit addition (matches Qwen SRF)")
    p.add_argument("--bias_mode", default=None,
                   choices=["additive_logit", "budget_shift", "prob_interp", "prob_scale"],
                   help="How the bias is applied (default: additive_logit)")
    p.add_argument("--interp_lambda", type=float, default=None,
                   help="Mixing weight for prob_interp bias mode (default: 1.0)")
    p.add_argument("--sys_beta",     type=float, default=None,
                   help="System-prompt token logit suppression (default: 0.30)")
    p.add_argument("--text_beta",    type=float, default=None,
                   help="Text-token logit suppression (default: 0.0, disabled)")
    p.add_argument("--text_layer_start", type=int, default=None,
                   help="First layer for text suppression zone (default: 20)")
    p.add_argument("--text_layer_end",   type=int, default=None,
                   help="Last layer for text suppression zone (default: 27)")
    p.add_argument("--prob_floor",   type=float, default=None,
                   help="Minimum attention probability floor (default: 0.005)")
    p.add_argument("--img_scale",    type=float, default=None,
                   help="Image token scale for global_redistribute mode (default: 1.5)")
    p.add_argument("--vr_target", type=float, default=None,
                   help="B1 visual reliance target fraction; 0=off (default). "
                        "If model attends < vr_target to image, alpha is scaled up.")
    p.add_argument("--vr_k", type=float, default=None,
                   help="B1 deficit amplification factor (default: 3.0)")

    # ── Saliency combination params ───────────────────────────────────────────
    p.add_argument("--clip_model", default=None,
                   help="CLIP model ID (default: openai/clip-vit-base-patch32)")
    p.add_argument("--hssa_layer_idx",       type=int,   default=None,
                   help="Decoder layer index for HSSA saliency (default: 12)")
    p.add_argument("--clip_saliency_method", default=None,
                   choices=["clip_patch", "clip_gradcam", "srf2"],
                   help="CLIP saliency computation method (default: clip_patch)")
    p.add_argument("--srf2_clip_weight",  type=float, default=None,
                   help="CLIP weight in srf2 ensemble (default: 0.7)")
    p.add_argument("--srf2_hssa_weight",  type=float, default=None,
                   help="HSSA weight in srf2 ensemble (default: 0.3)")
    p.add_argument("--lta_layer_idx",     type=int,   default=None,
                   help="Decoder layer index for LTA saliency (-1=last, default: -1)")
    p.add_argument("--lta_weight",        type=float, default=None,
                   help="LTA weight in clip_lta combined mode (default: 0.6)")
    p.add_argument("--clip_weight",       type=float, default=None,
                   help="CLIP weight in clip_lta combined mode (default: 0.4)")

    # ── MMHal-Bench ───────────────────────────────────────────────────────────
    p.add_argument("--mmhalbench_json", default=None,
                   help="Path to MMHal-Bench response_template.json "
                        "(default: auto-detect from ILVAD repo at ../ILVAD/data/MMHal-Bench/)")
    p.add_argument("--openai_model", default="gpt-4o",
                   help="OpenAI model for MMHal-Bench scoring (default: gpt-4o). "
                        "Requires OPENAI_API_KEY env var; skipped if not set.")
    p.add_argument("--mmhalbench_n", type=int, default=None,
                   help="Limit MMHal-Bench to first N samples (None=all 96)")

    # ── VCD-specific ──────────────────────────────────────────────────────────
    p.add_argument("--noise_step", type=int, default=None,
                   help="VCD: diffusion noise step 0–999 (default: 500). Higher = more noise.")
    p.add_argument("--cd_alpha",   type=float, default=None,
                   help="VCD: contrastive strength α (default: 1.0)")
    p.add_argument("--cd_beta",    type=float, default=None,
                   help="VCD: Adaptive Plausibility Constraint threshold (default: 0.1)")

    # ── VAF-specific ──────────────────────────────────────────────────────────
    p.add_argument("--vaf_beta",   type=float, default=None,
                   help="VAF: system-token suppression β (default: 0.30)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    global patch
    print(f"Loading {model_id}…")

    cfg        = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", None)

    if model_type == "llava":
        from transformers import LlavaForConditionalGeneration
        import llava_attn_patch as _llava_patch
        patch = _llava_patch
        print(f"  Using LLaVA attention patch")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="eager",
        ).eval().cuda()
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        import qwen_attn_patch as _qwen_patch
        patch = _qwen_patch
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


def _encode_llava(processor, model, device, image, question: str):
    """LLaVA-1.5 input encoding — returns (inp, img_start, img_end).

    The single <image> placeholder expands to 24×24=576 embeddings inside LLaVA's
    forward pass, so img_end = img_start + 575 (not img_start == img_end).
    """
    prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        f"USER: <image>\n{question}\nASSISTANT:"
    )
    inp = processor(text=prompt, images=[image], return_tensors="pt", padding=True).to(device)
    s, e = patch.get_image_token_range(inp, model)
    return inp, s, e


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

def method_get_logits(method_mod, model, inp: dict, gamma: float) -> torch.Tensor:
    """Single-token logits [1, vocab]: SRF uses one pass, SRF-E uses two."""
    if hasattr(method_mod, "get_contrastive_logits"):
        return method_mod.get_contrastive_logits(model, inp, gamma=gamma)
    # SRF base: patched model, single pass
    patch._STATE["method"] = "srf"
    with torch.inference_mode():
        out = model(**inp)
    return out.logits[:, -1, :].float()


def _decode_ab(text: str) -> str:
    """Extract A or B from model-generated text. Used for MMVP binary-choice eval."""
    patterns = [
        r"^\s*\(?([AB])\)?\.?\s*$",
        r"^\s*\(?([AB])\)?[\.\s]",
        r"answer\s+is\s+\(?([AB])\)?",
        r"option\s+\(?([AB])\)?",
        r"\(?([AB])\)?(?:\s|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    if text and text[0].upper() in ("A", "B"):
        return text[0].upper()
    return ""


def method_generate(method_mod, model, inp: dict, processor, gamma: float,
                    max_new_tokens: int = 20, content_offset: int = 0) -> list[int]:
    """Token generation: SRF uses model.generate, SRF-E uses contrastive loop."""
    if hasattr(method_mod, "generate_contrastive"):
        return method_mod.generate_contrastive(
            model, inp, processor, gamma=gamma, max_new_tokens=max_new_tokens,
            content_offset=content_offset)
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
        neg_absent_alpha=args.neg_absent_alpha,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        head_top_k_pct=args.head_top_k_pct,
        clip_coarse_grid=args.clip_coarse_grid,
        clip_top_k_pct=args.clip_top_k_pct,
        clip_fallback_thresh=args.clip_fallback_thresh,
        clip_patch_thresh=args.clip_patch_thresh,
        saliency_mode=args.saliency_mode,
        bias_mode=args.bias_mode,
        interp_lambda=args.interp_lambda,
        sys_beta=args.sys_beta,
        text_beta=args.text_beta,
        text_layer_start=args.text_layer_start,
        text_layer_end=args.text_layer_end,
        prob_floor=args.prob_floor,
        img_scale=args.img_scale,
        vr_target=args.vr_target,
        vr_k=args.vr_k,
        # saliency combination
        clip_model=args.clip_model,
        hssa_layer_idx=args.hssa_layer_idx,
        clip_saliency_method=args.clip_saliency_method,
        srf2_clip_weight=args.srf2_clip_weight,
        srf2_hssa_weight=args.srf2_hssa_weight,
        lta_layer_idx=args.lta_layer_idx,
        lta_weight=args.lta_weight,
        clip_weight=args.clip_weight,
        # VCD
        noise_step=args.noise_step,
        cd_alpha=args.cd_alpha,
        cd_beta=args.cd_beta,
        # VAF
        vaf_beta=args.vaf_beta,
    )


# ---------------------------------------------------------------------------
# POPE
# ---------------------------------------------------------------------------

def run_pope(method_mod, model, processor, img_token_id, device, args) -> dict:
    gammas = args.gamma if args.method == "srfe" else [0.0]

    splits_filter = {s.lower() for s in args.pope_splits}
    splits_label  = "+".join(sorted(splits_filter))

    print("\n" + "="*60)
    n_req     = args.n_pope if args.n_pope > 0 else "all"
    ds_name   = "RePOPE" if getattr(args, "repope_dir", None) else "POPE"
    print(f"DATASET: {ds_name} ({splits_label}, n={n_req}/split, seed={args.seed})")
    print("="*60)

    method_mod.reset_for_dataset(dataset="pope", **_reset_overrides(args))

    # ── Load RePOPE corrected labels (if requested) ───────────────────────────
    # Key: (split, str(question_id)) — compound key because each split
    # independently uses question_ids 1-3000, so split alone is not unique.
    repope_labels: dict = {}   # (split, question_id_str) → corrected label ("yes"/"no")
    if getattr(args, "repope_dir", None):
        import json as _json
        for split in splits_filter:
            fname = os.path.join(args.repope_dir, f"coco_repope_{split}.json")
            if os.path.exists(fname):
                with open(fname) as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if _line:
                            _e = _json.loads(_line)
                            repope_labels[(split, str(_e["question_id"]))] = _e["label"].lower()
        print(f"  [RePOPE] Loaded {len(repope_labels)} corrected labels")

    ds   = hf_load("lmms-lab/POPE", split="test")
    rows = [r for r in ds
            if str(r.get("category", r.get("type", ""))).lower() in splits_filter]
    if repope_labels:
        n_orig = len(rows)
        rows   = [r for r in rows
                  if (str(r.get("category", r.get("type", ""))).lower(),
                      str(r["question_id"])) in repope_labels]
        print(f"  [RePOPE] {len(rows)}/{n_orig} samples retained ({n_orig - len(rows)} removed as ambiguous)")
        label_str = "RePOPE"
    else:
        label_str = "POPE"
    rng  = random.Random(args.seed)
    rng.shuffle(rows)
    if args.n_pope > 0:
        rows = rows[:args.n_pope]
    print(f"  Loaded {len(rows)} {label_str} samples ({splits_label})")

    def _empty_stats():
        return {b: {"correct": 0, "tp": 0, "fp": 0, "fn": 0, "n": 0} for b in gammas}

    # per-split stats + overall
    split_stats: dict[str, dict] = {s: _empty_stats() for s in splits_filter}
    overall_stats = _empty_stats()
    is_llava = "llava" in args.model.lower()

    for i, r in enumerate(rows):
        image    = r["image"].convert("RGB")
        q        = str(r["question"]).strip() + "\nAnswer with Yes or No only."
        split_key = str(r.get("category", r.get("type", ""))).lower()
        gt    = (repope_labels[(split_key, str(r["question_id"]))] if repope_labels
                 else ("yes" if str(r.get("answer", "")).strip().lower() == "yes" else "no"))

        if is_llava:
            inp, s, e = _encode_llava(processor, model, device, image, q)
        else:
            msgs  = [{"role": "user", "content": [{"type": "image", "image": image},
                                                   {"type": "text",  "text":  q}]}]
            text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            vis, _ = process_vision_info(msgs)
            inp    = processor(text=[text], images=vis, return_tensors="pt",
                               padding=True).to(device)
            s, e   = get_img_range(inp["input_ids"], img_token_id)

        method_mod.prepare_sample(inp, s, e, image, q, model, processor)
        for gamma in gammas:
            logits = method_get_logits(method_mod, model, inp, gamma)
            pred = "yes" if decode_first_token(logits, processor).startswith("yes") else "no"
            for stats in (split_stats.get(split_key, {}), overall_stats):
                if not stats:
                    continue
                st = stats[gamma]
                st["n"] += 1
                if pred == gt:
                    st["correct"] += 1
                if pred == "yes" and gt == "yes":
                    st["tp"] += 1
                elif pred == "yes" and gt == "no":
                    st["fp"] += 1
                elif pred == "no" and gt == "yes":
                    st["fn"] += 1
        method_mod.cleanup()

        if (i + 1) % 500 == 0 or (i + 1) == len(rows):
            n_so_far = i + 1
            srf_str = "  ".join(
                f"γ={b}:{overall_stats[b]['correct']/n_so_far:.4f}" for b in gammas
            )
            print(f"  [{n_so_far:5d}/{len(rows)}]  {srf_str}")

    def _compute_metrics(st: dict) -> dict:
        n    = st["n"]
        if n == 0:
            return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
        acc  = st["correct"] / n
        tp, fp, fn = st["tp"], st["fp"], st["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "n": n}

    results_out: dict = {"splits": {}, "overall": {}}
    for b in gammas:
        label = f"γ={b}" if args.method == "srfe" else "SRF"
        print(f"\n{'─'*56}")
        print(f"POPE  {label}")
        split_metrics = {}
        for sp in sorted(splits_filter):
            m = _compute_metrics(split_stats[sp][b])
            split_metrics[sp] = m
            print(f"  {sp:12s}  acc={m['acc']:.4f}  prec={m['precision']:.4f}  "
                  f"rec={m['recall']:.4f}  F1={m['f1']:.4f}  (n={m['n']})")
        ov = _compute_metrics(overall_stats[b])
        print(f"  {'overall':12s}  acc={ov['acc']:.4f}  prec={ov['precision']:.4f}  "
              f"rec={ov['recall']:.4f}  F1={ov['f1']:.4f}  (n={ov['n']})")
        results_out["splits"][b] = split_metrics
        results_out["overall"][b] = ov

    return results_out


# ---------------------------------------------------------------------------
# MMVP
# ---------------------------------------------------------------------------

def run_mmvp(method_mod, model, processor, img_token_id, device, args) -> dict:
    import pandas as pd
    gammas = args.gamma if args.method == "srfe" else [0.0]

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

    pair_srf = {b: defaultdict(dict) for b in gammas}
    n_srf_ok = {b: 0 for b in gammas}

    for csv_1idx in range(1, 301):
        row_idx  = csv_1idx - 1
        row      = df.iloc[row_idx]
        opts     = re.findall(r'\(([ab])\)\s*([^(]+)', str(row["Options"]), re.IGNORECASE)
        if not opts:
            continue
        opt_text = "\n".join(f"{m[0].upper()}. {m[1].strip()}" for m in opts)
        # NOTE: "from the given choices" added to match validated autoresearch harness (44%).
        # Old prompt "Answer with the option's letter directly." + raw logit compare gave ~40.7%.
        prompt   = f"{row['Question']}\n{opt_text}\nAnswer with the option's letter from the given choices directly."
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

        method_mod.prepare_sample(inp, s, e, image, row["Question"], model, processor)
        for gamma in gammas:
            # NOTE: switched from raw A/B logit comparison to model.generate() + text decode.
            # Raw logit path (kept below for reference) gave ~40.7%; generate() gives ~44%.
            # --- old logit path ---
            # logits = method_get_logits(method_mod, model, inp, gamma)
            # pred   = "A" if logits[0, a_id] >= logits[0, b_id] else "B"
            # --- new generate path ---
            tok_ids = method_generate(method_mod, model, inp, processor, gamma,
                                      max_new_tokens=16)
            raw  = processor.decode(tok_ids, skip_special_tokens=True)
            pred = _decode_ab(raw)
            ok   = (pred == gt)
            if ok:
                n_srf_ok[gamma] += 1
            pair_srf[gamma][pair_id][img_key] = ok
        method_mod.cleanup()

        if csv_1idx % 60 == 0:
            srf_str = "  ".join(f"γ={b}:{n_srf_ok[b]/csv_1idx:.4f}" for b in gammas)
            print(f"  [{csv_1idx:3d}/300]  {srf_str}")

    def _pair_acc(pd_):
        pairs = [(pid, r) for pid, r in pd_.items() if "a" in r and "b" in r]
        return sum(1 for _, r in pairs if r["a"] and r["b"]) / max(1, len(pairs))

    acc_srf  = {b: _pair_acc(pair_srf[b]) for b in gammas}
    img_srf  = {b: n_srf_ok[b] / 300 for b in gammas}

    for b in gammas:
        label = f"β={b}" if args.method == "srfe" else "SRF"
        print(f"\nMMVP  {label}: pair={acc_srf[b]:.4f}  img={img_srf[b]:.4f}")

    return {"method_pair": acc_srf, "method_img": img_srf}


# ---------------------------------------------------------------------------
# VLM Bias
# ---------------------------------------------------------------------------

def run_vlmbias(method_mod, model, processor, img_token_id, device, args) -> dict:
    gammas     = args.gamma if args.method == "srfe" else [0.0]
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

    correct_srf  = {b: 0 for b in gammas}
    # per-category tracking: {gamma: {cat: [correct, total]}}
    cat_srf = {b: {cat: [0, 0] for cat in CFG.VLM_BIAS_CATEGORIES} for b in gammas}

    for i, sample in enumerate(samples):
        msgs  = [{"role": "user", "content": [{"type": "image", "image": sample["image"]},
                                               {"type": "text",  "text":  sample["question"]}]}]
        text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = process_vision_info(msgs)
        inp    = processor(text=[text], images=vis, return_tensors="pt",
                           padding=True).to(device)
        s, e   = get_img_range(inp["input_ids"], img_token_id)

        method_mod.prepare_sample(inp, s, e, sample["image"], sample["question"],
                                   model, processor)
        # SRF-E: skip contrastive on the '{' format prefix token (content_offset=1)
        _vlmbias_offset = 1 if hasattr(method_mod, "generate_contrastive") else 0
        for gamma in gammas:
            gen_ids = method_generate(method_mod, model, inp, processor,
                                      gamma=gamma, max_new_tokens=20,
                                      content_offset=_vlmbias_offset)
            raw_c   = processor.decode(gen_ids, skip_special_tokens=True)
            pred_c  = normalise(extract_answer(raw_c))
            hit = int(pred_c == sample["gt"])
            correct_srf[gamma] += hit
            cat_srf[gamma][sample["category"]][0] += hit
            cat_srf[gamma][sample["category"]][1] += 1
        method_mod.cleanup()

        if (i + 1) % 20 == 0:
            n = i + 1
            srf_str = "  ".join(f"γ={b}:{correct_srf[b]/n:.4f}" for b in gammas)
            print(f"  [{n:4d}/{len(samples)}]  {srf_str}")

    total   = len(samples)
    acc_srf = {b: correct_srf[b] / total for b in gammas}

    for b in gammas:
        label = f"SRF" if args.method != "srfe" else f"γ={b}"
        print(f"\nVLM Bias  {label}: {acc_srf[b]:.4f}  ({correct_srf[b]}/{total})")
        print(f"  {'Category':<20}  {'Correct':>7}  {'Total':>6}  {'Acc':>6}")
        print(f"  {'-'*20}  {'-'*7}  {'-'*6}  {'-'*6}")
        for cat in CFG.VLM_BIAS_CATEGORIES:
            c, t = cat_srf[b][cat]
            print(f"  {cat:<20}  {c:>7}  {t:>6}  {c/t:>6.1%}")

    # per_category in result: {cat: {"correct": c, "total": t, "acc": c/t}}
    per_category = {
        b: {cat: {"correct": cat_srf[b][cat][0],
                  "total":   cat_srf[b][cat][1],
                  "acc":     cat_srf[b][cat][0] / max(cat_srf[b][cat][1], 1)}
            for cat in CFG.VLM_BIAS_CATEGORIES}
        for b in gammas
    }

    return {"n": total, "method": acc_srf, "per_category": per_category}


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
    gammas = args.gamma if args.method == "srfe" else [0.0]

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
    n_total     = len(samples)
    correct_srf = {b: 0 for b in gammas}

    # Per-category: {cat: {"srf": {b: 0}, "total": 0}}
    cat_stats: dict = defaultdict(lambda: {"srf": {b: 0 for b in gammas}, "total": 0})

    # Pair accuracy: {pair_id: {"srf": {b: []}}}
    pair_stats: dict = defaultdict(lambda: {"srf": {b: [] for b in gammas}})

    for i, r in enumerate(samples):
        image    = r["image"].convert("RGB")
        q        = str(r["question"]).strip()
        gt_lower = str(r["answer"]).strip().lower()
        cat      = str(r.get("category", "unknown")).strip()
        pair_id  = str(r.get("question_id", f"{cat}_{i}"))

        msgs  = [{"role": "user", "content": [{"type": "image", "image": image},
                                               {"type": "text",  "text":  q}]}]
        text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = process_vision_info(msgs)
        inp    = processor(text=[text], images=vis, return_tensors="pt",
                           padding=True).to(device)
        s, e   = get_img_range(inp["input_ids"], img_token_id)

        method_mod.prepare_sample(inp, s, e, image, q, model, processor)
        for gamma in gammas:
            logits = method_get_logits(method_mod, model, inp, gamma)
            pred   = "yes" if logits[0, yes_id] >= logits[0, no_id] else "no"
            ok     = (pred == gt_lower)
            if ok:
                correct_srf[gamma] += 1
            cat_stats[cat]["srf"][gamma]  += int(ok)
            cat_stats[cat]["total"]      += 1 if gamma == gammas[0] else 0
            pair_stats[pair_id]["srf"][gamma].append(ok)
        method_mod.cleanup()

        if (i + 1) % 500 == 0 or (i + 1) == n_total:
            n = i + 1
            srf_str = "  ".join(f"γ={b}:{correct_srf[b]/n:.4f}" for b in gammas)
            print(f"  [{n:4d}/{n_total}]  {srf_str}")

    # ── Compute pair accuracy ──────────────────────────────────────────────────
    def _pair_acc(gamma):
        pairs = [v for v in pair_stats.values() if len(v["srf"][gamma]) == 2]
        if not pairs:
            return 0.0
        return sum(1 for p in pairs if all(p["srf"][gamma])) / len(pairs)

    acc_srf  = {b: correct_srf[b] / n_total for b in gammas}
    pair_srf = {b: _pair_acc(b) for b in gammas}

    # ── Perception / Cognition sub-scores ─────────────────────────────────────
    perc_srf: dict = {}
    cogn_srf: dict = {}
    for b in gammas:
        perc, cogn = 0, 0
        for cat, s in cat_stats.items():
            if cat in _MME_COGNITION:
                cogn += s["srf"][b]
            else:
                perc += s["srf"][b]
        perc_srf[b], cogn_srf[b] = perc, cogn

    # ── Print results ──────────────────────────────────────────────────────────
    for b in gammas:
        label = f"β={b}" if args.method == "srfe" else "SRF"
        print(f"\nMME  {label}: acc={acc_srf[b]:.4f}  pair={pair_srf[b]:.4f}  "
              f"score={correct_srf[b]} (perc={perc_srf[b]}, cogn={cogn_srf[b]})")

    print("\n  Per-category (SRF):")
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        srf_str = "  ".join(f"γ={b}:{s['srf'][b]}" for b in gammas)
        marker = "[C]" if cat in _MME_COGNITION else "[P]"
        print(f"    {marker} {cat:28s}: {srf_str}/{s['total']}")

    return {
        "n": n_total,
        "method_acc":        acc_srf,
        "method_pair":       pair_srf,
        "method_score":      correct_srf,
        "method_perception": perc_srf,
        "method_cognition":  cogn_srf,
        "cat_stats": {k: dict(v) for k, v in cat_stats.items()},
    }


# ---------------------------------------------------------------------------
# VLind-Bench
# ---------------------------------------------------------------------------

def _get_vlind_data_dir(download: bool = False) -> pathlib.Path:
    """Return the local VLind-Bench dataset root.

    By default uses only locally cached files (local_files_only=True).
    Pass download=True (via --vlind_download flag) to trigger a full HF download.
    """
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        repo_id=CFG.VLIND_BENCH_REPO_ID,
        repo_type="dataset",
        cache_dir=os.environ.get("HF_HOME"),
        local_files_only=not download,
    )
    return pathlib.Path(local_dir) / "VLind-Bench Dataset"


def run_vlindbench(method_mod, model, processor, img_token_id, device, args) -> dict:
    """VLind-Bench evaluation.

    Each of 421 counterfactual samples has:
      - counterfactual image (best_img_id): an AI-generated image showing a surprising
        but visually verified scene (e.g. swans in desert sand).
      - true_statement: what IS true per the image (e.g. "The swans are found in desert sands.")
      - false_statement: what language priors say (e.g. "Swans live near water.")
      - existent_noun: the main object in the image — used directly for CLIP saliency.

    Two questions per sample (True/False format):
      Q1: "Statement: {true_statement} ... is the given statement true or false?"  → GT: true
      Q2: "Statement: {false_statement} ... is the given statement true or false?" → GT: false

    Metrics:
      - q_acc:   question-level accuracy (correct / total_questions)
      - pair_acc: both Q1 and Q2 correct for the same image (=resistance to language priors)
      - per_concept breakdown
    """
    gammas = args.gamma if args.method == "srfe" else [0.0]

    print("\n" + "="*60)
    print(f"DATASET: VLind-Bench (421 counterfactual samples, 10 concepts)")
    print("="*60)

    method_mod.reset_for_dataset(dataset="vlind", **_reset_overrides(args))

    data_dir = _get_vlind_data_dir(download=args.vlind_download)
    with open(data_dir / "data.json") as f:
        items = json.load(f)

    _prompt_template = (
        "Only respond in True or False.\n"
        "Statement: {statement}\n"
        "Based on the image, is the given statement true or false?"
    )

    # Filter to samples with at least one good image (vote >= thresh)
    thresh = CFG.VLIND_BENCH_VOTE_THRESH
    valid_items = [
        it for it in items
        if any(v >= thresh for v in it["aggregated_human_label_good_images"].values())
    ]
    print(f"  Valid samples (≥{thresh} votes): {len(valid_items)} / {len(items)}")

    if getattr(args, "vlind_n", None) is not None:
        import random as _rng
        _rng.seed(getattr(args, "seed", 42))
        _rng.shuffle(valid_items)
        valid_items = valid_items[:args.vlind_n]
        print(f"  Subsetting to {len(valid_items)} samples (vlind_n={args.vlind_n})")

    def _load_image(item: dict):
        cf_dir = (data_dir / "images" / "counterfactual"
                  / item["concept"]
                  / f"{item['context_id']}_{item['context']}")
        img_path = cf_dir / f"{item['best_img_id']}.jpg"
        from PIL import Image as _PIL
        return _PIL.open(img_path).convert("RGB")

    correct_q   = {b: 0 for b in gammas}   # question-level
    correct_pair = {b: 0 for b in gammas}   # pair-level (both Q1+Q2 correct)
    # per-concept: {gamma: {concept: [pair_correct, total_pairs]}}
    cat_srf = {b: {c: [0, 0] for c in CFG.VLIND_BENCH_CONCEPTS} for b in gammas}

    total_q = len(valid_items) * 2   # 2 questions per sample

    for i, item in enumerate(valid_items):
        try:
            image = _load_image(item)
        except FileNotFoundError:
            print(f"  [WARN] image not found for item {item.get('global_id', i)}, skipping")
            total_q -= 2
            continue

        concept = item["concept"]

        # Extract dual nouns per statement:
        #   subject_noun  → saliency MAP  (localise the subject in the image)
        #   context_noun  → presence GATE (is the counterfactual environment present?)
        # Q1 (TRUE):  context should be IN the image → gate fires → boost subject tokens
        # Q2 (FALSE): context should NOT be in image → gate fails → suppression/baseline
        q1_subj, q1_ctx = extract_vlind_nouns(item["true_statement"])
        q2_subj, q2_ctx = extract_vlind_nouns(item["false_statement"])
        noun_pairs = [(q1_subj, q1_ctx), (q2_subj, q2_ctx)]

        q1_text = _prompt_template.format(statement=item["true_statement"])
        q2_text = _prompt_template.format(statement=item["false_statement"])

        pair_results: dict[float, bool] = {}

        for qi, (q_text, gt, (subj_noun, ctx_noun)) in enumerate(
                zip([q1_text, q2_text], ["true", "false"], noun_pairs)):
            msgs  = [{"role": "user", "content": [{"type": "image", "image": image},
                                                   {"type": "text",  "text":  q_text}]}]
            text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            vis, _ = process_vision_info(msgs)
            inp    = processor(text=[text], images=vis, return_tensors="pt",
                               padding=True).to(device)
            s, e   = get_img_range(inp["input_ids"], img_token_id)

            # subject_noun → saliency map; context_noun → presence gate
            method_mod.prepare_sample(inp, s, e, image, q_text, model, processor,
                                      noun_override=subj_noun,
                                      gate_noun_override=ctx_noun)

            for gamma in gammas:
                tok_ids = method_generate(method_mod, model, inp, processor, gamma,
                                          max_new_tokens=5)
                raw  = processor.decode(tok_ids, skip_special_tokens=True).strip().lower()
                pred = "true" if raw.startswith("true") else "false"
                hit  = int(pred == gt)
                correct_q[gamma] += hit
                if qi == 0:
                    pair_results[gamma] = bool(hit)
                else:
                    pair_hit = int(pair_results[gamma] and bool(hit))
                    correct_pair[gamma] += pair_hit
                    cat_srf[gamma][concept][0] += pair_hit
                    cat_srf[gamma][concept][1] += 1

            method_mod.cleanup()

        if (i + 1) % 50 == 0 or (i + 1) == len(valid_items):
            n_pairs = sum(cat_srf[gammas[0]][c][1] for c in CFG.VLIND_BENCH_CONCEPTS)
            n_q_done = max(total_q - (len(valid_items) - (i + 1)) * 2, 1)
            srf_str = "  ".join(
                f"γ={b}: q_acc={correct_q[b]/n_q_done:.4f} pair={correct_pair[b]/max(n_pairs,1):.4f}"
                for b in gammas
            )
            print(f"  [{i+1:4d}/{len(valid_items)}]  {srf_str}")

    n_pairs = sum(cat_srf[gammas[0]][c][1] for c in CFG.VLIND_BENCH_CONCEPTS)
    q_acc   = {b: correct_q[b] / max(total_q, 1) for b in gammas}
    pair_acc = {b: correct_pair[b] / max(n_pairs, 1) for b in gammas}

    for b in gammas:
        label = "SRF" if args.method != "srfe" else f"γ={b}"
        print(f"\nVLind  {label}: q_acc={q_acc[b]:.4f}  pair_acc={pair_acc[b]:.4f}  "
              f"({correct_pair[b]}/{n_pairs} pairs correct)")
        print(f"  {'Concept':<12}  {'Pairs':>6}  {'Correct':>7}  {'Acc':>6}")
        print(f"  {'-'*12}  {'-'*6}  {'-'*7}  {'-'*6}")
        for c in CFG.VLIND_BENCH_CONCEPTS:
            corr, tot = cat_srf[b][c]
            acc_str = f"{corr/tot:.1%}" if tot > 0 else "—"
            print(f"  {c:<12}  {tot:>6}  {corr:>7}  {acc_str:>6}")

    per_concept = {
        b: {c: {"correct": cat_srf[b][c][0],
                "total":   cat_srf[b][c][1],
                "pair_acc": cat_srf[b][c][0] / max(cat_srf[b][c][1], 1)}
            for c in CFG.VLIND_BENCH_CONCEPTS}
        for b in gammas
    }

    return {
        "n_samples": len(valid_items),
        "n_pairs":   n_pairs,
        "q_acc":     q_acc,
        "pair_acc":  pair_acc,
        "per_concept": per_concept,
    }


# ---------------------------------------------------------------------------
# MMHal-Bench
# ---------------------------------------------------------------------------

# Exact scoring template from ILVAD/eval/mmhal_bench.py — do not modify,
# ensures our scores are directly comparable to Table 1 in the ILVAD paper.
_MMHAL_SCORING_TEMPLATE = '''Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

For clarity, consider these examples:

### Example 1:
- Image Contents: Lynx, Cat
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
- Hallucination: Yes.

### Example 2:
- Image Contents: Tree, Plant, Panda, Animal
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
- Hallucination: No.

### Example 3:
- Image Contents: Fashion accessory, Mammal, Dog, Umbrella
- Question: Describe this image in details.
- Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
- LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
- Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
- Hallucination: Yes.

### Example 4:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
- Hallucination: No.

### Example 5:
- Image Contents: Tree, Tower
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: No.

With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM\'s response and the factual information provided in the image contents, question, and the standard human-generated answer below.

Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evalutation. LMM\'s detailed analysis or reasoning should be encouraged.

To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response by choosing from the following options:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination

### Image Contents
{}

### Question
{}

### Standard Human-Generated Answer
{}

### LMM Response to Evaluate
{}
'''

_MMHAL_DEFAULT_JSON   = pathlib.Path("/home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json")
_MMHAL_DEFAULT_IMAGES = pathlib.Path("/home/sgowda/workspace/ILVAD/data/MMHal-Bench/images")


def _find_mmhalbench_json(cli_path: str | None) -> pathlib.Path:
    if cli_path:
        p = pathlib.Path(cli_path)
        if not p.exists():
            raise FileNotFoundError(f"--mmhalbench_json not found: {p}")
        return p
    if _MMHAL_DEFAULT_JSON.exists():
        return _MMHAL_DEFAULT_JSON
    raise FileNotFoundError(
        "MMHal-Bench JSON not found. Pass --mmhalbench_json <path/to/response_template.json>"
    )


def _score_mmhal_with_gpt(records: list[dict], openai_model: str) -> list[int]:
    """Call GPT-4 to score all records. Returns list of 0–6 scores."""
    import openai as _openai
    import time

    client = _openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    scores = []
    for i, rec in enumerate(records):
        image_content = ", ".join(rec["image_content"])
        prompt = _MMHAL_SCORING_TEMPLATE.format(
            image_content, rec["question"], rec["gt_answer"], rec["model_answer"]
        )
        response = None
        while response is None:
            try:
                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                    temperature=0.0,
                    max_tokens=512,
                )
            except Exception as e:
                print(f"    GPT error ({e}), retrying in 10s…")
                time.sleep(10)

        text = response.choices[0].message.content
        scores_found = [s for s in range(7) if f"rating: {s}" in text.lower()]
        if len(scores_found) == 1:
            scores.append(scores_found[0])
        else:
            print(f"  [WARN] sample {i}: ambiguous rating in GPT response → defaulting 0")
            print(f"    {text[:200]}")
            scores.append(0)
        time.sleep(1)
    return scores


def run_mmhalbench(method_mod, model, processor, img_token_id, device, args) -> dict:
    """MMHal-Bench evaluation — 96 open-ended image-question pairs.

    Metrics (matching ILVAD paper Table 1):
      Score  = average GPT-4 rating 0–6 (higher is better)
      Hal%   = fraction of responses with score < 3 (lower is better)

    Response generation always runs and is saved to output_dir/mmhalbench_responses.json.
    GPT-4 scoring runs only if OPENAI_API_KEY is set; otherwise just save responses.
    To score later: python srf/score_mmhalbench.py --response <responses.json> --api-key ...
    """
    from PIL import Image as _PIL

    gammas   = args.gamma if args.method == "srfe" else [0.0]
    is_llava = "llava" in args.model.lower()

    print("\n" + "=" * 60)
    print("DATASET: MMHal-Bench (96 open-ended, 8 question types)")
    print("=" * 60)

    method_mod.reset_for_dataset(dataset="mmhalbench", **_reset_overrides(args))

    json_path  = _find_mmhalbench_json(getattr(args, "mmhalbench_json", None))
    images_dir = json_path.parent / "images"
    with open(json_path) as f:
        template_records = json.load(f)

    n_limit = getattr(args, "mmhalbench_n", None)
    if n_limit:
        template_records = template_records[:n_limit]
    print(f"  Loaded {len(template_records)} samples from {json_path}")

    # Per-gamma filled record lists (deep copy so we don't share mutable dicts)
    import copy
    filled: dict[float, list[dict]] = {b: copy.deepcopy(template_records) for b in gammas}

    for i, rec in enumerate(template_records):
        img_fname = rec["image_src"].split("/")[-1]
        image     = _PIL.open(images_dir / img_fname).convert("RGB")
        question  = rec["question"]

        if is_llava:
            inp, s, e = _encode_llava(processor, model, device, image, question)
        else:
            msgs   = [{"role": "user", "content": [{"type": "image", "image": image},
                                                    {"type": "text",  "text":  question}]}]
            text   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            vis, _ = process_vision_info(msgs)
            inp    = processor(text=[text], images=vis, return_tensors="pt",
                               padding=True).to(device)
            s, e   = get_img_range(inp["input_ids"], img_token_id)

        method_mod.prepare_sample(inp, s, e, image, question, model, processor)

        for gamma in gammas:
            tok_ids  = method_generate(method_mod, model, inp, processor, gamma,
                                       max_new_tokens=512)
            response = processor.decode(tok_ids, skip_special_tokens=True).strip()
            filled[gamma][i]["model_answer"] = response

        method_mod.cleanup()

        if (i + 1) % 24 == 0 or (i + 1) == len(template_records):
            print(f"  [{i+1:3d}/{len(template_records)}]  responses generated")

    # ── Save responses (always) ────────────────────────────────────────────────
    if args.output:
        out = pathlib.Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        for gamma in gammas:
            suffix = f"_g{gamma}" if args.method == "srfe" else ""
            resp_path = out / f"mmhalbench_responses{suffix}.json"
            with open(resp_path, "w") as f:
                json.dump(filled[gamma], f, indent=2)
            print(f"  Responses saved → {resp_path}")

    # ── GPT-4 scoring (optional) ───────────────────────────────────────────────
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    scores_by_gamma: dict[float, dict] = {}

    if openai_key:
        openai_model = getattr(args, "openai_model", "gpt-4o")
        print(f"\n  Scoring with {openai_model} …")
        for gamma in gammas:
            label = "SRF" if args.method != "srfe" else f"γ={gamma}"
            print(f"  [{label}]")
            raw_scores = _score_mmhal_with_gpt(filled[gamma], openai_model)

            hallucination = [1 if s < 3 else 0 for s in raw_scores]
            avg_score = sum(raw_scores) / len(raw_scores)
            hal_pct   = 100.0 * sum(hallucination) / len(hallucination)

            # Per-question-type breakdown (index % 8 matches ILVAD convention)
            per_type: dict[str, list[int]] = {}
            for idx, rec in enumerate(filled[gamma]):
                qt = rec["question_type"]
                per_type.setdefault(qt, []).append(raw_scores[idx])

            scores_by_gamma[gamma] = {
                "score":   avg_score,
                "hal_pct": hal_pct,
                "n":       len(raw_scores),
                "raw_scores": raw_scores,
                "per_type": {qt: sum(vs)/len(vs) for qt, vs in per_type.items()},
            }

            print(f"\nMMHal-Bench  {label}: Score={avg_score:.2f}  Hal%={hal_pct:.1f}%")
            print(f"  Per question type:")
            for qt, vs in per_type.items():
                print(f"    {qt:<14s}: {sum(vs)/len(vs):.2f}  (n={len(vs)})")

            if args.output:
                score_path = pathlib.Path(args.output) / f"mmhalbench_scores{'' if args.method != 'srfe' else f'_g{gamma}'}.json"
                with open(score_path, "w") as f:
                    json.dump(scores_by_gamma[gamma], f, indent=2)
                print(f"  Scores saved → {score_path}")
    else:
        print("\n  [INFO] OPENAI_API_KEY not set — responses saved, GPT-4 scoring skipped.")
        print("  To score: python srf/score_mmhalbench.py --response <responses.json> --api-key sk-...")

    return {
        "n": len(template_records),
        "scores": scores_by_gamma,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Import method module (srf/ is on sys.path — import directly by filename)
    if args.method == "srf":
        import srf as method_mod
    elif args.method == "srfe":
        import srf_e as method_mod
    elif args.method == "vcd":
        import vcd as method_mod
    elif args.method == "vaf":
        import vaf as method_mod
    elif args.method == "baseline":
        import baseline as method_mod
    else:
        raise ValueError(f"Unknown method: {args.method}")

    model, processor = load_model(args.model)
    arch = CFG.get_arch(args.model)
    if arch["image_token"] is not None:
        img_token_id = processor.tokenizer.convert_tokens_to_ids(arch["image_token"])
    else:
        img_token_id = model.config.image_token_index   # LLaVA-style
    device = next(model.parameters()).device

    if args.method == "srfe":
        gamma_str = f"β={args.gamma}"
    else:
        gamma_str = "base"
    print(f"\n[{args.method.upper()}] Setup  model={args.model}  {gamma_str}")

    # Inject the correct patch module into srf.py and srf_e.py.
    # Must happen before setup() so calibration uses the right patch.
    if args.method != "baseline":
        import srf as _srf_base
        _srf_base.patch = patch
        if hasattr(method_mod, "patch"):
            method_mod.patch = patch

    # LLaVA only: set boost mode before setup (controls multiplicative vs additive).
    if hasattr(patch, "_STATE") and args.llava_boost_mode is not None:
        patch._STATE["boost_mode"] = args.llava_boost_mode

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

    if "vlind" in args.datasets:
        results["vlind"] = run_vlindbench(method_mod, model, processor, img_token_id, device, args)
        save_results(results["vlind"], args.output, "vlind")

    if "mmhalbench" in args.datasets:
        results["mmhalbench"] = run_mmhalbench(method_mod, model, processor, img_token_id, device, args)
        save_results(results["mmhalbench"], args.output, "mmhalbench")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"FINAL SUMMARY — {args.method.upper()}")
    print("="*70)
    print(f"Model: {args.model}  |  method={args.method}  |  {gamma_str}")

    gammas = args.gamma if args.method == "srfe" else [0.0]

    def _p(v): return f"{v*100:.2f}%"

    if "mmvp" in results:
        r = results["mmvp"]
        for b in gammas:
            label = f"β={b}" if args.method == "srfe" else "SRF"
            print(f"\nMMVP  {label}: pair={_p(r['method_pair'][b])}  img={_p(r['method_img'][b])}")

    if "pope" in results:
        r = results["pope"]
        for b in gammas:
            label = f"β={b}" if args.method == "srfe" else "SRF"
            ov = r["overall"][b]
            print(f"\nPOPE (n={ov['n']})  {label}: "
                  f"acc={_p(ov['acc'])}  prec={_p(ov['precision'])}  "
                  f"rec={_p(ov['recall'])}  F1={_p(ov['f1'])}")

    if "vlmbias" in results:
        r = results["vlmbias"]
        for b in gammas:
            label = f"β={b}" if args.method == "srfe" else "SRF"
            print(f"\nVLM Bias (n={r['n']})  {label}: {_p(r['method'][b])}")

    if "mme" in results:
        r = results["mme"]
        for b in gammas:
            label = f"β={b}" if args.method == "srfe" else "SRF"
            print(f"\nMME  {label}: score={r['method_score'][b]}  acc={_p(r['method_acc'][b])}  "
                  f"pair={_p(r['method_pair'][b])}  "
                  f"perc={r['method_perception'][b]}  cogn={r['method_cognition'][b]}")

    if "vlind" in results:
        r = results["vlind"]
        for b in gammas:
            label = f"β={b}" if args.method == "srfe" else "SRF"
            print(f"\nVLind-Bench (n={r['n_samples']})  {label}: "
                  f"pair_acc={_p(r['pair_acc'][b])}  q_acc={_p(r['q_acc'][b])}")

    if "mmhalbench" in results:
        r = results["mmhalbench"]
        if r["scores"]:
            for b in gammas:
                if b in r["scores"]:
                    s = r["scores"][b]
                    label = "SRF" if args.method != "srfe" else f"γ={b}"
                    print(f"\nMMHal-Bench (n={r['n']})  {label}: "
                          f"Score={s['score']:.2f}  Hal%={s['hal_pct']:.1f}%")
        else:
            print(f"\nMMHal-Bench (n={r['n']}): responses saved, scoring pending (no OPENAI_API_KEY)")

    if args.output:
        save_results(results, args.output, "summary")
        print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main()
