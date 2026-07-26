#!/usr/bin/env python3
"""
20-sample diagnostic: debug why SRF = baseline on POPE prefill evaluation.

Checks:
  1. CLIP saliency gate — how many samples are "absent"? Save images + heatmaps.
  2. Phase gate — confirm phase="generation" is no-op during prefill (model(**inp))
  3. Compare: baseline logits vs SRF(phase=generation) vs SRF(phase=both)

Usage:
    cd /volumes2/mllm/lmms-eval
    conda run -n mllm python srf/diag_20sample.py \
        --repope_dir /volumes2/mllm/RePOPE \
        --out_dir results/diag_20sample
"""
from __future__ import annotations

import argparse
import os
import pathlib
import random
import sys

_SRF_DIR      = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR / "saliency"))
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import config as CFG
os.environ.setdefault("HF_HOME", CFG.HF_HOME)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datasets import load_dataset as hf_load

import qwen_attn_patch as patch
import srf
import clip_salience as clip_sal
from noun_extract import extract_clip_noun


# ---------------------------------------------------------------------------
# Arg parse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default=CFG.DEFAULT_MODEL)
    p.add_argument("--repope_dir",  default="/volumes2/mllm/RePOPE")
    p.add_argument("--out_dir",     default="results/diag_20sample")
    p.add_argument("--n",           type=int, default=20)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--splits",      nargs="+", default=["adversarial", "popular", "random"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_img_range(input_ids: torch.Tensor, img_token_id: int):
    ids   = input_ids[0].tolist()
    start = next(i for i, t in enumerate(ids) if t == img_token_id)
    end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return start, end


def logits_for_phase(model, inp, phase: str) -> torch.Tensor:
    """Run prefill pass with given SRF phase. Returns logits[1, vocab]."""
    patch._STATE["srf_apply_phase"] = phase
    patch._STATE["method"] = "srf"
    with torch.inference_mode():
        out = model(**inp)
    return out.logits[:, -1, :].float()


def baseline_logits(model, inp) -> torch.Tensor:
    """Run prefill pass with method=baseline (no SRF)."""
    patch._STATE["method"] = "baseline"
    with torch.inference_mode():
        out = model(**inp)
    return out.logits[:, -1, :].float()


def generate_answer(model, inp, phase: str, processor, max_new_tokens: int = 5) -> str:
    """model.generate() with SRF patch active at given phase."""
    patch._STATE["srf_apply_phase"] = phase
    patch._STATE["method"] = "srf"
    with torch.inference_mode():
        out_ids = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out_ids[0, inp["input_ids"].shape[1]:]
    return processor.decode(new_ids, skip_special_tokens=True).strip().lower()


def baseline_generate(model, inp, processor, max_new_tokens: int = 5) -> str:
    """model.generate() baseline (no SRF)."""
    patch._STATE["method"] = "baseline"
    with torch.inference_mode():
        out_ids = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out_ids[0, inp["input_ids"].shape[1]:]
    return processor.decode(new_ids, skip_special_tokens=True).strip().lower()


def sal_to_heatmap_array(saliency: torch.Tensor) -> np.ndarray:
    """Convert saliency vector to (H_g, W_g) float32 [0,1] for imshow."""
    sal = saliency.cpu().float().numpy()
    n   = len(sal)
    g   = int(round(n ** 0.5))
    if g * g != n:
        g = int(n ** 0.5)
    grid = sal[:g*g].reshape(g, g)
    mn, mx = grid.min(), grid.max()
    return (grid - mn) / (mx - mn + 1e-8)


def save_sample_figure(records: list, out_path: str) -> None:
    """
    One matplotlib row per sample:
      col 0: original image  (question + GT label)
      col 1: saliency heatmap overlay (jet, alpha=0.45)
      col 2: CLIP gate statistics
      col 3: answer comparison table (baseline / SRF-gen / SRF-both)
    """
    n = len(records)
    fig, axes = plt.subplots(n, 4, figsize=(22, 6.0 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle("SRF 20-Sample Diagnostic", fontsize=13, fontweight="bold")

    for i, rec in enumerate(records):
        image  = rec["_image"]
        result = rec["_clip_result"]
        noun   = rec["noun"]
        gt     = rec["gt"]
        q      = rec["question"]
        split  = rec["split"]

        # ── Col 0: original image ─────────────────────────────────────────────
        axes[i][0].imshow(image)
        axes[i][0].axis("off")
        q_wrap = "\n".join([q[j:j+45] for j in range(0, min(len(q), 135), 45)])
        bl_ans = rec["bl_gen_ans"]
        gate_correct = (result is not None
                        and result.object_present == (gt == "yes"))
        title_col = "green" if gate_correct else "red"
        axes[i][0].set_title(
            f"[{split}] {q_wrap}\nGT={gt.upper()}  noun='{noun}'",
            fontsize=7, loc="left", pad=3, color=title_col,
        )

        # ── Col 1: saliency heatmap overlay ──────────────────────────────────
        axes[i][1].imshow(image)
        if result is not None:
            if result.object_present and result.saliency is not None:
                hm = sal_to_heatmap_array(result.saliency)
                axes[i][1].imshow(hm, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            color  = "lime" if result.object_present else "orange"
            status = "PRESENT" if result.object_present else "ABSENT"
            axes[i][1].text(0.02, 0.02,
                f"CLIP→{status}\nsim={result.full_img_sim:.3f}",
                transform=axes[i][1].transAxes, fontsize=8, color=color,
                bbox=dict(facecolor="black", alpha=0.65, pad=2))
        axes[i][1].axis("off")
        axes[i][1].set_title(f"Saliency heatmap  noun='{noun}'", fontsize=8, pad=3)

        # ── Col 2: CLIP gate statistics ───────────────────────────────────────
        axes[i][2].axis("off")
        if result is not None:
            stats = (
                f"noun:            '{result.query_noun}'\n"
                f"GT:              {gt.upper()}\n"
                f"gate decision:   {'PRESENT' if result.object_present else 'ABSENT'}\n"
                f"\n"
                f"full_img_sim:    {result.full_img_sim:.4f}\n"
                f"patch_max_sim:   {result.max_sim:.4f}\n"
                f"patch_contrast:  {result.patch_contrast:.4f}\n"
                f"contrastive_gap: {result.contrastive_gap:.4f}\n"
                f"patch_entropy:   {result.patch_entropy:.4f}\n"
                f"raw_entropy:     {result.raw_entropy:.4f}\n"
                f"cross_scale_iou: {result.cross_scale_iou:.4f}\n"
                f"blur_delta:      {result.blur_delta:.4f}\n"
                f"\n"
                f"gate_full:       {result.gate_full}\n"
                f"gate_patch:      {result.gate_patch}\n"
                f"gate_contrast:   {result.gate_contrast}\n"
                f"gate_contrastive:{result.gate_contrastive}\n"
                f"gate_entropy:    {result.gate_entropy}\n"
                f"gate_raw_entr:   {result.gate_raw_entropy}\n"
                f"gate_cross_scl:  {result.gate_cross_scale}\n"
                f"gate_blur:       {result.gate_blur_delta}"
            )
        else:
            stats = f"noun: '{noun}'\nCLIP result: None"
        axes[i][2].text(0.03, 0.97, stats,
            transform=axes[i][2].transAxes,
            fontsize=7.5, va="top", ha="left", fontfamily="monospace",
            bbox=dict(facecolor="#f0f0f0", alpha=0.8, pad=4, boxstyle="round"))

        # ── Col 3: Answer comparison ──────────────────────────────────────────
        axes[i][3].axis("off")

        def tick(pred, ref): return "✓" if pred.startswith(ref) or ref.startswith(pred) else "✗"

        bl_pf  = rec["bl_pred_prefill"]
        gen_pf = rec["gen_pred_prefill"]
        bth_pf = rec["both_pred_prefill"]
        bl_g   = rec["bl_gen_ans"]
        gen_g  = rec["srf_gen_ans"]
        bth_g  = rec["srf_both_ans"]
        answer_text = (
            f"Question:\n{q_wrap}\n\n"
            f"GT: {gt.upper()}\n\n"
            f"─── PREFILL logit (model(**inp)) ───\n"
            f"  baseline:        {bl_pf:3s} {tick(bl_pf, gt)}\n"
            f"  SRF(phase=gen):  {gen_pf:3s} {tick(gen_pf, gt)}  diff={rec['phase_gen_diff']:.4f}\n"
            f"  SRF(phase=both): {bth_pf:3s} {tick(bth_pf, gt)}  diff={rec['phase_both_diff']:.4f}\n"
            f"\n"
            f"─── model.generate() ───────────────\n"
            f"  baseline:        {bl_g[:10]:10s} {tick(bl_g, gt)}\n"
            f"  SRF(phase=gen):  {gen_g[:10]:10s} {tick(gen_g, gt)}\n"
            f"  SRF(phase=both): {bth_g[:10]:10s} {tick(bth_g, gt)}"
        )
        axes[i][3].text(0.03, 0.97, answer_text,
            transform=axes[i][3].transAxes,
            fontsize=7.5, va="top", ha="left", fontfamily="monospace",
            bbox=dict(facecolor="#e8f4e8", alpha=0.8, pad=4, boxstyle="round"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.model}…")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()

    img_tok_str = CFG.get_arch(args.model)["image_token"]
    img_token_id = processor.tokenizer.convert_tokens_to_ids(img_tok_str)

    # ── Setup SRF (calibrate heads) ───────────────────────────────────────────
    srf.setup(model, processor, calib_dataset="pope")

    # ── Load RePOPE samples ───────────────────────────────────────────────────
    splits_filter = {s.lower() for s in args.splits}
    repope_labels: dict = {}
    for split in splits_filter:
        fname = os.path.join(args.repope_dir, f"coco_repope_{split}.json")
        if os.path.exists(fname):
            with open(fname) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        e = json.loads(line)
                        repope_labels[(split, str(e["question_id"]))] = e["label"].lower()
    print(f"  Loaded {len(repope_labels)} RePOPE labels from {args.repope_dir}")

    ds = hf_load("lmms-lab/POPE", split="test")
    rows = [r for r in ds
            if str(r.get("category", r.get("type", ""))).lower() in splits_filter]
    # Filter to RePOPE
    rows = [r for r in rows
            if (str(r.get("category", r.get("type", ""))).lower(),
                str(r["question_id"])) in repope_labels]
    print(f"  {len(rows)} RePOPE samples after filtering")

    # Select samples biased toward GT=yes (2:1 yes:no) with unique images
    rng = random.Random(args.seed)
    yes_rows = [r for r in rows if repope_labels[(
        str(r.get("category", r.get("type",""))).lower(), str(r["question_id"]))] == "yes"]
    no_rows  = [r for r in rows if repope_labels[(
        str(r.get("category", r.get("type",""))).lower(), str(r["question_id"]))] == "no"]
    rng.shuffle(yes_rows); rng.shuffle(no_rows)

    n_yes = (args.n * 2) // 3   # 2/3 yes
    n_no  = args.n - n_yes

    seen_imgs: set = set()
    selected = []
    for pool, n_want in [(yes_rows, n_yes), (no_rows, n_no)]:
        for r in pool:
            img_src = str(r.get("image_source", r.get("image", str(r["question_id"]))))
            if img_src not in seen_imgs:
                seen_imgs.add(img_src)
                selected.append(r)
            if sum(1 for s in selected if repope_labels[(
                    str(s.get("category", s.get("type",""))).lower(),
                    str(s["question_id"]))] == ("yes" if pool is yes_rows else "no")) >= n_want:
                break

    # Shuffle so yes/no are interleaved
    rng.shuffle(selected)
    n_yes_sel = sum(1 for r in selected if repope_labels[(
        str(r.get("category", r.get("type",""))).lower(), str(r["question_id"]))] == "yes")
    print(f"  Selected {len(selected)} samples ({n_yes_sel} GT=yes, {len(selected)-n_yes_sel} GT=no)\n")

    # ── Reset SRF for POPE ────────────────────────────────────────────────────
    srf.reset_for_dataset("pope")

    # ── Diagnostic loop ───────────────────────────────────────────────────────
    results = []
    n_absent = 0
    n_phase_gen_differs = 0   # SRF(phase=gen) != baseline → SRF IS active
    n_phase_both_differs = 0  # SRF(phase=both) != baseline → expected to differ

    yes_id = processor.tokenizer.convert_tokens_to_ids("Yes")
    no_id  = processor.tokenizer.convert_tokens_to_ids("No")

    for idx, r in enumerate(selected):
        split    = str(r.get("category", r.get("type", ""))).lower()
        qid      = str(r["question_id"])
        image    = r["image"].convert("RGB")
        question = str(r["question"]).strip()
        q_prompt = question + "\nAnswer with Yes or No only."
        gt       = repope_labels[(split, qid)]
        orig_ans = ("yes" if str(r.get("answer", "")).strip().lower() == "yes" else "no")

        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text",  "text":  q_prompt}]}]
        text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = process_vision_info(msgs)
        inp    = processor(text=[text], images=vis, return_tensors="pt",
                           padding=True).to(device)
        s, e   = get_img_range(inp["input_ids"], img_token_id)
        grid_h, grid_w = clip_sal.get_grid_dims(inp, srf._spatial)

        # ── CLIP saliency (v3 gate) ───────────────────────────────────────────
        noun = extract_clip_noun(question, mode="pope")
        v3_thresh = srf.SALIENCY.get("clip_fallback_thresh") or clip_sal._FULL_IMG_THRESH_V3
        clip_result = clip_sal.compute_clip_salience_full_gate_v3(
            image, noun, grid_h, grid_w,
            top_k_pct=srf.SALIENCY["clip_top_k_pct"],
            clip_model_name=srf.SALIENCY.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL),
            backup="none",
            full_img_thresh=v3_thresh,
        )
        object_present = clip_result.object_present
        if not object_present:
            n_absent += 1

        # ── Set up SRF state for this sample ─────────────────────────────────
        srf.prepare_sample(inp, s, e, image, q_prompt, model, processor)

        # ── Baseline logits (no SRF) ──────────────────────────────────────────
        bl_logits = baseline_logits(model, inp)
        bl_yes = bl_logits[0, yes_id].item()
        bl_no  = bl_logits[0, no_id].item()
        bl_pred_prefill = "yes" if bl_yes > bl_no else "no"

        # ── SRF prefill logits: phase=generation (should be no-op) ───────────
        srf.prepare_sample(inp, s, e, image, q_prompt, model, processor)  # re-setup
        gen_logits = logits_for_phase(model, inp, "generation")
        gen_yes = gen_logits[0, yes_id].item()
        gen_no  = gen_logits[0, no_id].item()
        gen_pred_prefill = "yes" if gen_yes > gen_no else "no"
        phase_gen_diff = abs(gen_yes - bl_yes) + abs(gen_no - bl_no)

        # ── SRF prefill logits: phase=both (should differ when object present) ─
        srf.prepare_sample(inp, s, e, image, q_prompt, model, processor)  # re-setup
        both_logits = logits_for_phase(model, inp, "both")
        both_yes = both_logits[0, yes_id].item()
        both_no  = both_logits[0, no_id].item()
        both_pred_prefill = "yes" if both_yes > both_no else "no"
        phase_both_diff = abs(both_yes - bl_yes) + abs(both_no - bl_no)

        # ── model.generate() paths: baseline, SRF(gen), SRF(both) ────────────
        srf.prepare_sample(inp, s, e, image, q_prompt, model, processor)
        bl_gen_ans   = baseline_generate(model, inp, processor)

        srf.prepare_sample(inp, s, e, image, q_prompt, model, processor)
        srf_gen_ans  = generate_answer(model, inp, "generation", processor)

        srf.prepare_sample(inp, s, e, image, q_prompt, model, processor)
        srf_both_ans = generate_answer(model, inp, "both", processor)
        srf.cleanup()

        if phase_gen_diff > 0.01:
            n_phase_gen_differs += 1
        if phase_both_diff > 0.01:
            n_phase_both_differs += 1

        rec = {
            "_image": image,           # PIL — for visualization only (not serialized)
            "_clip_result": clip_result,  # ClipSalienceResult — for visualization only
            "idx": idx, "split": split, "qid": qid,
            "question": question, "noun": noun,
            "gt": gt, "orig_pope_ans": orig_ans,
            "label_flipped": (gt != orig_ans),
            # CLIP gate
            "object_present": object_present,
            "full_img_sim": round(clip_result.full_img_sim, 4),
            "patch_max_sim": round(clip_result.max_sim, 4),
            "gate_full": clip_result.gate_full,
            "v3_thresh": v3_thresh,
            # Prefill logits
            "bl_yes_logit": round(bl_yes, 4), "bl_no_logit": round(bl_no, 4),
            "gen_yes_logit": round(gen_yes, 4), "gen_no_logit": round(gen_no, 4),
            "both_yes_logit": round(both_yes, 4), "both_no_logit": round(both_no, 4),
            "phase_gen_diff": round(phase_gen_diff, 4),
            "phase_both_diff": round(phase_both_diff, 4),
            # Predictions (prefill logit)
            "bl_pred_prefill": bl_pred_prefill,
            "gen_pred_prefill": gen_pred_prefill,
            "both_pred_prefill": both_pred_prefill,
            # Predictions (model.generate)
            "bl_gen_ans": bl_gen_ans,
            "srf_gen_ans": srf_gen_ans,
            "srf_both_ans": srf_both_ans,
        }
        results.append(rec)

        # ── Free GPU cache between samples to avoid OOM ───────────────────────
        torch.cuda.empty_cache()

        # ── Print per-sample ──────────────────────────────────────────────────
        absent_str = "ABSENT " if not object_present else "present"
        flip_str   = " [FLIPPED]" if gt != orig_ans else ""
        print(f"\n[{idx+1:2d}/{len(selected)}] {split} qid={qid}{flip_str}")
        print(f"  Q: {question}")
        print(f"  GT={gt}  orig={orig_ans}  noun={noun!r}")
        print(f"  CLIP: {absent_str}  full_sim={clip_result.full_img_sim:.3f}  "
              f"patch_max={clip_result.max_sim:.3f}  thresh={v3_thresh:.2f}")
        print(f"  PREFILL logits  Yes/No:")
        print(f"    baseline       : {bl_yes:+.3f} / {bl_no:+.3f}  → {bl_pred_prefill}")
        print(f"    SRF(gen-phase) : {gen_yes:+.3f} / {gen_no:+.3f}  → {gen_pred_prefill}  "
              f"diff={phase_gen_diff:.4f}  {'ACTIVE!' if phase_gen_diff > 0.01 else 'noop ✗'}")
        print(f"    SRF(both-phase): {both_yes:+.3f} / {both_no:+.3f}  → {both_pred_prefill}  "
              f"diff={phase_both_diff:.4f}  {'ACTIVE!' if phase_both_diff > 0.01 else 'noop ✗'}")
        print(f"  GENERATE answers:")
        print(f"    baseline       : {bl_gen_ans}")
        print(f"    SRF(gen-phase) : {srf_gen_ans}")
        print(f"    SRF(both-phase): {srf_both_ans}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    print("\n" + "="*70)
    print(f"DIAGNOSTIC SUMMARY  (n={n})")
    print("="*70)
    print(f"  CLIP absent         : {n_absent}/{n}  "
          f"({100*n_absent/n:.0f}%) — these get neg_absent_alpha suppression")
    print(f"  SRF(phase=gen) active (prefill): {n_phase_gen_differs}/{n}  "
          f"(expected 0 — generation phase = no-op in prefill)")
    print(f"  SRF(phase=both) active (prefill): {n_phase_both_differs}/{n}  "
          f"(expected > 0 — 'both' mode applies during prefill)")

    # ── CLIP score distributions by GT ───────────────────────────────────────
    yes_recs = [r for r in results if r["gt"] == "yes"]
    no_recs  = [r for r in results if r["gt"] == "no"]

    def _mean(recs, key): return sum(r[key] for r in recs) / max(len(recs), 1)

    full_yes = _mean(yes_recs, "full_img_sim")
    full_no  = _mean(no_recs,  "full_img_sim")
    patch_yes = _mean(yes_recs, "patch_max_sim")
    patch_no  = _mean(no_recs,  "patch_max_sim")
    n_yes_present = sum(1 for r in yes_recs if r["object_present"])
    n_no_present  = sum(1 for r in no_recs  if r["object_present"])

    print(f"\n  CLIP score distributions (thresh={results[0]['v3_thresh']:.2f}):")
    print(f"    {'':20s}  GT=yes (n={len(yes_recs)})   GT=no (n={len(no_recs)})    Δ")
    print(f"    {'full_img_sim':20s}  {full_yes:.3f}           {full_no:.3f}         {full_yes-full_no:+.3f}")
    print(f"    {'patch_max_sim':20s}  {patch_yes:.3f}           {patch_no:.3f}         {patch_yes-patch_no:+.3f}")
    print(f"  CLIP gate (object_present=True):")
    print(f"    GT=yes: {n_yes_present}/{len(yes_recs)} correct  ({100*n_yes_present/max(len(yes_recs),1):.0f}% TPR)")
    print(f"    GT=no : {n_no_present}/{len(no_recs)} wrong    ({100*n_no_present/max(len(no_recs),1):.0f}% FPR)")

    # ── Accuracy breakdown ────────────────────────────────────────────────────
    def acc(preds, gts): return sum(p.startswith(g) or g.startswith(p)
                                    for p, g in zip(preds, gts)) / max(len(gts), 1)

    gts = [r["gt"] for r in results]
    print(f"\n  Accuracy (prefill logits):")
    print(f"    baseline       : {acc([r['bl_pred_prefill'] for r in results], gts):.3f}")
    print(f"    SRF(phase=gen) : {acc([r['gen_pred_prefill'] for r in results], gts):.3f}")
    print(f"    SRF(phase=both): {acc([r['both_pred_prefill'] for r in results], gts):.3f}")
    print(f"\n  Accuracy (model.generate):")
    print(f"    baseline       : {acc([r['bl_gen_ans'] for r in results], gts):.3f}")
    print(f"    SRF(phase=gen) : {acc([r['srf_gen_ans'] for r in results], gts):.3f}")
    print(f"    SRF(phase=both): {acc([r['srf_both_ans'] for r in results], gts):.3f}")

    # ── Save combined matplotlib figures (5 samples per page) ──────────────
    page_size = 5
    for page_start in range(0, n, page_size):
        page_recs = results[page_start:page_start + page_size]
        page_num  = page_start // page_size
        fig_path  = str(out_dir / f"diag_page{page_num:02d}.png")
        save_sample_figure(page_recs, fig_path)

    # Save JSON (strip non-serializable PIL/tensor fields)
    out_json = out_dir / "diag_results.json"
    serializable = [{k: v for k, v in r.items() if not k.startswith("_")}
                    for r in results]
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Figures           → {out_dir}/diag_page*.png")
    print(f"  Full results      → {out_json}")


if __name__ == "__main__":
    main()
