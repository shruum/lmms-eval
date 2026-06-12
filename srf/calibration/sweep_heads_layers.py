#!/usr/bin/env python3
"""
Head/layer calibration sweep for SRF on the fixed 60-sample POPE val set.

Three phases (all on the same 60 samples, seed=42):
  Phase 1 — layer range  : sweep (layer_start, layer_end), head_top_k_pct=0.20 fixed
  Phase 2 — head fraction: sweep head_top_k_pct, best (layer_start, layer_end) from Phase 1
  Phase 3 — sys_beta     : sweep system-prompt suppression, best params from Phases 1+2

Key optimisation: CLIP saliency is pre-computed once per sample and reused across all
combos. Only changing head_top_k_pct requires re-running identify_visual_heads (20 passes).
Changing layer_start/end or sys_beta is free (direct _STATE update).

Usage (from /volumes2/mllm/lmms-eval/):
  conda activate mllm
  python srf/calibration/sweep_heads_layers.py

Output:
  results/sweep_heads_layers/sweep_results.json
  Sorted table printed to stdout at the end of each phase.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRF  = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_SRF,  ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, _SRF)
sys.path.insert(0, os.path.join(_SRF, "saliency"))
sys.path.insert(0, os.path.join(_REPO, "my_analysis"))

os.environ.setdefault("HF_HOME",        "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch

# ── Constants ─────────────────────────────────────────────────────────────────
SPLITS  = ["adversarial", "popular", "random"]
ANSWERS = ["Yes", "No"]
MODEL_ID      = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_PIXELS    = 512 * 28 * 28
SYSTEM_PROMPT = "You are a helpful assistant."
POST_PROMPT   = "\nAnswer with Yes or No only."

# ── Sweep grids ───────────────────────────────────────────────────────────────
LAYER_STARTS   = [4, 6, 8, 10, 12]
LAYER_ENDS     = [12, 14, 15, 17, 20, 24]
HEAD_TOP_K_PCTS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
SYS_BETAS      = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

RESULTS_DIR = os.path.join(_REPO, "results", "sweep_heads_layers")


# ── POPE loader ───────────────────────────────────────────────────────────────
def load_balanced_pope(n_per_cell: int = 10, seed: int = 42) -> List[Dict]:
    from datasets import load_dataset
    print("Loading POPE dataset…")
    ds  = load_dataset("lmms-lab/POPE", split="test")
    rng = random.Random(seed)
    buckets: Dict[tuple, list] = {(sp, ans): [] for sp in SPLITS for ans in ANSWERS}
    for r in ds:
        sp  = str(r.get("category", "")).strip().lower()
        ans = str(r.get("answer",   "")).strip().capitalize()
        if sp in SPLITS and ans in ANSWERS:
            buckets[(sp, ans)].append(r)
    samples: List[Dict] = []
    for (sp, ans), rows in sorted(buckets.items()):
        rng.shuffle(rows)
        for r in rows[:n_per_cell]:
            samples.append({
                "image":    r["image"].convert("RGB"),
                "question": str(r["question"]).strip(),
                "gt":       ans,
                "split":    sp,
            })
    rng.shuffle(samples)
    print(f"  Loaded {len(samples)} samples.")
    return samples


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass

    print(f"Loading {MODEL_ID} (attn=eager)…")
    model = ModelClass.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, max_pixels=MAX_PIXELS, min_pixels=256 * 28 * 28
    )
    print("  Model loaded.")
    return model, processor, process_vision_info


# ── Answer parsing ────────────────────────────────────────────────────────────
def parse_answer(raw: str) -> str:
    r = raw.strip().lower()
    if r.startswith("yes"): return "Yes"
    if r.startswith("no"):  return "No"
    if "yes" in r: return "Yes"
    if "no"  in r: return "No"
    return raw.strip()


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(preds: List[str], gts: List[str]) -> Dict:
    n   = len(preds)
    tp  = sum(1 for p, g in zip(preds, gts) if p == "Yes" and g == "Yes")
    tn  = sum(1 for p, g in zip(preds, gts) if p == "No"  and g == "No")
    fp  = sum(1 for p, g in zip(preds, gts) if p == "Yes" and g == "No")
    fn  = sum(1 for p, g in zip(preds, gts) if p == "No"  and g == "Yes")
    acc  = (tp + tn) / n
    tpr  = tp / (tp + fn + 1e-9)
    fpr  = fp / (fp + tn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    f1   = 2 * prec * tpr / (prec + tpr + 1e-9)
    yes_ratio = sum(1 for p in preds if p == "Yes") / n
    return {"acc": acc, "tpr": tpr, "fpr": fpr, "f1": f1,
            "yes_ratio": yes_ratio,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ── Pre-process samples (build inputs, cache img ranges + grid dims) ──────────
def preprocess_samples(samples, model, processor, process_vision_info):
    """Build tokenized inputs for all samples. Returns list of dicts."""
    import clip_salience as clip_sal
    import qwen_attn_patch as patch

    arch_spatial = getattr(model.config.vision_config, "spatial_merge_size", 2)
    processed = []
    print("Pre-processing samples (tokenising)…")
    for i, s in enumerate(samples):
        q = s["question"] + POST_PROMPT
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": s["image"]},
                {"type": "text",  "text":  q},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        image_inp, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inp,
                           padding=True, return_tensors="pt").to("cuda:0")
        img_start, img_end = patch.get_image_token_range(inputs, processor=processor)
        grid_h, grid_w     = clip_sal.get_grid_dims(inputs, arch_spatial)
        processed.append({
            "inputs":    inputs,
            "img_start": img_start,
            "img_end":   img_end,
            "grid_h":    grid_h,
            "grid_w":    grid_w,
            "image":     s["image"],
            "question":  q,
            "gt":        s["gt"],
            "split":     s["split"],
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)}")
    print("  Done.")
    return processed


# ── Pre-compute CLIP saliency for all samples ─────────────────────────────────
def precompute_clip_saliency(processed, clip_model_name: str) -> List[Dict]:
    """Run CLIP once per sample. Returns list of {present, saliency, clip_conf}."""
    from clip_salience import compute_clip_salience_full_gate_v3, _FULL_IMG_THRESH_V3

    print(f"Pre-computing CLIP saliency ({clip_model_name}) for {len(processed)} samples…")
    cache = []
    for i, s in enumerate(processed):
        result = compute_clip_salience_full_gate_v3(
            s["image"], s["question"], s["grid_h"], s["grid_w"],
            top_k_pct=0.30, coarse_scales=(3, 5, 7),
            clip_model_name=clip_model_name,
            backup="none",
        )
        clip_conf = min(result.full_img_sim / _FULL_IMG_THRESH_V3, 1.0) if result.object_present else 0.0
        cache.append({
            "present":   result.object_present,
            "saliency":  result.saliency,   # tensor on CPU
            "clip_conf": clip_conf,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(processed)}")
    print("  Done.")
    return cache


# ── Inject cached saliency into patch state ───────────────────────────────────
def inject_saliency(proc_sample: Dict, cache_entry: Dict,
                    patch_module, boost_alpha: float,
                    bias_mode: str, interp_lambda: float,
                    prob_floor: float, img_scale: float,
                    text_beta: float, apply_phase: str) -> None:
    """Set patch._STATE for one sample using pre-computed saliency."""
    patch_module.update_sample(proc_sample["img_start"], proc_sample["img_end"])
    patch_module._STATE["method"]            = "srf"
    patch_module._STATE["srf_apply_phase"]   = apply_phase
    patch_module._STATE["srf_bias_mode"]     = bias_mode
    patch_module._STATE["srf_interp_lambda"] = interp_lambda
    patch_module._STATE["srf_prob_floor"]    = prob_floor
    patch_module._STATE["srf_img_scale"]     = img_scale
    patch_module._STATE["srf_text_beta"]     = text_beta
    if cache_entry["present"]:
        patch_module._STATE["value"]         = boost_alpha * cache_entry["clip_conf"]
        patch_module._STATE["salience_mask"] = cache_entry["saliency"]
    else:
        patch_module._STATE["value"]         = 0.0
        patch_module._STATE["salience_mask"] = None


# ── Evaluate one combo ────────────────────────────────────────────────────────
def eval_combo(processed, clip_cache, model, processor,
               patch_module, srf_module, verbose: bool = False) -> Dict:
    """Run inference on all 60 samples with current patch state config."""
    import srf as _srf
    preds, gts = [], []
    alpha        = _srf.BIAS["boost_alpha"]
    bias_mode    = _srf.BIAS["bias_mode"]
    interp_lam   = _srf.BIAS["interp_lambda"]
    prob_floor   = _srf.BIAS["prob_floor"]
    img_scale    = _srf.BIAS["img_scale"]
    text_beta    = _srf.BIAS["text_beta"]
    apply_phase  = _srf.BIAS["srf_apply_phase"]

    for proc_sample, cache_entry in zip(processed, clip_cache):
        inject_saliency(proc_sample, cache_entry, patch_module,
                        alpha, bias_mode, interp_lam, prob_floor,
                        img_scale, text_beta, apply_phase)

        with torch.inference_mode():
            out = model.generate(
                **proc_sample["inputs"],
                max_new_tokens=8,
                do_sample=False,
            )
        _srf.cleanup()

        trimmed = out[:, proc_sample["inputs"]["input_ids"].shape[1]:]
        raw  = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        pred = parse_answer(raw)
        preds.append(pred)
        gts.append(proc_sample["gt"])

    return compute_metrics(preds, gts)


# ── Print sorted table ────────────────────────────────────────────────────────
def print_table(rows: List[Dict], sort_key: str = "acc") -> None:
    rows_s = sorted(rows, key=lambda r: -r["metrics"][sort_key])
    header = f"{'config':<40}  {'acc':>6}  {'TPR':>6}  {'FPR':>6}  {'F1':>6}  {'yes%':>5}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in rows_s:
        tag = r.get("tag", "")
        m   = r["metrics"]
        mark = "  ← baseline" if tag == "baseline" else ""
        print(f"{tag:<40}  {m['acc']:.3f}  {m['tpr']:.3f}  {m['fpr']:.3f}  "
              f"{m['f1']:.3f}  {m['yes_ratio']:.2f}{mark}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="SRF head/layer sweep on POPE val set")
    parser.add_argument("--n_per_cell",   type=int,   default=10)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--phase1_only",  action="store_true", help="Run Phase 1 only")
    parser.add_argument("--skip_phase1",  action="store_true",
                        help="Skip Phase 1, use --best_ls/--best_le directly")
    parser.add_argument("--best_ls",  type=int, default=6,  help="Best layer_start from Phase 1")
    parser.add_argument("--best_le",  type=int, default=12, help="Best layer_end from Phase 1")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load model + setup SRF ────────────────────────────────────────────────
    model, processor, process_vision_info = load_model()

    import srf as srf_module
    import qwen_attn_patch as patch

    print("\nSetting up SRF (initial calibration)…")
    srf_module.setup(model, processor, calib_dataset="pope")
    srf_module.reset_for_dataset("pope")   # phase=generation, alpha=4.0, layers=8-15
    clip_model_name = srf_module.SALIENCY.get("clip_model", "openai/clip-vit-base-patch32")
    print(f"  SRF ready.  clip_model={clip_model_name}")

    # ── Load + pre-process val samples ────────────────────────────────────────
    samples   = load_balanced_pope(args.n_per_cell, args.seed)
    processed = preprocess_samples(samples, model, processor, process_vision_info)

    # ── Pre-compute CLIP saliency once ────────────────────────────────────────
    clip_cache = precompute_clip_saliency(processed, clip_model_name)
    present_n  = sum(1 for c in clip_cache if c["present"])
    print(f"  CLIP gate: {present_n}/{len(clip_cache)} samples detected as present.")

    # ── Baseline: current config (layer_start=8, layer_end=15, head_top_k=0.20) ─
    print("\n── Baseline (current config: ls=8 le=15 htk=0.20 sys_beta=0.30) ──")
    baseline_metrics = eval_combo(processed, clip_cache, model, processor,
                                  patch, srf_module)
    all_results = [{"tag": "baseline  ls=8  le=15  htk=0.20", "metrics": baseline_metrics,
                    "layer_start": 8, "layer_end": 15, "head_top_k_pct": 0.20,
                    "sys_beta": 0.30, "phase": 1}]
    print(f"  baseline  acc={baseline_metrics['acc']:.3f}  tpr={baseline_metrics['tpr']:.3f}  "
          f"fpr={baseline_metrics['fpr']:.3f}  f1={baseline_metrics['f1']:.3f}")

    if args.skip_phase1:
        best_ls = args.best_ls
        best_le = args.best_le
        print(f"\n  Skipping Phase 1. Using ls={best_ls} le={best_le} (from --best_ls/--best_le).")
        phase1_results = []
    else:
        # ─────────────────────────────────────────────────────────────────────
        # Phase 1: Layer range sweep (head_top_k_pct=0.20 fixed)
        # Cheap: just update patch._STATE, no re-calibration.
        # ─────────────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Phase 1 — Layer range sweep (head_top_k_pct=0.20)")
        print("=" * 60)

        phase1_results = []
        total_p1 = sum(1 for ls in LAYER_STARTS for le in LAYER_ENDS if le > ls + 2)
        done_p1  = 0

        for ls in LAYER_STARTS:
            for le in LAYER_ENDS:
                if le <= ls + 2:
                    continue
                srf_module.BIAS["layer_start"] = ls
                srf_module.BIAS["layer_end"]   = le
                patch._STATE["vaf_layer_start"] = ls
                patch._STATE["vaf_layer_end"]   = le

                m    = eval_combo(processed, clip_cache, model, processor, patch, srf_module)
                tag  = f"ls={ls:<2}  le={le:<2}  htk=0.20"
                done_p1 += 1
                print(f"  [{done_p1:02d}/{total_p1}] {tag}  acc={m['acc']:.3f}  "
                      f"tpr={m['tpr']:.3f}  fpr={m['fpr']:.3f}  f1={m['f1']:.3f}")

                row = {"tag": tag, "metrics": m,
                       "layer_start": ls, "layer_end": le,
                       "head_top_k_pct": 0.20, "sys_beta": 0.30, "phase": 1}
                phase1_results.append(row)
                all_results.append(row)

        print_table(phase1_results + [all_results[0]], sort_key="acc")

        best_p1 = max(phase1_results, key=lambda r: r["metrics"]["acc"])
        best_ls  = best_p1["layer_start"]
        best_le  = best_p1["layer_end"]
        print(f"\n  Best Phase 1: layer_start={best_ls}  layer_end={best_le}  "
              f"acc={best_p1['metrics']['acc']:.3f}")

        if args.phase1_only:
            _save_and_exit(all_results, RESULTS_DIR)
            return

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Head fraction sweep (best layer range from Phase 1)
    # Expensive: identify_visual_heads re-runs for each head_top_k_pct.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Phase 2 — Head fraction sweep (ls={best_ls} le={best_le})")
    print("=" * 60)

    phase2_results = []
    for htkp in HEAD_TOP_K_PCTS:
        print(f"\n  Re-identifying heads with head_top_k_pct={htkp}…")
        # reset_for_dataset re-identifies heads and calls _sync_patch_state
        srf_module.reset_for_dataset("pope",
                                      layer_start=best_ls,
                                      layer_end=best_le,
                                      head_top_k_pct=htkp)
        # Restore layer range in _STATE (reset_for_dataset uses arch defaults otherwise)
        patch._STATE["vaf_layer_start"] = best_ls
        patch._STATE["vaf_layer_end"]   = best_le

        m   = eval_combo(processed, clip_cache, model, processor, patch, srf_module)
        tag = f"ls={best_ls:<2}  le={best_le:<2}  htk={htkp:.2f}"
        print(f"  {tag}  acc={m['acc']:.3f}  tpr={m['tpr']:.3f}  "
              f"fpr={m['fpr']:.3f}  f1={m['f1']:.3f}")

        row = {"tag": tag, "metrics": m,
               "layer_start": best_ls, "layer_end": best_le,
               "head_top_k_pct": htkp, "sys_beta": 0.30, "phase": 2}
        phase2_results.append(row)
        all_results.append(row)

    print_table(phase2_results, sort_key="acc")

    best_p2   = max(phase2_results, key=lambda r: r["metrics"]["acc"])
    best_htkp = best_p2["head_top_k_pct"]
    print(f"\n  Best Phase 2: head_top_k_pct={best_htkp}  "
          f"acc={best_p2['metrics']['acc']:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: sys_beta sweep (best layers + heads from Phases 1+2)
    # Free: just update patch._STATE["vaf_beta"].
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Phase 3 — sys_beta sweep (ls={best_ls} le={best_le} htk={best_htkp})")
    print("=" * 60)

    # Set best head mask
    srf_module.reset_for_dataset("pope",
                                  layer_start=best_ls,
                                  layer_end=best_le,
                                  head_top_k_pct=best_htkp)
    patch._STATE["vaf_layer_start"] = best_ls
    patch._STATE["vaf_layer_end"]   = best_le

    phase3_results = []
    for beta in SYS_BETAS:
        srf_module.BIAS["sys_beta"]  = beta
        patch._STATE["vaf_beta"]     = beta

        m   = eval_combo(processed, clip_cache, model, processor, patch, srf_module)
        tag = f"ls={best_ls:<2}  le={best_le:<2}  htk={best_htkp:.2f}  beta={beta:.2f}"
        print(f"  {tag}  acc={m['acc']:.3f}  tpr={m['tpr']:.3f}  "
              f"fpr={m['fpr']:.3f}  f1={m['f1']:.3f}")

        row = {"tag": tag, "metrics": m,
               "layer_start": best_ls, "layer_end": best_le,
               "head_top_k_pct": best_htkp, "sys_beta": beta, "phase": 3}
        phase3_results.append(row)
        all_results.append(row)

    print_table(phase3_results, sort_key="acc")

    best_p3   = max(phase3_results, key=lambda r: r["metrics"]["acc"])
    best_beta = best_p3["sys_beta"]
    print(f"\n  Best Phase 3: sys_beta={best_beta}  acc={best_p3['metrics']['acc']:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Current config: ls=8  le=15  htk=0.20  beta=0.30  "
          f"→ acc={all_results[0]['metrics']['acc']:.3f}")
    print(f"  Best layer range : ls={best_ls}  le={best_le}  "
          f"→ acc={best_p1['metrics']['acc']:.3f}")
    print(f"  Best head frac   : htk={best_htkp}  "
          f"→ acc={best_p2['metrics']['acc']:.3f}")
    print(f"  Best sys_beta    : beta={best_beta}  "
          f"→ acc={best_p3['metrics']['acc']:.3f}")

    _save_and_exit(all_results, RESULTS_DIR)


def _save_and_exit(all_results: List[Dict], out_dir: str) -> None:
    out_path = os.path.join(out_dir, "sweep_results.json")
    # tensors are not JSON-serialisable — strip them from metrics dict
    save_data = []
    for r in all_results:
        save_data.append({k: v for k, v in r.items() if k != "saliency"})
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
