#!/usr/bin/env python3
"""
SRF-v2 autoresearch sweep.

Loads the model ONCE, then iterates through all experiment configs.
Parses RESULT lines and writes results/sweep.csv.

Usage:
    cd /volumes2/mllm/lmms-eval
    conda run -n mllm python my_analysis/autoresearch_srf_v2/sweep.py
    conda run -n mllm python my_analysis/autoresearch_srf_v2/sweep.py --groups absent adaptive
    conda run -n mllm python my_analysis/autoresearch_srf_v2/sweep.py --exp e00_baseline e02_suppress_salient
    conda run -n mllm python my_analysis/autoresearch_srf_v2/sweep.py --list
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("HF_HOME", "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

_SWEEP_DIR   = pathlib.Path(__file__).parent
_ANALYSIS    = _SWEEP_DIR.parent
sys.path.insert(0, str(_SWEEP_DIR))
sys.path.insert(0, str(_ANALYSIS))

MODEL_ID    = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_TOKEN = "<|image_pad|>"
N_SAMPLES   = 200
SEED        = 42

# =============================================================================
# Experiment registry
# =============================================================================

# Base config — all experiments start from this and override specific keys.
# This matches srf_v2.py defaults exactly so "e00_baseline" is a clean anchor.
BASE = {
    "saliency_source":          "clip",
    "clip_model":               "openai/clip-vit-base-patch32",
    "clip_scales":              [7],
    "clip_top_k_pct":           0.30,
    "clip_use_soft":            True,
    "dino_model":               "facebook/dinov2-small",
    "dino_top_k_pct":           0.30,
    "dino_weight":              0.4,
    "hssa_layer":               16,
    "hssa_top_k_pct":           0.30,
    "hssa_use_soft":            True,
    "hssa_weight":              0.3,
    "absence_thresh":           0.248,
    "absent_strategy":          "suppress_salient",
    "suppress_alpha":           5.0,
    "adaptive_alpha_mode":      "fixed",
    "adaptive_sigmoid_k":       20.0,
    "boost_alpha":              4.0,
    "head_mode":                "static",
    "head_top_k_pct":           0.20,
    "n_calib":                  20,
    "layer_start":              8,
    "layer_end":                15,
    "bias_mode":                "additive_logit",
    "background_eps":           0.0,
    "sys_beta":                 0.10,
    "interp_lambda":            1.0,
    "prob_floor":               0.005,
    "img_scale":                1.5,
    "phase":                    "both",
    "text_suppress_beta":       0.3,
    "text_suppress_layer_start":20,
    "text_suppress_layer_end":  27,
    "logit_blend_gamma":        0.0,
    "logit_blend_k":            3,
}


def _exp(name: str, group: str, desc: str, **overrides) -> dict:
    cfg = copy.deepcopy(BASE)
    cfg.update(overrides)
    return {"name": name, "group": group, "desc": desc, "cfg": cfg}


EXPERIMENTS = [

    # ── G0: anchors ──────────────────────────────────────────────────────────
    _exp("e00_no_srf", "anchor",
         "No SRF: zero boost/suppress; raw model logits",
         boost_alpha=0.0, suppress_alpha=0.0, absent_strategy="nothing"),
    _exp("e01_baseline", "anchor",
         "Baseline SRF-v2 defaults (suppress_salient, fixed alpha, ViT-B/32, scales=[7])"),

    # ── G1: absent_strategy ──────────────────────────────────────────────────
    _exp("e02_absent_nothing",        "absent", "absent=nothing",         absent_strategy="nothing"),
    _exp("e03_absent_suppress_sal",   "absent", "absent=suppress_salient",absent_strategy="suppress_salient"),
    _exp("e04_absent_suppress_text",  "absent", "absent=suppress_text",   absent_strategy="suppress_text"),
    _exp("e05_absent_suppress_both",  "absent", "absent=suppress_both",   absent_strategy="suppress_both"),

    # ── G2: adaptive alpha mode ───────────────────────────────────────────────
    _exp("e06_alpha_fixed",   "adaptive", "alpha=fixed",   adaptive_alpha_mode="fixed"),
    _exp("e07_alpha_linear",  "adaptive", "alpha=linear",  adaptive_alpha_mode="linear"),
    _exp("e08_alpha_sigmoid", "adaptive", "alpha=sigmoid", adaptive_alpha_mode="sigmoid"),
    _exp("e09_alpha_zone",    "adaptive", "alpha=zone",    adaptive_alpha_mode="zone"),

    # ── G3: CLIP model size ───────────────────────────────────────────────────
    _exp("e10_clip_b32", "clip_model",
         "CLIP ViT-B/32 (default, 32px patches)",
         clip_model="openai/clip-vit-base-patch32"),
    _exp("e11_clip_b16", "clip_model",
         "CLIP ViT-B/16 (16px patches — 2× finer spatial)",
         clip_model="openai/clip-vit-base-patch16"),
    _exp("e12_clip_l14", "clip_model",
         "CLIP ViT-L/14 (14px patches, 86GB — best quality)",
         clip_model="openai/clip-vit-large-patch14"),

    # ── G4: multi-scale CLIP ─────────────────────────────────────────────────
    _exp("e13_scales_7",     "multiscale", "scales=[7] (single, default)",  clip_scales=[7]),
    _exp("e14_scales_5_9",   "multiscale", "scales=[5,9] (2-scale)",        clip_scales=[5, 9]),
    _exp("e15_scales_5_7_9", "multiscale", "scales=[5,7,9] (3-scale)",      clip_scales=[5, 7, 9]),
    _exp("e16_scales_3_7",   "multiscale", "scales=[3,7] (coarse+default)", clip_scales=[3, 7]),

    # ── G5: saliency source ──────────────────────────────────────────────────
    _exp("e17_sal_clip",      "saliency", "saliency=clip (default)",     saliency_source="clip"),
    _exp("e18_sal_hssa",      "saliency", "saliency=hssa",               saliency_source="hssa"),
    _exp("e19_sal_dino",      "saliency", "saliency=dino",               saliency_source="dino"),
    _exp("e20_sal_clip_hssa", "saliency", "saliency=clip+hssa ensemble", saliency_source="clip_hssa"),
    _exp("e21_sal_clip_dino", "saliency", "saliency=clip+dino ensemble", saliency_source="clip_dino"),

    # ── G6: head selection mode ──────────────────────────────────────────────
    _exp("e22_head_static",     "head_mode", "head=static (default)",     head_mode="static"),
    _exp("e23_head_per_sample", "head_mode", "head=per_sample (1 extra pass)", head_mode="per_sample"),

    # ── G7: logit blend gamma (SLA) ──────────────────────────────────────────
    _exp("e24_gamma_0",    "logit_blend", "SLA gamma=0 (disabled)",    logit_blend_gamma=0.0),
    _exp("e25_gamma_010",  "logit_blend", "SLA gamma=0.10",            logit_blend_gamma=0.10),
    _exp("e26_gamma_015",  "logit_blend", "SLA gamma=0.15",            logit_blend_gamma=0.15),
    _exp("e27_gamma_020",  "logit_blend", "SLA gamma=0.20",            logit_blend_gamma=0.20),

    # ── G8: suppress_alpha strength ──────────────────────────────────────────
    _exp("e28_supp_3",  "suppress_strength", "suppress_alpha=3.0", suppress_alpha=3.0),
    _exp("e29_supp_5",  "suppress_strength", "suppress_alpha=5.0 (default)", suppress_alpha=5.0),
    _exp("e30_supp_8",  "suppress_strength", "suppress_alpha=8.0", suppress_alpha=8.0),
    _exp("e31_supp_12", "suppress_strength", "suppress_alpha=12.0", suppress_alpha=12.0),

    # ── G9: boost_alpha strength ─────────────────────────────────────────────
    _exp("e32_boost_2", "boost_strength", "boost_alpha=2.0", boost_alpha=2.0),
    _exp("e33_boost_4", "boost_strength", "boost_alpha=4.0 (default)", boost_alpha=4.0),
    _exp("e34_boost_6", "boost_strength", "boost_alpha=6.0", boost_alpha=6.0),
    _exp("e35_boost_8", "boost_strength", "boost_alpha=8.0", boost_alpha=8.0),

    # ── G10: "always suppress" — inverted logic for adversarial POPE ─────────
    # KEY INSIGHT from per-sample analysis: SRF logit shifts are BACKWARDS.
    # GT=no (absent) get avg +0.111 shift (should be negative).
    # GT=yes (present) get avg -0.124 shift (should be positive).
    # Reason: adversarial POPE objects co-occur with scene → high CLIP sim →
    # classified as "present" → boosted → hallucination worsened.
    # Hypothesis: in adversarial POPE, high CLIP sim = adversarial trap → always suppress.
    # Implementation: set absence_thresh=1.0 so EVERYTHING is "absent" → suppress all.
    _exp("e36_always_suppress",   "invert", "always suppress salient (thresh=1.0)",
         absence_thresh=1.0, absent_strategy="suppress_salient", suppress_alpha=5.0),
    _exp("e37_always_supp_strong","invert", "always suppress strong (thresh=1.0, alpha=10)",
         absence_thresh=1.0, absent_strategy="suppress_salient", suppress_alpha=10.0),
    _exp("e38_always_supp_weak",  "invert", "always suppress weak (thresh=1.0, alpha=3)",
         absence_thresh=1.0, absent_strategy="suppress_salient", suppress_alpha=3.0),

    # ── G11: high absence threshold (classify more as absent → suppress more) ─
    # The 0.248 threshold is too low: many adversarial absent objects exceed it
    # and get BOOSTED. Raising it forces more suppression.
    _exp("e39_thresh_026",  "abs_thresh", "absence_thresh=0.26",      absence_thresh=0.26),
    _exp("e40_thresh_028",  "abs_thresh", "absence_thresh=0.28",      absence_thresh=0.28),
    _exp("e41_thresh_030",  "abs_thresh", "absence_thresh=0.30",      absence_thresh=0.30),
    _exp("e42_thresh_032",  "abs_thresh", "absence_thresh=0.32",      absence_thresh=0.32),
    _exp("e43_thresh_035",  "abs_thresh", "absence_thresh=0.35",      absence_thresh=0.35),

    # ── G12: no-boost (suppress only, never boost present objects) ─────────────
    # If boosting present objects is neutral/harmful, remove it entirely.
    # Only suppress: when absent → suppress, when present → do nothing.
    _exp("e44_no_boost_supp5",   "no_boost", "no boost, suppress_alpha=5",
         boost_alpha=0.0, absent_strategy="suppress_salient", suppress_alpha=5.0),
    _exp("e45_no_boost_supp10",  "no_boost", "no boost, suppress_alpha=10",
         boost_alpha=0.0, absent_strategy="suppress_salient", suppress_alpha=10.0),
    _exp("e46_no_boost_thresh030","no_boost", "no boost, thresh=0.30, supp=5",
         boost_alpha=0.0, absent_strategy="suppress_salient",
         suppress_alpha=5.0, absence_thresh=0.30),

    # ── G13: strong boost for hard FN samples ─────────────────────────────────
    # The 14 false negatives have logit margins of -1.5 to -5.0.
    # Current boost_alpha=4 only shifts by ~0.5. Try much larger.
    _exp("e47_boost_10",   "strong_boost", "boost_alpha=10.0", boost_alpha=10.0),
    _exp("e48_boost_15",   "strong_boost", "boost_alpha=15.0", boost_alpha=15.0),
    _exp("e49_boost_20",   "strong_boost", "boost_alpha=20.0", boost_alpha=20.0),

    # ── G14: strong suppress for remaining FP samples ─────────────────────────
    _exp("e50_supp_20",  "strong_suppress", "suppress_alpha=20", suppress_alpha=20.0),
    _exp("e51_supp_30",  "strong_suppress", "suppress_alpha=30", suppress_alpha=30.0),

    # ── G15: combined — high thresh + strong suppression ────────────────────
    _exp("e52_combo_thresh030_supp15", "combo_supp",
         "thresh=0.30 + suppress_alpha=15 (many absent + strong signal)",
         absence_thresh=0.30, suppress_alpha=15.0, boost_alpha=4.0),
    _exp("e53_combo_always_supp20", "combo_supp",
         "always suppress alpha=20 (extreme adversarial treatment)",
         absence_thresh=1.0, absent_strategy="suppress_salient", suppress_alpha=20.0,
         boost_alpha=0.0),
]

# Name → experiment lookup
EXP_BY_NAME  = {e["name"]: e for e in EXPERIMENTS}
GROUPS       = sorted({e["group"] for e in EXPERIMENTS})


# =============================================================================
# Inner eval loop (model loaded once by sweep.py, not per-experiment)
# =============================================================================

def _eval_one(model, processor, samples, img_token_id, yes_id, no_id,
              device, exp: dict) -> dict:
    import srf_v2 as srf
    from harness import get_img_range, N_SAMPLES as NS
    from qwen_vl_utils import process_vision_info

    exp_name = exp["name"]

    # Patch srf.CONFIG with this experiment's settings
    srf.CONFIG.update(exp["cfg"])
    srf.setup(model, processor)

    correct   = 0
    n_yes     = 0
    per_sample = []
    t0        = time.time()

    for i, s in enumerate(samples):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": s["image"]},
            {"type": "text",  "text":  s["question"]},
        ]}]
        text   = processor.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
        vis, _ = process_vision_info(msgs)
        inputs = processor(text=[text], images=vis, return_tensors="pt",
                           padding=True).to(device)
        img_start, img_end = get_img_range(inputs["input_ids"], img_token_id)

        srf.prepare_sample(inputs, img_start, img_end, s["image"],
                           s["question"], model, processor)
        logits = srf.get_logits(model, inputs)   # (1, vocab)
        srf.cleanup()

        pred = "yes" if logits[0, yes_id] >= logits[0, no_id] else "no"
        ok   = (pred == s["gt"])
        if ok:     correct += 1
        if pred == "yes": n_yes += 1
        per_sample.append({
            "gt": s["gt"], "pred": pred, "correct": ok,
            "yes_logit": float(logits[0, yes_id]),
            "no_logit":  float(logits[0, no_id]),
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{exp_name}] [{i+1:3d}/{len(samples)}] "
                  f"acc={correct/(i+1):.4f}  "
                  f"yes_rate={n_yes/(i+1):.3f}  "
                  f"elapsed={elapsed:.0f}s")

    n   = len(samples)
    acc = correct / n
    elapsed = time.time() - t0

    out = {
        "exp":     exp_name,
        "group":   exp["group"],
        "desc":    exp["desc"],
        "acc":     acc,
        "n":       n,
        "n_yes":   n_yes,
        "n_no":    n - n_yes,
        "elapsed": round(elapsed, 1),
        "config":  exp["cfg"].copy(),
        "samples": per_sample,
    }

    print(f"\nRESULT  acc={acc:.4f}  yes={n_yes}  "
          f"no={n - n_yes}  n={n}  exp={exp_name}  t={elapsed:.0f}s")
    return out


# =============================================================================
# CSV helpers
# =============================================================================

_CSV_COLS = ["exp", "group", "desc", "acc", "n_yes", "n_no", "n", "elapsed"]


def _write_csv(rows: list[dict], path: pathlib.Path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _append_csv(row: dict, path: pathlib.Path):
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", metavar="GROUP",
                        help=f"Run only these groups. Available: {GROUPS}")
    parser.add_argument("--exp", nargs="+", metavar="NAME",
                        help="Run only these specific experiment names")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments and exit")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip experiments whose JSON already exists")
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-run experiments even if JSON exists")
    args = parser.parse_args()

    if args.list:
        print(f"{'Name':<30} {'Group':<18} Description")
        print("-" * 90)
        for e in EXPERIMENTS:
            print(f"{e['name']:<30} {e['group']:<18} {e['desc']}")
        return

    # Select subset to run
    to_run = EXPERIMENTS
    if args.exp:
        to_run = [EXP_BY_NAME[n] for n in args.exp if n in EXP_BY_NAME]
        missing = [n for n in args.exp if n not in EXP_BY_NAME]
        if missing:
            print(f"[sweep] WARNING: unknown experiments: {missing}")
    elif args.groups:
        to_run = [e for e in EXPERIMENTS if e["group"] in args.groups]

    skip_existing = args.skip_existing and not args.no_skip
    if skip_existing:
        results_dir = _SWEEP_DIR / "results"
        to_run = [e for e in to_run
                  if not (results_dir / f"{e['name']}.json").exists()]
        print(f"[sweep] {len(to_run)} experiments to run (skip_existing=True)")
    else:
        print(f"[sweep] {len(to_run)} experiments to run")

    if not to_run:
        print("[sweep] Nothing to run.")
        return

    # ── Load model once ──────────────────────────────────────────────────────
    import torch
    from datasets import load_dataset as hf_load
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import srf_v2 as srf
    from harness import load_pope

    print(f"\n[sweep] Loading {MODEL_ID}…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager",
    ).eval()
    processor    = AutoProcessor.from_pretrained(MODEL_ID,
                                                  max_pixels=512 * 28 * 28)
    img_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    yes_id       = processor.tokenizer.convert_tokens_to_ids("Yes")
    no_id        = processor.tokenizer.convert_tokens_to_ids("No")
    device       = next(model.parameters()).device
    print(f"[sweep] Model loaded on {device}. Running {len(to_run)} experiments.\n")

    # ── Load POPE samples once ───────────────────────────────────────────────
    samples = load_pope(N_SAMPLES)

    # ── Results dir ─────────────────────────────────────────────────────────
    results_dir = _SWEEP_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "sweep.csv"

    # ── Run ──────────────────────────────────────────────────────────────────
    sweep_results = []
    for idx, exp in enumerate(to_run):
        print(f"\n{'='*70}")
        print(f"[sweep] ({idx+1}/{len(to_run)})  {exp['name']}  |  {exp['desc']}")
        print(f"{'='*70}")

        try:
            out = _eval_one(model, processor, samples,
                            img_token_id, yes_id, no_id, device, exp)
        except Exception as exc:
            import traceback
            print(f"[sweep] ERROR in {exp['name']}: {exc}")
            traceback.print_exc()
            out = {
                "exp": exp["name"], "group": exp["group"], "desc": exp["desc"],
                "acc": -1.0, "n_yes": -1, "n_no": -1, "n": N_SAMPLES,
                "elapsed": 0.0, "error": str(exc),
            }

        # Save per-experiment JSON
        json_path = results_dir / f"{exp['name']}.json"
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)

        # Append to CSV immediately (so partial sweep is recoverable)
        _append_csv(out, csv_path)
        sweep_results.append(out)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SWEEP SUMMARY  ({len(sweep_results)} experiments)")
    print(f"{'='*70}")
    print(f"{'Exp':<30}  {'Group':<18}  {'Acc':>6}  {'Yes':>5}  {'No':>5}  {'t(s)':>6}")
    print(f"{'-'*30}  {'-'*18}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*6}")
    sorted_res = sorted(sweep_results, key=lambda r: r.get("acc", 0), reverse=True)
    for r in sorted_res:
        acc_str = f"{r['acc']:.4f}" if r['acc'] >= 0 else " ERROR"
        print(f"{r['exp']:<30}  {r['group']:<18}  {acc_str}  "
              f"{r.get('n_yes','?'):>5}  {r.get('n_no','?'):>5}  "
              f"{r.get('elapsed',0):>6.0f}")

    print(f"\n[sweep] Results written to {csv_path}")
    print(f"[sweep] Per-experiment JSON in {results_dir}/")


if __name__ == "__main__":
    main()
