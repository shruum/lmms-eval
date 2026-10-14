#!/usr/bin/env python3
"""
VLIND-Bench hyperparameter sweep — thin wrapper around eval.run_vlindbench.

Loads model ONCE; each config builds a SimpleNamespace args and calls
eval_mod.run_vlindbench directly. No eval loop reimplementation.

NOTE: max_pixels set to 256*28*28 (vs default 512*28*28) for RTX 2080 Ti (11 GiB).
      Re-run with CFG.DEFAULT_MAX_PIXELS = 512*28*28 on a larger GPU for paper results.

Usage:
  source activate mllm && python srf/sweep_vlind.py --method srf --n 50   # quick test
  source activate mllm && python srf/sweep_vlind.py --method srf           # full 302
  source activate mllm && python srf/sweep_vlind.py --method srfe
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import types

_HERE = pathlib.Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "saliency"))
sys.path.insert(0, str(_REPO / "my_analysis"))

os.environ.setdefault("HF_HOME", "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import config as CFG
import eval as eval_mod  # authoritative eval loop — do NOT reimplement

# NOTE: reduced from CFG.DEFAULT_MAX_PIXELS (512*28*28) for RTX 2080 Ti (11 GiB).
# Re-run with 512*28*28 on a larger GPU for paper-quality results.
_MAX_PIXELS = 256 * 28 * 28


# ── Sweep grid ────────────────────────────────────────────────────────────────

SRF_SWEEP = [
    # Baseline reference
    {"name": "baseline_ref",     "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0, "sys_beta": 0.30, "layer_end": 16, "clip_fallback_thresh": 0.21},

    # neg_absent_alpha sweep — suppresses absent-context samples
    {"name": "neg1.0",           "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 1.0, "sys_beta": 0.30, "layer_end": 16, "clip_fallback_thresh": 0.21},
    {"name": "neg2.0",           "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "clip_fallback_thresh": 0.21},
    {"name": "neg3.0",           "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 3.0, "sys_beta": 0.30, "layer_end": 16, "clip_fallback_thresh": 0.21},

    # clip_fallback_thresh sweep — higher = stricter gate, better Q1/Q2 discrimination
    {"name": "thresh0.30",       "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "clip_fallback_thresh": 0.30},
    {"name": "thresh0.35",       "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "clip_fallback_thresh": 0.35},
    {"name": "thresh0.40",       "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "clip_fallback_thresh": 0.40},
    {"name": "thresh0.45",       "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "clip_fallback_thresh": 0.45},

    # alpha sweep
    {"name": "alpha1.0",         "alpha": 1.0, "eps": 0.1, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "clip_fallback_thresh": 0.35},
    {"name": "alpha3.0",         "alpha": 3.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "clip_fallback_thresh": 0.35},
    {"name": "alpha4.0",         "alpha": 4.0, "eps": 0.3, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "clip_fallback_thresh": 0.35},

    # sys_beta sweep
    {"name": "sysbeta0.0",       "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.00, "layer_end": 16, "clip_fallback_thresh": 0.35},
    {"name": "sysbeta0.15",      "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.15, "layer_end": 16, "clip_fallback_thresh": 0.35},
    {"name": "sysbeta0.50",      "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.50, "layer_end": 16, "clip_fallback_thresh": 0.35},
    {"name": "sysbeta1.0",       "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 1.00, "layer_end": 16, "clip_fallback_thresh": 0.35},

    # layer_end sweep
    {"name": "le12",             "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 12, "clip_fallback_thresh": 0.35},
    {"name": "le20",             "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 20, "clip_fallback_thresh": 0.35},
    {"name": "le24",             "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 24, "clip_fallback_thresh": 0.35},

    # Best combo candidates
    {"name": "best_candidate_A", "alpha": 3.0, "eps": 0.2, "neg_absent_alpha": 3.0, "sys_beta": 0.50, "layer_end": 20, "clip_fallback_thresh": 0.35},
    {"name": "best_candidate_B", "alpha": 4.0, "eps": 0.3, "neg_absent_alpha": 4.0, "sys_beta": 0.30, "layer_end": 24, "clip_fallback_thresh": 0.40},
    {"name": "best_candidate_C", "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.50, "layer_end": 20, "clip_fallback_thresh": 0.40},
]

SRFE_SWEEP = [
    # Baseline reference (current best: 52.32%)
    {"name": "baseline_ref",     "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 3.0, "clip_fallback_thresh": 0.21},

    # gamma sweep
    {"name": "gamma1.0",         "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 1.0, "clip_fallback_thresh": 0.21},
    {"name": "gamma2.0",         "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 2.0, "clip_fallback_thresh": 0.21},
    {"name": "gamma4.0",         "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 4.0, "clip_fallback_thresh": 0.21},
    {"name": "gamma5.0",         "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 5.0, "clip_fallback_thresh": 0.21},

    # clip_fallback_thresh + gamma
    {"name": "thresh0.30_g3",    "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 3.0, "clip_fallback_thresh": 0.30},
    {"name": "thresh0.35_g3",    "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 3.0, "clip_fallback_thresh": 0.35},
    {"name": "thresh0.40_g3",    "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 3.0, "clip_fallback_thresh": 0.40},
    {"name": "thresh0.35_g4",    "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 4.0, "clip_fallback_thresh": 0.35},

    # neg_absent + gamma
    {"name": "neg2_g3",          "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 3.0, "clip_fallback_thresh": 0.21},
    {"name": "neg3_g3",          "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 3.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 3.0, "clip_fallback_thresh": 0.21},
    {"name": "neg2_g4",          "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 4.0, "clip_fallback_thresh": 0.21},

    # alpha + gamma
    {"name": "a3_g3",            "alpha": 3.0, "eps": 0.2, "neg_absent_alpha": 0.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 3.0, "clip_fallback_thresh": 0.21},
    {"name": "a4_g4",            "alpha": 4.0, "eps": 0.3, "neg_absent_alpha": 0.0, "sys_beta": 0.30, "layer_end": 16, "gamma": 4.0, "clip_fallback_thresh": 0.21},

    # sys_beta + gamma
    {"name": "sysbeta0_g3",      "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0, "sys_beta": 0.00, "layer_end": 16, "gamma": 3.0, "clip_fallback_thresh": 0.21},
    {"name": "sysbeta0.5_g3",    "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0, "sys_beta": 0.50, "layer_end": 16, "gamma": 3.0, "clip_fallback_thresh": 0.21},
    {"name": "sysbeta1.0_g3",    "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0, "sys_beta": 1.00, "layer_end": 16, "gamma": 3.0, "clip_fallback_thresh": 0.21},

    # layer_end + gamma
    {"name": "le20_g3",          "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0, "sys_beta": 0.30, "layer_end": 20, "gamma": 3.0, "clip_fallback_thresh": 0.21},
    {"name": "le24_g3",          "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0, "sys_beta": 0.30, "layer_end": 24, "gamma": 3.0, "clip_fallback_thresh": 0.21},

    # Best combo candidates
    {"name": "best_candidate_A", "alpha": 3.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.50, "layer_end": 20, "gamma": 4.0, "clip_fallback_thresh": 0.35},
    {"name": "best_candidate_B", "alpha": 4.0, "eps": 0.3, "neg_absent_alpha": 3.0, "sys_beta": 0.30, "layer_end": 24, "gamma": 3.0, "clip_fallback_thresh": 0.35},
    {"name": "best_candidate_C", "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0, "sys_beta": 0.50, "layer_end": 20, "gamma": 5.0, "clip_fallback_thresh": 0.40},
]


# ── Build fake args namespace ─────────────────────────────────────────────────

def _make_args(method: str, cfg: dict, n: int | None, seed: int) -> types.SimpleNamespace:
    """Build SimpleNamespace matching eval.py's argparse output for run_vlindbench."""
    return types.SimpleNamespace(
        method=method,
        datasets=["vlind"],
        gamma=[cfg.get("gamma", 3.0)] if method == "srfe" else [0.0],
        vlind_download=False,
        vlind_n=n,
        seed=seed,
        # Sweep params — override only what's in cfg; None → use dataset config default
        phase=None,
        alpha=cfg.get("alpha"),
        eps=cfg.get("eps"),
        neg_absent_alpha=cfg.get("neg_absent_alpha"),
        layer_start=None,
        layer_end=cfg.get("layer_end"),
        head_top_k_pct=None,
        clip_coarse_grid=None,
        clip_top_k_pct=None,
        clip_fallback_thresh=cfg.get("clip_fallback_thresh"),
        saliency_mode=None,
        bias_mode=None,
        interp_lambda=None,
        sys_beta=cfg.get("sys_beta"),
        text_beta=None,
        text_layer_start=None,
        text_layer_end=None,
        prob_floor=None,
        img_scale=None,
        vr_target=None,
        vr_k=None,
        clip_model=None,
        hssa_layer_idx=None,
        clip_saliency_method=None,
        srf2_clip_weight=None,
        srf2_hssa_weight=None,
        lta_layer_idx=None,
        lta_weight=None,
        clip_weight=None,
        noise_step=None,
        cd_alpha=None,
        cd_beta=None,
        vaf_beta=None,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["srf", "srfe"], default="srf")
    ap.add_argument("--n",    type=int, default=None, help="sample subset (None=all 302)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",  type=str, default="results/sweep_vlind/")
    cli = ap.parse_args()

    out_dir = pathlib.Path(cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = SRFE_SWEEP if cli.method == "srfe" else SRF_SWEEP
    print(f"Method: {cli.method}  |  {len(sweep)} configs  |  n={cli.n or 'all'}")

    # Load model with reduced max_pixels for RTX 2080 Ti (11 GiB)
    # NOTE: set _MAX_PIXELS = CFG.DEFAULT_MAX_PIXELS (512*28*28) for paper results on larger GPU
    orig_px = CFG.DEFAULT_MAX_PIXELS
    CFG.DEFAULT_MAX_PIXELS = _MAX_PIXELS
    model, processor = eval_mod.load_model(CFG.DEFAULT_MODEL)
    CFG.DEFAULT_MAX_PIXELS = orig_px

    img_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    device = next(p for p in model.parameters() if p.is_cuda).device

    import srf_e as srfe_mod
    import srf as srf_mod
    method_mod = srfe_mod if cli.method == "srfe" else srf_mod

    # One-time calibration on VLIND (same domain as eval — methodologically correct).
    # Calibration images are resized to 224×224 in srf._build_calib_inputs to avoid OOM.
    print(f"Calibrating {cli.method.upper()} on vlind...")
    method_mod.setup(model, processor, calib_dataset="vlind")

    all_results: list[dict] = []
    best_pair = 0.0

    for ci, cfg in enumerate(sweep):
        print(f"\n[{ci+1}/{len(sweep)}] {cfg['name']}  cfg={cfg}")
        t0 = time.time()
        try:
            fake_args = _make_args(cli.method, cfg, cli.n, cli.seed)
            r = eval_mod.run_vlindbench(method_mod, model, processor, img_token_id, device, fake_args)
            elapsed = time.time() - t0

            # run_vlindbench returns {gamma: value} dicts; extract the single gamma we used
            gamma_key = fake_args.gamma[0]
            pair_acc = r["pair_acc"][gamma_key]
            q_acc    = r["q_acc"][gamma_key]

            print(f"  pair_acc={pair_acc*100:.2f}%  q_acc={q_acc*100:.2f}%  ({elapsed:.0f}s)")
            if pair_acc > best_pair:
                best_pair = pair_acc
                print(f"  *** NEW BEST: {best_pair*100:.2f}% ***")

            all_results.append({
                "name": cfg["name"],
                "config": {k: v for k, v in cfg.items() if k != "name"},
                "pair_acc": pair_acc,
                "q_acc": q_acc,
                "per_concept": r["per_concept"][gamma_key],
            })
        except Exception as ex:
            import traceback
            print(f"  ERROR: {ex}")
            traceback.print_exc()
            all_results.append({"name": cfg["name"], "error": str(ex)})

        torch.cuda.empty_cache()

        # Save after every config
        out_file = out_dir / f"sweep_{cli.method}.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)

    # Final leaderboard
    print("\n" + "=" * 60)
    print(f"SWEEP COMPLETE — {cli.method.upper()} — Top results:")
    print("=" * 60)
    valid_r = [r for r in all_results if "pair_acc" in r]
    valid_r.sort(key=lambda x: x["pair_acc"], reverse=True)
    print(f"{'Name':<25} {'pair_acc':>9} {'q_acc':>8}  config")
    print("-" * 80)
    for r in valid_r[:10]:
        cfg_str = "  ".join(f"{k}={v}" for k, v in r["config"].items())
        print(f"{r['name']:<25} {r['pair_acc']*100:>8.2f}% {r['q_acc']*100:>7.2f}%  {cfg_str}")

    print(f"\nFull results: {cli.out}/sweep_{cli.method}.json")


if __name__ == "__main__":
    main()
