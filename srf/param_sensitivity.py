#!/usr/bin/env python3
"""
param_sensitivity.py — one-parameter-at-a-time sensitivity sweeps on MMVP.

Sweeps each SRF parameter in turn while holding every other parameter at the
published MMVP configuration, so each curve isolates one knob. Produces the
"Parameter sensitivity" appendix table/figures.

Parameters swept (notation as settled 2026-09-17):

  lambda_sem  boost_alpha           amplify relevant image tokens
  lambda_bg   background_eps        attenuate background image tokens
  lambda_sys  sys_beta              suppress system-prompt tokens
  sigma       srf_fovea.SIGMA       foveal blur scale, pixels
  tau         clip_fallback_thresh  presence threshold
  k           head_top_k_pct        head fraction per layer (triggers re-calibration)
  layers      [layer_start, layer_end]

Sweeping lambda_bg and lambda_sys independently also SEPARATES them, which the
cumulative ablation could not do because it moved both in a single step.

Reuses eval.run_mmvp (the function that produced every published MMVP number),
eval.parse_args and ablation_components for the anchor, so no evaluation loop,
saliency function or noun extractor is reimplemented here.

Usage
-----
  cd /volumes2/mllm/lmms-eval

  # everything (~30 passes, ~2 h)
  source activate mllm && stdbuf -oL -eL python -u srf/param_sensitivity.py \\
    --output results/sensitivity/ 2>&1 | tee /tmp/param_sensitivity_mmvp.log

  # just the two suppression terms (separates lambda_bg from lambda_sys)
  python srf/param_sensitivity.py --params lambda_bg lambda_sys
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
from typing import Dict, List

_SRF_DIR      = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR / "saliency"))
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import config as CFG

os.environ.setdefault("HF_HOME", CFG.HF_HOME)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import ablation_components as abl_comp
import eval as eval_mod
import head_calibration as hc
import srf as srf_mod
import srf_fovea as srf_fovea_mod

# Sweep grids. Each entry maps an args attribute to the values to try. `sigma`
# and `layers` are handled specially below.
SWEEPS: Dict[str, dict] = {
    "lambda_sem": {"attr": "alpha",                "values": [0.5, 1.0, 2.0, 4.0, 8.0]},
    # lambda_bg and lambda_sys grids extended upward: under the old anchor the
    # best value sat at the top of the grid, so the response was not
    # characterised on that side.
    "lambda_bg":  {"attr": "eps",                  "values": [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]},
    "lambda_sys": {"attr": "sys_beta",             "values": [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]},
    # tau: the old grid was flat at baseline for every value >= 0.3, which spent
    # three passes measuring "the gate rejects everything". Resampled below 0.3.
    "tau":        {"attr": "clip_fallback_thresh", "values": [0.05, 0.10, 0.15, 0.20, 0.30]},
    "k":          {"attr": "head_top_k_pct",       "values": [0.10, 0.20, 0.30, 0.50]},
    "sigma":      {"attr": "__sigma__",            "values": [0.0, 10.0, 20.0, 30.0, 50.0]},
    # Bands around the adopted [6,31]. Includes the old anchor [8,16] and the
    # full decoder [0,35] so the appendix can state what the band buys.
    "layers":     {"attr": "__layers__",           "values": [(6, 31), (8, 16), (0, 35),
                                                               (6, 20), (18, 31), (12, 24)]},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-at-a-time parameter sensitivity on MMVP.")
    p.add_argument("--model",  default=CFG.DEFAULT_MODEL)
    p.add_argument("--params", nargs="+", default=list(SWEEPS), choices=list(SWEEPS))
    p.add_argument("--saliency_mode", default=None,
                   help="Override saliency mode (default: the anchor's, clip_full_gate_v3).")
    p.add_argument("--anchor", default="current", choices=list(abl_comp.ANCHORS),
                   help="Configuration to hold the un-swept parameters at.")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    anchor = abl_comp.ANCHORS[args.anchor]

    model, processor = eval_mod.load_model(args.model)
    img_token_id     = processor.tokenizer.convert_tokens_to_ids(CFG.IMAGE_TOKEN)
    device           = next(model.parameters()).device

    srf_mod.setup(model, processor, calib_dataset="mmvp")

    base = abl_comp.anchor_args(abl_comp.build_eval_args(args.model), anchor, suppress=True)
    if args.saliency_mode:
        base.saliency_mode = args.saliency_mode
    base_sigma = abl_comp.FOVEA_SIGMA

    print("\n" + "=" * 84)
    print("PARAMETER SENSITIVITY — MMVP, Qwen2.5-VL-3B-Instruct")
    print(f"  anchor:  {args.anchor} — {anchor['note']}")
    print(f"  held at: saliency={base.saliency_mode}  lambda_sem={base.alpha}  "
          f"lambda_bg={base.eps}  lambda_sys={base.sys_beta}  "
          f"layers=[{base.layer_start},{base.layer_end}]  sigma={base_sigma}  "
          f"phase={base.phase}  head_mode={getattr(base, 'head_mode', 'global')}  "
          f"k={getattr(base, 'head_top_k_pct', None)}")
    print("=" * 84)

    results: Dict[str, List[dict]] = {}

    for name in args.params:
        spec = SWEEPS[name]
        rows: List[dict] = []
        print(f"\n{'='*84}\nSWEEP  {name}  ({spec['attr']})  values={spec['values']}\n{'='*84}")

        for val in spec["values"]:
            eargs = copy.deepcopy(base)
            sigma = base_sigma

            if spec["attr"] == "__sigma__":
                sigma = float(val)
            elif spec["attr"] == "__layers__":
                eargs.layer_start, eargs.layer_end = int(val[0]), int(val[1])
            else:
                setattr(eargs, spec["attr"], val)

            srf_fovea_mod.SIGMA = sigma
            label = f"{name}={val}"
            print(f"\n--- {label}  (sigma={sigma}) ---")

            # Head masks depend on k and on the band, so re-install per point.
            # _install_head_mode is the same entry point eval.py uses, so the
            # selection logic has one implementation.
            handles = eval_mod._install_head_mode(model, processor, eargs, "mmvp")
            try:
                res = eval_mod.run_mmvp(srf_fovea_mod, model, processor,
                                         img_token_id, device, eargs)
            finally:
                hc.remove_hooks(handles)
            pair = res["method_pair"][0.0]
            img  = res["method_img"][0.0]
            rows.append({"value": val, "pair_acc": pair, "img_acc": img})
            print(f"  → {label}:  pair={pair*100:.2f}%  img={img*100:.2f}%")

        results[name] = rows
        print(f"\n  {name} sweep summary")
        print(f"  {'value':>12} {'pair':>8} {'img':>8}")
        print("  " + "-" * 30)
        for r in rows:
            print(f"  {str(r['value']):>12} {r['pair_acc']*100:>8.2f} {r['img_acc']*100:>8.2f}")

    # restore
    srf_fovea_mod.SIGMA = base_sigma

    print("\n" + "=" * 84)
    print("ALL SWEEPS")
    print("=" * 84)
    for name, rows in results.items():
        best = max(rows, key=lambda r: r["pair_acc"])
        span = max(r["pair_acc"] for r in rows) - min(r["pair_acc"] for r in rows)
        print(f"  {name:<12} best={str(best['value']):<10} "
              f"pair={best['pair_acc']*100:.2f}%   spread={span*100:.2f}pp")

    if args.output:
        out = pathlib.Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "param_sensitivity_mmvp_qwen3b.json"
        path.write_text(json.dumps({
            "model": args.model, "dataset": "mmvp", "anchor": anchor,
            "base_sigma": base_sigma, "results": results,
        }, indent=2, default=str))
        print(f"\n  Saved → {path}")


if __name__ == "__main__":
    main()
