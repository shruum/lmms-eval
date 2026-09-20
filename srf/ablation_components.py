#!/usr/bin/env python3
"""
ablation_components.py — cumulative component ablation for the ICLR paper.

Builds SRF up one component at a time on MMVP / Qwen2.5-VL-3B and reports the
marginal contribution of each stage described in the method section:

  A baseline            no intervention
  B uniform_boost       boost every image token, every head, every layer
  C sem_mask            + semantic relevance map (CLIP)          [§ semantic relevance]
  D sem_heads           + vision-responsive head calibration     [§ head calibration]
  E sem_heads_layers    + fusion-layer targeting                 [§ fusion-layer targeting]
  F srf_decoder         + background / system suppression (λ, η) [§ decoder re-focus]
  G srf_full            + semantic foveation (σ=20)              [§ semantic foveation]

Row A is a true no-op and row G is the full published method, so both re-derive
known numbers and act as a consistency check on the whole pass.

Two anchors are run because the docs disagree with config.py on MMVP:

  published    saliency_mode=clip_full_gate_v3, alpha=2.0, layer_end=16
               → the configuration behind the 43.3% pair accuracy in the paper
  mmvp_tuned   saliency_mode=clip, alpha=4.0, layer_end=15
               → srf/docs/MMVP.md best params, confirmed by results/srf_base_best/
                 (clip 42.00% vs clip_full_gate_v3 39.33% at matched layer_end)

alpha is held fixed within an anchor so each table isolates *targeting* rather
than boost strength.

The evaluation loop itself is NOT reimplemented: every variant is handed to
eval.run_mmvp(), the same function that produced the published MMVP numbers
(validated prompt, MMVP_GT_CORRECTIONS, pair-accuracy definition). This module
only adds the variant loop and the per-variant state overrides.

Usage
-----
  cd /volumes2/mllm/lmms-eval

  source activate mllm && stdbuf -oL -eL python -u srf/ablation_components.py \\
    --output results/ablation/ 2>&1 | tee /tmp/ablation_components_mmvp.log

  # single anchor
  python srf/ablation_components.py --anchors mmvp_tuned

  # smoke test (2 rows, still 300 images each)
  python srf/ablation_components.py --anchors published --variants baseline srf_full
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import random
import sys
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

import eval as eval_mod
import eval_ablation as abl_mod       # reused: _prepare_saliency (uniform saliency path)
import qwen_attn_patch as patch
import srf as srf_mod
import srf_fovea as srf_fovea_mod


# ---------------------------------------------------------------------------
# Anchor configurations
# ---------------------------------------------------------------------------
# Each anchor fixes the hyperparameters that every row in its table shares.
# `eps` and `sys_beta` are the λ / η values switched on only at row F.

ANCHORS: Dict[str, dict] = {
    "published": {
        "saliency_mode": "clip_full_gate_v3",
        "alpha":         2.0,
        "layer_start":   8,
        "layer_end":     16,
        "eps":           0.2,
        "sys_beta":      0.30,
        "phase":         "both",
        "note":          "config.py defaults — the configuration behind the paper's 43.3%",
    },
    "mmvp_tuned": {
        "saliency_mode": "clip",
        "alpha":         4.0,
        "layer_start":   8,
        "layer_end":     15,
        "eps":           0.2,
        "sys_beta":      0.30,
        "phase":         "both",
        "note":          "srf/docs/MMVP.md best params (basic clip gate, alpha=4.0)",
    },
}

FOVEA_SIGMA = 20.0    # srf_fovea.SIGMA default; best on MMVP (EXPERIMENTS.md sigma sweep)


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
# saliency : "anchor"   use the anchor's saliency_mode (CLIP semantic relevance)
#            "uniform"  all-ones mask — boost every image token equally
# heads    : "cal"      calibrated vision-responsive heads (top head_top_k_pct)
#            "all"      every head (patch treats head_mask=None as all heads)
# layers   : "anchor"   the anchor's [layer_start, layer_end] fusion interval
#            "all"      every decoder layer [0, n_layers-1]
# suppress : True       apply the anchor's eps (λ) and sys_beta (η)
#            False      eps=0.0, sys_beta=0.0 — boost only, no suppression
# fovea    : True       route through srf_fovea (pre-encoder foveal blur, σ=20)

VARIANTS: List[dict] = [
    {"key": "baseline",         "label": "Baseline (no intervention)",
     "baseline": True},
    {"key": "uniform_boost",    "label": "+ uniform attention boost",
     "saliency": "uniform", "heads": "all", "layers": "all",
     "suppress": False, "fovea": False},
    {"key": "sem_mask",         "label": "+ semantic relevance mask",
     "saliency": "anchor",  "heads": "all", "layers": "all",
     "suppress": False, "fovea": False},
    {"key": "sem_heads",        "label": "+ vision-responsive head calibration",
     "saliency": "anchor",  "heads": "cal", "layers": "all",
     "suppress": False, "fovea": False},
    {"key": "sem_heads_layers", "label": "+ fusion-layer targeting",
     "saliency": "anchor",  "heads": "cal", "layers": "anchor",
     "suppress": False, "fovea": False},
    {"key": "srf_decoder",      "label": "+ background/system suppression  [SRF decoder-only]",
     "saliency": "anchor",  "heads": "cal", "layers": "anchor",
     "suppress": True,  "fovea": False},
    {"key": "srf_full",         "label": "+ semantic foveation  [full SRF]",
     "saliency": "anchor",  "heads": "cal", "layers": "anchor",
     "suppress": True,  "fovea": True},
]

VARIANT_BY_KEY = {v["key"]: v for v in VARIANTS}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cumulative SRF component ablation on MMVP.")
    p.add_argument("--model",    default=CFG.DEFAULT_MODEL)
    p.add_argument("--anchors",  nargs="+", default=list(ANCHORS),
                   choices=list(ANCHORS),
                   help="Anchor configurations to run (default: both).")
    p.add_argument("--variants", nargs="+", default=[v["key"] for v in VARIANTS],
                   choices=[v["key"] for v in VARIANTS],
                   help="Subset of rows to run (default: all 7).")
    p.add_argument("--sigma",    type=float, default=FOVEA_SIGMA,
                   help=f"Foveal blur sigma for the srf_full row (default {FOVEA_SIGMA}).")
    p.add_argument("--output",   default=None,
                   help="Directory for the results JSON (default: no file written).")
    return p.parse_args()


def build_eval_args(model: str) -> argparse.Namespace:
    """
    Construct eval.py's Namespace by calling its own parse_args with a patched
    argv, so the override set stays in sync with eval.py automatically instead
    of being duplicated here.
    """
    saved = sys.argv
    try:
        sys.argv = ["eval.py", "--method", "srf", "--model", model,
                    "--datasets", "mmvp"]
        return eval_mod.parse_args()
    finally:
        sys.argv = saved


def anchor_args(base: argparse.Namespace, anchor: dict,
                suppress: bool) -> argparse.Namespace:
    """Copy of `base` carrying one anchor's hyperparameters."""
    a = copy.deepcopy(base)
    a.saliency_mode = anchor["saliency_mode"]
    a.alpha         = anchor["alpha"]
    a.layer_start   = anchor["layer_start"]
    a.layer_end     = anchor["layer_end"]
    a.phase         = anchor["phase"]
    # λ and η are the components switched on at row F.
    a.eps           = anchor["eps"]      if suppress else 0.0
    a.sys_beta      = anchor["sys_beta"] if suppress else 0.0
    return a


# ---------------------------------------------------------------------------
# Method-module shims
# ---------------------------------------------------------------------------
# eval.run_mmvp() expects a module-like object exposing reset_for_dataset /
# prepare_sample / cleanup. These shims delegate to srf (or srf_fovea) and then
# re-apply the per-variant state overrides.
#
# The re-apply after cleanup() is load-bearing: srf.cleanup() restores
# vaf_layer_start/end from BIAS on every sample, which would otherwise undo the
# layer-range override after the first image.


class _BaselineShim:
    """True no-op on an already-patched model.

    Defines generate_contrastive so eval.method_generate() takes that branch
    instead of forcing _STATE["method"] = "srf" on a patched model.
    """

    def reset_for_dataset(self, dataset: Optional[str] = None, **kwargs) -> None:
        patch._STATE["method"] = "baseline"

    def prepare_sample(self, inp, img_start: int, img_end: int, image,
                       question: str, model, processor, **kwargs) -> None:
        patch.update_sample(img_start, img_end)
        patch._STATE["method"]        = "baseline"
        patch._STATE["salience_mask"] = None

    def generate_contrastive(self, model, inp, processor, gamma: float = 0.0,
                             max_new_tokens: int = 20,
                             content_offset: int = 0) -> List[int]:
        patch._STATE["method"] = "baseline"
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=max_new_tokens,
                                 do_sample=False)
        return out[0, inp["input_ids"].shape[1]:].tolist()

    def cleanup(self) -> None:
        patch._STATE["salience_mask"] = None
        patch._STATE["method"]        = "baseline"


class _SRFShim:
    """SRF (optionally foveated) with per-variant head / layer / saliency overrides."""

    def __init__(self, variant: dict, calibrated_mask: torch.Tensor,
                 n_layers: int, anchor: dict, sigma: float) -> None:
        self.variant    = variant
        self.base       = srf_fovea_mod if variant["fovea"] else srf_mod
        self.sigma      = sigma
        self.calibrated = calibrated_mask     # this anchor's calibrated mask

        # None = all heads, per qwen_attn_patch._STATE["head_mask"] semantics.
        self.use_calibrated_heads = (variant["heads"] == "cal")
        self.layers = ((anchor["layer_start"], anchor["layer_end"])
                       if variant["layers"] == "anchor" else (0, n_layers - 1))
        self._rng = random.Random(0)          # unused by the uniform path

    def _apply_overrides(self) -> None:
        patch._STATE["head_mask"]       = (self.calibrated.clone()
                                           if self.use_calibrated_heads else None)
        patch._STATE["vaf_layer_start"] = self.layers[0]
        patch._STATE["vaf_layer_end"]   = self.layers[1]

    def reset_for_dataset(self, dataset: Optional[str] = None, **kwargs) -> None:
        if self.variant["fovea"]:
            srf_fovea_mod.SIGMA = self.sigma
        # Restore the calibrated mask before resetting: srf.reset_for_dataset
        # re-runs calibration when head_mask is None, which a preceding
        # all-heads row would otherwise trigger on every subsequent row.
        patch._STATE["head_mask"] = self.calibrated.clone()
        # Reset through srf so BIAS/SALIENCY follow the anchor. The layer range
        # is overridden afterwards rather than passed in, because a layer_end
        # change would trigger head re-calibration and alter the head ranking.
        self.base.reset_for_dataset(dataset=dataset, **kwargs)
        self._apply_overrides()

    def prepare_sample(self, inp, img_start: int, img_end: int, image,
                       question: str, model, processor, **kwargs) -> None:
        if self.variant["saliency"] == "uniform":
            # Reuse eval_ablation's uniform-saliency path rather than
            # reconstructing the patch-state contract here.
            abl_mod._prepare_saliency("uniform", inp, img_start, img_end,
                                       image, question, model, processor,
                                       self._rng)
        else:
            self.base.prepare_sample(inp, img_start, img_end, image, question,
                                      model, processor, **kwargs)
        self._apply_overrides()

    def cleanup(self) -> None:
        self.base.cleanup()
        self._apply_overrides()   # srf.cleanup() restores BIAS layer range


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(x: float) -> float:
    return round(100.0 * x, 2)


def print_summary(anchor_name: str, anchor: dict,
                  rows: List[Tuple[str, str, float, float]]) -> None:
    print("\n" + "=" * 84)
    print(f"ABLATION SUMMARY — anchor '{anchor_name}'  ({anchor['note']})")
    print(f"  saliency_mode={anchor['saliency_mode']}  alpha={anchor['alpha']}  "
          f"layers=[{anchor['layer_start']},{anchor['layer_end']}]  "
          f"eps={anchor['eps']}  sys_beta={anchor['sys_beta']}")
    print("=" * 84)
    print(f"{'Row':<48} {'Pair':>7} {'Δprev':>7} {'Δbase':>7} {'Img':>7}")
    print("-" * 84)

    base_pair = rows[0][2] if rows else float("nan")
    prev_pair = None
    for key, label, pair, img in rows:
        d_prev = "" if prev_pair is None else f"{_pct(pair - prev_pair):+.2f}"
        d_base = f"{_pct(pair - base_pair):+.2f}" if key != rows[0][0] else ""
        print(f"{label:<48} {_pct(pair):>7.2f} {d_prev:>7} {d_base:>7} "
              f"{_pct(img):>7.2f}")
        prev_pair = pair

    print("\nLaTeX rows (booktabs):")
    for key, label, pair, img in rows:
        clean = label.replace("[", "(").replace("]", ")").replace("σ", r"$\sigma$")
        print(f"  {clean} & {_pct(pair):.1f} & {_pct(img):.1f} \\\\")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    model, processor = eval_mod.load_model(args.model)
    img_token_id     = processor.tokenizer.convert_tokens_to_ids(CFG.IMAGE_TOKEN)
    device           = next(model.parameters()).device
    n_layers         = CFG.get_arch(args.model)["n_layers"]

    # Patch the model and identify vision-aware heads once.
    srf_mod.setup(model, processor, calib_dataset="mmvp")

    base_args = build_eval_args(args.model)
    results: Dict[str, Dict[str, dict]] = {}

    for anchor_name in args.anchors:
        anchor = ANCHORS[anchor_name]
        results[anchor_name] = {}
        rows: List[Tuple[str, str, float, float]] = []

        # Warm-up reset: applies this anchor's params and triggers head
        # re-calibration if layer_end changed since the previous anchor. Every
        # row in this anchor then shares one calibrated mask, so head selection
        # is never a confound between rows.
        warm = anchor_args(base_args, anchor, suppress=True)
        srf_mod.reset_for_dataset(dataset="mmvp", **eval_mod._reset_overrides(warm))
        calibrated_mask = patch._STATE["head_mask"].clone()
        print(f"\n  [{anchor_name}] calibrated heads: "
              f"{int(calibrated_mask.sum())}/{len(calibrated_mask)}")

        for key in args.variants:
            variant = VARIANT_BY_KEY[key]
            print("\n" + "=" * 84)
            print(f"[{anchor_name}]  {variant['label']}")
            print("=" * 84)

            eargs = anchor_args(base_args, anchor,
                                suppress=variant.get("suppress", False))

            if variant.get("baseline"):
                shim = _BaselineShim()
            else:
                shim = _SRFShim(variant, calibrated_mask, n_layers,
                                anchor, args.sigma)

            res  = eval_mod.run_mmvp(shim, model, processor, img_token_id,
                                      device, eargs)
            pair = res["method_pair"][0.0]
            img  = res["method_img"][0.0]

            results[anchor_name][key] = {
                "label":    variant["label"],
                "pair_acc": pair,
                "img_acc":  img,
                "config": {
                    "saliency": variant.get("saliency", "none"),
                    "heads":    variant.get("heads",    "none"),
                    "layers":   variant.get("layers",   "none"),
                    "suppress": variant.get("suppress", False),
                    "fovea":    variant.get("fovea",    False),
                    "sigma":    args.sigma if variant.get("fovea") else None,
                    "alpha":    eargs.alpha,
                    "eps":      eargs.eps,
                    "sys_beta": eargs.sys_beta,
                    "layer_range": (list(shim.layers)
                                    if isinstance(shim, _SRFShim) else None),
                },
            }
            rows.append((key, variant["label"], pair, img))
            print(f"  → pair={_pct(pair):.2f}%  img={_pct(img):.2f}%")

        print_summary(anchor_name, anchor, rows)

    # Cross-anchor consistency check: the baseline row must be identical.
    if len(args.anchors) > 1 and "baseline" in args.variants:
        pairs = {a: results[a]["baseline"]["pair_acc"] for a in args.anchors}
        if len(set(pairs.values())) > 1:
            print("\n  WARNING: baseline pair accuracy differs across anchors "
                  f"({pairs}) — indicates state leaking between variants. "
                  "Treat the rest of this run as untrustworthy.")
        else:
            print(f"\n  Baseline consistent across anchors: "
                  f"{_pct(list(pairs.values())[0]):.2f}%")

    if args.output:
        out_dir = pathlib.Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "ablation_components_mmvp_qwen3b.json"
        payload  = {
            "model":    args.model,
            "dataset":  "mmvp",
            "sigma":    args.sigma,
            "anchors":  {a: ANCHORS[a] for a in args.anchors},
            "results":  results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
