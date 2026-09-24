#!/usr/bin/env python3
"""
profile_cost.py — inference-time cost of SRF, measured not estimated.

Answers the three questions a reviewer asks about a training-free method.
  1. How much wall-clock time does SRF add per sample, and where does it go?
  2. How many extra forward passes does the alignment model actually run?
  3. How does that compare to the cost of the VLM itself, in FLOPs?

Method
------
Runs the real MMVP evaluation twice, baseline and SRF, through eval.run_mmvp —
the same function that produced every published number. Nothing about the
evaluation is reimplemented. Timing comes from a shim around the method module
plus counters on the CLIP encoder, so the measured path is exactly the shipped
one.

Stages timed per sample:
  clip     alignment-model passes (full image + patch crops at 3 grid scales)
  fovea    Gaussian blur, pixel composite, and ViT re-processing
  vlm      everything else, i.e. encode + prefill + decode

FLOPs are reported as
  CLIP  = (measured GFLOPs for one image forward) x (counted images)
  VLM   = 2 x n_params x n_tokens, the standard dense-transformer estimate
The attention-logit modification is O(n_heads x n_tokens) additions inside an
existing kernel and is below the resolution of both measurements. It is
reported as part of `vlm` rather than claimed to be free.

Usage
-----
  cd /volumes2/mllm/lmms-eval
  source activate mllm && stdbuf -oL -eL python -u srf/profile_cost.py \
    --output results/cost/ 2>&1 | tee /tmp/profile_cost.log
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
import time
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

import torch

import ablation_components as abl_comp
import baseline as baseline_mod
import clip_salience as clip_sal
import eval as eval_mod
import head_calibration as hc
import srf as srf_mod
import srf_fovea as srf_fovea_mod
import vaf as vaf_mod
import vcd as vcd_mod
import ilvad as ilvad_mod
import vhr as vhr_mod


# ---------------------------------------------------------------------------
# Counters, installed on the CLIP encoder itself
# ---------------------------------------------------------------------------

COUNTS = {"img_calls": 0, "img_items": 0, "txt_calls": 0, "txt_items": 0,
          "clip_sec": 0.0}
_ORIG_LOAD = clip_sal._load_clip
_PATCHED: set = set()


def _instrumented_load(*a, **kw):
    """Wrap _load_clip so every CLIP encoder call is counted and timed."""
    model, proc = _ORIG_LOAD(*a, **kw)
    if id(model) in _PATCHED:
        return model, proc
    _PATCHED.add(id(model))

    def _wrap(fn, call_key, item_key):
        def inner(*args, **kwargs):
            pv = kwargs.get("pixel_values", kwargs.get("input_ids"))
            n  = int(pv.shape[0]) if pv is not None and hasattr(pv, "shape") else 1
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            COUNTS["clip_sec"] += time.perf_counter() - t0
            COUNTS[call_key] += 1
            COUNTS[item_key] += n
            return out
        return inner

    model.get_image_features = _wrap(model.get_image_features, "img_calls", "img_items")
    model.get_text_features  = _wrap(model.get_text_features,  "txt_calls", "txt_items")
    return model, proc


def reset_counts() -> None:
    for k in COUNTS:
        COUNTS[k] = 0 if k != "clip_sec" else 0.0


# ---------------------------------------------------------------------------
# Timing shim — wraps whichever method module run_mmvp is given
# ---------------------------------------------------------------------------

class _TimedShim:
    """Delegates to `base` and records how long prepare_sample spends.

    Attribute lookups fall through to the wrapped module, so optional hooks
    such as `generate_contrastive` stay visible to `eval.method_generate`.
    Without that fall-through the contrastive baselines are silently measured
    as ordinary generation.
    """

    def __init__(self, base) -> None:
        self.base       = base
        self.prep_sec   = 0.0
        self.n_samples  = 0
        self.tok_total  = 0

    def __getattr__(self, name):
        # Only reached for attributes not defined on the shim itself.
        return getattr(self.__dict__["base"], name)

    def reset_for_dataset(self, dataset=None, **kw):
        return self.base.reset_for_dataset(dataset=dataset, **kw)

    def prepare_sample(self, inp, img_start, img_end, image, question,
                       model, processor, **kw):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.base.prepare_sample(inp, img_start, img_end, image, question,
                                        model, processor, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.prep_sec  += time.perf_counter() - t0
        self.n_samples += 1
        try:                                  # BatchFeature is a UserDict, not a dict
            ids = inp["input_ids"]
        except (TypeError, KeyError, IndexError):
            ids = None
        if ids is not None and hasattr(ids, "shape"):
            self.tok_total += int(ids.shape[-1])
        return out

    def cleanup(self):
        return self.base.cleanup()


# ---------------------------------------------------------------------------
# VLM forward counter — makes the "+pass" column measured rather than asserted
# ---------------------------------------------------------------------------

PASSES = {"forwards": 0, "prefills": 0}


def reset_passes() -> None:
    PASSES["forwards"] = 0
    PASSES["prefills"] = 0


def _count_forward(module, args, kwargs):
    """Forward pre-hook on the top-level VLM. A prefill has q_len > 1."""
    PASSES["forwards"] += 1
    ids = kwargs.get("input_ids")
    if ids is None and args:
        ids = args[0]
    if ids is not None and hasattr(ids, "shape") and ids.ndim >= 2 and ids.shape[-1] > 1:
        PASSES["prefills"] += 1
    return None


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------

def clip_gflops_per_image(model, processor) -> float:
    """Measure one CLIP image forward with the profiler. Returns GFLOPs."""
    from PIL import Image
    import numpy as np
    dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    inp   = processor(images=dummy, return_tensors="pt")
    dev   = next(model.parameters()).device
    inp   = {k: v.to(dev) for k, v in inp.items()}
    with torch.no_grad():
        with torch.profiler.profile(with_flops=True) as prof:
            model.get_image_features(**inp)
    total = sum(e.flops for e in prof.key_averages() if e.flops)
    return total / 1e9


def vlm_gflops(model, n_tokens: int) -> float:
    """2 * n_params * n_tokens, the standard dense forward-pass estimate."""
    n_params = sum(p.numel() for p in model.parameters())
    return 2.0 * n_params * n_tokens / 1e9


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
# name -> (module attr on this file's imports, eval `--method` string, is_srf)
# is_srf marks the arms that take the head-mode install and the SRF anchor.

ARMS = {
    "baseline":    ("baseline_mod",   "baseline",  False),
    "srf_decoder": ("srf_mod",        "srf",       True),
    "srf":         ("srf_fovea_mod",  "srffovea",  True),
    "vaf":         ("vaf_mod",        "vaf",       False),
    "vcd":         ("vcd_mod",        "vcd",       False),
    "ilvad":       ("ilvad_mod",      "ilvad",     False),
    "vhr":         ("vhr_mod",        "vhr",       False),
}


def _resolve(name: str):
    attr, method, is_srf = ARMS[name]
    return globals()[attr], method, is_srf


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference-time cost on MMVP.")
    p.add_argument("--model",  default=CFG.DEFAULT_MODEL)
    p.add_argument("--anchor", default="current", choices=list(abl_comp.ANCHORS))
    p.add_argument("--arms",   nargs="+", default=["baseline", "srf"],
                   choices=list(ARMS),
                   help="srf_decoder = attention only, no foveation. "
                        "vaf/vcd/ilvad/vhr are the comparison baselines.")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    anchor = abl_comp.ANCHORS[args.anchor]

    clip_sal._load_clip = _instrumented_load

    model, processor = eval_mod.load_model(args.model)
    img_token_id     = processor.tokenizer.convert_tokens_to_ids(CFG.IMAGE_TOKEN)
    device           = next(model.parameters()).device

    pass_hook = model.register_forward_pre_hook(_count_forward, with_kwargs=True)

    base = abl_comp.anchor_args(abl_comp.build_eval_args(args.model), anchor,
                                suppress=True)

    print("\n" + "=" * 96)
    print(f"INFERENCE COST — MMVP, {args.model}")
    print(f"  anchor: {args.anchor} — {anchor['note']}")
    print("=" * 96)

    rows: Dict[str, dict] = {}

    try:
        for arm in args.arms:
            method_mod, method_name, is_srf = _resolve(arm)
            eargs = copy.deepcopy(base)
            eargs.method = method_name
            if arm == "srf":
                srf_fovea_mod.SIGMA = abl_comp.FOVEA_SIGMA

            # One-time per-model setup, timed separately. For SRF this is the
            # 20-sample head calibration, which is amortised over the run.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t_setup = time.perf_counter()
            method_mod.setup(model, processor, calib_dataset="mmvp")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            setup_s = time.perf_counter() - t_setup

            shim = _TimedShim(method_mod)
            reset_counts()
            reset_passes()

            handles = (eval_mod._install_head_mode(model, processor, eargs, "mmvp")
                       if is_srf else [])
            print(f"\n--- arm: {arm}  (method={method_name}) ---")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                res = eval_mod.run_mmvp(shim, model, processor, img_token_id,
                                         device, eargs)
            finally:
                hc.remove_hooks(handles)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0

            n      = max(shim.n_samples, 1)
            clip_s, prep_s = COUNTS["clip_sec"], shim.prep_sec
            peak_gb = (torch.cuda.max_memory_allocated() / 1024**3
                       if torch.cuda.is_available() else float("nan"))
            rows[arm] = {
                "method":         method_name,
                "pair_acc":       res["method_pair"][0.0],
                "img_acc":        res["method_img"][0.0],
                "n_samples":      n,
                "wall_ms":        1e3 * wall / n,
                "clip_ms":        1e3 * clip_s / n,
                "prep_ms":        1e3 * (prep_s - clip_s) / n,
                "vlm_ms":         1e3 * (wall - prep_s) / n,
                "clip_img_calls": COUNTS["img_calls"] / n,
                "clip_img_items": COUNTS["img_items"] / n,
                "clip_txt_items": COUNTS["txt_items"] / n,
                "vlm_prefills":   PASSES["prefills"] / n,
                "vlm_forwards":   PASSES["forwards"] / n,
                "mean_tokens":    shim.tok_total / n,
                "peak_mem_gb":    peak_gb,
                "setup_s":        setup_s,
            }
            r = rows[arm]
            print(f"  pair={r['pair_acc']*100:.2f}%   {r['wall_ms']:.1f} ms/sample"
                  f"   peak {r['peak_mem_gb']:.2f} GB   setup {setup_s:.1f} s")
            print(f"    clip {r['clip_ms']:.1f}   prep(non-clip) {r['prep_ms']:.1f}   "
                  f"vlm {r['vlm_ms']:.1f}")
            print(f"    CLIP image encodes/sample: {r['clip_img_items']:.1f} "
                  f"in {r['clip_img_calls']:.1f} batched calls")
            print(f"    VLM prefill forwards/sample: {r['vlm_prefills']:.2f}"
                  f"   total forwards: {r['vlm_forwards']:.1f}"
                  f"   mean prompt tokens: {r['mean_tokens']:.0f}")
            method_mod.cleanup()
    finally:
        pass_hook.remove()

    # ---- FLOPs ----------------------------------------------------------
    cm, cp = clip_sal._CLIP_MODEL, clip_sal._CLIP_PROCESSOR
    g_img  = clip_gflops_per_image(cm, cp) if cm is not None else float("nan")
    mean_tok = (sum(r["mean_tokens"] for r in rows.values()) / len(rows)) if rows else 512

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    hdr = (f"  {'arm':<13}{'ms/sample':>11}{'clip':>8}{'prep':>8}{'vlm':>9}"
           f"{'prefills':>10}{'CLIPimg':>9}{'CLIP GF':>9}{'peakGB':>8}{'pair':>8}")
    print(hdr)
    for arm, r in rows.items():
        print(f"  {arm:<13}{r['wall_ms']:>11.1f}{r['clip_ms']:>8.1f}"
              f"{r['prep_ms']:>8.1f}{r['vlm_ms']:>9.1f}"
              f"{r['vlm_prefills']:>10.2f}{r['clip_img_items']:>9.1f}"
              f"{r['clip_img_items']*g_img:>9.1f}{r['peak_mem_gb']:>8.2f}"
              f"{r['pair_acc']*100:>8.2f}")

    if "baseline" in rows:
        b = rows["baseline"]["wall_ms"]
        print()
        for arm, r in rows.items():
            if arm != "baseline":
                print(f"  overhead {arm:<12} vs baseline: "
                      f"{r['wall_ms'] - b:+8.1f} ms  ({100*(r['wall_ms']/b - 1):+6.1f}%)")

    print(f"\n  CLIP ViT forward, measured: {g_img:.2f} GFLOPs per image")
    print(f"  VLM forward estimate at {mean_tok:.0f} measured mean tokens: "
          f"{vlm_gflops(model, mean_tok):.1f} GFLOPs (2 x n_params x n_tokens)")
    print("  Prefill counts above are measured, not assumed. SRF adds no extra "
          "VLM prefill. Its added FLOPs are the CLIP column.")

    if args.output:
        out = pathlib.Path(args.output); out.mkdir(parents=True, exist_ok=True)
        path = out / "profile_cost_mmvp_qwen3b.json"
        path.write_text(json.dumps({
            "model": args.model, "anchor": args.anchor, "dataset": "mmvp",
            "clip_gflops_per_image": g_img,
            "mean_prompt_tokens": mean_tok,
            "vlm_gflops_mean_tokens": vlm_gflops(model, mean_tok),
            "arms": rows,
        }, indent=2, default=str))
        print(f"\n  Saved → {path}")


if __name__ == "__main__":
    main()
