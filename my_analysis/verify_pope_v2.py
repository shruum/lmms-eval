#!/usr/bin/env python3
"""
POPE verification for SRF-V2.

Runs three configurations through the same eval path:
  (A) Baseline   — SRF-V2 with both ideas disabled (value_beta=0, drift_scale=0)
                   → should reproduce 83.3% if eval harness is correct
  (B) V-Amp only — value_beta=0.5, drift_scale=0 (uniform alpha)
  (C) Drift-α only — value_beta=0, drift_scale=0.5 (no value scaling)
  (D) V2 full    — value_beta=0.5, drift_scale=0.5

n=150 (larger than original 100 for stability; ~15 min total).
"""
from __future__ import annotations

import os, random, sys, pathlib

os.environ.setdefault("HF_HOME", "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

ANALYSIS_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(ANALYSIS_DIR))

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datasets import load_dataset as hf_load

from srf import srf_v2
import qwen_attn_patch as patch

MODEL_ID    = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_TOKEN = "<|image_pad|>"
N_SAMPLES   = 150
SEED        = 42


def get_img_range(input_ids, img_token_id):
    ids = input_ids[0].tolist()
    s   = next(i for i, t in enumerate(ids) if t == img_token_id)
    e   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return s, e


def load_pope(n, seed=SEED):
    ds   = hf_load("lmms-lab/POPE", split="test")
    rows = [r for r in ds if str(r.get("category", r.get("type", ""))).lower() == "adversarial"]
    rng  = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def run_pope_config(model, processor, img_token_id, device, rows,
                    value_beta: float, drift_scale: float, tag: str) -> float:
    """Run POPE with given V2 params. Model already patched."""
    print(f"\n--- {tag}  (value_beta={value_beta}, drift_scale={drift_scale}) ---")

    # Update BIAS and re-calibrate/re-hook
    srf_v2.BIAS["value_beta"]  = value_beta
    srf_v2.BIAS["drift_scale"] = drift_scale
    srf_v2.BIAS["srf_apply_phase"] = "both"
    srf_v2.BIAS["boost_alpha"]     = 4.0
    srf_v2.BIAS["layer_end"]       = 15
    srf_v2.BIAS["background_eps"]  = 0.2
    srf_v2._noun_mode = "pope"

    # Re-calibrate drift alphas
    print("  Calibrating…")
    calib_inputs, img_ranges = srf_v2._build_calib_inputs("pope", n=20, seed=0)
    srf_v2._calibrate(calib_inputs, img_ranges)
    del calib_inputs, img_ranges
    torch.cuda.empty_cache()

    # Re-register V-Amp hooks
    srf_v2._remove_value_hooks()
    if value_beta > 0.0:
        srf_v2._register_value_hooks()
    patch._STATE["srf_apply_phase"] = "both"
    patch._STATE["srf_background_eps"] = 0.2

    correct = 0
    for i, r in enumerate(rows):
        image = r["image"].convert("RGB")
        q     = str(r["question"]).strip() + "\nAnswer with Yes or No only."
        gt    = "yes" if str(r.get("answer", "")).strip().lower() == "yes" else "no"

        msgs  = [{"role": "user", "content": [{"type": "image", "image": image},
                                               {"type": "text",  "text":  q}]}]
        text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _= process_vision_info(msgs)
        inp   = processor(text=[text], images=vis, return_tensors="pt", padding=True).to(device)
        s, e  = get_img_range(inp["input_ids"], img_token_id)

        srf_v2.prepare_sample(inp, s, e, image, q, model, processor)
        with torch.inference_mode():
            out_ids = model.generate(**inp, max_new_tokens=10, do_sample=False)
        srf_v2.cleanup()

        pred_str = processor.decode(out_ids[0, inp["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip().lower()
        pred = "yes" if pred_str.startswith("yes") else "no"
        if pred == gt:
            correct += 1

    acc = correct / len(rows)
    print(f"  POPE accuracy ({tag}): {acc:.4f}  ({correct}/{len(rows)})")
    return acc


def main():
    print(f"Loading {MODEL_ID}…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager",
    ).eval()
    processor    = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=512 * 28 * 28)
    img_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    device       = next(model.parameters()).device

    # Initial setup (hooks + patch registered once)
    srf_v2.setup(model, processor, calib_dataset="pope")

    rows = load_pope(N_SAMPLES)
    print(f"Loaded {len(rows)} POPE adversarial samples")

    results = {}
    # (A) Baseline through V2 harness — should give ~83.3%
    results["A_baseline"]  = run_pope_config(model, processor, img_token_id, device, rows,
                                              value_beta=0.0, drift_scale=0.0, tag="A: baseline (V2 disabled)")
    # (B) V-Amp only
    results["B_vamp_only"] = run_pope_config(model, processor, img_token_id, device, rows,
                                              value_beta=0.5, drift_scale=0.0, tag="B: V-Amp only")
    # (C) Drift-α only
    results["C_drift_only"]= run_pope_config(model, processor, img_token_id, device, rows,
                                              value_beta=0.0, drift_scale=0.5, tag="C: Drift-α only")
    # (D) Both (SRF-V2 full)
    results["D_v2_full"]   = run_pope_config(model, processor, img_token_id, device, rows,
                                              value_beta=0.5, drift_scale=0.5, tag="D: V2 full")

    print("\n" + "="*55)
    print(f"POPE VERIFICATION  (adversarial, n={N_SAMPLES})")
    print("="*55)
    print(f"  A. Baseline (V2 disabled):  {results['A_baseline']:.4f}")
    print(f"  B. V-Amp only (β=0.5):      {results['B_vamp_only']:.4f}  Δ={(results['B_vamp_only']-results['A_baseline']):+.4f}")
    print(f"  C. Drift-α only (ds=0.5):   {results['C_drift_only']:.4f}  Δ={(results['C_drift_only']-results['A_baseline']):+.4f}")
    print(f"  D. V2 full:                 {results['D_v2_full']:.4f}  Δ={(results['D_v2_full']-results['A_baseline']):+.4f}")
    print(f"\n  SRF-V1 reference (n=100):   0.8333")


if __name__ == "__main__":
    main()
