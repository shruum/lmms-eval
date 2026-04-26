#!/usr/bin/env python3
"""
MME fast eval harness — DO NOT MODIFY.

Perception categories only, N_SAMPLES=200. ~3-4 min per run with CLIP.
Uses logit comparison (Yes vs No) — no generation, faster than pope_eval.

The one line Claude extracts from run.log:
    MME score: 168/200  baseline=160/200

Usage:
    cd /volumes2/mllm/lmms-eval
    conda run -n mllm python my_analysis/autoresearch_mme/mme_eval.py > run.log 2>&1
    grep "MME score:" run.log
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import sys

os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

SCRIPT_DIR   = pathlib.Path(__file__).parent
ANALYSIS_DIR = SCRIPT_DIR.parent
SRF_DIR      = ANALYSIS_DIR.parent / "srf"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(SRF_DIR / "saliency"))

MODEL_ID    = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_TOKEN = "<|image_pad|>"
N_SAMPLES   = 200
SEED        = 42

# MME cognition categories (the 4 non-perception ones)
_MME_COGNITION = {"code_reasoning", "numerical_calculation", "text_translation",
                  "commonsense_reasoning"}

import torch
from datasets import load_dataset as hf_load
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import qwen_attn_patch as patch
import srf


def load_mme_perception(n: int, seed: int = SEED) -> list[dict]:
    print(f"  Loading MME perception (n={n}, seed={seed})…", flush=True)
    ds = hf_load("lmms-lab/MME", split="test")
    ds_meta = ds.remove_columns(["image"])
    meta = [
        {"idx": i,
         "category": str(r.get("category", "")).strip().lower(),
         "question": str(r["question"]).strip(),
         "gt":       str(r["answer"]).strip().lower()}
        for i, r in enumerate(ds_meta)
        if str(r.get("category", "")).strip().lower() not in _MME_COGNITION
    ]
    rng = random.Random(seed)
    rng.shuffle(meta)
    meta = meta[:n]
    print(f"  Decoding {n} selected images…", flush=True)
    out = []
    for m in meta:
        r = ds[m["idx"]]
        out.append({
            "image":    r["image"].convert("RGB"),
            "question": m["question"],
            "gt":       m["gt"],
            "category": m["category"],
        })
    print(f"  → {len(out)} perception samples loaded", flush=True)
    return out


def get_img_range(input_ids: torch.Tensor, img_token_id: int) -> tuple[int, int]:
    ids   = input_ids[0].tolist()
    start = next(i for i, t in enumerate(ids) if t == img_token_id)
    end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return start, end


def run() -> tuple[int, int]:
    samples = load_mme_perception(N_SAMPLES)

    print(f"\n  Loading {MODEL_ID}…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    ).eval()
    processor    = AutoProcessor.from_pretrained(MODEL_ID)
    img_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    yes_id       = processor.tokenizer.convert_tokens_to_ids("Yes")
    no_id        = processor.tokenizer.convert_tokens_to_ids("No")
    device       = next(model.parameters()).device
    print(f"  img_token_id={img_token_id}  yes_id={yes_id}  no_id={no_id}")

    srf.setup(model, processor)

    # SRF-e contrastive beta: logits_final = logits_full + β*(logits_full - logits_noval)
    SRFE_BETA = 1.0

    correct_base  = 0
    correct_srf   = 0
    correct_e     = 0    # pure-E: baseline + β*(baseline - noval)
    correct_srfe  = 0    # SRF-E:  srf     + β*(srf     - noval)
    results       = []

    for i, s in enumerate(samples):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": s["image"], "max_pixels": 448 * 448},
            {"type": "text",  "text":  s["question"] + "\nAnswer with Yes or No only."},
        ]}]
        text     = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_in, _ = process_vision_info(msgs)
        inputs   = processor(text=[text], images=img_in, return_tensors="pt", padding=True
                             ).to(device)

        img_start, img_end = get_img_range(inputs["input_ids"], img_token_id)

        # ── Baseline ───────────────────────────────────────────────────────────
        patch._STATE["method"] = "baseline"
        with torch.inference_mode():
            logits_base = model(**inputs).logits[:, -1, :].float()
        pred_base = "yes" if logits_base[0, yes_id] >= logits_base[0, no_id] else "no"
        ok_base   = (pred_base == s["gt"])
        if ok_base:
            correct_base += 1

        # ── SRF ───────────────────────────────────────────────────────────────
        srf.prepare_sample(inputs, img_start, img_end, s["image"], s["question"], model, processor)
        with torch.inference_mode():
            logits_srf = model(**inputs).logits[:, -1, :].float()
        pred_srf = "yes" if logits_srf[0, yes_id] >= logits_srf[0, no_id] else "no"
        ok_srf   = (pred_srf == s["gt"])
        if ok_srf:
            correct_srf += 1
        srf.cleanup()

        # ── No-visual pass (shared by both SRF-e variants) ────────────────────
        # Zero pixel_values → ViT sees black image → language-prior-only logits.
        inp_noval = {k: (torch.zeros_like(v) if k == "pixel_values" else v)
                     for k, v in inputs.items()}
        patch._STATE["method"] = "baseline"
        with torch.inference_mode():
            logits_noval = model(**inp_noval).logits[:, -1, :].float()

        # ── Pure-E: baseline + β*(baseline − noval) ───────────────────────────
        logits_e     = logits_base + SRFE_BETA * (logits_base - logits_noval)
        pred_e       = "yes" if logits_e[0, yes_id] >= logits_e[0, no_id] else "no"
        ok_e         = (pred_e == s["gt"])
        if ok_e:
            correct_e += 1

        # ── SRF-E: srf + β*(srf − noval) ─────────────────────────────────────
        logits_srfe  = logits_srf + SRFE_BETA * (logits_srf - logits_noval)
        pred_srfe    = "yes" if logits_srfe[0, yes_id] >= logits_srfe[0, no_id] else "no"
        ok_srfe      = (pred_srfe == s["gt"])
        if ok_srfe:
            correct_srfe += 1

        results.append({
            "gt": s["gt"], "cat": s["category"],
            "base": pred_base, "srf": pred_srf,
            "pure_e": pred_e, "srfe": pred_srfe,
            "ok_base": ok_base, "ok_srf": ok_srf,
            "ok_e": ok_e, "ok_srfe": ok_srfe,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{N_SAMPLES}] "
                  f"base={correct_base/(i+1):.4f}  "
                  f"srf={correct_srf/(i+1):.4f}  "
                  f"pure-E={correct_e/(i+1):.4f}  "
                  f"srf-E={correct_srfe/(i+1):.4f}", flush=True)

    out_path = SCRIPT_DIR / "last_run.json"
    with open(out_path, "w") as f:
        json.dump({
            "base": correct_base, "srf": correct_srf,
            "pure_e": correct_e, "srfe": correct_srfe,
            "n": N_SAMPLES, "srfe_beta": SRFE_BETA,
            "base_acc":  correct_base  / N_SAMPLES,
            "srf_acc":   correct_srf   / N_SAMPLES,
            "pure_e_acc": correct_e    / N_SAMPLES,
            "srfe_acc":  correct_srfe  / N_SAMPLES,
            "samples":   results,
        }, f)

    print(f"\n{'='*50}")
    print(f"MME score: {correct_srf}/{N_SAMPLES}  baseline={correct_base}/{N_SAMPLES}  "
          f"pure-E={correct_e}/{N_SAMPLES}  srf-E={correct_srfe}/{N_SAMPLES}")
    print(f"Deltas: srf={correct_srf-correct_base:+d}  "
          f"pure-E={correct_e-correct_base:+d}  "
          f"srf-E={correct_srfe-correct_base:+d}")
    return correct_srf, correct_base


if __name__ == "__main__":
    run()
