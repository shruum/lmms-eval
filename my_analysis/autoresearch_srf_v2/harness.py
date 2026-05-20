#!/usr/bin/env python3
"""
POPE adversarial harness for SRF-v2 autoresearch — DO NOT MODIFY.

Evaluates on N_SAMPLES POPE adversarial samples using single-token logit
comparison (yes_id vs no_id) instead of model.generate(). This is faster
and gives per-sample confidence values for analysis.

The one line sweep.py extracts:
    RESULT  acc=0.8600  yes=102  no=98  n=200  exp=<name>

Usage:
    cd /volumes2/mllm/lmms-eval
    conda run -n mllm python my_analysis/autoresearch_srf_v2/harness.py \
        --exp_name baseline 2>&1 | tail -5
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

os.environ.setdefault("HF_HOME", "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

_HARNESS_DIR = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _HARNESS_DIR.parent
sys.path.insert(0, str(_HARNESS_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

MODEL_ID    = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_TOKEN = "<|image_pad|>"
N_SAMPLES   = 200
SEED        = 42

import torch
from datasets import load_dataset as hf_load
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import srf_v2 as srf


def load_pope(n: int, seed: int = SEED) -> list[dict]:
    print(f"  [harness] Loading POPE adversarial n={n} seed={seed}…")
    ds   = hf_load("lmms-lab/POPE", split="test")
    rows = [r for r in ds
            if str(r.get("category", r.get("type", ""))).strip().lower() == "adversarial"]
    rng  = random.Random(seed)
    rng.shuffle(rows)
    return [
        {
            "image":    r["image"].convert("RGB"),
            "question": str(r["question"]).strip() + "\nAnswer with Yes or No only.",
            "gt":       "yes" if str(r.get("answer", "")).strip().lower() == "yes" else "no",
        }
        for r in rows[:n]
    ]


def get_img_range(input_ids: torch.Tensor, img_token_id: int):
    ids   = input_ids[0].tolist()
    start = next(i for i, t in enumerate(ids) if t == img_token_id)
    end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return start, end


def run(exp_name: str = "exp") -> dict:
    samples = load_pope(N_SAMPLES)

    print(f"  [harness] Loading {MODEL_ID}…")
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

    srf.setup(model, processor)

    correct = 0
    n_yes_pred = 0
    results = []

    for i, s in enumerate(samples):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": s["image"]},
            {"type": "text",  "text":  s["question"]},
        ]}]
        text      = processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
        vis, _    = process_vision_info(msgs)
        inputs    = processor(text=[text], images=vis, return_tensors="pt",
                              padding=True).to(device)
        img_start, img_end = get_img_range(inputs["input_ids"], img_token_id)

        srf.prepare_sample(inputs, img_start, img_end, s["image"],
                           s["question"], model, processor)

        # Single-token logit eval: compare Yes vs No logit directly
        # This is faster than generate() and gives cleaner confidence values.
        logits = srf.get_logits(model, inputs)   # (1, vocab)

        srf.cleanup()

        pred = "yes" if logits[0, yes_id] >= logits[0, no_id] else "no"
        ok   = (pred == s["gt"])
        if ok:
            correct += 1
        if pred == "yes":
            n_yes_pred += 1
        results.append({
            "gt": s["gt"], "pred": pred, "correct": ok,
            "yes_logit": float(logits[0, yes_id]),
            "no_logit":  float(logits[0, no_id]),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:3d}/{N_SAMPLES}] acc={correct/(i+1):.4f}  "
                  f"yes_rate={n_yes_pred/(i+1):.3f}")

    acc = correct / N_SAMPLES
    out = {
        "exp": exp_name, "acc": acc, "n": N_SAMPLES,
        "n_yes": n_yes_pred, "n_no": N_SAMPLES - n_yes_pred,
        "config": srf.CONFIG.copy(),
    }

    out_dir  = _HARNESS_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{exp_name}.json"
    with open(out_path, "w") as f:
        json.dump({**out, "samples": results}, f, indent=2)

    print(f"\nRESULT  acc={acc:.4f}  yes={n_yes_pred}  "
          f"no={N_SAMPLES - n_yes_pred}  n={N_SAMPLES}  exp={exp_name}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="exp")
    args = p.parse_args()
    run(args.exp_name)
