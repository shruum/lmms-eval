#!/usr/bin/env python3
"""
POPE autoresearch eval harness — DO NOT MODIFY.

Loads POPE adversarial split (N_SAMPLES=100), runs Qwen2.5-VL-3B-Instruct
with the SRF method defined in srf.py, and reports accuracy.

The one line Claude extracts from run.log:
    POPE accuracy: 0.8333  (adversarial n=100)

Usage:
    cd /volumes2/mllm/lmms-eval
    python my_analysis/autoresearch/pope_eval.py > run.log 2>&1
    grep "POPE accuracy:" run.log
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import sys

# ── env ───────────────────────────────────────────────────────────────────
os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ── paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR   = pathlib.Path(__file__).parent          # my_analysis/autoresearch/
ANALYSIS_DIR = SCRIPT_DIR.parent                      # my_analysis/

sys.path.insert(0, str(SCRIPT_DIR))    # for srf
sys.path.insert(0, str(ANALYSIS_DIR))  # for qwen_attn_patch, clip_salience, etc.

# ── fixed constants ───────────────────────────────────────────────────────
MODEL_ID       = "Qwen/Qwen2.5-VL-3B-Instruct"
SPLIT          = "adversarial"
N_SAMPLES      = 100
SEED           = 42
IMAGE_TOKEN    = "<|image_pad|>"
MAX_NEW_TOKENS = 10
GEN_KWARGS     = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

# ── imports ───────────────────────────────────────────────────────────────
import torch
from datasets import load_dataset as hf_load
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import srf  # agent-modifiable method — must expose setup / prepare_sample / cleanup


# ── data loader ───────────────────────────────────────────────────────────

def load_pope(n: int, seed: int = SEED) -> list[dict]:
    print(f"  Loading POPE ({SPLIT}, n={n}, seed={seed})…")
    ds   = hf_load("lmms-lab/POPE", split="test")
    rows = [
        r for r in ds
        if str(r.get("category", r.get("type", ""))).strip().lower() == SPLIT
    ]
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]

    samples = []
    for r in rows:
        gt = "yes" if str(r.get("answer", "")).strip().lower() == "yes" else "no"
        q  = str(r.get("question", "")).strip() + "\nAnswer with Yes or No only."
        samples.append({
            "image":        r["image"].convert("RGB"),
            "question":     q,
            "ground_truth": gt,
        })
    print(f"  → {len(samples)} samples loaded")
    return samples


# ── image-token range helper ──────────────────────────────────────────────

def get_img_range(input_ids: torch.Tensor, img_token_id: int) -> tuple[int, int]:
    ids   = input_ids[0].tolist()
    start = next(i for i, t in enumerate(ids) if t == img_token_id)
    end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return start, end


# ── main eval ─────────────────────────────────────────────────────────────

def run() -> float:
    samples = load_pope(N_SAMPLES)

    print(f"\n  Loading {MODEL_ID}…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    processor    = AutoProcessor.from_pretrained(MODEL_ID)
    img_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    # ── one-time SRF setup (calibration, hook registration) ───────────────
    srf.setup(model, processor)

    correct = 0
    results = []

    for i, s in enumerate(samples):
        image    = s["image"]
        question = s["question"]
        gt       = s["ground_truth"]

        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  question},
        ]}]
        text      = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_in, _ = process_vision_info(msgs)
        inputs    = processor(
            text=[text], images=img_in, return_tensors="pt", padding=True
        ).to(next(model.parameters()).device)

        img_start, img_end = get_img_range(inputs["input_ids"], img_token_id)

        # ── per-sample SRF setup ──────────────────────────────────────────
        srf.prepare_sample(inputs, img_start, img_end, image, question, model, processor)

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **GEN_KWARGS)

        srf.cleanup()

        pred_str = processor.decode(
            out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()
        pred = "yes" if pred_str.startswith("yes") else "no"
        ok   = pred == gt

        if ok:
            correct += 1

        results.append({"gt": gt, "pred": pred, "correct": ok, "response": pred_str})

        if (i + 1) % 20 == 0:
            running = correct / (i + 1)
            print(f"  [{i+1:3d}/{N_SAMPLES}] running acc: {running:.4f}")

    acc = correct / N_SAMPLES

    # ── write per-sample JSON ─────────────────────────────────────────────
    out_path = SCRIPT_DIR / "last_run.json"
    with open(out_path, "w") as f:
        json.dump({"accuracy": acc, "n": N_SAMPLES, "split": SPLIT, "samples": results}, f)

    # ── final report — Claude greps this line ─────────────────────────────
    print(f"\nPOPE accuracy: {acc:.4f}  ({SPLIT} n={N_SAMPLES})")
    return acc


if __name__ == "__main__":
    run()
