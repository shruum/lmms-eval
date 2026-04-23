#!/usr/bin/env python3
"""
POPE fast eval harness — DO NOT MODIFY.

Adversarial split only, N_SAMPLES=50. ~2 min per run with CLIP.
Use this during the search loop for rapid iteration.

The one line Claude extracts from run.log:
    POPE accuracy: 0.8400  (adversarial n=50)

Usage:
    cd /volumes2/mllm/lmms-eval
    conda run -n mllm python my_analysis/autoresearch/pope_eval_fast.py > run.log 2>&1
    grep "POPE accuracy:" run.log
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

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

MODEL_ID       = "Qwen/Qwen2.5-VL-3B-Instruct"
SPLIT          = "adversarial"
N_SAMPLES      = 50
SEED           = 42
IMAGE_TOKEN    = "<|image_pad|>"
MAX_NEW_TOKENS = 10
GEN_KWARGS     = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

import torch
from datasets import load_dataset as hf_load
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import srf


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
        samples.append({"image": r["image"].convert("RGB"), "question": q, "ground_truth": gt})
    print(f"  → {len(samples)} samples loaded")
    return samples


def get_img_range(input_ids: torch.Tensor, img_token_id: int) -> tuple[int, int]:
    ids   = input_ids[0].tolist()
    start = next(i for i, t in enumerate(ids) if t == img_token_id)
    end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return start, end


def run() -> float:
    samples = load_pope(N_SAMPLES)

    print(f"\n  Loading {MODEL_ID}…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    ).eval()
    processor    = AutoProcessor.from_pretrained(MODEL_ID)
    img_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    srf.setup(model, processor)

    correct = 0
    results = []

    for i, s in enumerate(samples):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": s["image"]},
            {"type": "text",  "text":  s["question"]},
        ]}]
        text      = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_in, _ = process_vision_info(msgs)
        inputs    = processor(text=[text], images=img_in, return_tensors="pt", padding=True
                              ).to(next(model.parameters()).device)

        img_start, img_end = get_img_range(inputs["input_ids"], img_token_id)
        srf.prepare_sample(inputs, img_start, img_end, s["image"], s["question"], model, processor)

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **GEN_KWARGS)

        srf.cleanup()

        pred_str = processor.decode(
            out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()
        pred = "yes" if pred_str.startswith("yes") else "no"
        ok   = pred == s["ground_truth"]
        if ok:
            correct += 1
        results.append({"gt": s["ground_truth"], "pred": pred, "correct": ok})

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:2d}/{N_SAMPLES}] running acc: {correct/(i+1):.4f}")

    acc = correct / N_SAMPLES

    out_path = SCRIPT_DIR / "last_run.json"
    with open(out_path, "w") as f:
        json.dump({"accuracy": acc, "n": N_SAMPLES, "split": SPLIT, "samples": results}, f)

    print(f"\nPOPE accuracy: {acc:.4f}  ({SPLIT} n={N_SAMPLES})")
    return acc


if __name__ == "__main__":
    run()
