#!/usr/bin/env python3
"""
VLM Bias autoresearch eval harness — DO NOT MODIFY.

Loads anvo25/vlms-are-biased (all 7 categories), n=15 per category (~105 total),
runs Qwen2.5-VL-3B-Instruct with the SRF method defined in srf.py, and reports accuracy.

The one line Claude extracts from run.log:
    VLM Bias accuracy: 0.1714  (all categories n=105)

Usage:
    cd /volumes2/mllm/lmms-eval
    python my_analysis/autoresearch_vlmbias/vlmbias_eval.py > run.log 2>&1
    grep "VLM Bias accuracy:" run.log
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import re
import sys

os.environ.setdefault("HF_HOME", "/volumes2/hugging_face_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

SCRIPT_DIR   = pathlib.Path(__file__).parent
ANALYSIS_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

MODEL_ID       = "Qwen/Qwen2.5-VL-3B-Instruct"
N_PER_CATEGORY = 15
SEED           = 42
IMAGE_TOKEN    = "<|image_pad|>"
MAX_NEW_TOKENS = 20
GEN_KWARGS     = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

CATEGORIES = [
    "Animals", "Chess Pieces", "Flags",
    "Game Boards", "Logos", "Optical Illusion", "Patterned Grid",
]

import torch
from datasets import load_dataset as hf_load
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import srf


def extract_answer(text: str) -> str:
    """Extract answer from model output — handles {X} format or raw."""
    m = re.search(r'\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()
    return text.strip().split()[0] if text.strip() else ""


def normalise(s: str) -> str:
    return s.strip().lower().lstrip("{").rstrip("}")


def load_vlmbias(n_per_cat: int, seed: int = SEED) -> list[dict]:
    print(f"  Loading VLM Bias (all categories, n={n_per_cat}/cat, seed={seed})…")
    ds = hf_load("anvo25/vlms-are-biased", split="main")

    from collections import defaultdict
    by_cat: dict = defaultdict(list)
    for r in ds:
        by_cat[r["topic"]].append(r)

    samples = []
    rng = random.Random(seed)
    for cat in CATEGORIES:
        rows = by_cat.get(cat, [])
        rng.shuffle(rows)
        for r in rows[:n_per_cat]:
            samples.append({
                "image":    r["image"].convert("RGB"),
                "question": r["prompt"],
                "gt":       normalise(str(r["ground_truth"])),
                "category": cat,
            })

    rng.shuffle(samples)
    print(f"  → {len(samples)} samples across {len(CATEGORIES)} categories")
    return samples


def get_img_range(input_ids: torch.Tensor, img_token_id: int) -> tuple[int, int]:
    ids   = input_ids[0].tolist()
    start = next(i for i, t in enumerate(ids) if t == img_token_id)
    end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return start, end


def run() -> float:
    samples = load_vlmbias(N_PER_CATEGORY)

    print(f"\n  Loading {MODEL_ID}…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    # max_pixels caps image tokens to avoid OOM on 11 GiB GPUs (VLM Bias images are
    # higher-res than POPE; 512*28*28 ≈ 401K pixels is the same cap used in run_h100_qwen.sh)
    processor    = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=512 * 28 * 28)
    img_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    srf.setup(model, processor)

    correct   = 0
    results   = []
    cat_stats: dict = {c: {"correct": 0, "total": 0} for c in CATEGORIES}

    for i, s in enumerate(samples):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": s["image"]},
            {"type": "text",  "text":  s["question"]},
        ]}]
        text      = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_in, _ = process_vision_info(msgs)
        inputs    = processor(
            text=[text], images=img_in, return_tensors="pt", padding=True
        ).to(next(model.parameters()).device)

        img_start, img_end = get_img_range(inputs["input_ids"], img_token_id)

        srf.prepare_sample(inputs, img_start, img_end,
                           s["image"], s["question"], model, processor)

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **GEN_KWARGS)

        srf.cleanup()

        raw  = processor.decode(
            out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        pred = normalise(extract_answer(raw))
        ok   = pred == s["gt"]

        if ok:
            correct += 1
        cat_stats[s["category"]]["correct"] += int(ok)
        cat_stats[s["category"]]["total"]   += 1

        results.append({"gt": s["gt"], "pred": pred, "correct": ok,
                        "category": s["category"], "response": raw})

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(samples)}] running acc: {correct/(i+1):.4f}")

    n_total = len(samples)
    acc     = correct / n_total

    print("\n  Per-category:")
    for cat in CATEGORIES:
        st = cat_stats[cat]
        cat_acc = st["correct"] / st["total"] if st["total"] else 0.0
        print(f"    {cat:20s}  {cat_acc:.4f}  ({st['correct']}/{st['total']})")

    out_path = SCRIPT_DIR / "last_run.json"
    with open(out_path, "w") as f:
        json.dump({"accuracy": acc, "n": n_total,
                   "per_category": cat_stats, "samples": results}, f, indent=2)

    print(f"\nVLM Bias accuracy: {acc:.4f}  (all categories n={n_total})")
    return acc


if __name__ == "__main__":
    run()
