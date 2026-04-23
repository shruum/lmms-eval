#!/usr/bin/env python3
"""
POPE autoresearch eval harness (all three splits) — DO NOT MODIFY.

Loads POPE adversarial + popular + random splits (N_SAMPLES=100 each),
runs Qwen2.5-VL-3B-Instruct with the SRF method defined in srf.py, and
reports per-split accuracy and the average.

The one line Claude extracts from run.log:
    POPE average: 0.8333  (adversarial=0.8200 popular=0.8000 random=0.8800 n=100 each)

Usage:
    cd /volumes2/mllm/lmms-eval
    conda run -n mllm python my_analysis/autoresearch/pope_eval_all.py > run.log 2>&1
    grep "POPE average:" run.log
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
SPLITS         = ["adversarial", "popular", "random"]
N_SAMPLES      = 100        # per split → 300 total
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

def load_pope_split(split: str, n: int, seed: int = SEED) -> list[dict]:
    print(f"  Loading POPE ({split}, n={n}, seed={seed})…")
    ds   = hf_load("lmms-lab/POPE", split="test")
    rows = [
        r for r in ds
        if str(r.get("category", r.get("type", ""))).strip().lower() == split
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
            "split":        split,
        })
    print(f"  → {len(samples)} samples loaded")
    return samples


# ── image-token range helper ──────────────────────────────────────────────

def get_img_range(input_ids: torch.Tensor, img_token_id: int) -> tuple[int, int]:
    ids   = input_ids[0].tolist()
    start = next(i for i, t in enumerate(ids) if t == img_token_id)
    end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return start, end


# ── eval one split ────────────────────────────────────────────────────────

def eval_split(split: str, model, processor, img_token_id: int) -> tuple[float, list]:
    samples = load_pope_split(split, N_SAMPLES)
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

        results.append({"split": split, "gt": gt, "pred": pred,
                        "correct": ok, "response": pred_str})

        if (i + 1) % 20 == 0:
            print(f"  [{split} {i+1:3d}/{N_SAMPLES}] running acc: {correct/(i+1):.4f}")

    acc = correct / N_SAMPLES
    print(f"  {split}: {acc:.4f}")
    return acc, results


# ── main eval ─────────────────────────────────────────────────────────────

def run() -> float:
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

    all_results = {}
    accs        = {}

    for split in SPLITS:
        print(f"\n── {split} ──────────────────────────")
        acc, results = eval_split(split, model, processor, img_token_id)
        accs[split]        = acc
        all_results[split] = results

    avg = sum(accs.values()) / len(accs)

    # ── write per-sample JSON ─────────────────────────────────────────────
    out_path = SCRIPT_DIR / "last_run.json"
    with open(out_path, "w") as f:
        json.dump({
            "average":  avg,
            "per_split": accs,
            "n_per_split": N_SAMPLES,
            "results":  all_results,
        }, f)

    # ── final report — Claude greps this line ─────────────────────────────
    print(f"\nPOPE average: {avg:.4f}  "
          f"(adversarial={accs['adversarial']:.4f} "
          f"popular={accs['popular']:.4f} "
          f"random={accs['random']:.4f} "
          f"n={N_SAMPLES} each)")
    return avg


if __name__ == "__main__":
    run()
