#!/usr/bin/env python3
"""
MMVP autoresearch eval harness — DO NOT MODIFY.

Loads MMVP/MMVP (all 150 pairs = 300 images), runs Qwen2.5-VL-7B-Instruct
with the SRF method defined in srf.py, and reports pair accuracy.

Pair accuracy = fraction of pairs where BOTH images answered correctly.
This is the primary metric because MMVP pairs test the same pattern with
opposite correct answers — a model relying on language prior would get 0%.

The two lines Claude extracts from run.log:
    MMVP pair accuracy:  X.XXXX  (150 pairs)
    MMVP image accuracy: X.XXXX  (300 images)

Usage:
    cd /volumes2/mllm/lmms-eval
    python my_analysis/autoresearch_mmvp/mmvp_eval.py > my_analysis/autoresearch_mmvp/run.log 2>&1
    grep "MMVP.*accuracy:" my_analysis/autoresearch_mmvp/run.log
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

os.environ.setdefault("HF_HOME", "/volumes2/hugging_face_cache")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"      # 7B needs both 2080Ti + 1080Ti
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

SCRIPT_DIR   = pathlib.Path(__file__).parent
ANALYSIS_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

MODEL_ID       = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_TOKEN    = "<|image_pad|>"
MAX_NEW_TOKENS = 16
GEN_KWARGS     = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

# Ground truth corrections (0-indexed from CSV, i.e. csv_1idx - 1)
GROUND_TRUTH_CORRECTIONS: dict[int, str] = {
    99:  "A",   # row 100: elephant tusks are long, not short
    279: "A",   # row 280: person is standing, not sitting
}

QUESTIONS_CSV = "/volumes2/hugging_face_cache/mmvp_questions/Questions.csv"

import torch
import pandas as pd
from datasets import load_dataset as hf_load
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import srf


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _build_csv_to_hf_map() -> dict[int, int]:
    """Map CSV 1-indexed row number → HF dataset index.

    The HF imagefolder loads filenames in lexicographic order
    (1.jpg, 10.jpg, 100.jpg, 101.jpg, … 2.jpg, 20.jpg, …) while the CSV
    uses numeric order (1, 2, 3, …, 300). This mapping corrects the mismatch.
    """
    lex_sorted = sorted(range(1, 301), key=str)
    return {csv_1idx: hf_idx for hf_idx, csv_1idx in enumerate(lex_sorted)}


def load_mmvp() -> list[dict]:
    print("  Loading MMVP Questions.csv…")
    df = pd.read_csv(QUESTIONS_CSV)   # columns: Index, Question, Options, Correct Answer

    print("  Loading MMVP/MMVP images from HuggingFace…")
    img_ds    = hf_load("MMVP/MMVP", split="train")   # 300 images, lex order
    csv_to_hf = _build_csv_to_hf_map()

    samples = []
    for csv_1idx in range(1, 301):
        row_idx = csv_1idx - 1
        row     = df.iloc[row_idx]

        # Parse options: "(a) Open (b) Closed" → "A. Open\nB. Closed"
        opt_matches = re.findall(r'\(([ab])\)\s*([^(]+)', str(row["Options"]), re.IGNORECASE)
        if not opt_matches:
            continue
        choices  = {m[0].upper(): m[1].strip() for m in opt_matches}
        opt_text = "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
        prompt   = (f"{row['Question']}\n{opt_text}\n"
                    "Answer with the option's letter from the given choices directly.")

        gt_raw  = str(row["Correct Answer"]).strip().strip("()").upper()   # "A" or "B"
        gt      = GROUND_TRUTH_CORRECTIONS.get(row_idx, gt_raw)

        hf_idx  = csv_to_hf[csv_1idx]
        image   = img_ds[hf_idx]["image"].convert("RGB")

        pair_id = (csv_1idx - 1) // 2   # 0-indexed pair (0..149)

        samples.append({
            "image":    image,
            "question": row["Question"],
            "prompt":   prompt,
            "gt":       gt,
            "pair_id":  pair_id,
            "csv_1idx": csv_1idx,
        })

    print(f"  → {len(samples)} samples, {len(samples) // 2} pairs")
    return samples


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_answer(text: str) -> str:
    """Extract A or B from model output."""
    text = text.strip()
    patterns = [
        r"^\s*\(?([AB])\)?\.?\s*$",
        r"^\s*\(?([AB])\)?[\.\s]",
        r"answer\s+is\s+\(?([AB])\)?",
        r"option\s+\(?([AB])\)?",
        r"\(?([AB])\)?(?:\s|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    if text and text[0].upper() in ("A", "B"):
        return text[0].upper()
    return ""


def get_img_range(input_ids: torch.Tensor, img_token_id: int) -> tuple[int, int]:
    ids   = input_ids[0].tolist()
    start = next(i for i, t in enumerate(ids) if t == img_token_id)
    end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return start, end


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def run() -> tuple[float, float]:
    samples = load_mmvp()

    print(f"\n  Loading {MODEL_ID}…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    processor    = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=512 * 28 * 28)
    img_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    srf.setup(model, processor)

    img_correct = 0
    pair_correct: dict[int, list[bool]] = {}   # pair_id → [correct_A, correct_B]
    results = []

    for i, s in enumerate(samples):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": s["image"]},
            {"type": "text",  "text":  s["prompt"]},
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

        raw  = processor.decode(out_ids[0, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        pred = _extract_answer(raw)
        ok   = (pred == s["gt"])

        img_correct += int(ok)
        pair_correct.setdefault(s["pair_id"], []).append(ok)

        results.append({
            "csv_1idx": s["csv_1idx"],
            "pair_id":  s["pair_id"],
            "gt":       s["gt"],
            "pred":     pred,
            "correct":  ok,
            "response": raw,
        })

        if (i + 1) % 30 == 0:
            n_pairs_done   = sum(1 for v in pair_correct.values() if len(v) == 2)
            n_pairs_ok     = sum(1 for v in pair_correct.values() if len(v) == 2 and all(v))
            pair_acc_so_far = n_pairs_ok / n_pairs_done if n_pairs_done else 0.0
            print(f"  [{i+1:3d}/300] img acc: {img_correct/(i+1):.4f}  pair acc: {pair_acc_so_far:.4f}")

    n_images    = len(samples)
    img_acc     = img_correct / n_images
    n_pairs     = sum(1 for v in pair_correct.values() if len(v) == 2)
    n_pairs_ok  = sum(1 for v in pair_correct.values() if len(v) == 2 and all(v))
    pair_acc    = n_pairs_ok / n_pairs if n_pairs else 0.0

    out_path = SCRIPT_DIR / "last_run.json"
    with open(out_path, "w") as f:
        json.dump({"pair_accuracy": pair_acc, "image_accuracy": img_acc,
                   "n_pairs": n_pairs, "n_images": n_images, "samples": results}, f, indent=2)

    print(f"\nMMVP pair accuracy:  {pair_acc:.4f}  ({n_pairs_ok}/{n_pairs} pairs)")
    print(f"MMVP image accuracy: {img_acc:.4f}  ({img_correct}/{n_images} images)")
    return pair_acc, img_acc


if __name__ == "__main__":
    run()
