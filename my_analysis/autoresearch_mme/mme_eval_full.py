#!/usr/bin/env python3
"""
MME full eval harness — DO NOT MODIFY.

Full 2374-question MME. Use only for final validation after autoresearch finds a strong config.
Reports: question accuracy, pair accuracy, perception score, cognition score, total MME score.

The one line Claude extracts from run.log:
    MME FULL: srf=1842  base=1750  perc_srf=1542  cogn_srf=300  pair_srf=0.7421

Usage:
    cd /volumes2/mllm/lmms-eval
    conda run -n mllm python my_analysis/autoresearch_mme/mme_eval_full.py > run_full.log 2>&1
    grep "MME FULL:" run_full.log
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import defaultdict

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

_MME_COGNITION = {"code_reasoning", "numerical_calculation", "text_translation",
                  "commonsense_reasoning"}

import torch
from datasets import load_dataset as hf_load
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import qwen_attn_patch as patch
import srf


def get_img_range(input_ids: torch.Tensor, img_token_id: int) -> tuple[int, int]:
    ids   = input_ids[0].tolist()
    start = next(i for i, t in enumerate(ids) if t == img_token_id)
    end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    return start, end


def run() -> dict:
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

    srf.setup(model, processor)

    print("  Loading MME (full, 2374 samples)…")
    ds      = hf_load("lmms-lab/MME", split="test")
    samples = list(ds)
    n_total = len(samples)
    print(f"  → {n_total} samples")

    correct_base = 0
    correct_srf  = 0
    cat_stats    = defaultdict(lambda: {"base": 0, "srf": 0, "total": 0})
    pair_stats   = defaultdict(lambda: {"base": [], "srf": []})

    for i, r in enumerate(samples):
        image   = r["image"].convert("RGB")
        q       = str(r["question"]).strip()
        gt      = str(r["answer"]).strip().lower()
        cat     = str(r.get("category", "unknown")).strip()
        pair_id = str(r.get("question_id", f"{cat}_{i}"))

        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image, "max_pixels": 448 * 448},
            {"type": "text",  "text":  q + "\nAnswer with Yes or No only."},
        ]}]
        text      = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_in, _ = process_vision_info(msgs)
        inputs    = processor(text=[text], images=img_in, return_tensors="pt", padding=True
                              ).to(device)

        img_start, img_end = get_img_range(inputs["input_ids"], img_token_id)

        # Baseline
        patch._STATE["method"] = "baseline"
        with torch.inference_mode():
            logits_base = model(**inputs).logits[:, -1, :].float()
        pred_base = "yes" if logits_base[0, yes_id] >= logits_base[0, no_id] else "no"
        ok_base   = (pred_base == gt)
        if ok_base:
            correct_base += 1
        cat_stats[cat]["base"]  += int(ok_base)
        cat_stats[cat]["total"] += 1
        pair_stats[pair_id]["base"].append(ok_base)

        # SRF
        srf.prepare_sample(inputs, img_start, img_end, image, q, model, processor)
        with torch.inference_mode():
            logits_srf = model(**inputs).logits[:, -1, :].float()
        pred_srf = "yes" if logits_srf[0, yes_id] >= logits_srf[0, no_id] else "no"
        ok_srf   = (pred_srf == gt)
        if ok_srf:
            correct_srf += 1
        srf.cleanup()
        cat_stats[cat]["srf"]  += int(ok_srf)
        pair_stats[pair_id]["srf"].append(ok_srf)

        if (i + 1) % 200 == 0 or (i + 1) == n_total:
            print(f"  [{i+1:4d}/{n_total}] base={correct_base/(i+1):.4f}  srf={correct_srf/(i+1):.4f}")

    # Pair accuracy
    pairs_base = [v for v in pair_stats.values() if len(v["base"]) == 2]
    pairs_srf  = [v for v in pair_stats.values() if len(v["srf"])  == 2]
    pair_acc_base = sum(1 for p in pairs_base if all(p["base"])) / max(len(pairs_base), 1)
    pair_acc_srf  = sum(1 for p in pairs_srf  if all(p["srf"]))  / max(len(pairs_srf),  1)

    # Perception / cognition sub-scores
    perc_base = perc_srf = cogn_base = cogn_srf = 0
    for cat, s in cat_stats.items():
        if cat in _MME_COGNITION:
            cogn_base += s["base"]
            cogn_srf  += s["srf"]
        else:
            perc_base += s["base"]
            perc_srf  += s["srf"]

    print(f"\n{'='*60}")
    print(f"Category breakdown:")
    for cat in sorted(cat_stats):
        s = cat_stats[cat]
        marker = "[C]" if cat in _MME_COGNITION else "[P]"
        print(f"  {marker} {cat:30s}  base={s['base']:3d}/{s['total']}  "
              f"srf={s['srf']:3d}/{s['total']}  "
              f"delta={s['srf']-s['base']:+d}")
    print(f"\nPair accuracy:  base={pair_acc_base:.4f}  srf={pair_acc_srf:.4f}")
    print(f"Perception:     base={perc_base}  srf={perc_srf}  delta={perc_srf-perc_base:+d}")
    print(f"Cognition:      base={cogn_base}  srf={cogn_srf}  delta={cogn_srf-cogn_base:+d}")
    print(f"Total:          base={correct_base}  srf={correct_srf}  delta={correct_srf-correct_base:+d}")
    print(f"\nMME FULL: srf={correct_srf}  base={correct_base}  "
          f"perc_srf={perc_srf}  cogn_srf={cogn_srf}  "
          f"pair_srf={pair_acc_srf:.4f}")

    out = {
        "base": correct_base, "srf": correct_srf, "n": n_total,
        "perc_base": perc_base, "perc_srf": perc_srf,
        "cogn_base": cogn_base, "cogn_srf": cogn_srf,
        "pair_base": pair_acc_base, "pair_srf": pair_acc_srf,
        "cat_stats": {k: dict(v) for k, v in cat_stats.items()},
    }
    with open(SCRIPT_DIR / "last_run_full.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    run()
