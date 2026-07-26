#!/usr/bin/env python3
"""
Find POPE samples where baseline performs poorly.

Usage:
    python srf/find_hard_pope_samples.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --pope_splits adversarial \
        --n_samples 100 \
        --output srf/hard_samples_pope.json

Output: JSON file with sample indices where baseline gets wrong answers.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

_SRF_DIR = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import config as CFG
import os
os.environ.setdefault("HF_HOME", CFG.HF_HOME)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import os

# Import eval utilities
from eval import (
    get_token_id, format_qwen_msgs, cleanup_qwen_temp_images,
    apply_chat_template, get_img_range, decode_first_token
)

from qwen_vl_utils import process_vision_info
from datasets import load_dataset as hf_load

# Global for patch module
patch = None


def load_model(model_id: str):
    """Load model and set patch module."""
    global patch

    print(f"Loading {model_id}…")

    # Detect model type from config and model_id
    from transformers import AutoConfig, AutoModelForCausalLM, LlavaForConditionalGeneration

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = config.model_type if hasattr(config, 'model_type') else None

    # Determine model family from model_id for more specific routing
    is_qwen_vl_chat = "Qwen-VL-Chat" in model_id or "Qwen-VL" in model_id
    is_qwen2_vl = "Qwen2" in model_id or "qwen2" in model_id.lower()

    print(f"  Detected model type: {model_type}")
    print(f"  Model family: {'Qwen-VL-Chat (v1)' if is_qwen_vl_chat else 'Qwen2-VL' if is_qwen2_vl else 'Unknown'}")

    # Load appropriate patch module and model class
    if model_type == "llava":
        import llava_attn_patch as llava_patch
        patch = llava_patch
        print(f"  Using LLaVA attention patch")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval()
        model.cuda()
    elif is_qwen_vl_chat:
        import qwen_v1_attn_patch as qwen_v1_patch
        patch = qwen_v1_patch
        print(f"  Using Qwen-VL-Chat (v1) attention patch")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True
        ).eval()
        model.cuda()
    else:
        # Default to Qwen2.5-VL (includes Qwen2-VL and Qwen2.5-VL)
        import qwen_attn_patch as qwen_patch
        patch = qwen_patch
        print(f"  Using Qwen2-VL attention patch")
        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError("Qwen2.5-VL requires transformers >= 4.40.0. Please upgrade: pip install transformers --upgrade")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval()
        model.cuda()

    # For LLaVA, use slow tokenizer to avoid compatibility issues
    from transformers import AutoProcessor, LlavaProcessor, CLIPImageProcessor

    use_fast = False if model_type == "llava" else True
    if model_type == "llava":
        from transformers import AutoTokenizer, LlavaProcessor
        from transformers import CLIPImageProcessor
        # Load slow tokenizer to avoid compatibility issues with transformers 4.39.3
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        # Load image processor
        image_processor = CLIPImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        # Manually construct processor
        processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    else:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def parse_args():
    p = argparse.ArgumentParser(description="Find hard POPE samples (baseline errors)")
    p.add_argument("--model", default=CFG.DEFAULT_MODEL, help="Model to use")
    p.add_argument("--pope_splits", nargs="+", default=CFG.POPE_SPLITS,
                   choices=["adversarial", "popular", "random"])
    p.add_argument("--n_samples", type=int, default=100,
                   help="Number of hard samples to find (default: 100)")
    p.add_argument("--seed", type=int, default=CFG.POPE_SEED)
    p.add_argument("--output", default="srf/hard_samples_pope.json",
                   help="Output JSON path")
    return p.parse_args()


def main():
    print("="*60)
    print("Finding hard POPE samples (baseline errors)")
    print("="*60)

    args = parse_args()

    # Load model
    global patch
    model, processor = load_model(args.model)

    # Initialize patch module
    if patch is None:
        # patch should have been set by load_model
        raise ValueError("patch module not initialized by load_model")

    arch = CFG.get_arch(args.model)
    if arch["image_token"] is not None:
        if hasattr(processor, 'tokenizer'):
            img_token_id = get_token_id(processor, arch["image_token"])
        else:
            img_token_id = processor.convert_tokens_to_ids(arch["image_token"])
    else:
        img_token_id = model.config.image_token_index
    device = next(model.parameters()).device

    # Check if Qwen-VL
    is_qwen_vl = hasattr(processor, 'from_list_format') and callable(processor.from_list_format)

    # Load POPE
    splits_filter = {s.lower() for s in args.pope_splits}
    splits_label = "+".join(sorted(splits_filter))

    ds = hf_load("lmms-lab/POPE", split="test")
    rows = [r for r in ds if str(r.get("category", r.get("type", ""))).lower() in splits_filter]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    print(f"\nLoaded {len(rows)} POPE samples ({splits_label})")
    print(f"Searching for {args.n_samples} samples where baseline fails...\n")

    # Track baseline errors
    hard_samples = []
    total_checked = 0

    for i, r in enumerate(rows):
        if len(hard_samples) >= args.n_samples:
            break

        image = r["image"].convert("RGB")
        q = str(r["question"]).strip() + "\nAnswer with Yes or No only."
        gt = "yes" if str(r.get("answer", "")).strip().lower() == "yes" else "no"
        split = str(r.get("category", r.get("type", "unknown"))).lower()

        # Format input
        if is_qwen_vl:
            msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                                   {"type": "text", "text": q}]}]
            text, img_paths = format_qwen_msgs(msgs, processor, model)
            pad_token_id = processor.eod_id
            inp = processor(text, return_tensors="pt", padding=False).to(device)
        else:
            msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                                  {"type": "text", "text": q}]}]
            text = apply_chat_template(processor, msgs, tokenize=False, add_generation_prompt=True)
            vis, _ = process_vision_info(msgs)
            inp = processor(text=[text], images=vis, return_tensors="pt", padding=True).to(device)

        s, e = get_img_range(inp["input_ids"], img_token_id)

        # Run baseline
        patch._STATE["method"] = "baseline"

        if is_qwen_vl:
            with torch.inference_mode():
                max_new_tokens = 20
                eos_token_id = getattr(processor, 'eod_id', getattr(processor, 'eos_token_id', 151644))
                generated = model.generate(
                    **inp,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos_token_id
                )
                generated_ids = generated[0][inp["input_ids"].shape[1]:]
                generated_part = processor.decode(generated_ids, skip_special_tokens=True).strip().lower()
                pred = "yes" if "yes" in generated_part else "no"
        else:
            with torch.inference_mode():
                logits_base = model(**inp).logits[:, -1, :].float()
                yes_with_space_id = processor.tokenizer.encode(" Yes", add_special_tokens=False)[0]
                no_with_space_id = processor.tokenizer.encode(" No", add_special_tokens=False)[0]
                yes_logit = logits_base[0, yes_with_space_id].item()
                no_logit = logits_base[0, no_with_space_id].item()
                pred = "yes" if yes_logit > no_logit else "no"

        # Check if baseline got it wrong
        if pred != gt:
            hard_samples.append({
                "index": i,
                "split": split,
                "question": str(r["question"]).strip(),
                "ground_truth": gt,
                "baseline_pred": pred,
                "sample_id": r.get("image_id", f"{split}_{i}")
            })

        total_checked += 1

        if (total_checked % 50) == 0:
            print(f"  Checked {total_checked} samples, found {len(hard_samples)} hard samples...")

    # Cleanup Qwen temp images
    if is_qwen_vl:
        cleanup_qwen_temp_images([])

    # Save results
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "model": args.model,
        "pope_splits": args.pope_splits,
        "n_requested": args.n_samples,
        "n_found": len(hard_samples),
        "total_checked": total_checked,
        "samples": hard_samples
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Found {len(hard_samples)} hard samples (checked {total_checked} total)")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}")

    # Print summary by split
    by_split = {}
    for s in hard_samples:
        by_split[s["split"]] = by_split.get(s["split"], 0) + 1

    print("\nHard samples by split:")
    for split, count in sorted(by_split.items()):
        print(f"  {split}: {count}")


if __name__ == "__main__":
    import os
    main()
