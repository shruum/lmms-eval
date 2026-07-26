#!/usr/bin/env python3
"""
Worker process for layer-specific sweep - runs a subset of configurations on one GPU.
Called by sweep_layer_specific_parallel.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import os

_SRF_DIR = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR / "saliency"))
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import config as CFG

# Set GPU before importing torch
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
print(f"[Layer-Specific Worker GPU {gpu_id}] Starting...", flush=True)

import torch
from datasets import load_dataset as hf_load

# Import eval utilities
from eval import (
    load_model, get_token_id, apply_chat_template,
    format_qwen_msgs, cleanup_qwen_temp_images,
    get_img_range, decode_first_token
)
from qwen_vl_utils import process_vision_info

global patch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hard_samples", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--configs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu_id", type=int, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    gpu_id = args.gpu_id

    print(f"[GPU {gpu_id}] Loading hard samples...", flush=True)
    with open(args.hard_samples, "r") as f:
        hard_data = json.load(f)
    hard_samples = hard_data["samples"]
    print(f"[GPU {gpu_id}] Loaded {len(hard_samples)} hard samples", flush=True)

    print(f"[GPU {gpu_id}] Loading configs...", flush=True)
    with open(args.configs, "r") as f:
        configs = json.load(f)
    print(f"[GPU {gpu_id}] Loaded {len(configs)} configs", flush=True)

    model, processor = load_model(args.model)
    arch = CFG.get_arch(args.model)
    if arch["image_token"] is not None:
        if hasattr(processor, 'tokenizer'):
            img_token_id = get_token_id(processor, arch["image_token"])
        else:
            img_token_id = processor.convert_tokens_to_ids(arch["image_token"])
    else:
        img_token_id = model.config.image_token_index
    device = next(model.parameters()).device

    is_qwen_vl = hasattr(processor, 'from_list_format') and callable(processor.from_list_format)

    # Load full dataset
    ds = hf_load("lmms-lab/POPE", split="test")

    # Results storage
    results = {cfg["name"]: {"correct": 0, "total": 0, "predictions": []} for cfg in configs}

    print(f"[GPU {gpu_id}] Starting layer-specific sweep on {len(hard_samples)} samples...", flush=True)

    # Import SRF module
    import srf as srf_mod
    srf_mod.patch = patch

    for sample_idx, sample in enumerate(hard_samples):
        if sample_idx % 5 == 0:
            print(f"[GPU {gpu_id}] Processed {sample_idx}/{len(hard_samples)}", flush=True)

        idx = sample["index"]
        row = ds[idx]

        image = row["image"].convert("RGB")
        q = str(row["question"]).strip() + "\nAnswer with Yes or No only."
        gt = sample["ground_truth"]

        # Format input
        if is_qwen_vl:
            msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                                   {"type": "text", "text": q}]}]
            text, img_paths = format_qwen_msgs(msgs, processor, model)
            inp = processor(text, return_tensors="pt", padding=False).to(device)
        else:
            msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                                  {"type": "text", "text": q}]}]
            text = apply_chat_template(processor, msgs, tokenize=False, add_generation_prompt=True)
            vis, _ = process_vision_info(msgs)
            inp = processor(text=[text], images=vis, return_tensors="pt", padding=True).to(device)

        s, e = get_img_range(inp["input_ids"], img_token_id)

        # Test each config
        for cfg in configs:
            cfg_name = cfg["name"]

            try:
                if cfg["method"] == "baseline":
                    patch._STATE["method"] = "baseline"

                    if is_qwen_vl:
                        with torch.inference_mode():
                            max_new_tokens = 20
                            eos_token_id = getattr(processor, 'eod_id', getattr(processor, 'eos_token_id', 151644))
                            generated = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos_token_id)
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

                else:
                    # For layer-specific, we still use SRF reset but with different params
                    # This is a simplified version - full layer-specific would need more implementation
                    srf_mod.reset_for_dataset(
                        dataset="pope",
                        layer_start=cfg.get("layer_early_end", 8),
                        layer_end=cfg.get("layer_mid_end", 15),
                        head_top_k_pct=0.20,
                        alpha=cfg.get("alpha_mid", 2.0),
                        text_beta=cfg.get("beta_late"),
                        sys_beta=0.0,
                        phase="both",
                    )

                    srf_mod.setup(model, processor, calib_dataset="pope")
                    srf_mod.prepare_sample(inp, s, e, image, q, model, processor)

                    if is_qwen_vl:
                        with torch.inference_mode():
                            max_new_tokens = 20
                            eos_token_id = getattr(processor, 'eod_id', getattr(processor, 'eos_token_id', 151644))
                            generated = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos_token_id)
                            generated_ids = generated[0][inp["input_ids"].shape[1]:]
                            pred_text = processor.decode(generated_ids, skip_special_tokens=True).strip().lower()
                            pred = "yes" if "yes" in pred_text else "no"
                    else:
                        patch._STATE["method"] = "srf"
                        with torch.inference_mode():
                            logits = model(**inp).logits[:, -1, :].float()
                            pred_text = decode_first_token(logits, processor)
                            pred = "yes" if pred_text.startswith("yes") else "no"

                    srf_mod.cleanup()

                # Track results
                results[cfg_name]["total"] += 1
                if pred == gt:
                    results[cfg_name]["correct"] += 1
                results[cfg_name]["predictions"].append({
                    "sample_idx": sample_idx,
                    "predicted": pred,
                    "ground_truth": gt,
                    "correct": pred == gt
                })

            except Exception as e:
                print(f"[GPU {gpu_id}] ERROR in config {cfg_name}: {e}", flush=True)
                results[cfg_name]["total"] += 1
                results[cfg_name]["predictions"].append({
                    "sample_idx": sample_idx,
                    "predicted": "error",
                    "ground_truth": gt,
                    "correct": False,
                    "error": str(e)
                })

        if is_qwen_vl:
            cleanup_qwen_temp_images([])

    # Calculate summary
    summary = []
    for cfg in configs:
        cfg_name = cfg["name"]
        res = results[cfg_name]
        acc = res["correct"] / res["total"] if res["total"] > 0 else 0.0
        summary.append({
            "config": cfg,
            "accuracy": acc,
            "correct": res["correct"],
            "total": res["total"]
        })

    # Save results
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "gpu_id": gpu_id,
            "results": results,
            "summary": summary
        }, f, indent=2)

    print(f"[GPU {gpu_id}] Completed! Results saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
