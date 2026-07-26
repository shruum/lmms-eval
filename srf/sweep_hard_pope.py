#!/usr/bin/env python3
"""
Run systematic hyperparameter sweep on hard POPE samples.

Tests:
1. Baseline (no intervention)
2. SRF (pre-softmax) with different configs
3. Post-softmax redistribution (different configs)

All parameters via CLI - NO hardcoding.

Usage:
    python srf/sweep_hard_pope.py \
        --hard_samples srf/hard_samples_pope.json \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --output results/sweep_hard_pope/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import itertools
from collections import defaultdict

_SRF_DIR = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR / "saliency"))
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import config as CFG
os.environ.setdefault("HF_HOME", CFG.HF_HOME)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import os

from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM, LlavaForConditionalGeneration

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

from qwen_vl_utils import process_vision_info
from datasets import load_dataset as hf_load

# Global for patch module
patch = None


def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter sweep on hard POPE samples")
    p.add_argument("--hard_samples", required=True,
                   help="Path to hard_samples JSON from find_hard_pope_samples.py")
    p.add_argument("--model", default=CFG.DEFAULT_MODEL)
    p.add_argument("--output", default="results/sweep_hard_pope/",
                   help="Output directory")

    # Layer ranges to test
    p.add_argument("--layer_ranges", nargs="+", default=["5-10", "8-15", "10-15", "15-25"],
                   help="Layer ranges to test (format: start-end)")

    # Head percentages
    p.add_argument("--head_pcts", nargs="+", type=float, default=[0.3, 0.5, 0.8],
                   help="Head top-k percentages to test")

    # Alpha values
    p.add_argument("--alphas", nargs="+", type=float, default=[0.15, 0.5, 1.0, 2.0],
                   help="Alpha values to test")

    # Text suppression options
    p.add_argument("--with_text_suppression", action="store_true",
                   help="Include text suppression (text_beta)")
    p.add_argument("--with_sys_suppression", action="store_true",
                   help="Include system prompt suppression (sys_beta)")

    # Post-softmax variant
    p.add_argument("--include_post_softmax", action="store_true",
                   help="Include post-softmax redistribution variant")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (from eval.py)
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    global patch

    print(f"Loading {model_id}…")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = config.model_type if hasattr(config, 'model_type') else None

    is_qwen_vl_chat = "Qwen-VL-Chat" in model_id or "Qwen-VL" in model_id
    is_qwen2_vl = "Qwen2" in model_id or "qwen2" in model_id.lower()

    print(f"  Detected model type: {model_type}")

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
        import qwen_attn_patch as qwen_patch
        patch = qwen_patch
        print(f"  Using Qwen2-VL attention patch")
        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError("Qwen2.5-VL requires transformers >= 4.40.0")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval()
        model.cuda()

    if model_type == "llava":
        from transformers import AutoTokenizer, LlavaProcessor
        from transformers import CLIPImageProcessor
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        image_processor = CLIPImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    else:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def apply_chat_template(processor, msgs: list, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
    if hasattr(processor, 'tokenizer'):
        tok = processor.tokenizer
        if not hasattr(tok, 'chat_template') or tok.chat_template is None:
            user_content = []
            for msg in msgs:
                if msg["role"] == "user":
                    for item in msg["content"]:
                        if item["type"] == "image":
                            user_content.append("<image>")
                        elif item["type"] == "text":
                            user_content.append(item["text"])
            user_text = " ".join(user_content)
            prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {user_text} ASSISTANT:"
            return prompt

    if hasattr(processor, 'apply_chat_template') and callable(processor.apply_chat_template):
        return processor.apply_chat_template(msgs, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
    elif hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'apply_chat_template'):
        return processor.tokenizer.apply_chat_template(msgs, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
    else:
        raise AttributeError(f"No apply_chat_template method found on processor {type(processor)}")


def get_token_id(processor, token: str) -> int:
    tok_id = None

    if hasattr(processor, 'convert_tokens_to_ids') and callable(processor.convert_tokens_to_ids):
        try:
            tok_id = processor.convert_tokens_to_ids(token)
        except:
            pass

    if tok_id is None and hasattr(processor, 'tokenizer'):
        try:
            if hasattr(processor.tokenizer, 'convert_tokens_to_ids'):
                tok_id = processor.tokenizer.convert_tokens_to_ids(token)
        except:
            pass

    if tok_id is None:
        if hasattr(processor, 'special_tokens_map'):
            tok_id = processor.special_tokens_map.get(token, None)
        elif hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'special_tokens_map'):
            tok_id = processor.tokenizer.special_tokens_map.get(token, None)

    if tok_id is not None:
        if hasattr(tok_id, 'item'):
            tok_id = tok_id.item()
        return int(tok_id)

    raise ValueError(f"Cannot find token ID for {token!r} in processor {type(processor)}")


def format_qwen_msgs(msgs: list, processor, model) -> tuple:
    query = []
    images = []

    for msg in msgs:
        for item in msg["content"]:
            if item["type"] == "image":
                import uuid
                name = uuid.uuid4().hex.upper()[0:6]
                temp_path = f"/tmp/{name}.png"
                item["image"].save(temp_path)
                images.append(temp_path)
                query.append({"image": temp_path})
            elif item["type"] == "text":
                query.append({"text": item["text"]})

    text = processor.from_list_format(query)
    text = text + "\nAssistant:"
    return text, images


def cleanup_qwen_temp_images(images: list) -> None:
    for temp_path in images:
        try:
            os.unlink(temp_path)
        except (FileNotFoundError, PermissionError):
            pass


def get_img_range(input_ids: torch.Tensor, img_token_id: int) -> tuple[int, int]:
    ids = input_ids[0].tolist()

    try:
        start = next(i for i, t in enumerate(ids) if t == img_token_id)
        end = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
        return start, end
    except StopIteration:
        return 0, min(255, len(ids) - 1)


def decode_first_token(logits: torch.Tensor, processor) -> str:
    token_id = logits.argmax(dim=-1).item()

    decoded = processor.decode([token_id], skip_special_tokens=False).strip().lower()
    decoded = decoded.replace("", "").replace("<|im_start|>", "").strip()

    if "yes" in decoded:
        return "yes"
    elif "no" in decoded:
        return "no"

    decoded_clean = processor.decode([token_id], skip_special_tokens=True).strip().lower()
    if "yes" in decoded_clean:
        return "yes"
    elif "no" in decoded_clean:
        return "no"

    return decoded_clean if decoded_clean else ""


# ---------------------------------------------------------------------------
# Main sweep logic
# ---------------------------------------------------------------------------

def run_sweep():
    print("="*70)
    print("Hard POPE Sample Hyperparameter Sweep")
    print("="*70)

    args = parse_args()

    # Load hard samples
    print(f"\nLoading hard samples from: {args.hard_samples}")
    with open(args.hard_samples, "r") as f:
        hard_data = json.load(f)

    hard_samples = hard_data["samples"]
    print(f"Loaded {len(hard_samples)} hard samples")

    # Load model
    global patch
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

    # Generate all configurations
    configs = []

    # Parse layer ranges
    layer_ranges = []
    for lr in args.layer_ranges:
        start, end = map(int, lr.split("-"))
        layer_ranges.append((start, end))

    # Baseline config
    configs.append({
        "name": "baseline",
        "method": "baseline",
        "layer_start": None,
        "layer_end": None,
        "head_top_k_pct": None,
        "alpha": None,
        "text_beta": None,
        "sys_beta": None,
    })

    # Generate SRF configs
    for layer_start, layer_end in layer_ranges:
        for head_pct in args.head_pcts:
            for alpha in args.alphas:
                # No suppression
                configs.append({
                    "name": f"srf_l{layer_start}-{layer_end}_h{head_pct}_a{alpha}",
                    "method": "srf",
                    "layer_start": layer_start,
                    "layer_end": layer_end,
                    "head_top_k_pct": head_pct,
                    "alpha": alpha,
                    "text_beta": None,
                    "sys_beta": None,
                })

                # Text suppression
                if args.with_text_suppression:
                    configs.append({
                        "name": f"srf_l{layer_start}-{layer_end}_h{head_pct}_a{alpha}_txt",
                        "method": "srf",
                        "layer_start": layer_start,
                        "layer_end": layer_end,
                        "head_top_k_pct": head_pct,
                        "alpha": alpha,
                        "text_beta": 0.1,
                        "sys_beta": None,
                    })

                # Sys suppression
                if args.with_sys_suppression:
                    configs.append({
                        "name": f"srf_l{layer_start}-{layer_end}_h{head_pct}_a{alpha}_sys",
                        "method": "srf",
                        "layer_start": layer_start,
                        "layer_end": layer_end,
                        "head_top_k_pct": head_pct,
                        "alpha": alpha,
                        "text_beta": None,
                        "sys_beta": 0.1,
                    })

    # Post-softmax configs
    if args.include_post_softmax:
        for layer_start, layer_end in layer_ranges:
            for head_pct in args.head_pcts:
                for alpha in args.alphas:
                    configs.append({
                        "name": f"post_l{layer_start}-{layer_end}_h{head_pct}_a{alpha}",
                        "method": "post_softmax",
                        "layer_start": layer_start,
                        "layer_end": layer_end,
                        "head_top_k_pct": head_pct,
                        "alpha": alpha,
                        "text_beta": None,
                        "sys_beta": None,
                    })

    print(f"\nGenerated {len(configs)} configurations to test")
    print(f"  - 1 baseline")
    print(f"  - {len(configs) - 1} intervention configs")

    # Load full dataset for sample access
    ds = hf_load("lmms-lab/POPE", split="test")

    # Results storage
    results = {cfg["name"]: {"correct": 0, "total": 0, "predictions": []} for cfg in configs}

    print(f"\nRunning sweep on {len(hard_samples)} samples...")
    print("="*70)

    # Import SRF module
    import srf as srf_mod
    srf_mod.patch = patch

    for sample_idx, sample in enumerate(hard_samples):
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
                # Setup config
                srf_mod.reset_for_dataset(
                    dataset="pope",
                    layer_start=cfg["layer_start"],
                    layer_end=cfg["layer_end"],
                    head_top_k_pct=cfg["head_top_k_pct"],
                    alpha=cfg["alpha"],
                    text_beta=cfg["text_beta"],
                    sys_beta=cfg["sys_beta"],
                    phase="both",
                )

                srf_mod.setup(model, processor, calib_dataset="pope")
                srf_mod.prepare_sample(inp, s, e, image, q, model, processor)

                # Run with method
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

        # Progress
        if (sample_idx + 1) % 10 == 0 or (sample_idx + 1) == len(hard_samples):
            print(f"  Processed {sample_idx + 1}/{len(hard_samples)} samples")

        # Cleanup Qwen temp images
        if is_qwen_vl:
            cleanup_qwen_temp_images([])

    # Calculate final metrics
    print("\n" + "="*70)
    print("SWEEP RESULTS")
    print("="*70)

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

        # Print
        method_str = "BASELINE" if cfg["method"] == "baseline" else cfg["method"].upper()
        print(f"{cfg_name:40s}: {acc:.4f}  ({res['correct']}/{res['total']})  [{method_str}]")

    # Sort by accuracy
    summary.sort(key=lambda x: x["accuracy"], reverse=True)

    print("\n" + "="*70)
    print("TOP 10 CONFIGURATIONS")
    print("="*70)
    for i, entry in enumerate(summary[:10]):
        cfg = entry["config"]
        print(f"\n#{i+1}: {cfg['name']}")
        print(f"  Accuracy: {entry['accuracy']:.4f} ({entry['correct']}/{entry['total']})")
        print(f"  Method: {cfg['method']}")
        if cfg["layer_start"] is not None:
            print(f"  Layers: {cfg['layer_start']}-{cfg['layer_end']}")
        if cfg["head_top_k_pct"] is not None:
            print(f"  Heads: {cfg['head_top_k_pct']:.2f}")
        if cfg["alpha"] is not None:
            print(f"  Alpha: {cfg['alpha']}")
        if cfg["text_beta"] is not None:
            print(f"  Text beta: {cfg['text_beta']}")
        if cfg["sys_beta"] is not None:
            print(f"  Sys beta: {cfg['sys_beta']}")

    # Save results
    output_path = pathlib.Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / "sweep_results.json"
    with open(result_file, "w") as f:
        json.dump({
            "args": vars(args),
            "hard_samples_info": hard_data,
            "results": results,
            "summary": summary
        }, f, indent=2)

    print(f"\nResults saved to: {result_file}")
    print("="*70)


if __name__ == "__main__":
    import os
    run_sweep()
