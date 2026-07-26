#!/usr/bin/env python3
"""
Fixed POPE VCD evaluation using actual image files.

This script evaluates POPE VCD-format datasets by loading images from
local files instead of trying to match with HuggingFace POPE.

Usage:
    python srf/eval_pope_vcd_fixed.py \\
        --method baseline \\
        --model llava-hf/llava-1.5-7b-hf \\
        --pope_vcd_file /path/to/coco_adversarial.json \\
        --pope_vcd_name coco_adversarial \\
        --image_dir /path/to/images/val2014 \\
        --output results/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from eval_datasets import is_correct


def apply_chat_template(processor, msgs: list, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
    """Apply chat template, handling both processor and tokenizer.

    For LLaVA with old transformers (no chat_template), use manual format.
    """
    # For LLaVA with old transformers (no chat_template), use manual format
    if hasattr(processor, 'tokenizer'):
        tok = processor.tokenizer
        # Check if chat_template is None or missing (transformers < 4.40)
        if not hasattr(tok, 'chat_template') or tok.chat_template is None:
            # LLaVA-1.5 Vicuna format with intro
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

    # Standard chat template
    return processor.apply_chat_template(msgs, tokenize=tokenize, add_generation_prompt=add_generation_prompt)


def load_pope_vcd_data(json_file: str, image_dir: str) -> List[Dict]:
    """Load POPE VCD data with actual image files.

    Args:
        json_file: Path to VCD-format JSON file (JSONL or JSON array format)
        image_dir: Path to directory containing the actual images

    Returns:
        List of dicts with keys: question_id, image (PIL Image), question, label
    """
    items = []

    with open(json_file, 'r') as f:
        content = f.read()

        # Check if it's a JSON array (A-OKVQA, GQA) or JSONL (COCO)
        if content.strip().startswith('['):
            # JSON array format
            data_list = json.loads(content)
        else:
            # JSONL format (one JSON per line)
            data_list = []
            for line in content.split('\n'):
                if line.strip():
                    data_list.append(json.loads(line))

        for data in data_list:

                # Load image from file
                image_path = os.path.join(image_dir, data["image"])
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue

                image = Image.open(image_path).convert("RGB")

                # Convert question format to match AIR paper
                # From: "Is there a [object] in the image?"
                # To: "Is [object] in this image? Please answer yes or no."
                question = str(data.get("text", "")).strip()

                # Extract object name and convert format
                if "Is there a" in question and "in the image" in question:
                    obj_start = question.find("Is there a") + len("Is there a")
                    obj_end = question.find("in the image")
                    object_name = question[obj_start:obj_end].strip().rstrip('?').strip()
                    question = f"Is {object_name} in this image? Please answer yes or no."
                else:
                    # Fallback: just add "Please answer yes or no."
                    question = question + " Please answer yes or no."

                items.append({
                    "question_id": data.get("question_id", 0),
                    "image": image,
                    "question": question,
                    "label": str(data.get("label", "")).strip().lower()
                })

    return items


def parse_yes_no(response: str) -> str:
    """Parse model response to extract yes/no answer.

    Uses the same logic as VCD/VAF papers:
    - Take first sentence
    - Remove commas
    - Check for "No"/"not"/"no" -> "no", else -> "yes"
    """
    response = response.split('.')[0]
    response = response.replace(',', '')
    words = response.split(' ')

    if 'No' in words or 'not' in words or 'no' in words:
        return 'no'
    else:
        return 'yes'


def evaluate_pope_vcd(
    model,
    processor,
    items: List[Dict],
    method: str,
    device: str,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 20
) -> Dict:
    """Evaluate POPE VCD dataset.

    Args:
        model: VLM model
        processor: Model processor/tokenizer
        items: List of POPE samples (from load_pope_vcd_data)
        method: "baseline" or "srf"
        device: Device to run on
        max_new_tokens: Max tokens to generate

    Returns:
        Dict with metrics: accuracy, precision, recall, f1, yes_ratio
    """
    tp = tn = fp = fn = 0
    yes_count = 0

    for item in tqdm(items, desc=f"Evaluating {method}"):
        image = item["image"]
        question = item["question"]
        gt_label = item["label"]

        # Prepare input
        messages = [{"role": "user", "content": [
            {"type": "image", "url": image},
            {"type": "text", "text": question}
        ]}]
        prompt = apply_chat_template(processor, messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")

        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            if do_sample:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p
                )
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )

        # Decode response
        generated_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True).strip().lower()

        # Parse answer
        pred_label = parse_yes_no(response)

        # Update metrics
        if pred_label == "yes":
            yes_count += 1

        if gt_label == "yes":
            if pred_label == "yes":
                tp += 1
            else:
                fn += 1
        else:  # gt == "no"
            if pred_label == "yes":
                fp += 1
            else:
                tn += 1

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    yes_ratio = yes_count / len(items)

    return {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "yes_ratio": yes_ratio,
        "n_samples": len(items),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate POPE VCD datasets with actual image files")
    parser.add_argument("--method", type=str, required=True, choices=["baseline", "srf"])
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--pope_vcd_file", type=str, required=True, help="Path to VCD JSON file")
    parser.add_argument("--pope_vcd_name", type=str, required=True, help="Dataset name (e.g., coco_adversarial)")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling decoding (VCD paper method)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for sampling (default: 0.9)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("="*60)
    print(f"POPE VCD Evaluation: {args.pope_vcd_name}")
    print(f"Method: {args.method.upper()}")
    print(f"Model: {args.model}")
    print(f"Dataset file: {args.pope_vcd_file}")
    print(f"Image directory: {args.image_dir}")
    print("="*60)

    # Load model
    print(f"\nLoading model...")
    from transformers import AutoProcessor, AutoConfig
    from transformers import LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device
    ).eval()

    # Load processor with slow tokenizer to avoid fast tokenizer corruption
    # Load processor components separately, forcing slow tokenizer
    from transformers import AutoTokenizer, AutoImageProcessor, LlavaProcessor

    # Load slow tokenizer manually
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(args.model)

    # Load processor manually with both components
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)

    print(f"Model loaded on {args.device}")

    # Load data
    print(f"\nLoading POPE VCD data...")
    items = load_pope_vcd_data(args.pope_vcd_file, args.image_dir)
    print(f"Loaded {len(items)} samples")

    # Evaluate
    print(f"\nEvaluating...")
    decoding_method = "sampling" if args.do_sample else "greedy"
    print(f"Decoding: {decoding_method}", end="")
    if args.do_sample:
        print(f" (temp={args.temperature}, top_p={args.top_p})")
    else:
        print()

    results = evaluate_pope_vcd(
        model, processor, items, args.method, args.device,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Accuracy:  {results['accuracy']:.2f}%")
    print(f"Precision: {results['precision']:.2f}%")
    print(f"Recall:    {results['recall']:.2f}%")
    print(f"F1:        {results['f1']:.2f}%")
    print(f"Yes Ratio: {results['yes_ratio']:.2f}")
    print(f"Samples:   {results['n_samples']}")
    print("="*60)

    # Save results
    output_file = os.path.join(args.output, f"pope_{args.pope_vcd_name}_{args.method}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
