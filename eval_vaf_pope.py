#!/usr/bin/env python3
"""
VAF (Visual Amplification Fusion) - POPE Evaluation Script.

Tests ClearSight method on POPE benchmark with LLaVA-1.5-7B.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vaf import setup_vaf, cleanup_vaf, update_token_lengths
from transformers import AutoProcessor, LlavaForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VAF on POPE")
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf",
                       help="Model path or HuggingFace ID")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["coco", "aokvqa", "gqa"],
                       help="POPE dataset")
    parser.add_argument("--split", type=str, required=True,
                       choices=["random", "popular", "adversarial"],
                       help="POPE split")
    parser.add_argument("--image_dir", type=str,
                       default="/home/anna2/shruthi/dataset/POPE_images/images/val2014",
                       help="Image directory")
    parser.add_argument("--data_dir", type=str,
                       default="/home/anna2/shruthi/VCD/experiments/data/POPE",
                       help="POPE data directory")
    parser.add_argument("--output_dir", type=str, default="results/vaf_pope",
                       help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.15,
                       help="VAF enhancement parameter")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="VAF suppression parameter")
    parser.add_argument("--layer_start", type=int, default=10,
                       help="First layer to patch")
    parser.add_argument("--layer_end", type=int, default=15,
                       help="Last layer to patch")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples (for testing)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    return parser.parse_args()


def load_pope_data(dataset, split, data_dir):
    """Load POPE data."""
    file_path = os.path.join(data_dir, dataset, f"{dataset}_pope_{split}.json")
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def get_image_token_range(inputs, model):
    """Get image token range for LLaVA."""
    image_token_id = model.config.image_token_index
    ids_cpu = inputs["input_ids"][0].cpu()
    positions = (ids_cpu == image_token_id).nonzero(as_tuple=False)[0]

    if len(positions) == 0:
        return 0, 0

    img_start = int(positions[0].item())

    # Calculate number of image tokens
    vis_cfg = model.vision_tower.config
    n_img_tokens = (vis_cfg.image_size // vis_cfg.patch_size) ** 2

    return img_start, img_start + n_img_tokens - 1


def evaluate_vaf(args):
    """Evaluate VAF on POPE."""

    print("=" * 60)
    print(f"VAF Evaluation on POPE {args.dataset} {args.split}")
    print("=" * 60)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.model}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(args.model)
    print("Model loaded")

    # Load data
    print(f"Loading POPE data: {args.dataset} {args.split}")
    data = load_pope_data(args.dataset, split, args.data_dir)
    print(f"Loaded {len(data)} samples")

    if args.limit:
        data = data[:args.limit]
        print(f"Limited to {len(data)} samples")

    # Setup VAF
    print(f"Setting up VAF: alpha={args.alpha}, beta={args.beta}, layers={args.layer_start}-{args.layer_end}")
    setup_vaf(
        model,
        alpha=args.alpha,
        beta=args.beta,
        layer_start=args.layer_start,
        layer_end=args.layer_end
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Output file
    output_file = os.path.join(
        args.output_dir,
        f"vaf_{args.dataset}_{args.split}_alpha{args.alpha}_beta{args.beta}.jsonl"
    )

    # Evaluation
    correct = 0
    total = 0
    yes_count = 0

    with open(output_file, 'w') as f_out:
        for sample in tqdm(data, desc=f"Evaluating {args.split}"):
            image_path = os.path.join(args.image_dir, sample['image'])
            question = sample['text']
            question_id = sample['question_id']

            # Load image
            try:
                from PIL import Image
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            # Prepare prompt
            prompt = processor.apply_chat_template([
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question + " Please answer this question with one word."}
                ]}
            ], add_generation_prompt=True)

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

            # Update token lengths for VAF
            img_start, img_end = get_image_token_range(inputs, model)
            update_token_lengths(img_start, img_end)

            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    use_cache=True
                )

            # Decode response
            response = processor.decode(output_ids[0], skip_special_tokens=True)

            # Extract answer
            if "ASSISTANT:" in response:
                answer = response.split("ASSISTANT:")[-1].strip().split()[0].lower()
            else:
                answer = response.strip().split()[0].lower()

            # Get ground truth
            gt_answer = sample.get("label", "").lower()

            # Check correctness
            is_correct = (answer == gt_answer)
            if is_correct:
                correct += 1

            total += 1
            if answer == "yes":
                yes_count += 1

            # Write output
            result = {
                "question_id": question_id,
                "prompt": question,
                "text": answer,
                "gt_answer": gt_answer,
                "correct": is_correct
            }
            f_out.write(json.dumps(result) + "\n")

    # Cleanup VAF
    cleanup_vaf(model)

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    yes_ratio = yes_count / total if total > 0 else 0

    print("\n" + "=" * 60)
    print(f"Results: {args.dataset} {args.split}")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Yes ratio: {yes_ratio:.4f} ({yes_ratio*100:.2f}%)")
    print(f"Output: {output_file}")

    # Save metrics
    metrics_file = output_file.replace(".jsonl", "_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            "dataset": args.dataset,
            "split": args.split,
            "alpha": args.alpha,
            "beta": args.beta,
            "layer_start": args.layer_start,
            "layer_end": args.layer_end,
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "yes_ratio": yes_ratio
        }, f, indent=2)

    print(f"Metrics saved to: {metrics_file}")

    return accuracy


def main():
    args = parse_args()
    accuracy = evaluate_vaf(args)
    print(f"\nFinal accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
