#!/usr/bin/env python3
"""
SRF evaluation on RePOPE COCO - standalone script for autosearch.

Usage:
    python srf/eval_repope.py --head_top_k_pct 0.30 --alpha 2.0 --clip_top_k_pct 0.25
"""
import os
import sys
import json
import argparse
from pathlib import Path

SRF_DIR = Path(__file__).parent
LMMS_EVAL_DIR = SRF_DIR.parent
os.chdir(LMMS_EVAL_DIR)

sys.path.insert(0, str(SRF_DIR))
sys.path.insert(0, str(SRF_DIR / "saliency"))

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
import config as CFG

# Import SRF components
from srf import SRF
from noun_extract import extract_clip_noun

# RePOPE data paths
REPOPE_ANNOTATIONS = "/home/anna2/shruthi/RePOPE/annotations"
COCO_IMAGES = "/home/anna2/shruthi/POPE/coco/images"

MODEL = "liuhaotian/llava-v1.5-7b"

def load_repope_split(split):
    """Load RePOPE split data."""
    file_path = f"{REPOPE_ANNOTATIONS}/coco_repope_{split}.json"
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def evaluate_config(head_top_k_pct, alpha, clip_top_k_pct, split="adversarial", n_samples=None):
    """Evaluate SRF with given config on RePOPE."""
    
    # Load data
    data = load_repope_split(split)
    if n_samples:
        data = data[:n_samples]
    
    print(f"\nEvaluating on {len(data)} samples from {split}")
    print(f"Config: head_top_k_pct={head_top_k_pct}, alpha={alpha}, clip_top_k_pct={clip_top_k_pct}")
    
    # Load model
    print("Loading model...")
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(MODEL)
    model = LlavaForCausalLM.from_pretrained(
        MODEL, 
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    
    # Setup SRF with custom config
    srf_config = {
        "layer_start": 10,
        "layer_end": 15,
        "head_top_k_pct": head_top_k_pct,
        "clip_coarse_grid": 6,  # LLaVA
        "clip_top_k_pct": clip_top_k_pct,
        "clip_fallback_thresh": 0.20,
        "alpha": alpha,
        "eps": 0.0,
        "clip_suppress_thresh": 0.0,
        "clip_suppress_alpha": 5.0,
        "phase": "both",
    }
    
    srf = SRF()
    srf.setup(model, processor, calib_dataset="pope")
    srf.reset_for_dataset("pope", **srf_config)
    
    # Evaluate
    correct = 0
    total = 0
    
    print("\nRunning evaluation...")
    for item in tqdm(data):
        question = item["text"]
        image_path = os.path.join(COCO_IMAGES, item["image"])
        label = item["label"].lower()
        
        # Prepare inputs
        prompt = f"USER: <image>\n{question} Please answer yes or no. ASSISTANT:"
        inputs = processor(text=prompt, images=image_path, return_tensors="pt").to(device)
        
        # Apply SRF
        img_start, img_end = srf.prepare_sample(inputs, image_path, question, model, processor)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                output_hidden_states=True,
            )
        
        # Decode answer
        answer = processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        answer = answer.lower().strip()
        
        # Check correctness
        if label in answer:
            correct += 1
        total += 1
        
        # Cleanup
        srf.cleanup()
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Results: {correct}/{total} correct")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"{'='*60}\n")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="SRF evaluation on RePOPE")
    parser.add_argument("--head_top_k_pct", type=float, default=0.20)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--clip_top_k_pct", type=float, default=0.30)
    parser.add_argument("--split", default="adversarial", choices=["random", "popular", "adversarial"])
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    acc = evaluate_config(
        args.head_top_k_pct,
        args.alpha, 
        args.clip_top_k_pct,
        args.split,
        args.n_samples
    )
    
    result = {
        "config": {
            "head_top_k_pct": args.head_top_k_pct,
            "alpha": args.alpha,
            "clip_top_k_pct": args.clip_top_k_pct,
        },
        "split": args.split,
        "n_samples": args.n_samples,
        "accuracy": acc,
    }
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return acc

if __name__ == "__main__":
    main()
