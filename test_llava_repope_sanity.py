#!/usr/bin/env python3
"""
Sanity test: Load LLaVA model and test on 1 RePOPE sample.
REAL TESTING - NO FAKING.
"""

import os
import sys
import json
from pathlib import Path

# Set environment BEFORE any imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/home/anna2/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/anna2/.cache/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/home/anna2/.cache/huggingface/datasets"
os.environ["HF_HUB_CACHE"] = "/home/anna2/.cache/huggingface/hub"

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "my_analysis"))
sys.path.insert(0, str(Path(__file__).parent / "srf"))
sys.path.insert(0, str(Path(__file__).parent / "srf" / "saliency"))

import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

print("=" * 70)
print("LLaVA RePOPE Sanity Test - REAL TESTING")
print("=" * 70)

def test_llava_on_repope():
    """Test LLaVA loading and inference on 1 RePOPE sample."""

    # RePOPE paths
    repope_file = "/home/anna2/shruthi/RePOPE/annotations/coco_repope_adversarial.json"
    image_dir = "/home/anna2/shruthi/dataset/POPE_images/images/val2014/"

    print(f"RePOPE annotations: {repope_file}")
    print(f"Image directory: {image_dir}")

    # Load 1 RePOPE sample
    if not os.path.exists(repope_file):
        print(f"❌ FAILED: RePOPE file not found: {repope_file}")
        return False

    with open(repope_file) as f:
        sample = json.loads(f.readline())

    print(f"✓ Loaded RePOPE sample: {sample['text'][:50]}...")

    # Check image exists
    image_path = os.path.join(image_dir, sample["image"])
    if not os.path.exists(image_path):
        print(f"❌ FAILED: Image not found: {image_path}")
        return False

    print(f"✓ Found image: {sample['image']}")

    # Load LLaVA model
    print("\nLoading LLaVA model...")
    model_id = "llava-hf/llava-1.5-7b-hf"

    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id)
        print(f"✓ Model loaded: {model_id}")
    except Exception as e:
        print(f"❌ FAILED: Model loading error: {e}")
        return False

    # Run inference
    print("\nRunning inference on 1 sample...")
    try:
        image = Image.open(image_path).convert("RGB")
        question = sample["text"]
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        inputs = processor(text=prompt, images=image, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                num_beams=1,
            )

        response = processor.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("ASSISTANT:")[-1].strip().lower()
        gt_answer = sample["label"].lower()

        print(f"✓ Inference completed")
        print(f"Question: {question}")
        print(f"Generated: {answer}")
        print(f"Ground truth: {gt_answer}")
        print(f"Correct: {answer == gt_answer}")

    except Exception as e:
        print(f"❌ FAILED: Inference error: {e}")
        return False

    print("\n" + "=" * 70)
    print("✅ SUCCESS: LLaVA loads and runs on RePOPE data")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_llava_on_repope()
    sys.exit(0 if success else 1)
