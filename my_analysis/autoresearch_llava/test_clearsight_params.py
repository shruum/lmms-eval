#!/usr/bin/env python3
"""Test with ClearSight's actual parameters on 50 samples."""
from __future__ import annotations
import os, pathlib, sys, torch
os.environ["HF_HOME"] = "/home/anna2/.cache/huggingface"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset as hf_load
import random
from tqdm import tqdm
import llava_attn_patch_working as patch

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Load POPE adversarial samples
ds = hf_load("lmms-lab/POPE", split="test")
rows = [r for r in ds if str(r.get("category", "")).strip().lower() == "adversarial"]
random.Random(42).shuffle(rows)
test_samples = rows[:50]  # Test on 50 samples

def evaluate(samples, method="baseline", enh_para=1.0, sup_para=1.0):
    """Evaluate on samples."""
    patch.unpatch_model(model)
    patch.patch_model(model, method, enh_para, sup_para, layer_start=9, layer_end=14)

    correct = 0
    for sample in tqdm(samples, desc=f"{method} enh={enh_para} sup={sup_para}"):
        image = sample["image"].convert("RGB")
        question = str(sample.get("question", "")).strip() + "\nAnswer with Yes or No only."

        prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}], add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        img_start, img_end = patch.get_image_token_range(inputs, model)
        patch.update_sample(img_start, img_end)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        response = processor.decode(output[0], skip_special_tokens=True)

        # Extract answer
        if "ASSISTANT:" in response:
            answer = response.split("ASSISTANT:")[-1].strip().split()[0].lower()
        else:
            answer = response.strip().split()[0].lower()

        # Get ground truth
        gt = str(sample.get("answer", "")).strip().lower()
        if answer == gt:
            correct += 1

    return correct / len(samples)

# Test baseline
print("\n=== Evaluating Baseline ===")
acc_baseline = evaluate(test_samples, "baseline", 1.0, 1.0)
print(f"Baseline accuracy: {acc_baseline:.1%}")

# Test ClearSight parameters
print("\n=== Evaluating ClearSight (enh=1.15, sup=0.95) ===")
acc_clearsight = evaluate(test_samples, "srf", 1.15, 0.95)
print(f"ClearSight accuracy: {acc_clearsight:.1%}")

# Compare
if acc_clearsight > acc_baseline:
    improvement = (acc_clearsight - acc_baseline) * 100
    print(f"\n✓ ClearSight improves accuracy by {improvement:.1f}%")
elif acc_clearsight < acc_baseline:
    degradation = (acc_baseline - acc_clearsight) * 100
    print(f"\n⚠️  ClearSight degrades accuracy by {degradation:.1f}%")
else:
    print(f"\n= ClearSight matches baseline")

patch.unpatch_model(model)
print("\n✓ Evaluation complete")
