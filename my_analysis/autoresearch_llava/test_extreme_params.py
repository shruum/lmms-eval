#!/usr/bin/env python3
"""Test with extreme parameters to verify effect."""
from __future__ import annotations
import os, pathlib, sys, torch
os.environ["HF_HOME"] = "/home/anna2/.cache/huggingface"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset as hf_load
import random
import llava_attn_patch_working as patch

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Load one sample
ds = hf_load("lmms-lab/POPE", split="test")
rows = [r for r in ds if str(r.get("category", "")).strip().lower() == "adversarial"]
random.Random(42).shuffle(rows)
sample = rows[0]
image = sample["image"].convert("RGB")
question = str(sample.get("question", "")).strip() + "\nAnswer with Yes or No only."

prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}], add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

img_start, img_end = patch.get_image_token_range(inputs, model)
print(f"Image tokens: [{img_start}, {img_end}] = {img_end - img_start + 1} tokens")

# Test 1: Baseline
print("\n=== Test 1: Baseline ===")
patch.patch_model(model, "baseline", 1.0, 1.0)
patch.update_sample(img_start, img_end)
with torch.no_grad():
    out1 = model.generate(**inputs, max_new_tokens=5, do_sample=False)
resp1 = processor.decode(out1[0], skip_special_tokens=True)
print(f"Output: {resp1}")

# Test 2: EXTREME enhancement (enh=10.0) - should definitely change output
print("\n=== Test 2: EXTREME enhancement (enh=10.0, sup=0.1) ===")
patch.unpatch_model(model)
patch.patch_model(model, "srf", 10.0, 0.1, layer_start=9, layer_end=14)
patch.update_sample(img_start, img_end)
print(f"Enhancement: {patch._STATE['enh_para']}, Suppression: {patch._STATE['sup_para']}")
with torch.no_grad():
    out2 = model.generate(**inputs, max_new_tokens=5, do_sample=False)
resp2 = processor.decode(out2[0], skip_special_tokens=True)
print(f"Output: {resp2}")

# Check if outputs differ
if torch.equal(out1, out2):
    print("\n⚠️  Even extreme params produce same output - VAF not working during generation!")
else:
    print("\n✓ Extreme params change output - VAF is working!")

# Extract answers
def extract_answer(response):
    if "ASSISTANT:" in response:
        return response.split("ASSISTANT:")[-1].strip().split()[0].lower()
    return response.strip().split()[0].lower()

ans1 = extract_answer(resp1)
ans2 = extract_answer(resp2)

print(f"\nAnswers:")
print(f"  Baseline: {ans1}")
print(f"  Extreme VAF: {ans2}")

patch.unpatch_model(model)
print("\n✓ Test completed")
