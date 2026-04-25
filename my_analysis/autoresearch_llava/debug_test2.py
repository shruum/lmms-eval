#!/usr/bin/env python3
"""Test if sys_beta is causing the issue."""
from __future__ import annotations
import os, pathlib, sys, torch
os.environ["HF_HOME"] = "/home/anna2/.cache/huggingface"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset as hf_load
import random
import llava_attn_patch as patch

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto").eval()
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
print(f"System prompt ends at: {img_start - 1}")

# Test 1: Baseline
print("\n=== Test 1: Baseline ===")
patch.patch_model(model, "baseline", 1.0)
patch.update_sample(img_start, img_end)
with torch.no_grad():
    out1 = model.generate(**inputs, max_new_tokens=5, do_sample=False)
print(f"Output: {processor.decode(out1[0], skip_special_tokens=True)}")

# Test 2: Image boost ONLY (no system suppression)
print("\n=== Test 2: Image boost alpha=1.0, sys_beta=0 ===")
patch.patch_model(model, "srf", 1.0)
patch._STATE["salience_mask"] = torch.ones(img_end - img_start + 1, dtype=torch.float32) * 0.5
patch._STATE["vaf_beta"] = 0.0  # Disable system suppression
patch.update_sample(img_start, img_end)
with torch.no_grad():
    out2 = model.generate(**inputs, max_new_tokens=5, do_sample=False)
print(f"Output: {processor.decode(out2[0], skip_special_tokens=True)}")

# Test 3: System suppression ONLY (no image boost)
print("\n=== Test 3: System suppression beta=0.1, no image boost ===")
patch.patch_model(model, "srf", 0.0)  # No image boost
patch._STATE["salience_mask"] = None
patch._STATE["vaf_beta"] = 0.1  # System suppression only
patch.update_sample(img_start, img_end)
with torch.no_grad():
    out3 = model.generate(**inputs, max_new_tokens=5, do_sample=False)
print(f"Output: {processor.decode(out3[0], skip_special_tokens=True)}")

# Check if they differ
if torch.equal(out1, out2):
    print("\n⚠️  Image boost alone produces same output as baseline")
else:
    print("\n✓ Image boost changes output")

if torch.equal(out1, out3):
    print("⚠️  System suppression alone produces same output as baseline")
else:
    print("✓ System suppression changes output")
