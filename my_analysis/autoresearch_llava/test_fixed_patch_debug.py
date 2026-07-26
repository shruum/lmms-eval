#!/usr/bin/env python3
"""Debug test to verify patch is actually being called."""
from __future__ import annotations
import os, pathlib, sys, torch
os.environ["HF_HOME"] = "/home/anna2/.cache/huggingface"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset as hf_load
import random

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

# Import patch AFTER model is loaded
import llava_attn_patch_fixed as patch

img_start, img_end = patch.get_image_token_range(inputs, model)
print(f"Image tokens: [{img_start}, {img_end}] = {img_end - img_start + 1} tokens")

# Check attention implementation
print(f"\nModel attention implementation: {model.config._attn_implementation}")
print(f"Language model type: {type(model.language_model)}")

# Check layers
lm = model.language_model
if hasattr(lm, 'model'):
    layers = lm.model.layers
elif hasattr(lm, 'layers'):
    layers = lm.layers
else:
    raise AttributeError("Cannot find layers")

print(f"Number of layers: {len(layers)}")
print(f"Layer 9 self_attn type: {type(layers[9].self_attn)}")

# Patch with debug logging
print("\n=== Patching model ===")
patch.patch_model(model, "srf", 1.5, 0.9)
patch.update_sample(img_start, img_end)

print(f"Patch enabled: {patch._STATE['enabled']}")
print(f"Patch method: {patch._STATE['method']}")
print(f"Enhancement para: {patch._STATE['enh_para']}")
print(f"Suppression para: {patch._STATE['sup_para']}")
print(f"Layer range: {patch._STATE['layer_start']}-{patch._STATE['layer_end']}")

# Now generate with debug counter
call_count = [0]
original_softmax = torch.nn.functional.softmax

def counting_softmax(*args, **kwargs):
    call_count[0] += 1
    return original_softmax(*args, **kwargs)

torch.nn.functional.softmax = counting_softmax

print("\n=== Generating ===")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=3, do_sample=False)

print(f"\nTotal softmax calls: {call_count[0]}")
print(f"Output: {processor.decode(out[0], skip_special_tokens=True)}")

patch.unpatch_model(model)
