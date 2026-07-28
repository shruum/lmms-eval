#!/usr/bin/env python3
"""Check if adapter is called during generation."""
from __future__ import annotations
import os, pathlib, sys, torch
os.environ["HF_HOME"] = "/home/anna2/.cache/huggingface"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from transformers import AutoProcessor, LlavaForConditionalGeneration
import llava_attn_patch_working as patch

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Simple text-only input
inputs = processor(text="Hello, how are you?", return_tensors="pt").to(model.device)

# Patch and count calls
call_count = {"layer_9": 0}

original_forward = None

def counting_forward(self, *args, **kwargs):
    call_count["layer_9"] += 1
    q_len = args[0].shape[1] if len(args) > 0 else None
    print(f"  Layer 9 call #{call_count['layer_9']}: q_len={q_len}, method={patch._STATE.get('method', 'unknown')}")
    return original_forward(*args, **kwargs)

patch.patch_model(model, "srf", 10.0, 0.1, layer_start=9, layer_end=14)

# Get the language model layers
lm = model.language_model
if hasattr(lm, 'model'):
    layers = lm.model.layers
elif hasattr(lm, 'layers'):
    layers = lm.layers
else:
    raise AttributeError("Cannot find layers")

original_forward = layers[9].self_attn.original_attn.forward

# Monkey-patch the original to count calls
layers[9].self_attn.original_attn.forward = lambda *args, **kwargs: counting_forward(layers[9].self_attn.original_attn, *args, **kwargs)

print("\n=== Generating 5 tokens ===")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=5, do_sample=False)

print(f"\nTotal layer 9 calls: {call_count['layer_9']}")
print(f"Output: {processor.decode(output[0], skip_special_tokens=True)}")

patch.unpatch_model(model)
