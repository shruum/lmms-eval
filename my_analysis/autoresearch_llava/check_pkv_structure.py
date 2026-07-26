#!/usr/bin/env python3
"""Check past_key_values structure."""
from __future__ import annotations
import os, pathlib, sys, torch
os.environ["HF_HOME"] = "/home/anna2/.cache/huggingface"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Check layer 9 attention
lm = model.language_model
if hasattr(lm, 'model'):
    layers = lm.model.layers
elif hasattr(lm, 'layers'):
    layers = lm.layers
else:
    raise AttributeError("Cannot find layers")

attn = layers[9].self_attn

print(f"Attention type: {type(attn)}")
print(f"Attention forward signature: {attn.forward.__code__.co_varnames[:attn.forward.__code__.co_argcount]}")

# Create a simple test
inputs = processor(text="Hello", return_tensors="pt").to(model.device)
outputs = model.model(**inputs, output_attentions=True)

if outputs.past_key_values is not None:
    print(f"\nPast key values type: {type(outputs.past_key_values)}")
    print(f"Length: {len(outputs.past_key_values)}")
    print(f"First element type: {type(outputs.past_key_values[0])}")
    if isinstance(outputs.past_key_values[0], (list, tuple)):
        print(f"First element length: {len(outputs.past_key_values[0])}")
        print(f"First element[0] shape: {outputs.past_key_values[0][0].shape}")
        print(f"First element[1] shape: {outputs.past_key_values[0][1].shape}")
