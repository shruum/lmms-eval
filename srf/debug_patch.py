#!/usr/bin/env python3
"""
Comprehensive debug script to check if SRF patch is working.
"""
import os
import sys
import torch
sys.path.insert(0, 'my_analysis')

print("=" * 60)
print("SRF Patch Debug Test")
print("=" * 60)

# Import and patch
import llava_attn_patch as patch

print("\n1. Checking initial patch state:")
print(f"   enabled: {patch._STATE.get('enabled')}")
print(f"   method: {patch._STATE.get('method')}")
print(f"   enh_para: {patch._STATE.get('enh_para')}")
print(f"   Hooks: {len(patch._HOOKS)}")
print(f"   Original softmax: {patch._ORIGINAL_SOFTMAX}")

print("\n2. Loading LLaVA model...")
from transformers import LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # Force eager
).eval()

model.cuda()

print("\n3. Patching model...")
patch.patch_model(model, "srf", enh_para=2.0)

print(f"\n4. After patching:")
print(f"   enabled: {patch._STATE.get('enabled')}")
print(f"   method: {patch._STATE.get('method')}")
print(f"   enh_para: {patch._STATE.get('enh_para')}")
print(f"   Hooks: {len(patch._HOOKS)}")
print(f"   Original softmax: {patch._ORIGINAL_SOFTMAX}")
print(f"   torch.nn.functional.softmax: {torch.nn.functional.softmax}")

print("\n5. Testing if patch is active...")
# Create test input
test_input = torch.randn(1, 8, 32, 32).cuda()  # (batch, heads, q_len, kv_len)

# Set up state for a target layer
patch._STATE.update({
    "img_start": 10,
    "img_end": 20,
    "sys_end": 5,
    "current_layer": 12,  # Within layer_start=9, layer_end=14 range
    "in_language_model": True,
})

print(f"   State set: layer=12, in_language_model=True")

# Call softmax
result = torch.nn.functional.softmax(test_input, dim=-1)

print(f"   Softmax called successfully")
print(f"   Result shape: {result.shape}")

# Check if the patched function was called
if patch._ORIGINAL_SOFTMAX is not None:
    original_softmax = patch._ORIGINAL_SOFTMAX
    print(f"   Patch was installed (original softmax saved)")

print("\n6. Running actual inference test...")
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Create a simple test case
from PIL import Image
import numpy as np

# Create dummy image
img = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))

msg = [{"role": "user", "content": [{"type": "image", "image": img},
                                   {"type": "text", "text": "Is there a dog? Answer with Yes or No."}]}]

# Format prompt (LLaVA-1.5 Vicuna format)
user_text = "<image> Is there a dog? Answer with Yes or No."
prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {user_text} ASSISTANT:"

inputs = processor(text=[prompt], images=[img], return_tensors="pt", padding=True).to(model.device)

print(f"   Input shape: {inputs['input_ids'].shape}")
print(f"   Image tokens in input: {(inputs['input_ids'] == model.config.image_token_index).sum().item()}")

# Reset patch state for inference
patch.update_sample(100, 200)
patch._STATE["salience_mask"] = torch.ones(101).cuda() * 0.5  # Dummy saliency

print(f"\n7. Before inference:")
print(f"   enh_para: {patch._STATE.get('enh_para')}")
print(f"   img_start: {patch._STATE.get('img_start')}")
print(f"   img_end: {patch._STATE.get('img_end')}")
print(f"   salience_mask: {patch._STATE.get('salience_mask')}")

# Run inference
print(f"\n8. Running inference...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=5)

print(f"   Output: {processor.decode(outputs[0], skip_special_tokens=True)}")

print(f"\n9. After inference:")
print(f"   Patch still enabled: {patch._STATE.get('enabled')}")
print(f"   Hooks still registered: {len(patch._HOOKS)}")

print("\n" + "=" * 60)
print("Debug test complete!")
print("=" * 60)
