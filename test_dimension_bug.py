#!/usr/bin/env python3
"""
Direct test of CLIP saliency upsampling in the actual SRF pipeline
"""
import sys
sys.path.insert(0, '/home/anna2/shruthi/lmms-eval')

from srf.saliency import clip_salience as clip_sal
from PIL import Image
import torch

# Test CLIP saliency upsampling directly
print("Testing CLIP saliency upsampling...")

# Create dummy image (or load real one if available)
try:
    image = Image.open("/home/anna2/shruthi/dataset/POPE_images/images/val2014/COCO_val2014_000000312192.jpg")
    print(f"✅ Image loaded: {image.size}")
except:
    # Create dummy image
    from PIL import Image
    import numpy as np
    arr = np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
    image = Image.fromarray(arr)
    print(f"Created dummy image: {image.size}")

noun = "dog"
print(f"Testing with noun: '{noun}'")

# Test 1: WITHOUT upsampling (baseline)
print("\n--- WITHOUT upsampling (should give 36 elements) ---")
result_no_upsample = clip_sal.compute_clip_salience(
    image, noun,
    grid_h=6, grid_w=6,  # 6x6 = 36 elements
    top_k_pct=0.30,
    coarse_n=6,
    target_n_tokens=None  # No upsampling
)
print(f"Saliency shape: {result_no_upsample.saliency.shape}")
print(f"Mask shape: {result_no_upsample.mask.shape}")

# Test 2: WITH upsampling to 576 (should work)
print("\n--- WITH upsampling to 576 tokens (should give 576 elements) ---")
result_with_upsample = clip_sal.compute_clip_salience(
    image, noun,
    grid_h=6, grid_w=6,  # 6x6 = 36 elements
    top_k_pct=0.30,
    coarse_n=6,
    target_n_tokens=576  # Upsample to 576
)
print(f"Saliency shape: {result_with_upsample.saliency.shape}")
print(f"Mask shape: {result_with_upsample.mask.shape}")

# Test 3: Simulate the dimension check from attention patch
print("\n--- Simulating dimension check from attention patch ---")
img_start = 35
img_end = 610  # 35 + 576 - 1

# Check without upsampling
check_no_upsample = result_no_upsample.saliency.numel() == (img_end - img_start + 1)
print(f"Without upsampling: {result_no_upsample.saliency.numel()} == {img_end - img_start + 1} = {check_no_upsample}")

# Check with upsampling
check_with_upsample = result_with_upsample.saliency.numel() == (img_end - img_start + 1)
print(f"With upsampling: {result_with_upsample.saliency.numel()} == {img_end - img_start + 1} = {check_with_upsample}")

if check_with_upsample:
    print("✅ DIMENSION CHECK PASSES - CLIP saliency will be used!")
else:
    print("❌ DIMENSION CHECK FAILS - Will fall back to uniform enhancement")

# Verify config
print("\n--- Checking config settings ---")
from srf.config import SRF_CONFIG
upsampling_enabled = SRF_CONFIG["llava-hf/llava-1.5-7b-hf"]["saliency"]["clip_upsample_to_tokens"]
print(f"clip_upsample_to_tokens in config: {upsampling_enabled}")