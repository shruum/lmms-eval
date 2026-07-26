#!/usr/bin/env python3
"""
Step-by-step SRF debugging test - runs full pipeline on small samples
"""
import sys
sys.path.insert(0, '/home/anna2/shruthi/lmms-eval')

import json
import torch
from PIL import Image
from srf.saliency import clip_salience as clip_sal

def test_clip_saliency_upsampling():
    """Test if CLIP saliency upsampling works correctly"""
    print("=" * 80)
    print("STEP 1: Test CLIP Saliency Upsampling")
    print("=" * 80)

    # Load a sample image
    image_path = "/home/anna2/shruthi/dataset/POPE_images/images/val2014/COCO_val2014_000000391895.jpg"
    try:
        image = Image.open(image_path)
        print(f"✅ Image loaded: {image.size}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return False

    # Test with sample noun
    noun = "dog"
    print(f"Testing with noun: '{noun}'")

    # Test WITHOUT upsampling (should give 36 elements)
    print("\n--- Test 1: WITHOUT upsampling ---")
    result_no_upsample = clip_sal.compute_clip_salience(
        image, noun,
        grid_h=6, grid_w=6,  # 6x6 = 36 elements
        top_k_pct=0.30,
        coarse_n=6,
        target_n_tokens=None  # No upsampling
    )
    print(f"Saliency shape: {result_no_upsample.saliency.shape}")
    print(f"Mask shape: {result_no_upsample.mask.shape}")
    print(f"Expected: torch.Size([36])")

    # Test WITH upsampling (should give 576 elements)
    print("\n--- Test 2: WITH upsampling to 576 tokens ---")
    result_with_upsample = clip_sal.compute_clip_salience(
        image, noun,
        grid_h=6, grid_w=6,  # 6x6 = 36 elements
        top_k_pct=0.30,
        coarse_n=6,
        target_n_tokens=576  # Upsample to 576
    )
    print(f"Saliency shape: {result_with_upsample.saliency.shape}")
    print(f"Mask shape: {result_with_upsample.mask.shape}")
    print(f"Expected: torch.Size([576])")

    # Verify upsampling worked
    if result_with_upsample.saliency.shape == torch.Size([576]):
        print("✅ Upsampling SUCCESS!")
        return True
    else:
        print(f"❌ Upsampling FAILED! Got {result_with_upsample.saliency.shape}")
        return False

def test_srf_image_token_detection():
    """Test if image token detection works correctly"""
    print("\n" + "=" * 80)
    print("STEP 2: Test Image Token Detection")
    print("=" * 80)

    # Simulate LLaVA input with placeholder token
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # Create sample input (like LLaVA processes)
    sample_text = "<image>\nIs there a dog in the image?"
    input_ids = tokenizer(sample_text, return_tensors="pt").input_ids[0]

    print(f"Input IDs length: {len(input_ids)}")
    print(f"Input IDs: {input_ids[:20]}...")

    # Find image placeholder token
    img_token_id = tokenizer.convert_tokens_to_ids("<image>")
    print(f"Image token ID: {img_token_id}")

    # Find placeholder position
    try:
        img_start = next(i for i, t in enumerate(input_ids) if t == img_token_id)
        print(f"Placeholder token position: {img_start}")
    except StopIteration:
        print("❌ No <image> token found!")
        return False

    # The issue: placeholder is at position 35, but actual image tokens are 576
    print(f"⚠️  Placeholder at {img_start}, but LLaVA expands to 576 image tokens internally")
    print(f"⚠️  Actual image token range should be: [{img_start}, {img_start + 576}] = [{img_start}, {img_start + 576}]")

    return True

def test_dimension_check_logic():
    """Test the dimension check that's failing in attention patch"""
    print("\n" + "=" * 80)
    print("STEP 3: Test Dimension Check Logic")
    print("=" * 80)

    # Simulate the condition from llava_attn_patch.py line 137
    img_start = 35
    img_end = 610  # Should be 35 + 576 - 1 = 610

    # WITHOUT upsampling (36 elements)
    saliency_36 = torch.randn(36)
    check_36 = saliency_36.numel() == (img_end - img_start + 1)
    print(f"Check with 36 elements: {saliency_36.numel()} == {img_end - img_start + 1} = {check_36}")
    if not check_36:
        print("❌ Dimension check FAILS with 36 elements → Falls back to uniform enhancement")

    # WITH upsampling (576 elements)
    saliency_576 = torch.randn(576)
    check_576 = saliency_576.numel() == (img_end - img_start + 1)
    print(f"Check with 576 elements: {saliency_576.numel()} == {img_end - img_start + 1} = {check_576}")
    if check_576:
        print("✅ Dimension check PASSES with 576 elements → Uses CLIP saliency correctly")

    return check_576

def test_config_parameter_flow():
    """Test if config parameters are being read correctly"""
    print("\n" + "=" * 80)
    print("STEP 4: Test Config Parameter Flow")
    print("=" * 80)

    from srf.config import SRF_CONFIG

    # Check if upsampling is enabled in config
    upsampling_enabled = SRF_CONFIG["llava-hf/llava-1.5-7b-hf"]["saliency"]["clip_upsample_to_tokens"]
    print(f"clip_upsample_to_tokens in config: {upsampling_enabled}")

    if upsampling_enabled:
        print("✅ Upsampling is ENABLED in config")
    else:
        print("❌ Upsampling is DISABLED in config - THIS IS THE BUG!")

    return upsampling_enabled

def main():
    """Run all tests"""
    print("\n" + "🔍 " * 40)
    print("SRF STEP-BY-STEP DEBUGGING TEST")
    print("🔍 " * 40 + "\n")

    results = {}

    # Run all tests
    results['clip_saliency'] = test_clip_saliency_upsampling()
    results['token_detection'] = test_srf_image_token_detection()
    results['dimension_check'] = test_dimension_check_logic()
    results['config_flow'] = test_config_parameter_flow()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✅ ALL TESTS PASSED - SRF should work correctly!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - SRF has bugs that need fixing!")
        return 1

if __name__ == "__main__":
    exit(main())