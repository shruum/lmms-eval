#!/usr/bin/env python3
"""
Direct component test - verify each SRF component individually
"""
import sys
sys.path.insert(0, '/home/anna2/shruthi/lmms-eval')

print("🔍 Direct Component Testing - SRF Verification")
print("=" * 80)

# Test 1: Config reading
print("\n1️⃣ Testing config reading...")
try:
    from srf.config import SRF_ARCH_PARAMS
    llava_config = SRF_ARCH_PARAMS['llava-hf/llava-1.5-7b-hf']
    upsampling = llava_config.get('clip_upsample_to_tokens', False)
    print(f"   ✅ Config loaded: clip_upsample_to_tokens = {upsampling}")
    if upsampling:
        print("   ✅ Upsampling ENABLED in config")
    else:
        print("   ❌ Upsampling DISABLED in config")
except Exception as e:
    print(f"   ❌ Config test failed: {e}")

# Test 2: _make_saliency function
print("\n2️⃣ Testing _make_saliency function...")
try:
    # Setup minimal environment
    from srf.srf import _make_saliency

    # Mock model setup
    import torch
    class MockModel:
        class config:
            _name_or_path = 'llava-hf/llava-1.5-7b-hf'

    # Initialize minimal state
    from srf import srf
    srf._model_id = 'llava-hf/llava-1.5-7b-hf'

    result = _make_saliency(overrides={})
    upsampling = result['clip_upsample_to_tokens']
    print(f"   ✅ _make_saliency result: clip_upsample_to_tokens = {upsampling}")

    if upsampling:
        print("   ✅ Upsampling parameter passed through correctly")
    else:
        print("   ❌ Upsampling parameter NOT passed through (BUG)")

except Exception as e:
    print(f"   ❌ _make_saliency test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: CLIP upsampling directly
print("\n3️⃣ Testing CLIP upsampling directly...")
try:
    from srf.saliency import clip_salience as clip_sal
    from PIL import Image
    import numpy as np

    # Create dummy image
    arr = np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
    image = Image.fromarray(arr)

    # Test without upsampling
    result_no_up = clip_sal.compute_clip_salience(
        image, "dog", grid_h=6, grid_w=6, target_n_tokens=None
    )
    print(f"   Without upsampling: saliency.shape = {result_no_up.saliency.shape}")

    # Test with upsampling
    result_with_up = clip_sal.compute_clip_salience(
        image, "dog", grid_h=6, grid_w=6, target_n_tokens=576
    )
    print(f"   With upsampling: saliency.shape = {result_with_up.saliency.shape}")

    if result_with_up.saliency.shape[0] == 576:
        print("   ✅ Upsampling works correctly (576 elements)")
    else:
        print(f"   ❌ Upsampling failed ({result_with_up.saliency.shape[0]} elements)")

except Exception as e:
    print(f"   ❌ CLIP upsampling test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Image token detection
print("\n4️⃣ Testing image token detection...")
try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
    sample_text = "<image>\nIs there a dog in the image?"
    input_ids = tokenizer(sample_text, return_tensors="pt").input_ids[0]

    # Find image placeholder
    img_token_id = tokenizer.convert_tokens_to_ids("<image>")
    placeholder_pos = None
    for i, token_id in enumerate(input_ids):
        if token_id == img_token_id:
            placeholder_pos = i
            break

    if placeholder_pos is not None:
        print(f"   ✅ Placeholder found at position {placeholder_pos}")

        # Test our fixed function
        from srf.eval import get_img_range
        start, end = get_img_range(input_ids[0:1], img_token_id)
        print(f"   ✅ get_img_range returns: [{start}, {end}]")

        if end - start == 575:  # 576 tokens
            print("   ✅ Correct token range (576 tokens)")
        else:
            print(f"   ❌ Wrong token range ({end - start + 1} tokens)")
    else:
        print("   ❌ Could not find image placeholder")

except Exception as e:
    print(f"   ❌ Token detection test failed: {e}")

print("\n" + "=" * 80)
print("🎯 Component testing complete!")
print("\nIf all tests show ✅, then SRF is working correctly.")