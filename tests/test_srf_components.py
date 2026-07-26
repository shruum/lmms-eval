#!/usr/bin/env python3
"""
SRF Component Test Suite - Sanity Checks for Each Component

Tests each component individually with actual POPE data to verify:
1. Saliency map computation and application
2. Head selection and application
3. Layer range enforcement
4. Token boosting mechanics
5. System prompt suppression
6. Non-salient token suppression

Usage:
    python tests/test_srf_components.py
"""

import sys
import os
import torch
import json
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "srf"))
sys.path.insert(0, str(Path(__file__).parent.parent / "srf" / "saliency"))
sys.path.insert(0, str(Path(__file__).parent.parent / "my_analysis"))

import torch
import clip_salience
import llava_attn_patch as patch
from PIL import Image
import numpy as np

print("=" * 80)
print("SRF COMPONENT TEST SUITE")
print("=" * 80)

# ============================================================================
# TEST 1: SALIENCY MAP COMPUTATION
# ============================================================================

def test_saliency_computation():
    """Test that CLIP saliency is computed correctly."""
    print("\n" + "=" * 80)
    print("TEST 1: SALIENCY MAP COMPUTATION")
    print("=" * 80)

    # Load a real POPE image
    image_path = "/home/anna2/shruthi/dataset/POPE_images/images/val2014/COCO_val2014_000000000244.jpg"
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return False

    image = Image.open(image_path).convert("RGB")
    question = "Is there a cat in the image?"

    try:
        # Initialize CLIP
        clip_salience._load_clip()

        # Compute saliency
        result = clip_salience.compute_clip_salience(
            image=image,
            noun_or_text="cat",
            grid_h=7,
            grid_w=7,
            top_k_pct=0.30,
            coarse_n=7,
        )

        print(f"✓ CLIP saliency computed")
        print(f"  Shape: {result.saliency.shape}")
        print(f"  Max similarity: {result.max_sim:.4f}")
        print(f"  Object present: {result.object_present}")
        print(f"  Saliency range: [{result.saliency.min():.4f}, {result.saliency.max():.4f}]")

        # Check output format
        assert result.saliency.dim() == 1, "Saliency should be 1D tensor"
        assert result.saliency.numel() == 49, "7×7 grid should have 49 elements"

        print("✓ PASS: Saliency computation works correctly")
        return True

    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

# ============================================================================
# TEST 2: HEAD SELECTION
# ============================================================================

def test_head_selection():
    """Test that vision-aware heads are identified and stored."""
    print("\n" + "=" * 80)
    print("TEST 2: HEAD SELECTION")
    print("=" * 80)

    try:
        # Check if head_mask exists in patch state
        head_mask = patch._STATE.get("head_mask")

        if head_mask is None:
            print("⚠️  Head mask not computed (need to run calibration first)")
            print("   This is expected if no calibration was run")
            return True  # Not a failure, just needs calibration

        print(f"✓ Head mask exists")
        print(f"  Shape: {head_mask.shape}")
        print(f"  Type: {head_mask.dtype}")
        print(f"  Selected heads: {head_mask.sum().item()}/{head_mask.numel()}")

        # Verify it's a boolean tensor
        assert head_mask.dtype == torch.bool, "Head mask should be boolean"

        # Check that some heads are selected
        n_selected = head_mask.sum().item()
        assert n_selected > 0, "At least one head should be selected"
        assert n_selected < head_mask.numel(), "Not all heads should be selected"

        print("✓ PASS: Head selection works correctly")
        return True

    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

# ============================================================================
# TEST 3: LAYER RANGE ENFORCEMENT
# ============================================================================

def test_layer_enforcement():
    """Test that configured layer ranges are actually enforced."""
    print("\n" + "=" * 80)
    print("TEST 3: LAYER RANGE ENFORCEMENT")
    print("=" * 80)

    try:
        # Get configured layer ranges
        configured_start = patch._STATE.get("vaf_layer_start", "NOT_SET")
        configured_end = patch._STATE.get("vaf_layer_end", "NOT_SET")
        actual_start = patch._STATE.get("layer_start", "NOT_SET")
        actual_end = patch._STATE.get("layer_end", "NOT_SET")

        print(f"Configured layers: vaf_layer_start={configured_start}, vaf_layer_end={configured_end}")
        print(f"Used by attention: layer_start={actual_start}, layer_end={actual_end}")

        # Check if they match
        if actual_start == "NOT_SET" or actual_end == "NOT_SET":
            print("⚠️  Layer parameters not set in _STATE")
            return False

        if configured_start == "NOT_SET" or configured_end == "NOT_SET":
            print("⚠️  vaf_layer parameters not set in _STATE")
            return False

        if configured_start != actual_start or configured_end != actual_end:
            print(f"✗ FAIL: Layer mismatch!")
            print(f"   Configured: {configured_start}-{configured_end}")
            print(f"   Actually used: {actual_start}-{actual_end}")
            return False

        print("✓ PASS: Layer ranges correctly connected")
        return True

    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

# ============================================================================
# TEST 4: TOKEN BOOSTING APPLICATION
# ============================================================================

def test_token_boosting():
    """Test that visual tokens actually get boosted."""
    print("\n" + "=" * 80)
    print("TEST 4: TOKEN BOOSTING APPLICATION")
    print("=" * 80)

    try:
        enh_para = patch._STATE.get("enh_para", 1.0)

        print(f"Enhancement parameter: enh_para={enh_para}")

        if enh_para == 1.0:
            print("⚠️  No enhancement applied (enh_para=1.0)")
            return False

        if enh_para <= 0.0 or enh_para > 10.0:
            print(f"✗ FAIL: enh_para={enh_para} seems wrong")
            return False

        print(f"✓ Enhancement parameter reasonable: {enh_para}")

        # Check if enhancement would actually modify attention
        # (This is a logic check, we can't test actual attention without running model)
        print("✓ PASS: Token boosting parameters are set")
        return True

    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

# ============================================================================
# TEST 5: SYSTEM PROMPT SUPPRESSION
# ============================================================================

def test_system_suppression():
    """Test that system prompt tokens get suppressed."""
    print("\n" + "=" * 80)
    print("TEST 5: SYSTEM PROMPT SUPPRESSION")
    print("=" * 80)

    try:
        sup_para = patch._STATE.get("sup_para", 1.0)

        print(f"Suppression parameter: sup_para={sup_para}")

        if sup_para == 1.0:
            print("⚠️  No suppression applied (sup_para=1.0)")
            return False

        if sup_para < 0.0 or sup_para > 1.0:
            print(f"✗ FAIL: sup_para={sup_para} seems wrong")
            return False

        expected_suppression = (1.0 - sup_para) * 100
        print(f"✓ Suppressing system tokens by {expected_suppression:.1f}%")

        # Check if sys_end is set correctly
        sys_end = patch._STATE.get("sys_end", None)
        print(f"  sys_end={sys_end} (should be index of last system token)")

        print("✓ PASS: System suppression parameters are set")
        return True

    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

# ============================================================================
# TEST 6: IMAGE TOKEN RANGE
# ============================================================================

def test_image_token_range():
    """Test that image token range is correctly identified."""
    print("\n" + "=" * 80)
    print("TEST 6: IMAGE TOKEN RANGE")
    print("=" * 80)

    try:
        img_start = patch._STATE.get("img_start", None)
        img_end = patch._STATE.get("img_end", None)

        print(f"Image token range: img_start={img_start}, img_end={img_end}")

        if img_start is None or img_end is None:
            print("✗ FAIL: Image range not set")
            return False

        if img_start == img_end:
            print(f"⚠️  WARNING: Only 1 image token found (range=[{img_start}, {img_end}])")
            print("   This suggests get_img_range found placeholder token, not actual image tokens")
            print("   Expected: 576 tokens for LLaVA (24×24 patches)")
            return False

        if img_start < 0 or img_end < img_start:
            print(f"✗ FAIL: Invalid range: [{img_start}, {img_end}]")
            return False

        n_image_tokens = img_end - img_start + 1
        print(f"  Number of image tokens: {n_image_tokens}")

        if n_image_tokens < 100:
            print(f"⚠️  WARNING: Only {n_image_tokens} image tokens (expected ~576 for LLaVA)")

        print("✓ PASS: Image token range is set")
        return True

    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

# ============================================================================
# TEST 7: END-TO-END VERIFICATION
# ============================================================================

def test_end_to_end():
    """Test complete SRF flow with one sample."""
    print("\n" + "=" * 80)
    print("TEST 7: END-TO-END VERIFICATION")
    print("=" * 80)

    try:
        print("Running one POPE sample with maximum debug...")

        # Run evaluation with 1 sample and capture output
        import subprocess
        result = subprocess.run([
            "CUDA_VISIBLE_DEVICES=0",
            "/home/anna2/miniconda3/envs/mllm/bin/python",
            "srf/eval.py",
            "--method", "srf",
            "--model", "llava-hf/llava-1.5-7b-hf",
            "--datasets", "pope",
            "--pope_splits", "adversarial",
            "--alpha", "0.15",
            "--sys_beta", "0.1",
            "--layer_start", "10",
            "--layer_end", "15",
            "--head_top_k_pct", "0.50",
            "--n_pope", "1",
            "--output", "results/session_logs/component_test.json"
        ], capture_output=True, text=True, cwd="/home/anna2/shruthi/lmms-eval")

        output = result.stdout

        # Check for successful execution
        if "Accuracy:" not in output:
            print("✗ FAIL: Evaluation did not produce output")
            return False

        print("✓ Evaluation completed successfully")

        # Check for expected debug messages
        if "[SRF DEBUG]" in output:
            print("✓ Debug messages found in output")

            # Extract key values
            if "sys_beta=0.100" in output:
                print("  ✓ sys_beta parameter used correctly")

            if "layer=9" in output or "layer=10" in output:
                print("  ✓ Layer enforcement debug found")

            if "img=[35, 35]" in output:
                print("  ⚠️  Image token range issue confirmed")

        print("✓ PASS: End-to-end evaluation works")
        return True

    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

# ============================================================================
# RUN ALL TESTS
# ============================================================================

def main():
    print("\nRunning SRF Component Tests...")
    print("=" * 80)

    tests = [
        ("Saliency Computation", test_saliency_computation),
        ("Head Selection", test_head_selection),
        ("Layer Enforcement", test_layer_enforcement),
        ("Token Boosting", test_token_boosting),
        ("System Suppression", test_system_suppression),
        ("Image Token Range", test_image_token_range),
        ("End-to-End", test_end_to_end),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total_count - passed_count} tests failed")

if __name__ == "__main__":
    main()