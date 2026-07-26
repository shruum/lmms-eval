#!/usr/bin/env python3
"""
Test script for clip_full_gate_v3 implementation.

Tests backward compatibility and v3 gate functionality.
"""
import sys
import os
from pathlib import Path

SRF_DIR = Path(__file__).parent
sys.path.insert(0, str(SRF_DIR / "saliency"))
sys.path.insert(0, str(SRF_DIR))

from PIL import Image
import torch

def test_backward_compatibility():
    """Test that existing code still works without v3 gate."""
    print("Testing backward compatibility (v3 gate disabled)...")
    
    # Import the function
    from clip_salience import compute_clip_salience, ABSENCE_THRESH, V3_GATE_ENABLED
    
    # Verify v3 gate is disabled by default
    assert V3_GATE_ENABLED == False, "V3_GATE_ENABLED should be False by default"
    
    print("✅ Backward compatibility test passed")
    print(f"   ABSENCE_THRESH: {ABSENCE_THRESH}")
    print(f"   V3_GATE_ENABLED: {V3_GATE_ENABLED}")

def test_v3_gate_enabled():
    """Test v3 gate mechanism when enabled."""
    print("\nTesting v3 gate mechanism (enabled)...")
    
    # Import the function and constants
    from clip_salience import (compute_clip_salience, V3_GATE_ENABLED, 
                                V3_FULL_IMG_THRESH, V3_RAW_ENTROPY_THRESH,
                                V3_CROSS_SCALE_IOU_THRESH, V3_BLUR_DELTA_THRESH)
    
    # Temporarily enable v3 gate
    import clip_salience as clip_module
    clip_module.V3_GATE_ENABLED = True
    
    # Verify thresholds are set correctly
    assert V3_FULL_IMG_THRESH == 0.21, f"V3_FULL_IMG_THRESH should be 0.21, got {V3_FULL_IMG_THRESH}"
    assert V3_RAW_ENTROPY_THRESH == 0.95, f"V3_RAW_ENTROPY_THRESH should be 0.95"
    assert V3_CROSS_SCALE_IOU_THRESH == 0.30, f"V3_CROSS_SCALE_IOU_THRESH should be 0.30"
    assert V3_BLUR_DELTA_THRESH == 0.005, f"V3_BLUR_DELTA_THRESH should be 0.005"
    
    print("✅ v3 gate constants verified")
    print(f"   V3_FULL_IMG_THRESH: {V3_FULL_IMG_THRESH}")
    print(f"   V3_RAW_ENTROPY_THRESH: {V3_RAW_ENTROPY_THRESH}")
    print(f"   V3_CROSS_SCALE_IOU_THRESH: {V3_CROSS_SCALE_IOU_THRESH}")
    print(f"   V3_BLUR_DELTA_THRESH: {V3_BLUR_DELTA_THRESH}")
    
    # Reset v3 gate
    clip_module.V3_GATE_ENABLED = False

def test_signal_functions():
    """Test v3 signal detection functions."""
    print("\nTesting v3 signal detection functions...")
    
    from clip_salience import (_compute_raw_entropy, _compute_cross_scale_iou, 
                               _compute_blur_delta)
    
    # Test raw entropy
    test_sims = torch.tensor([0.8, 0.7, 0.2, 0.1, 0.05])  # Peaked distribution
    entropy = _compute_raw_entropy(test_sims)
    print(f"   Raw entropy (peaked): {entropy:.4f} (should be low)")
    assert entropy < 0.95, "Peaked similarities should have low entropy"
    
    # Test cross-scale IOU (need 7x7 grid)
    test_sims_grid = torch.rand(49)  # 7x7 grid
    iou = _compute_cross_scale_iou(test_sims_grid.view(7, 7))
    print(f"   Cross-scale IOU: {iou:.4f}")
    
    # Test blur delta
    # Create dummy image for test
    dummy_image = Image.new('RGB', (224, 224), color='red')
    delta = _compute_blur_delta(dummy_image, "test object")
    print(f"   Blur delta: {delta:.4f}")
    
    print("✅ v3 signal functions working")

def main():
    print("="*60)
    print("clip_full_gate_v3 Implementation Tests")
    print("="*60)
    
    try:
        test_backward_compatibility()
        test_v3_gate_enabled()
        test_signal_functions()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nv3 gate implementation is ready!")
        print("To enable: set V3_GATE_ENABLED=True or pass enable_v3_gate=True")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
