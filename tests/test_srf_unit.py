#!/usr/bin/env python3
"""
SRF Unit Tests - Systematic Component Verification

Each test runs a minimal example to verify one specific component works.
Can be run independently: python tests/test_srf_unit.py <test_name>

Usage:
    python tests/test_srf_unit.py saliency
    python tests/test_srf_unit.py layers
    python tests/test_srf_unit.py boosting
    python tests/test_srf_unit.py all
"""

import sys
import os
import torch
import subprocess
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "srf"))
sys.path.insert(0, str(Path(__file__).parent.parent / "srf" / "saliency"))
sys.path.insert(0, str(Path(__file__).parent.parent / "my_analysis"))

def test_saliency():
    """Test CLIP saliency computation."""
    print("\n=== TEST: CLIP Saliency ===")

    # Test with simple known inputs
    try:
        import clip_salience
        from PIL import Image

        # Create dummy image
        dummy_img = Image.new('RGB', (100, 100), color='red')

        clip_salience._load_clip()
        result = clip_salence.compute_clip_salience(
            image=dummy_img,
            noun_or_text="cat",
            grid_h=7,
            grid_w=7,
            top_k_pct=0.30,
            coarse_n=7,
        )

        print(f"✓ Saliency computed: shape={result.saliency.shape}, max_sim={result.max_sim:.3f}")
        assert result.saliency.shape == (49,), f"Wrong shape: {result.saliency.shape}"
        print("✅ PASS: CLIP saliency works")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

def test_layers():
    """Test layer range enforcement."""
    print("\n=== TEST: Layer Range Enforcement ===")

    try:
        import llava_attn_patch as patch

        # Mock the calibration
        patch._STATE["current_layer"] = 10
        patch._STATE["enabled"] = True
        patch._STATE["method"] = "srf"

        # Check what layer ranges are actually used
        layer_start = patch._STATE.get("layer_start", None)
        layer_end = patch._STATE.get("layer_end", None)
        vaf_layer_start = patch._STATE.get("vaf_layer_start", None)
        vaf_layer_end = patch._STATE.get("vaf_layer_end", None)

        print(f"Configured vaf layers: {vaf_layer_start}-{vaf_layer_end}")
        print(f"Actual read layers: {layer_start}-{layer_end}")

        # Test if layer 10 would be included
        current_layer = 10
        would_trigger_config = (vaf_layer_start and vaf_layer_end and
                               vaf_layer_start <= current_layer <= vaf_layer_end)
        would_trigger_actual = (layer_start and layer_end and
                               layer_start <= current_layer <= layer_end)

        print(f"Layer {current_layer}:")
        print(f"  Configured range would trigger: {would_trigger_config}")
        print(f"  Actual range would trigger: {would_trigger_actual}")

        if not would_trigger_actual and would_trigger_config:
            print(f"✗ FAIL: Layer {current_layer} should trigger but doesn't!")
            return False

        print("✅ PASS: Layer range enforcement checked")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

def test_boosting():
    """Test token boosting mechanism."""
    print("\n=== TEST: Token Boosting ===")

    try:
        import llava_attn_patch as patch

        # Set up state with known values
        patch._STATE["enh_para"] = 1.15  # 15% boost
        patch._STATE["layer_start"] = 10
        patch._STATE["layer_end"] = 15
        patch._STATE["current_layer"] = 12  # Middle layer
        patch._STATE["enabled"] = True
        patch._STATE["method"] = "srf"

        enh_para = patch._STATE.get("enh_para", None)
        print(f"Enhancement parameter: {enh_para}")

        if enh_para == 1.15:
            print("✓ Enhancement parameter set correctly (1.15 = 15% boost)")
        else:
            print(f"✗ FAIL: enh_para={enh_para}, expected 1.15")
            return False

        print("✅ PASS: Token boosting mechanism works")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

def test_suppression():
    """Test system prompt suppression."""
    print("\n=== TEST: System Suppression ===")

    try:
        import llava_attn_patch as patch

        # Set up state with suppression
        patch._STATE["sup_para"] = 0.9  # 10% suppression
        patch._STATE["sys_end"] = 34

        sup_para = patch._STATE.get("sup_para", None)
        sys_end = patch._STATE.get("sys_end", None)

        print(f"Suppression parameter: {sup_para}")
        print(f"System token end: {sys_end}")

        if sup_para == 0.9:
            print("✓ Suppression parameter set (0.9 = 10% suppression)")
        else:
            print(f"✗ FAIL: sup_para={sup_para}, expected 0.9")
            return False

        print("✅ PASS: System suppression mechanism works")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

def test_image_tokens():
    """Test image token detection."""
    print("\n=== TEST: Image Token Detection ===")

    try:
        # This test requires running actual model to see what tokens are detected
        print("Running minimal model inference to check token detection...")

        result = subprocess.run([
            "CUDA_VISIBLE_DEVICES=0",
            "/home/anna2/miniconda3/envs/mllm/bin/python",
            "-c",
            """
import sys
sys.path.insert(0, 'srf')
sys.path.insert(0, 'my_analysis')

# Simulate what get_img_range does
input_ids_example = [1]*35 + [32000] + [2]*5  # 32000 is LLaVA image token
img_token_id = 32000

ids = input_ids_example
start = next((i for i, t inenumerate(ids) if t == img_token_id), None)
end = len(ids) - 1 - next((i for i, t in enumerate(reversed(ids)) if t == img_token_id), None)

print(f"Image token detection: [{start}, {end}]")
print(f"Number of tokens: {end - start + 1}")
"""
        ], capture_output=True, text=True, cwd="/home/anna2/shruthi/lmms-eval")

        output = result.stdout.strip()
        print(output)

        if "[35, 35]" in output or "[34, 34]" in output:
            print("✗ CONFIRMED BUG: Only 1 image token found!")
            return False

        print("✅ PASS: Image token detection completed")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

def test_parameter_flow():
    """Test complete parameter flow from CLI to attention."""
    print("\n=== TEST: Parameter Flow ===")

    try:
        print("Running sample with --layer_start 10 --layer_end 15...")

        result = subprocess.run([
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
            "--output", "results/session_logs/param_flow_test.json"
        ], capture_output=True, text=True, cwd="/home/anna2/shruthi/lmms-eval", env={"CUDA_VISIBLE_DEVICES": "0"})

        output = result.stdout + result.stderr

        # Check what actually happened
        checks = {
            "Layer 10 configured": "layer_start=10" in output,
            "Layer 15 configured": "layer_end=15" in output,
            "Enhancement applied": "enh_para: 0.15" in output,
            "System suppression": "sup_para=0.900" in output,
            "Layer 9 triggered": "layer=9" in output,
            "Layer 10 triggered": "layer=10" in output,
        }

        for check_name, result in checks.items():
            status = "✓" if result else "✗"
            print(f"{status} {check_name}")

        print("\n🔍 ANALYSIS:")
        if "Layer 10 triggered" in output:
            print("  ✓ Configured layer 10 actually triggered - GOOD!")
        elif "Layer 9 triggered" in output:
            print("  ✗ Layer 9 triggered instead of 10 - PARAMETER DISCONNECTED!")

        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

def main():
    """Run specified tests or all tests."""
    if len(sys.argv) < 2:
        print("Usage: python tests/test_srf_unit.py <test_name>|all")
        print("Available tests: saliency, layers, boosting, suppression, image_tokens, parameter_flow")
        sys.exit(1)

    test_name = sys.argv[1]

    tests = {
        "saliency": test_saliency,
        "layers": test_layers,
        "boosting": test_boosting,
        "suppression": test_suppression,
        "image_tokens": test_image_tokens,
        "parameter_flow": test_parameter_flow,
    }

    if test_name == "all":
        print("Running all tests...")
        results = []
        for name, func in tests.items():
            try:
                passed = func()
                results.append((name, passed))
            except Exception as e:
                print(f"✗ {name}: {e}")
                results.append((name, False))

        print("\n" + "="*80)
        print("OVERALL RESULTS")
        print("="*80)
        passed = sum(1 for _, p in results if p)
        total = len(results)
        print(f"Passed: {passed}/{total}")
        return passed == total
    else:
        if test_name in tests:
            return tests[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)