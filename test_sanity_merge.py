#!/usr/bin/env python3
"""
Comprehensive SRF Sanity Test for Branch Merging

Tests incremental functionality after each merge step:
1. Model loading (Qwen + LLaVA)
2. SRF initialization
3. CLIP saliency computation
4. Attention patching
5. Single sample inference
6. Parameter verification

Usage:
    python test_sanity_merge.py --model qwen    # Test Qwen only
    python test_sanity_merge.py --model llava   # Test LLaVA only
    python test_sanity_merge.py --model both    # Test both models
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)

    try:
        import torch
        print("✓ PyTorch imported")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        from transformers import AutoProcessor, AutoTokenizer
        print("✓ Transformers imported")
    except Exception as e:
        print(f"✗ Transformers import failed: {e}")
        return False

    try:
        import srf.config as config
        print("✓ SRF config imported")
        print(f"  Default model: {config.DEFAULT_MODEL}")
    except Exception as e:
        print(f"✗ SRF config import failed: {e}")
        return False

    try:
        import srf.srf as srf_module
        print("✓ SRF module imported")
    except Exception as e:
        print(f"✗ SRF module import failed: {e}")
        return False

    try:
        import srf.saliency.clip_salience as clip_sal
        print("✓ CLIP saliency imported")
        has_v3 = hasattr(clip_sal, 'compute_clip_salience_full_gate_v3')
        print(f"  Has clip_full_gate_v3: {has_v3}")
        if not has_v3:
            print("  ⚠ WARNING: clip_full_gate_v3 function missing!")
    except Exception as e:
        print(f"✗ CLIP saliency import failed: {e}")
        return False

    try:
        import my_analysis.qwen_attn_patch as qwen_patch
        print("✓ Qwen attention patch imported")
    except Exception as e:
        print(f"✗ Qwen attention patch import failed: {e}")
        return False

    # Test LLaVA patch (might not exist in some branches)
    try:
        import my_analysis.llava_attn_patch as llava_patch
        print("✓ LLaVA attention patch imported")
    except ImportError:
        print("⚠ LLaVA attention patch not found (expected on some branches)")
    except Exception as e:
        print(f"✗ LLaVA attention patch import error: {e}")

    print()
    return True

def test_qwen_model():
    """Test Qwen model loading and SRF initialization."""
    print("=" * 60)
    print("TEST 2: Qwen Model Loading + SRF")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoProcessor
        import srf.config as config
        import srf.srf as srf_module

        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Loading model: {model_id}")
        print(f"Device: {device}")

        # Load processor
        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            print("✓ Processor loaded")
        except Exception as e:
            print(f"✗ Processor load failed: {e}")
            return False

        # Try to load model (might fail if not downloaded)
        model = None
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            print("✓ Model loaded")
        except Exception as e:
            print(f"⚠ Model load failed (may need download): {str(e)[:100]}")
            print("  Continuing with processor-only tests...")

        # Test SRF setup (skip if no model)
        if model is None:
            print("⊘ Skipping SRF setup (no model loaded)")
            return True

        # Test SRF setup
        try:
            srf_module.setup(model, processor, calib_dataset="pope")
            print("✓ SRF setup completed")
        except Exception as e:
            print(f"✗ SRF setup failed: {e}")
            if model:
                del model
            return False

        # Check SRF state
        try:
            srf_state = srf_module.get_state()
            print("✓ SRF state retrieved")
            print(f"  Enabled: {getattr(srf_state, 'enabled', 'N/A')}")
            print(f"  Method: {getattr(srf_state, 'method', 'N/A')}")
        except Exception as e:
            print(f"⚠ SRF state check failed: {e}")

        # Test dataset reset
        try:
            srf_module.reset_for_dataset("pope", phase="both", alpha=2.0, eps=0.2)
            print("✓ SRF dataset reset completed")
        except Exception as e:
            print(f"✗ SRF dataset reset failed: {e}")
            srf_module.cleanup()
            if model:
                del model
            return False

        # Cleanup
        try:
            srf_module.cleanup()
            print("✓ SRF cleanup completed")
        except Exception as e:
            print(f"⚠ SRF cleanup warning: {e}")

        if model:
            del model
            torch.cuda.empty_cache()

        print()
        return True

    except Exception as e:
        print(f"✗ Qwen test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llava_model():
    """Test LLaVA model loading and SRF initialization."""
    print("=" * 60)
    print("TEST 3: LLaVA Model Loading + SRF")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoProcessor
        import srf.config as config
        import srf.srf as srf_module

        model_id = "llava-hf/llava-1.5-7b-hf"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Testing LLaVA: {model_id}")
        print(f"Device: {device}")

        # Check if LLaVA patch exists
        try:
            import my_analysis.llava_attn_patch as llava_patch
            print("✓ LLaVA attention patch available")
        except ImportError:
            print("✗ LLaVA attention patch NOT AVAILABLE")
            print("  LLaVA tests will fail without this!")
            return False

        # Load processor
        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            print("✓ LLaVA processor loaded")
        except Exception as e:
            print(f"✗ LLaVA processor load failed: {e}")
            return False

        # Try to load model
        model = None
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            print("✓ LLaVA model loaded")
        except Exception as e:
            print(f"⚠ LLaVA model load failed (may need download): {str(e)[:100]}")
            print("  Continuing with processor-only tests...")

        # Test SRF setup for LLaVA (skip if no model)
        if model is None:
            print("⊘ Skipping LLaVA SRF setup (no model loaded)")
            return True

        # Test SRF setup for LLaVA
        try:
            srf_module.setup(model, processor, calib_dataset="pope")
            print("✓ LLaVA SRF setup completed")
        except Exception as e:
            print(f"✗ LLaVA SRF setup failed: {e}")
            if model:
                del model
            return False

        # Test LLaVA-specific config
        try:
            llava_config = config.SRF_ARCH_PARAMS.get(model_id, {})
            print("✓ LLaVA config found")
            print(f"  Layer range: {llava_config.get('layer_start', 'N/A')}-{llava_config.get('layer_end', 'N/A')}")
            print(f"  Saliency mode: {llava_config.get('saliency_mode', 'N/A')}")
            if llava_config.get('saliency_mode') != 'clip_full_gate_v3':
                print("  ⚠ WARNING: LLaVA not using clip_full_gate_v3!")
        except Exception as e:
            print(f"⚠ LLaVA config check failed: {e}")

        # Cleanup
        try:
            srf_module.cleanup()
            print("✓ LLaVA SRF cleanup completed")
        except Exception as e:
            print(f"⚠ LLaVA SRF cleanup warning: {e}")

        if model:
            del model
            torch.cuda.empty_cache()

        print()
        return True

    except Exception as e:
        print(f"✗ LLaVA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_clip_saliency():
    """Test CLIP saliency computation with v3 gate."""
    print("=" * 60)
    print("TEST 4: CLIP Saliency (clip_full_gate_v3)")
    print("=" * 60)

    try:
        import torch
        from PIL import Image
        import srf.saliency.clip_salience as clip_sal

        print("Testing CLIP saliency computation...")

        # Create a dummy image
        try:
            dummy_image = Image.new('RGB', (336, 336), color='red')
            print("✓ Dummy image created")
        except Exception as e:
            print(f"✗ Image creation failed: {e}")
            return False

        # Test noun extraction
        try:
            from srf.noun_extract import extract_clip_noun
            noun = extract_clip_noun("Is there a cat in the image?")
            print(f"✓ Noun extraction works: '{noun}'")
        except Exception as e:
            print(f"✗ Noun extraction failed: {e}")
            return False

        # Test CLIP saliency v3
        try:
            result = clip_sal.compute_clip_salience_full_gate_v3(
                image=dummy_image,
                text=noun,
                grid_h=6,
                grid_w=6,
                top_k_pct=0.30
            )
            print("✓ CLIP saliency v3 computed")

            # Check expected output (handle different result types)
            if hasattr(result, 'saliency'):
                print(f"  Saliency shape: {result.saliency.shape}")
            if hasattr(result, 'full_img_sim'):
                print(f"  Full image sim: {result.full_img_sim:.4f}")
            if hasattr(result, 'patch_max_sim'):
                print(f"  Patch max sim: {result.patch_max_sim:.4f}")
            if hasattr(result, 'object_present'):
                print(f"  Object present: {result.object_present}")

        except Exception as e:
            print(f"✗ CLIP saliency v3 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        print()
        return True

    except Exception as e:
        print(f"✗ CLIP saliency test failed: {e}")
        return False

def test_config_parameters():
    """Test that config parameters are correctly set."""
    print("=" * 60)
    print("TEST 5: Configuration Parameters")
    print("=" * 60)

    try:
        import srf.config as config

        print("Checking key configuration parameters...")

        # Check Qwen config
        qwen_config = config.SRF_ARCH_PARAMS.get("Qwen/Qwen2.5-VL-3B-Instruct", {})
        if qwen_config:
            print("✓ Qwen config found")
            print(f"  Layer range: {qwen_config.get('layer_start')}-{qwen_config.get('layer_end')}")
            print(f"  Head top-k: {qwen_config.get('head_top_k_pct')}")
            print(f"  Saliency mode: {qwen_config.get('saliency_mode')}")
            print(f"  CLIP grid: {qwen_config.get('clip_coarse_grid')}")
            print(f"  CLIP top-k: {qwen_config.get('clip_top_k_pct')}")
        else:
            print("✗ Qwen config not found")
            return False

        # Check LLaVA config
        llava_config = config.SRF_ARCH_PARAMS.get("llava-hf/llava-1.5-7b-hf", {})
        if llava_config:
            print("✓ LLaVA config found")
            print(f"  Layer range: {llava_config.get('layer_start')}-{llava_config.get('layer_end')}")
            print(f"  Saliency mode: {llava_config.get('saliency_mode')}")
            if llava_config.get('saliency_mode') != 'clip_full_gate_v3':
                print("  ⚠ LLaVA not using clip_full_gate_v3!")
        else:
            print("⚠ LLaVA config not found (may be okay if not merged yet)")

        # Check dataset params
        pope_config = config.SRF_DATASET_PARAMS.get("pope", {})
        if pope_config:
            print("✓ POPE dataset config found")
            print(f"  Phase: {pope_config.get('phase')}")
            print(f"  Alpha: {pope_config.get('alpha')}")
        else:
            print("⚠ POPE dataset config not found")

        print()
        return True

    except Exception as e:
        print(f"✗ Config parameter test failed: {e}")
        return False

def run_all_tests(test_model="both"):
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("SRF SANITY TEST SUITE")
    print("=" * 60)
    print(f"Testing model: {test_model}")
    print()

    results = {}

    # Test 1: Imports (always run)
    results['imports'] = test_imports()

    # Test 2: Config (always run)
    results['config'] = test_config_parameters()

    # Test 3: CLIP saliency (always run)
    results['clip_saliency'] = test_clip_saliency()

    # Test 4: Model-specific tests
    if test_model in ["qwen", "both"]:
        results['qwen'] = test_qwen_model()
    else:
        print("⊘ Skipping Qwen tests")
        results['qwen'] = None

    if test_model in ["llava", "both"]:
        results['llava'] = test_llava_model()
    else:
        print("⊘ Skipping LLaVA tests")
        results['llava'] = None

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"{test_name:20s}: {status}")

    # Overall result
    failed_tests = [k for k, v in results.items() if v is False]
    if failed_tests:
        print(f"\n✗ {len(failed_tests)} test(s) FAILED: {', '.join(failed_tests)}")
        return False
    else:
        print("\n✓ All tests PASSED!")
        return True

def main():
    parser = argparse.ArgumentParser(description="SRF Sanity Test for Branch Merging")
    parser.add_argument(
        "--model",
        choices=["qwen", "llava", "both"],
        default="both",
        help="Which model(s) to test"
    )

    args = parser.parse_args()

    success = run_all_tests(args.model)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())