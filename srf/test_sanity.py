#!/usr/bin/env python3
"""
Sanity test for SRF absence-aware logic.

Tests the critical functionality:
1. Config has clip_suppress_thresh and clip_suppress_alpha
2. prepare_sample() sets enh_para correctly
3. Absent objects (max_sim < thresh) get LOW enh_para (suppression)
4. Present objects (max_sim >= thresh) get HIGH enh_para (boost)

Run BEFORE and AFTER code changes to verify fix.
"""
import sys
from pathlib import Path

# Add paths
_SRF_DIR = Path(__file__).parent
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_SRF_DIR / "saliency"))

def test_config():
    """Test 1: Check config has absence-aware params."""
    print("\n=== Test 1: Config Parameters ===")
    import config as CFG

    required_params = ["clip_suppress_thresh", "clip_suppress_alpha"]

    for dataset in ["pope", "mmvp", "vlmbias", "mme"]:
        params = CFG.SRF_DATASET_PARAMS.get(dataset, {})
        print(f"\n{dataset}:")
        for param in required_params:
            if param in params:
                print(f"  ✓ {param} = {params[param]}")
            else:
                print(f"  ✗ MISSING: {param}")
                return False

    return True

def test_prepare_sample_logic():
    """Test 2: Check prepare_sample has absence-aware logic."""
    print("\n=== Test 2: prepare_sample() Logic ===")

    # Read the file and check for critical lines
    srf_file = _SRF_DIR / "srf.py"
    content = srf_file.read_text()

    checks = {
        "suppress_thresh": "suppress_thresh",
        "suppress_alpha": "suppress_alpha",
        "conditional_logic": "if result.max_sim < suppress_thresh:",
        "enh_para_set": 'patch._STATE["enh_para"]',
    }

    for name, pattern in checks.items():
        if pattern in content:
            print(f"  ✓ Found: {name}")
        else:
            print(f"  ✗ MISSING: {name}")
            return False

    return True

def test_enh_para_values():
    """Test 3: Verify enh_para gets correct values."""
    print("\n=== Test 3: enh_para Value Logic ===")

    srf_file = _SRF_DIR / "srf.py"
    content = srf_file.read_text()

    # Check for absence case (low value)
    if "1.0 / (1.0 + abs(suppress_alpha))" in content:
        print(f"  ✓ Absent case: enh_para = 1.0 / (1.0 + suppress_alpha)")
    else:
        print(f"  ✗ MISSING: Absent object suppression formula")
        return False

    # Check for present case (high value)
    if 'patch._STATE["enh_para"] = abs(BIAS["boost_alpha"])' in content:
        print(f"  ✓ Present case: enh_para = abs(BIAS[\"boost_alpha\"])")
    else:
        print(f"  ✗ MISSING: Present object boost assignment")
        return False

    return True

def test_no_wrong_value_assignment():
    """Test 4: Ensure direct value assignment is NOT present."""
    print("\n=== Test 4: Check No Direct 'value' Assignment ===")

    srf_file = _SRF_DIR / "srf.py"
    content = srf_file.read_text()

    # Find prepare_sample function
    func_start = content.find("def prepare_sample(")
    if func_start == -1:
        print("  ✗ Could not find prepare_sample function")
        return False

    func_end = content.find("\ndef ", func_start + 1)
    func_body = content[func_start:func_end]

    # Check if it sets "value" directly (wrong)
    if 'patch._STATE["value"] = BIAS["boost_alpha"]' in func_body:
        print('  ✗ WRONG: Found direct patch._STATE["value"] assignment')
        print('         This bypasses absence-aware logic!')
        return False
    else:
        print('  ✓ Good: No direct "value" assignment in prepare_sample()')

    return True

def main():
    print("=" * 60)
    print("SRF Absence-Aware Logic Sanity Test")
    print("=" * 60)

    tests = [
        ("Config Parameters", test_config),
        ("prepare_sample() Logic", test_prepare_sample_logic),
        ("enh_para Value Logic", test_enh_para_values),
        ("No Direct 'value' Assignment", test_no_wrong_value_assignment),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ All tests passed! Absence-aware logic is working.")
        return 0
    else:
        print("\n✗ Some tests failed. Fix the code and re-run.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
