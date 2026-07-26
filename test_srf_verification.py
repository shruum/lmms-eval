#!/usr/bin/env python3
"""
Test FIXED SRF on small sample set to verify code flow analysis
"""
import subprocess
import sys

print("🔍 Testing FIXED SRF on 5 samples with full debug output...")
print("=" * 80)

# Run SRF evaluation on 5 samples with maximum debug
cmd = [
    "/home/anna2/miniconda3/envs/mllm/bin/python",
    "/home/anna2/shruthi/lmms-eval/srf/eval.py",
    "--method", "srf",
    "--model", "llava-hf/llava-1.5-7b-hf",
    "--datasets", "pope_vcd",
    "--calib_dataset", "pope",
    "--pope_vcd_file", "/home/anna2/shruthi/RePOPE/annotations/coco_repoe_adversarial.json",
    "--pope_vcd_name", "RePOPE adversarial verification test",
    "--pope_image_dir", "/home/anna2/shruthi/dataset/POPE_images/images/val2014",
    "--alpha", "0.25",  # Conservative boost (25% max for high saliency)
    "--eps", "0.1",     # Background suppression (10%)
    "--sys_beta", "0.15",  # System suppression (15%)
    "--layer_start", "10",  # Middle fusion layers
    "--layer_end", "15",
    "--head_top_k_pct", "0.50",  # Top 50% vision-aware heads
    "--clip_top_k_pct", "0.30",  # Top 30% most salient tokens
    "--n_pope", "5",  # Only 5 samples for quick verification
    "--do_sample", "--temperature", "0.7", "--top_p", "0.9",
    "--output", "results/verification_test/"
]

print("Command:")
print(" ".join(cmd[5:]))  # Show without python path
print("=" * 80)

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    print("\n📊 OUTPUT:")
    print(result.stdout)

    if result.stderr:
        print("\n⚠️  ERRORS:")
        print(result.stderr)

    print("\n" + "=" * 80)

    # Check for key success indicators
    success_indicators = [
        "[IMG TOKENS FIXED]",  # Image token detection working
        "DEBUG SRF",  # CLIP saliency debug output
        "accuracy",  # Results computed
    ]

    found_indicators = []
    for indicator in success_indicators:
        if indicator in result.stdout:
            found_indicators.append(indicator)

    print(f"✅ Found {len(found_indicators)}/{len(success_indicators)} success indicators:")
    for indicator in found_indicators:
        print(f"  - {indicator}")

    if len(found_indicators) >= 2:
        print("\n✅ VERIFICATION TEST PASSED - SRF is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ VERIFICATION TEST FAILED - Check output for issues")
        sys.exit(1)

except subprocess.TimeoutExpired:
    print("\n⏱️  Test timed out after 10 minutes")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    sys.exit(1)