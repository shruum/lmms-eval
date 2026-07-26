#!/usr/bin/env python3
"""
COMPREHENSIVE SRF VERIFICATION TEST
Runs SRF on actual samples and prints EVERY intermediate value to find all bugs
"""
import sys
import os
sys.path.insert(0, '/home/anna2/shruthi/lmms-eval')

print("🔍 COMPREHENSIVE SRF VERIFICATION - Step by Step")
print("=" * 100)

# Temporarily patch attention functions to print everything
original_forward = None

def debug_attention_forward(self, hidden_states, attention_mask=None, position_ids=None,
                         past_key_value=None, output_attentions=False, use_cache=False):
    """Debug version of attention forward that prints everything"""
    print(f"\n{'='*100}")
    print(f"📍 LAYER ATTENTION START")
    print(f"{'='*100}")

    # Input shapes
    bsz, q_len, _ = hidden_states.size()
    print(f"Input: batch={bsz}, q_len={q_len}, hidden_dim={hidden_states.shape[-1]}")

    # Check if we should apply SRF
    layer_num = getattr(self, '_layer_num', None)
    patch_state = getattr(self, '_patch_state', None)

    if patch_state is None:
        print("❌ No patch_state - SRF not initialized!")
        return original_forward(self, hidden_states, attention_mask, position_ids,
                              past_key_value, output_attentions, use_cache)

    method = patch_state.get("method", "baseline")
    if method != "srf":
        print(f"Method: {method} - skipping SRF")
        return original_forward(self, hidden_states, attention_mask, position_ids,
                              past_key_value, output_attentions, use_cache)

    # Check layer range
    layer_start = patch_state.get("layer_start", None)
    layer_end = patch_state.get("layer_end", None)

    print(f"Layer: {layer_num}")
    print(f"SRF layer range: [{layer_start}, {layer_end}]")

    if layer_num < layer_start or layer_num > layer_end:
        print("❌ Layer outside SRF range - skipping")
        return original_forward(self, hidden_states, attention_mask, position_ids,
                              past_key_value, output_attentions, use_cache)

    print("✅ Layer in SRF range - applying modifications")

    # Print SRF parameters
    enh_para = patch_state.get("enh_para", None)
    sup_para = patch_state.get("sup_para", None)
    background_eps = patch_state.get("srf_background_eps", 0.0)
    img_start = patch_state.get("img_start", None)
    img_end = patch_state.get("img_end", None)

    print(f"📊 SRF PARAMETERS:")
    print(f"  enh_para = {enh_para} (boosting strength)")
    print(f"  sup_para = {sup_para} (system suppression)")
    print(f"  background_eps = {background_eps} (background suppression)")
    print(f"  img_start = {img_start}")
    print(f"  img_end = {img_end}")
    print(f"  Image token count: {img_end - img_start + 1}")

    # Check saliency
    saliency = patch_state.get("salience", None)
    if saliency is not None:
        print(f"  saliency.shape = {saliency.shape}")
        print(f"  saliency.min = {saliency.min().item():.4f}")
        print(f"  saliency.max = {saliency.max().item():.4f}")
        print(f"  saliency.mean = {saliency.mean().item():.4f}")

        # Dimension check
        expected_tokens = img_end - img_start + 1
        actual_tokens = saliency.numel()
        print(f"  Dimension check: {actual_tokens} == {expected_tokens} = {actual_tokens == expected_tokens}")

        if actual_tokens != expected_tokens:
            print(f"❌ DIMENSION MISMATCH BUG! saliency has {actual_tokens} elements, expected {expected_tokens}")
    else:
        print("  ❌ No saliency available!")

    # Run original attention but print intermediate values
    result = original_forward(self, hidden_states, attention_mask, position_ids,
                            past_key_value, output_attentions, use_cache)

    attn_output, attn_weights, past_key_value = result

    print(f"\n📊 ATTENTION WEIGHTS (after modification):")
    print(f"  attn_weights.shape = {attn_weights.shape}")

    if attn_weights is not None:
        print(f"  attn_weights.min = {attn_weights.min().item():.6f}")
        print(f"  attn_weights.max = {attn_weights.max().item():.6f}")
        print(f"  attn_weights.mean = {attn_weights.mean().item():.6f}")

        # Check specific attention patterns
        if img_start is not None and img_end is not None:
            # Attention to image tokens
            img_attn = attn_weights[:, :, :, img_start:img_end+1]
            print(f"  Image token attention: min={img_attn.min().item():.6f}, max={img_attn.max().item():.6f}")

    print(f"{'='*100}\n")

    return attn_output, attn_weights, past_key_value

print("✅ Debug function defined")
print("✅ Ready to run comprehensive test")

# Now run actual test
try:
    print("\n" + "="*100)
    print("🚀 RUNNING SRF ON 2 SAMPLES WITH FULL DEBUG")
    print("="*100 + "\n")

    # Import after defining debug function
    from srf.eval import run_pope_vcd
    import torch

    # Monkey-patch the attention mechanism
    from my_analysis import llava_attn_patch
    original_forward = llava_attn_patch.SRFLlamaAttention.forward
    llava_attn_patch.SRFLlamaAttention.forward = debug_attention_forward

    # Add layer numbering
    original_init = llava_attn_patch.SRFLlamaAttention.__init__
    def numbered_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Try to get layer number from config
        try:
            self._layer_num = self.layer_idx
        except:
            self._layer_num = 99  # fallback

    llava_attn_patch.SRFLlamaAttention.__init__ = numbered_init

    print("✅ Monkey-patched attention with debug function")

    # Run evaluation on 2 samples
    import subprocess
    result = subprocess.run([
        "/home/anna2/miniconda3/envs/mllm/bin/python",
        "/home/anna2/shruthi/lmms-eval/srf/eval.py",
        "--method", "srf",
        "--model", "llava-hf/llava-1.5-7b-hf",
        "--datasets", "pope_vcd",
        "--calib_dataset", "pope",
        "--pope_vcd_file", "/home/anna2/shruthi/RePOPE/annotations/coco_repoe_adversarial.json",
        "--pope_vcd_name", "Debug test",
        "--pope_image_dir", "/home/anna2/shruthi/dataset/POPE_images/images/val2014",
        "--alpha", "0.25",  # Should give enh_para = 1.25
        "--eps", "0.1",
        "--sys_beta", "0.15",  # Should give sup_para = 0.85
        "--layer_start", "10",
        "--layer_end", "15",
        "--head_top_k_pct", "0.50",
        "--clip_top_k_pct", "0.30",
        "--n_pope", "2",  # Only 2 samples
        "--do_sample", "--temperature", "0.7", "--top_p", "0.9",
        "--output", "results/debug_comprehensive/"
    ], capture_output=True, text=True, timeout=600)

    print("\n" + "="*100)
    print("📊 TEST OUTPUT")
    print("="*100)
    print(result.stdout)

    if result.stderr:
        print("\n" + "="*100)
        print("⚠️  ERRORS")
        print("="*100)
        print(result.stderr)

    print("\n" + "="*100)
    print("🎯 ANALYSIS")
    print("="*100)

    # Check for expected patterns
    checks = {
        "enh_para value": "enh_para = 1.25" in result.stdout,
        "sup_para value": "sup_para = 0.85" in result.stdout,
        "saliency shape": "saliency.shape = torch.Size([576])" in result.stdout,
        "dimension check": "576 == 576 = True" in result.stdout,
        "layer in range": "Layer in SRF range" in result.stdout,
    }

    print("Verification Checks:")
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(checks.values())
    if all_passed:
        print("\n✅ ALL CHECKS PASSED - SRF working correctly!")
    else:
        print("\n❌ SOME CHECKS FAILED - BUGS DETECTED!")
        print("Review the debug output above to identify issues")

except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()