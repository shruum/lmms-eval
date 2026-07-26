# SRF Debug Context - 0.00% Delta Investigation

**Date:** 2026-04-30
**Status:** ACTIVE DEBUGGING

---

## 🔴 Critical Issue

SRF gets exactly **0.00% delta** across ALL datasets:
- POPE: +0.05% avg (3 categories, 9000 samples)
- MME: 0.00% (14 categories, 2374 samples)
- MMVP: 0.00% (150 pairs, 300 images)

**User's hypothesis:** "are you even applying SRF? Something is terribly wrong"

---

## 🔍 Investigation So Far

### Debug Prints Added

**srf/srf.py (lines 542-575) in `prepare_sample()`:**
```python
# Added debug output
print(f"[DEBUG SRF] max_sim: {result.max_sim:.4f}")
print(f"[DEBUG SRF] fallback_thresh: {SALIENCY['clip_fallback_thresh']:.4f}")
print(f"[DEBUG SRF] clip_top_k_pct: {SALIENCY['clip_top_k_pct']}")
print(f"[DEBUG SRF] noun: {noun}")

# Check if salience_mask is None
if result.max_sim < SALIENCY["clip_fallback_thresh"]:
    print(f"[DEBUG SRF] ❌ max_sim < thresh → salience_mask = None")
else:
    print(f"[DEBUG SRF] ✅ max_sim >= thresh → salience_mask set")
    if patch._STATE["salience_mask"] is not None:
        mask = patch._STATE["salience_mask"]
        if torch.is_tensor(mask):
            print(f"[DEBUG SRF] mask: shape={mask.shape}, sum={mask.sum().item():.2f}")

# Check enh_para
print(f"[DEBUG SRF] enh_para: {patch._STATE.get('enh_para', 'NOT SET')}")
```

**srf/eval.py (lines 377-397) in `method_get_logits()`:**
```python
print(f"[DEBUG method_get_logits] SRF mode - setting patch._STATE['method'] = 'srf'")
print(f"[DEBUG method_get_logits] enh_para: {patch._STATE.get('enh_para', 'NOT SET')}")
print(f"[DEBUG method_get_logits] salience_mask: {patch._STATE.get('salience_mask', 'NOT SET')}")
patch._STATE["method"] = "srf"
```

### Temp File Leak Found

**Location:** `srf/eval.py` lines 295-312 in `format_qwen_msgs()`
- Creates temp PNG files in `/tmp` for Qwen-VL images
- **Never cleans them up** → 96,122 files (~7GB leaked)
- Fixed by: Added `cleanup_qwen_temp_images()` function (lines 316-322)
- Status: Cleanup code partially implemented, needs testing

---

## 🎯 Latest Findings (May 2026)

### Small but Consistent Improvement on VLM Bias
- **Delta: +0.68%** on full VLM Bias dataset (vs baseline)
- **Alpha parameter has NO effect** - tested across multiple values
- **Per-category tracking implemented** - can now analyze which categories benefit most

### Parameter Sweep Results
Tested on Qwen2.5-VL-3B:
- Alpha values: 0.5, 1.0, 2.0, 4.0, 6.0, 8.0
- **Finding**: No correlation between alpha and accuracy
- **Hypothesis**: Boosting mechanism needs fundamental redesign

### Current Status
- ✅ Code works correctly (masks applied, logits differ)
- ✅ Small gain on VLM Bias (+0.68%)
- ❌ Zero gain on POPE, MME, MMVP
- 🔬 **Active investigation**: Why does VLM Bias respond but not others?

### Next Steps
1. **Analyze per-category VLM Bias results** - understand which categories improve
2. **Compare dataset characteristics** - why VLM Bias responds differently
3. **Test alternative mechanisms** - post-softmax redistribution, layer-specific modulation

### 2. Verify SRF is Actually Applied

**Check files:**
- `my_analysis/qwen_attn_patch.py` - Look for `enh_para` usage in forward pass
- Verify patch._STATE is being read correctly
- Check if boosting actually happens (add more debug prints if needed)

### 3. Complete Temp File Cleanup

**Status:** Partially done
- Need to register temp files in global list
- Call cleanup after each evaluation
- Test that /tmp doesn't fill up again

---

## 📂 Key Files

| File | Purpose | Status |
|------|---------|--------|
| `srf/srf.py` | SRF algorithm | Debug prints added ✅ |
| `srf/eval.py` | Evaluation CLI | Debug prints added ✅, cleanup partial ⚠️ |
| `my_analysis/qwen_attn_patch.py` | Attention patching | Needs inspection 🔍 |
| `srf/config.py` | Hyperparameters | BEST_SRF_CONFIG: alpha=6.0, eps=0.3 |

---

## 🔧 Best SRF Config (from POPE sweep)

```python
BEST_SRF_CONFIG = {
    "alpha": 6.0,              # Boosting strength
    "eps": 0.3,                # Suppression strength
    "clip_coarse_grid": 7,     # CLIP patch grid
    "clip_top_k_pct": 0.5,     # Top 50% image tokens boosted
    "clip_suppress_thresh": 0.0,
}
```

---

## 🐛 Possible Root Causes

1. **Salience mask is None** - If max_sim < clip_fallback_thresh, no boosting happens
2. **enh_para not used** - Parameter set but not actually applied in forward pass
3. **Wrong layer range** - Boosting happens outside cross-modal fusion zone
4. **Baseline run twice** - Actual bug in eval code (unlikely but possible)

---

## 📊 Results Summary (Updated May 2026)

| Dataset | Baseline | SRF | Δ | Status |
|---------|----------|-----|---|--------|
| POPE (avg) | 84.13% | 84.18% | +0.05% | ❌ No improvement |
| MME | 69.50% | 69.50% | 0.00% | ❌ No improvement |
| MMVP | 26.67% | 26.67% | 0.00% | ❌ No improvement |
| VLM Bias | ~19% | ~19.68% | **+0.68%** | ✅ Small improvement |

**Key Insight**: VLM Bias shows consistent small gains while others show zero.

---

## 💡 User's Explicit Requests

1. ✅ "DEBUG WHOLE REPO CODE"
2. ✅ "add unit tests to make sure salient regions are being focused well"
3. ✅ "maybe qualitatively? save few samples, I will also view it"
4. ✅ "check if boosting is happening fine"

**Current focus:** Run debug test to see what's actually happening inside SRF

---

**Last updated:** 2026-05-02
**Current focus:** Understanding why VLM Bias improves (+0.68%) but POPE/MME/MMVP don't (0.00%)
**Key insight:** Alpha parameter has no effect → fundamental mechanism needs redesign
