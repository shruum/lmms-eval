# SRF RePOPE Results - INVALIDATED DUE TO CODE BUGS

## ❌ **CRITICAL ISSUE**

All previous SRF results on RePOPE (in REPOPE_FINDINGS.md) are **INVALID** because they used the **BROKEN SRF implementation** with 7 critical bugs.

## 🐛 **Bugs in Previous Implementation:**

1. **Parameter name mismatch**: Configured layers 10-15 ignored, used hardcoded 9-14
2. **Image token failure**: Targeted 1 placeholder token instead of 576 actual tokens  
3. **CLIP saliency never applied**: Dimension mismatch caused fallback to uniform
4. **Head mask ignored**: Computed but never used in attention
5. **Hardcoded overrides**: User arguments ignored
6. **System suppression incomplete**: Implementation bugs
7. **Parameter flow broken**: Config values didn't reach attention code

## ✅ **All 7 Bugs Now Fixed (CRITICAL_BUG_REPORT.md)**

All bugs fixed and verified via unit tests (tests/test_srf_comprehensive.py).

## 📊 **INVALID Results (Do Not Use)**

The following SRF results from REPOPE_FINDINGS.md are INVALID and should be removed:

| Split | SRF α=4.0 | SRF α=5.0 |
|-------|-----------|-----------|
| Random | 87.20% | 87.27% | ❌ INVALID |
| Popular | 85.57% | 85.57% | ❌ INVALID |
| Adversarial | 82.93% | 82.93% | ❌ INVALID |

## 🔄 **Re-Run Required**

Need to re-run SRF with FIXED implementation to get valid results on RePOPE.

Status: ⏳ **PENDING** - Starting now with adversarial split on GPU 0.

---
*See CRITICAL_BUG_REPORT.md for complete bug details and fixes*
