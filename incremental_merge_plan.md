# Incremental Merge Plan for srf-remote-final

## Current Baseline: autoresearch/mmvp-srf (HEAD)
- ✅ All sanity tests pass
- ✅ clip_full_gate_v3 working
- ✅ Qwen support complete
- ❌ No LLaVA support

## Incremental Merge Steps:

### STEP 1: Create Base Branch (Current mmvp-srf)
```bash
git checkout -b srf-remote-final HEAD
# Test: python test_sanity_merge.py --model qwen (should pass)
```

### STEP 2: Add LLaVA Attention Patch
**Files to cherry-pick from srf-remote:**
- `my_analysis/llava_attn_patch.py`

**Test:**
```bash
python test_sanity_merge.py --model llava
# Should see: ✓ LLaVA attention patch imported
```

### STEP 3: Add Test Suite
**Files to cherry-pick from srf-remote:**
- `tests/test_srf_components.py`
- `tests/test_srf_comprehensive.py`
- `tests/test_srf_verification.py`
- `tests/test_srf_unit.py`

**Test:**
```bash
python -m pytest tests/test_srf_components.py -v
```

### STEP 4: Update LLaVA Config
**Modify: srf/config.py**

Change LLaVA entry:
```python
"llava-hf/llava-1.5-7b-hf": {
    "saliency_mode": "clip_full_gate_v3",  # ← Change from "clip"
    # Add other params from srf-remote...
}
```

**Test:**
```bash
python test_sanity_merge.py --model llava
# Should see: ✓ LLaVA config using clip_full_gate_v3
```

### STEP 5: Add Key Documentation
**Files to cherry-pick from srf-remote:**
- `REPOPE_FINDINGS.md` (root level)
- `CRITICAL_BUG_REPORT.md` (root level)
- `SRF_ANALYSIS.md` (root level)

**Test:** Visual check that files exist

### STEP 6: Add Validation Scripts
**Files to cherry-pick from srf-remote:**
- `validate_repope_proper.sh`
- Key test scripts: `test_*.py`, `test_*.sh`

**Test:**
```bash
ls -la validate_repope_proper.sh
```

### STEP 7: Final Comprehensive Test
**Test:**
```bash
python test_sanity_merge.py --model both
# Should see: All tests PASS for both Qwen and LLaVA
```

### STEP 8: Code Flow Verification
**Manual verification:**
1. Review LLaVA attention patch code
2. Verify clip_full_gate_v3 integration with LLaVA
3. Check layer ranges and parameters
4. Verify noun extraction works for LLaVA
5. Check absence detection integration

## Post-Merge Checklist:
- [ ] Both Qwen and LLaVA load successfully
- [ ] clip_full_gate_v3 works for both models
- [ ] CLIP saliency computation correct
- [ ] All test suite passes
- [ ] LLaVA parameters match findings from srf-remote
- [ ] Documentation is complete
- [ ] No regressions in Qwen functionality

## Rollback Strategy:
If any step fails:
```bash
git reset --hard HEAD~1  # Go back one step
git checkout srf-remote-final # Restart from clean state
```

## Priority Order:
1. **STEP 1** (Create base) - MUST DO FIRST
2. **STEP 2** (LLaVA patch) - CRITICAL
3. **STEP 4** (Config update) - CRITICAL
4. **STEP 3** (Tests) - IMPORTANT
5. **STEP 5-6** (Docs/Scripts) - OPTIONAL
6. **STEP 7** (Final test) - MUST DO
7. **STEP 8** (Code review) - SHOULD DO