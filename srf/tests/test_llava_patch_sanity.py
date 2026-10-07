"""
Sanity tests for my_analysis/llava_attn_patch.py

Tests the patched softmax in isolation — no model load required.
Run before and after any change to llava_attn_patch.py to confirm
nothing is broken and all four SRF bugs are fixed.

Usage:
    cd /home/sgowda/workspace/SRF/lmms-eval
    source activate mllm
    python -m pytest srf/tests/test_llava_patch_sanity.py -v

Each test documents what behaviour is expected and WHY.
Failures indicate a regression or an unfixed bug.

Bug map (see srf/docs/PATCH_COMPARISON.md):
    Bug 1 — lambda_bg (background attenuation, eps) not applied
    Bug 2 — lambda_sys (system suppression) no-op in additive mode
    Bug 3 — head_mask not applied in additive mode
    Bug 4 — phase gate missing
"""
from __future__ import annotations
import importlib
import sys
import os
import math
import torch
import pytest

# ---------------------------------------------------------------------------
# Helpers to load the patch fresh for each test (avoids global state leaks)
# ---------------------------------------------------------------------------

PATCH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "my_analysis", "llava_attn_patch.py"
)


def _load_patch():
    """Import llava_attn_patch with a clean module state."""
    spec = importlib.util.spec_from_file_location("llava_attn_patch_fresh", PATCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_logits(batch=1, n_heads=4, q_len=8, kv_len=20):
    """Return a reproducible (batch, n_heads, q_len, kv_len) float32 tensor."""
    torch.manual_seed(42)
    return torch.randn(batch, n_heads, q_len, kv_len)


def _activate_patch(mod, img_start=5, img_end=14, sys_end=4,
                    alpha=2.0, eps=0.2, beta=0.3,
                    head_mask=None, sal=None, phase="both",
                    boost_mode="additive"):
    """Set up patch _STATE for a controlled single-sample SRF run."""
    n_img = img_end - img_start + 1

    mod._STATE["enabled"] = True
    mod._STATE["method"] = "srf"
    mod._STATE["in_language_model"] = True
    mod._STATE["current_layer"] = 10
    mod._STATE["vaf_layer_start"] = 0
    mod._STATE["vaf_layer_end"] = 31
    mod._STATE["img_start"] = img_start
    mod._STATE["img_end"] = img_end
    mod._STATE["sys_end"] = sys_end
    mod._STATE["boost_mode"] = boost_mode
    mod._STATE["value"] = alpha
    mod._STATE["srf_background_eps"] = eps
    mod._STATE["vaf_beta"] = beta
    mod._STATE["srf_apply_phase"] = phase
    mod._STATE["head_mask"] = head_mask

    if sal is not None:
        mod._STATE["salience_mask"] = sal
    else:
        # uniform non-trivial saliency
        mod._STATE["salience_mask"] = torch.linspace(0.0, 1.0, n_img)

    # Install the patched softmax (module-level replace)
    import torch.nn.functional as F
    mod._ORIGINAL_SOFTMAX = F.softmax
    torch.nn.functional.softmax = mod._patched_softmax

    return mod


def _restore(mod):
    """Put the real softmax back and disable the patch."""
    if mod._ORIGINAL_SOFTMAX is not None:
        torch.nn.functional.softmax = mod._ORIGINAL_SOFTMAX
        mod._ORIGINAL_SOFTMAX = None
    mod._STATE["enabled"] = False
    mod._STATE["in_language_model"] = False


# ---------------------------------------------------------------------------
# Test 1 — Baseline: patch is a provable no-op
# ---------------------------------------------------------------------------

def test_baseline_is_identity():
    """With method='baseline', output must equal standard softmax exactly."""
    mod = _load_patch()
    logits = _make_logits()
    import torch.nn.functional as F

    mod._ORIGINAL_SOFTMAX = F.softmax
    torch.nn.functional.softmax = mod._patched_softmax
    mod._STATE["enabled"] = True
    mod._STATE["method"] = "baseline"
    mod._STATE["in_language_model"] = True

    try:
        out_patch = mod._patched_softmax(logits, dim=-1)
        out_real  = mod._ORIGINAL_SOFTMAX(logits, dim=-1)
        assert torch.allclose(out_patch, out_real, atol=1e-6), \
            "Baseline mode must be identity"
    finally:
        _restore(mod)


# ---------------------------------------------------------------------------
# Test 2 — Additive mode fires: SRF output != baseline output
# ---------------------------------------------------------------------------

def test_additive_mode_changes_output():
    """SRF in additive mode must produce output different from baseline."""
    mod = _load_patch()
    logits = _make_logits()
    _activate_patch(mod, boost_mode="additive")
    try:
        out_srf = mod._patched_softmax(logits, dim=-1)
    finally:
        _restore(mod)

    import torch.nn.functional as F
    out_base = F.softmax(logits, dim=-1)
    assert not torch.allclose(out_srf, out_base, atol=1e-5), \
        "SRF additive must change the output vs baseline"


# ---------------------------------------------------------------------------
# Test 3 — BUG 1: lambda_bg (eps) applied to background image tokens
# ---------------------------------------------------------------------------

def test_lambda_bg_applied(img_start=5, img_end=14):
    """
    BUG 1 check. With a saliency map of all zeros (pure background),
    the additive logit shift on image tokens must be -eps*(1-0) = -eps.
    If only lambda_sem is applied (bug), the shift is 0*sal=0.
    """
    mod = _load_patch()
    n_img = img_end - img_start + 1
    sal = torch.zeros(n_img)  # all background
    alpha, eps = 2.0, 0.5

    logits = _make_logits()
    # Disable head_mask so we can check all heads uniformly
    _activate_patch(mod, img_start=img_start, img_end=img_end,
                    alpha=alpha, eps=eps, beta=0.0,
                    head_mask=None, sal=sal, boost_mode="additive")
    try:
        out = mod._patched_softmax(logits, dim=-1)
    finally:
        _restore(mod)

    import torch.nn.functional as F
    # With sal=0 everywhere: bias_row = alpha*0 - eps*(1-0) = -eps
    expected_logits = logits.clone()
    expected_logits[..., img_start:img_end + 1] -= eps
    out_expected = F.softmax(expected_logits, dim=-1)

    assert torch.allclose(out, out_expected, atol=1e-5), (
        f"BUG 1: lambda_bg not applied. "
        f"Max diff = {(out - out_expected).abs().max():.6f}. "
        f"Expected image-token attn to be suppressed by eps={eps} when sal=0."
    )


# ---------------------------------------------------------------------------
# Test 4 — BUG 2: lambda_sys applied in additive mode
# ---------------------------------------------------------------------------

def test_lambda_sys_applied_additive(img_start=5, img_end=14, sys_end=4):
    """
    BUG 2 check. System tokens (0..sys_end) must receive a -beta logit shift
    in additive mode. In the buggy version, sup_para is applied to attn_weights
    which is then discarded; the returned softmax sees unmodified system logits.
    """
    mod = _load_patch()
    n_img = img_end - img_start + 1
    sal = torch.ones(n_img) * 0.5
    alpha, eps, beta = 2.0, 0.0, 0.3

    logits = _make_logits()
    _activate_patch(mod, img_start=img_start, img_end=img_end,
                    sys_end=sys_end, alpha=alpha, eps=eps, beta=beta,
                    head_mask=None, sal=sal, boost_mode="additive")
    try:
        out = mod._patched_softmax(logits, dim=-1)
    finally:
        _restore(mod)

    import torch.nn.functional as F
    expected_logits = logits.clone()
    expected_logits[..., :sys_end + 1] -= beta
    expected_logits[..., img_start:img_end + 1] += alpha * sal  # eps=0 here
    out_expected = F.softmax(expected_logits, dim=-1)

    assert torch.allclose(out, out_expected, atol=1e-5), (
        f"BUG 2: lambda_sys not applied in additive mode. "
        f"Max diff = {(out - out_expected).abs().max():.6f}. "
        f"System tokens 0..{sys_end} must receive -beta={-beta} logit shift."
    )


# ---------------------------------------------------------------------------
# Test 5 — BUG 3: head_mask respected in additive mode
# ---------------------------------------------------------------------------

def test_head_mask_applied_additive(img_start=5, img_end=14):
    """
    BUG 3 check. With head_mask=[True, False, True, False] (4 heads),
    only heads 0 and 2 should have their image-token logits shifted.
    Heads 1 and 3 must be byte-identical to baseline.
    """
    mod = _load_patch()
    n_heads = 4
    n_img = img_end - img_start + 1
    sal = torch.rand(n_img)
    head_mask = torch.tensor([True, False, True, False])
    alpha, eps = 2.0, 0.2

    logits = _make_logits(n_heads=n_heads)
    _activate_patch(mod, img_start=img_start, img_end=img_end,
                    alpha=alpha, eps=eps, beta=0.0,
                    head_mask=head_mask, sal=sal, boost_mode="additive")
    try:
        out = mod._patched_softmax(logits, dim=-1)
    finally:
        _restore(mod)

    import torch.nn.functional as F
    # Unmasked heads (1, 3) must be identical to vanilla softmax
    base = F.softmax(logits, dim=-1)
    unmasked = [1, 3]
    for h in unmasked:
        diff = (out[:, h, :, :] - base[:, h, :, :]).abs().max().item()
        assert diff < 1e-5, (
            f"BUG 3: head_mask not applied. Head {h} (masked=False) changed by "
            f"{diff:.6f}. Only masked heads should be modified."
        )

    # Masked heads (0, 2) must differ from vanilla softmax
    masked = [0, 2]
    for h in masked:
        diff = (out[:, h, :, :] - base[:, h, :, :]).abs().max().item()
        assert diff > 1e-4, (
            f"BUG 3: head_mask not applied. Head {h} (masked=True) unchanged "
            f"(diff={diff:.6f}). Masked heads must receive the logit shift."
        )


# ---------------------------------------------------------------------------
# Test 6 — BUG 4: phase gate — generation-only mode skips prefill
# ---------------------------------------------------------------------------

def test_phase_gate_generation_skips_prefill(img_start=5, img_end=14):
    """
    BUG 4 check. With phase='generation', the patch must be a no-op during
    prefill (q_len > 1). If the phase gate is missing, prefill is always boosted.
    """
    mod = _load_patch()
    n_img = img_end - img_start + 1
    sal = torch.ones(n_img) * 0.5

    # Use q_len=8 to simulate prefill
    logits = _make_logits(q_len=8)
    _activate_patch(mod, img_start=img_start, img_end=img_end,
                    alpha=2.0, eps=0.0, beta=0.0,
                    head_mask=None, sal=sal,
                    phase="generation", boost_mode="additive")
    try:
        out = mod._patched_softmax(logits, dim=-1)
    finally:
        _restore(mod)

    import torch.nn.functional as F
    base = F.softmax(logits, dim=-1)
    assert torch.allclose(out, base, atol=1e-5), (
        f"BUG 4: phase gate missing. phase='generation' but patch fired at q_len=8 "
        f"(prefill). Max diff={( out - base).abs().max():.6f}."
    )


def test_phase_gate_generation_fires_at_decode(img_start=5, img_end=14):
    """
    Companion to test above: phase='generation' MUST fire at q_len=1 (decode step).
    """
    mod = _load_patch()
    n_img = img_end - img_start + 1
    sal = torch.ones(n_img) * 0.5

    logits = _make_logits(q_len=1)
    _activate_patch(mod, img_start=img_start, img_end=img_end,
                    alpha=2.0, eps=0.0, beta=0.0,
                    head_mask=None, sal=sal,
                    phase="generation", boost_mode="additive")
    try:
        out = mod._patched_softmax(logits, dim=-1)
    finally:
        _restore(mod)

    import torch.nn.functional as F
    base = F.softmax(logits, dim=-1)
    assert not torch.allclose(out, base, atol=1e-5), \
        "phase='generation' must fire at q_len=1 (decode step)"


# ---------------------------------------------------------------------------
# Test 7 — Multiplicative mode unchanged (VAF baseline must not be broken)
# ---------------------------------------------------------------------------

def test_multiplicative_mode_renormalises(img_start=5, img_end=14):
    """
    Multiplicative mode (original, VAF-compatible) must:
      - boost image token weights by enh_para
      - renormalise so rows still sum to 1
    This mode must not be broken by any fix to the additive path.
    """
    mod = _load_patch()
    n_img = img_end - img_start + 1
    enh_para = 2.0

    logits = _make_logits()
    _activate_patch(mod, img_start=img_start, img_end=img_end,
                    alpha=enh_para, eps=0.0, beta=0.0,
                    head_mask=None, sal=None,
                    boost_mode="multiplicative")
    try:
        out = mod._patched_softmax(logits, dim=-1)
    finally:
        _restore(mod)

    # Rows must sum to 1
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        "Multiplicative mode: rows must sum to 1 after renormalisation"

    # Image tokens must have higher weight than baseline
    import torch.nn.functional as F
    base = F.softmax(logits, dim=-1)
    img_ratio = (out[..., img_start:img_end + 1].sum(-1) /
                 base[..., img_start:img_end + 1].sum(-1))
    assert (img_ratio > 1.0).all(), \
        "Multiplicative mode: image token attention must increase with enh_para>1"


# ---------------------------------------------------------------------------
# Test 8 — Calibration captures in language model context only
# ---------------------------------------------------------------------------

def test_calibration_only_fires_in_lm_context():
    """
    Calibration must accumulate only when in_language_model=True.
    When False (e.g. vision encoder softmax calls), it must be skipped.
    """
    mod = _load_patch()
    import torch.nn.functional as F

    mod._ORIGINAL_SOFTMAX = F.softmax
    torch.nn.functional.softmax = mod._patched_softmax
    mod._STATE["_calibrate_heads"] = True
    mod._STATE["_calib_head_acc"]  = None
    mod._STATE["_calib_head_count"] = 0
    mod._STATE["img_start"] = 5
    mod._STATE["img_end"]   = 14

    try:
        logits = _make_logits()

        # Call with in_language_model = False (vision encoder)
        mod._STATE["in_language_model"] = False
        mod._patched_softmax(logits, dim=-1)
        assert mod._STATE["_calib_head_count"] == 0, \
            "Calibration must not accumulate outside language model"

        # Call with in_language_model = True (decoder)
        mod._STATE["in_language_model"] = True
        mod._patched_softmax(logits, dim=-1)
        assert mod._STATE["_calib_head_count"] == 1, \
            "Calibration must accumulate inside language model"
    finally:
        mod._STATE["_calibrate_heads"] = False
        _restore(mod)


# ---------------------------------------------------------------------------
# Test 9 — Full pipeline: lambda_sem + lambda_bg + lambda_sys + head_mask
# ---------------------------------------------------------------------------

def test_full_srf_additive_correctness(img_start=5, img_end=14, sys_end=4):
    """
    End-to-end correctness check of the complete additive SRF equation:

        Z_img[j] += alpha * sal[j] - eps * (1 - sal[j])   for j in img tokens
        Z_sys[j] -= beta                                    for j in sys tokens
        Only for heads where head_mask=True.

    Tests that the patch output matches the manually constructed expected output.
    This is the single most important test — it catches all four bugs at once.
    """
    mod = _load_patch()
    n_heads = 4
    n_img = img_end - img_start + 1
    alpha, eps, beta = 2.0, 0.2, 0.3
    head_mask = torch.tensor([True, False, True, False])
    torch.manual_seed(7)
    sal = torch.rand(n_img)

    logits = _make_logits(n_heads=n_heads)
    _activate_patch(mod, img_start=img_start, img_end=img_end,
                    sys_end=sys_end, alpha=alpha, eps=eps, beta=beta,
                    head_mask=head_mask, sal=sal, phase="both",
                    boost_mode="additive")
    try:
        out = mod._patched_softmax(logits, dim=-1)
    finally:
        _restore(mod)

    import torch.nn.functional as F
    # Build expected modified logits manually
    expected = logits.clone()
    bias_img = alpha * sal - eps * (1.0 - sal)          # (n_img,)
    # Apply to masked heads only
    for h in range(n_heads):
        if head_mask[h]:
            expected[:, h, :, img_start:img_end + 1] += bias_img
            expected[:, h, :, :sys_end + 1] -= beta
    out_expected = F.softmax(expected, dim=-1)

    max_diff = (out - out_expected).abs().max().item()
    assert max_diff < 1e-5, (
        f"Full SRF additive pipeline mismatch (max_diff={max_diff:.6f}). "
        f"Check Bugs 1-4 in srf/docs/PATCH_COMPARISON.md."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_baseline_is_identity,
        test_additive_mode_changes_output,
        test_lambda_bg_applied,
        test_lambda_sys_applied_additive,
        test_head_mask_applied_additive,
        test_phase_gate_generation_skips_prefill,
        test_phase_gate_generation_fires_at_decode,
        test_multiplicative_mode_renormalises,
        test_calibration_only_fires_in_lm_context,
        test_full_srf_additive_correctness,
    ]
    passed = failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}")
            print(f"        {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} passed")
    if failed:
        sys.exit(1)
