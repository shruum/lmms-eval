"""
SRF-V — Value-side salient visual token amplification.

Extends SRF with a value-projection channel in self-attention layers.

Motivation
----------
SRF (attention channel): biases the answer token to PULL from salient patches
  by boosting attention logits before softmax (routing side).

SRF-V (value channel): makes salient patches BROADCAST a stronger content signal
  by scaling their V vectors before the attention output is computed. Every query
  that attends to a salient patch extracts a richer signal — without needing to
  be told to look there more.

The two channels are complementary:
  - Attention logit boost (SRF): fixes the routing  (how much weight flows to salient patches)
  - Value scaling        (SRF-V): fixes the content  (how strong the signal is when read)

KV-cache benefit
----------------
V vectors are computed once at prefill (step 0) and cached. Scaling them at
prefill means every subsequent generation step automatically reads amplified
values for salient patches — zero extra compute per generation step.

Scale rule per salient image token at each fusion layer:
    v_proj_out[i] *= (1 + alpha_v * salience[i])

  salience ≈ 1 (salient)    → multiplied by (1 + alpha_v)  [amplified]
  salience ≈ 0 (background)  → multiplied by ~1              [unchanged]

No core files are modified. Hooks attach to
  model.language_model.layers[l].self_attn.v_proj

Public API
----------
    setup_v(model, alpha_v=0.3, layer_start=None, layer_end=None)
        Register v_proj hooks. Call AFTER srf.setup().
        layer_start/end default to same range as SRF (vaf_layer_start/end).

    teardown_v()
        Remove all value hooks. Safe to call multiple times.
"""
from __future__ import annotations

import pathlib
import sys

_SRF_DIR      = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import torch
import qwen_attn_patch as patch

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_V_HOOKS: list = []

_V_STATE: dict = {
    "active":      False,
    "alpha_v":     0.3,    # amplification strength (try 0.1 – 1.0)
    "layer_start": None,   # None → inherit vaf_layer_start from patch._STATE
    "layer_end":   None,   # None → inherit vaf_layer_end  from patch._STATE
}


# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------

def _make_v_hook(layer_idx: int):
    """Return a forward hook for model.language_model.layers[layer_idx].self_attn.v_proj."""

    def _hook(module, input, output):  # noqa: ARG001
        if not _V_STATE["active"]:
            return output

        # Only amplify during SRF forward pass.
        if patch._STATE.get("method", "baseline") != "srf":
            return output

        # Layer range gate
        l_start = (
            _V_STATE["layer_start"]
            if _V_STATE["layer_start"] is not None
            else patch._STATE.get("vaf_layer_start", 8)
        )
        l_end = (
            _V_STATE["layer_end"]
            if _V_STATE["layer_end"] is not None
            else patch._STATE.get("vaf_layer_end", 16)
        )
        if not (l_start <= layer_idx <= l_end):
            return output

        s   = patch._STATE.get("img_start")
        e   = patch._STATE.get("img_end")
        sal = patch._STATE.get("salience_mask")  # (n_img_tokens,) float [0,1]

        if s is None or e is None or sal is None:
            return output

        n_img   = e - s + 1
        seq_len = output.shape[1]

        # At prefill (step 0) seq_len == full sequence length.
        # At generation steps (step > 0) seq_len == 1 (only new token) —
        # image V vectors are already cached; skip.
        if seq_len <= e or n_img <= 0 or sal.shape[0] != n_img:
            return output

        # output: (batch, seq_len, n_kv_heads * head_dim)
        alpha   = float(_V_STATE["alpha_v"])
        sal_dev = sal.to(device=output.device, dtype=output.dtype)  # (n_img,)
        scale   = (1.0 + alpha * sal_dev).unsqueeze(-1)             # (n_img, 1)

        out = output.clone()
        out[:, s : e + 1, :] = out[:, s : e + 1, :] * scale
        return out

    return _hook


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_v(
    model,
    alpha_v:     float    = 0.3,
    layer_start: int | None = None,
    layer_end:   int | None = None,
) -> None:
    """
    Register v_proj hooks for value-side salient token amplification.

    Call AFTER srf.setup() so qwen_attn_patch is active and
    patch._STATE["vaf_layer_start/end"] are already set.

    Safe to call multiple times — tears down existing V hooks first.

    Args:
        model       : Qwen2.5-VL (or LLaVA) model passed to srf.setup()
        alpha_v     : amplification strength (default 0.3).
                      scale = 1 + alpha_v * salience.
                      Suggested sweep: 0.1, 0.3, 0.5, 0.8, 1.0
        layer_start : first decoder layer index to apply V boost.
                      None (default) → inherits vaf_layer_start from patch._STATE.
        layer_end   : last  decoder layer index (inclusive).
                      None (default) → inherits vaf_layer_end from patch._STATE.
    """
    teardown_v()   # idempotent cleanup

    _V_STATE["active"]      = True
    _V_STATE["alpha_v"]     = alpha_v
    _V_STATE["layer_start"] = layer_start
    _V_STATE["layer_end"]   = layer_end

    # Resolve decoder layers
    lm = model.language_model
    if hasattr(lm, "layers"):
        layers = lm.layers
    elif hasattr(lm, "model") and hasattr(lm.model, "layers"):
        layers = lm.model.layers
    else:
        raise AttributeError(
            "Cannot find decoder layers on language_model. "
            "Expected .layers (Qwen) or .model.layers (LLaVA)."
        )

    n_hooked = 0
    for idx, layer in enumerate(layers):
        if not hasattr(layer, "self_attn") or not hasattr(layer.self_attn, "v_proj"):
            continue
        h = layer.self_attn.v_proj.register_forward_hook(_make_v_hook(idx))
        _V_HOOKS.append(h)
        n_hooked += 1

    l0 = layer_start if layer_start is not None else patch._STATE.get("vaf_layer_start", "?")
    l1 = layer_end   if layer_end   is not None else patch._STATE.get("vaf_layer_end",   "?")
    print(f"  [SRF-V] Registered v_proj hooks on {n_hooked} layers  "
          f"α_v={alpha_v}  active_range=[{l0}, {l1}]")


def teardown_v() -> None:
    """Remove all registered V hooks. Safe to call multiple times."""
    for h in _V_HOOKS:
        h.remove()
    _V_HOOKS.clear()
    _V_STATE["active"] = False
