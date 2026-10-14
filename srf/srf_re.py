"""
SRF-RE — Skip-connection re-injection of early visual hidden states.

Injects the hidden states of salient image tokens from an early layer
(src_layer) back into later fusion layers (layer_start–layer_end) at
the FFN input — a learned skip connection that refreshes the visual
signal without multiplicative scaling.

Motivation
----------
SRF-FN and SRF-V scaled intermediate tensors multiplicatively, which
compounds across layers and disrupts LayerNorm calibration. Re-injection
is ADDITIVE — it adds a bounded vector to the residual stream, avoiding
magnitude blowup.

As image token hidden states flow through layers 0→8→16, text context
accumulates via cross-attention. By the fusion layers the original visual
signal is diluted. Injecting h[salient_i][src_layer] at the MLP input of
later layers provides the FFN a "refreshed" visual content signal to
transform — allowing it to re-emphasise the original patch content in
the context of what the later layer is computing.

Injection formula (per salient image token, per fusion layer):
    h_mlp_in[i] += beta * salience[i] * h_captured[i]

  salience ≈ 1 (salient)    → full injection
  salience ≈ 0 (background)  → near-zero injection

No core files are modified. Two sets of hooks are used:
  1. Pre-forward hook on layers[src_layer]  → capture image token hidden states
  2. Pre-forward hook on layers[l].mlp      → inject into MLP input

Public API
----------
    setup_re(model, beta=0.3, src_layer=0, layer_start=None, layer_end=None)
        Register capture + inject hooks. Call AFTER srf.setup().

    teardown_re()
        Remove all hooks and clear captured state. Safe to call multiple times.
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

_RE_HOOKS: list = []

_RE_STATE: dict = {
    "active":      False,
    "beta":        0.3,
    "src_layer":   0,     # layer whose INPUT is captured (0 = visual projector output)
    "layer_start": None,  # injection range start (None → inherit vaf_layer_start)
    "layer_end":   None,  # injection range end   (None → inherit vaf_layer_end)
}

# Captured image token hidden states from src_layer (set during prefill, reused at gen steps)
_CAPTURED_HIDDEN: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------

def _make_capture_hook():
    """
    Pre-forward hook on layers[src_layer].
    Captures the INPUT hidden states of image tokens before that layer processes them.
    src_layer=0: captures visual projector output (pure visual, no LLM attention).
    src_layer=2: captures state after 2 rounds of attn+FFN (some text context mixed in).
    """
    def _hook(module, args):
        global _CAPTURED_HIDDEN
        if not _RE_STATE["active"]:
            return
        if patch._STATE.get("method", "baseline") != "srf":
            return

        hidden = args[0]   # (batch, seq_len, d_model)
        s = patch._STATE.get("img_start")
        e = patch._STATE.get("img_end")
        if s is None or e is None:
            return

        seq_len = hidden.shape[1]
        # Only capture at prefill (full sequence present)
        if seq_len <= e:
            return

        _CAPTURED_HIDDEN = hidden[:, s : e + 1, :].detach().clone()  # (batch, n_img, d_model)

    return _hook


def _make_inject_hook(layer_idx: int):
    """
    Pre-forward hook on layers[layer_idx].mlp.
    Injects beta * salience * captured_hidden into the MLP input for salient image tokens.
    Returns modified args so the MLP processes the injected signal.
    """
    def _hook(module, args):
        global _CAPTURED_HIDDEN
        if not _RE_STATE["active"]:
            return args
        if patch._STATE.get("method", "baseline") != "srf":
            return args
        if _CAPTURED_HIDDEN is None:
            return args

        # Layer range gate
        l_start = (
            _RE_STATE["layer_start"]
            if _RE_STATE["layer_start"] is not None
            else patch._STATE.get("vaf_layer_start", 8)
        )
        l_end = (
            _RE_STATE["layer_end"]
            if _RE_STATE["layer_end"] is not None
            else patch._STATE.get("vaf_layer_end", 16)
        )
        if not (l_start <= layer_idx <= l_end):
            return args

        s   = patch._STATE.get("img_start")
        e   = patch._STATE.get("img_end")
        sal = patch._STATE.get("salience_mask")  # (n_img,) float [0,1]
        if s is None or e is None or sal is None:
            return args

        hidden  = args[0]   # (batch, seq_len, d_model)
        n_img   = e - s + 1
        seq_len = hidden.shape[1]

        # Skip generation steps (seq_len == 1, image tokens are in KV cache)
        if seq_len <= e or n_img <= 0 or sal.shape[0] != n_img:
            return args

        beta    = float(_RE_STATE["beta"])
        dev     = hidden.device
        dtype   = hidden.dtype
        sal_dev = sal.to(device=dev, dtype=dtype)            # (n_img,)
        cap     = _CAPTURED_HIDDEN.to(device=dev, dtype=dtype)  # (batch, n_img, d_model)

        # injection: (batch, n_img, d_model) scaled per token by beta*salience
        injection = beta * sal_dev.unsqueeze(0).unsqueeze(-1) * cap   # (batch, n_img, d_model)

        new_hidden = hidden.clone()
        new_hidden[:, s : e + 1, :] += injection
        return (new_hidden,) + args[1:]

    return _hook


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_re(
    model,
    beta:        float    = 0.3,
    src_layer:   int      = 0,
    layer_start: int | None = None,
    layer_end:   int | None = None,
) -> None:
    """
    Register capture and injection hooks for skip-connection re-injection.

    Call AFTER srf.setup() so patch._STATE["vaf_layer_start/end"] are set.
    Safe to call multiple times — tears down existing hooks first.

    Args:
        model       : Qwen2.5-VL model
        beta        : injection strength (default 0.3). Suggested: 0.1, 0.3, 0.5, 1.0
        src_layer   : layer whose INPUT is captured and later re-injected.
                      0 = visual projector output (pure visual).
                      2 = after 2 LLM layers (light text context).
        layer_start : first layer to inject into (None → vaf_layer_start).
        layer_end   : last  layer to inject into (None → vaf_layer_end).
    """
    global _CAPTURED_HIDDEN
    teardown_re()
    _CAPTURED_HIDDEN = None

    _RE_STATE["active"]      = True
    _RE_STATE["beta"]        = beta
    _RE_STATE["src_layer"]   = src_layer
    _RE_STATE["layer_start"] = layer_start
    _RE_STATE["layer_end"]   = layer_end

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

    # Capture hook — pre-forward on src_layer
    if src_layer >= len(layers):
        raise ValueError(f"src_layer={src_layer} >= n_layers={len(layers)}")
    h_cap = layers[src_layer].register_forward_pre_hook(_make_capture_hook())
    _RE_HOOKS.append(h_cap)

    # Inject hooks — pre-forward on each layer's mlp
    n_inject = 0
    for idx, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            continue
        h_inj = layer.mlp.register_forward_pre_hook(_make_inject_hook(idx))
        _RE_HOOKS.append(h_inj)
        n_inject += 1

    l0 = layer_start if layer_start is not None else patch._STATE.get("vaf_layer_start", "?")
    l1 = layer_end   if layer_end   is not None else patch._STATE.get("vaf_layer_end",   "?")
    print(f"  [SRF-RE] Capture @ layer {src_layer} input  |  "
          f"Inject into {n_inject} MLP inputs  "
          f"β={beta}  active_range=[{l0}, {l1}]")


def teardown_re() -> None:
    """Remove all hooks and clear captured hidden state. Safe to call multiple times."""
    global _CAPTURED_HIDDEN
    for h in _RE_HOOKS:
        h.remove()
    _RE_HOOKS.clear()
    _CAPTURED_HIDDEN = None
    _RE_STATE["active"] = False
