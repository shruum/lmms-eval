"""
VHR — Vision-aware Head Reinforcement for Qwen2.5-VL.

Faithful adaptation of VHR (arXiv:2412.13949)
https://github.com/jinghan1he/VHR

Algorithm (per sample):
  1. Forward pass with full image       — capture pre-o_proj head outputs at target layers
  2. Forward pass with text-only input  — capture pre-o_proj head outputs at target layers
     (strip <|image_pad|> tokens from input_ids; omit pixel_values / image_grid_thw)
  3. VHD[layer][head] = ((output_full - output_text)**2).sum(-1)
     measured at LAST TOKEN POSITION to avoid sequence-length mismatch
  4. aug_heads[layer] = heads where VHD > median(VHD)  [top ~50%]
  5. Inference forward: scale pre-o_proj outputs of aug_heads by aug_ratio

Target layers (matching VHR paper, LLaVA-1.5-7B = 32 layers: last 14 + layer 1):
  n_layers = 28 (Qwen2.5-VL-3B): [1, 14, 15, ..., 27]
  n_layers = 32 (Qwen2.5-VL-7B): [1, 18, 19, ..., 31]
  General:  {1} ∪ {n_layers-14, ..., n_layers-1}

aug_ratio = 2.0 (from VHR repo default)

Interface matches srf.py / vaf.py so eval.py can dispatch with --method vhr.
"""
from __future__ import annotations

from typing import Any

import torch
import qwen_attn_patch as patch

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_VHR: dict = {
    "mode":                 "disabled",  # disabled | calib_full | calib_text | inference
    "aug_ratio":            2.0,
    "aug_heads_per_layer":  {},          # {layer_idx: LongTensor of head indices}
    "_calib_full":          {},          # {layer_idx: (1, n_heads, head_dim) float32 on CPU}
    "_calib_text":          {},          # {layer_idx: (1, n_heads, head_dim) float32 on CPU}
    "n_heads":              None,        # set in setup()
    "head_dim":             None,        # set in setup()
    "_target_layers":       set(),       # set of int layer indices
}
_HOOKS: list = []   # registered hook handles


# ---------------------------------------------------------------------------
# Target-layer logic
# ---------------------------------------------------------------------------

def _compute_target_layers(n_layers: int) -> set:
    """Return {1} ∪ {last 14 layers}, matching VHR's LLaVA-1.5-7B target."""
    last14 = set(range(max(2, n_layers - 14), n_layers))
    return {1} | last14


def _get_decoder_layers(model: Any):
    """Return list of decoder layers (Qwen or LLaVA)."""
    lm = model.language_model
    if hasattr(lm, "layers"):
        return lm.layers
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    raise AttributeError(f"Cannot find decoder layers on {type(lm).__name__}")


# ---------------------------------------------------------------------------
# o_proj hook — capture or scale per-head pre-o_proj outputs
# ---------------------------------------------------------------------------

def _make_o_proj_hook(layer_idx: int):
    """
    forward_pre_hook on self_attn.o_proj.

    Input  args[0]: (bsz, q_len, n_heads * head_dim)
    Output: modified args tuple when mode == "inference", else None (no change)
    """
    def hook(module, args):
        x       = args[0]   # (bsz, q_len, hidden)
        n_heads = _VHR["n_heads"]
        head_dim = _VHR["head_dim"]
        mode    = _VHR["mode"]

        if mode == "calib_full":
            # Capture last-position per-head output as VHD reference "with image"
            # (bsz, q_len, hidden) → last token → (bsz, n_heads, head_dim)
            x_last  = x[:, -1, :].detach().float().view(-1, n_heads, head_dim)
            # Store per-head squared norm proxy: (n_heads,) for VHD computation
            _VHR["_calib_full"][layer_idx] = x_last.cpu()

        elif mode == "calib_text":
            x_last  = x[:, -1, :].detach().float().view(-1, n_heads, head_dim)
            _VHR["_calib_text"][layer_idx] = x_last.cpu()

        elif mode == "inference":
            aug_heads = _VHR["aug_heads_per_layer"].get(layer_idx)
            if aug_heads is None or len(aug_heads) == 0:
                return   # no-op for this layer
            aug_ratio   = float(_VHR["aug_ratio"])
            bsz, q_len, hidden = x.shape
            x_heads     = x.view(bsz, q_len, n_heads, head_dim).clone()
            aug_heads_d = aug_heads.to(x.device)   # aug_heads is CPU; move to model device
            x_heads[:, :, aug_heads_d, :] = x_heads[:, :, aug_heads_d, :] * aug_ratio
            return (x_heads.view(bsz, q_len, hidden),)

    return hook


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup(model, processor, calib_dataset: str = "pope") -> None:
    """
    Discover model architecture and register o_proj hooks on target layers.
    No calibration data needed — VHR calibrates per sample in prepare_sample().
    """
    global _HOOKS

    # Activate baseline patch (needed for current_layer tracking by qwen_attn_patch)
    patch.patch_model(model, method="baseline", value=1.0)

    lm_cfg  = model.language_model.config
    n_layers = getattr(lm_cfg, "num_hidden_layers", 28)
    n_heads  = getattr(lm_cfg, "num_attention_heads", 16)
    hidden   = getattr(lm_cfg, "hidden_size", 2048)
    head_dim = hidden // n_heads

    _VHR["n_heads"]      = n_heads
    _VHR["head_dim"]     = head_dim
    _VHR["_target_layers"] = _compute_target_layers(n_layers)

    # Remove any existing VHR hooks
    for h in _HOOKS:
        h.remove()
    _HOOKS.clear()

    # Register hooks on o_proj for target layers
    layers = _get_decoder_layers(model)
    for idx in _VHR["_target_layers"]:
        if idx < len(layers):
            h = layers[idx].self_attn.o_proj.register_forward_pre_hook(
                _make_o_proj_hook(idx)
            )
            _HOOKS.append(h)

    n_target = len([i for i in _VHR["_target_layers"] if i < len(layers)])
    print(f"  [VHR] n_layers={n_layers}  n_heads={n_heads}  head_dim={head_dim}")
    print(f"  [VHR] target layers: {sorted(_VHR['_target_layers'])}  ({n_target} hooks)")


def reset_for_dataset(dataset: str = "pope", *, aug_ratio: float | None = None, **kwargs) -> None:
    """Reset per-dataset state. VHR has no dataset-specific hyperparameters (all per-sample)."""
    if aug_ratio is not None:
        _VHR["aug_ratio"] = aug_ratio
    _VHR["aug_heads_per_layer"] = {}
    _VHR["mode"] = "disabled"
    patch._STATE["method"] = "baseline"


def prepare_sample(
    inp,
    img_start: int,
    img_end: int,
    image,
    question: str,
    model,
    processor,
    **kwargs,
) -> None:
    """
    Per-sample VHR calibration:
      1. Forward pass (full image)  → capture pre-o_proj head outputs
      2. Forward pass (text only)   → capture pre-o_proj head outputs
      3. Compute VHD per layer, select aug_heads (VHD > median)
    """
    device = inp["input_ids"].device

    # ── Build text-only input (remove image tokens + pixel_values) ─────────────
    ids     = inp["input_ids"][0]
    new_ids = torch.cat([ids[:img_start], ids[img_end + 1:]], dim=0).unsqueeze(0).to(device)
    text_inp = {
        "input_ids":      new_ids,
        "attention_mask": torch.ones_like(new_ids),
    }

    # ── Pass 1: full image ──────────────────────────────────────────────────────
    _VHR["_calib_full"] = {}
    _VHR["mode"] = "calib_full"
    patch._STATE["method"] = "baseline"
    with torch.inference_mode():
        model(**inp)

    # ── Pass 2: text only ───────────────────────────────────────────────────────
    _VHR["_calib_text"] = {}
    _VHR["mode"] = "calib_text"
    patch._STATE["method"] = "baseline"
    with torch.inference_mode():
        model(**text_inp)

    # ── Compute VHD per layer, select aug_heads ─────────────────────────────────
    _VHR["mode"] = "disabled"
    aug_heads: dict = {}
    for layer_idx in sorted(_VHR["_target_layers"]):
        full = _VHR["_calib_full"].get(layer_idx)   # (bsz, n_heads, head_dim)
        text = _VHR["_calib_text"].get(layer_idx)
        if full is None or text is None:
            continue
        # VHD per head = squared L2 distance at last token position
        # full, text: (1, n_heads, head_dim) → squeeze batch
        vhd = ((full[0] - text[0]) ** 2).sum(-1)     # (n_heads,)
        # Select heads with VHD > median (top ~50%, matching VHR repo)
        selected = (vhd > vhd.median()).nonzero().flatten()
        aug_heads[layer_idx] = selected

    _VHR["aug_heads_per_layer"] = aug_heads

    # Switch to inference mode for the actual forward pass
    _VHR["mode"] = "inference"
    patch._STATE["method"] = "baseline"   # no softmax-level intervention for VHR


def cleanup() -> None:
    """Reset per-sample state."""
    _VHR["mode"] = "disabled"
    _VHR["aug_heads_per_layer"] = {}
    _VHR["_calib_full"] = {}
    _VHR["_calib_text"] = {}
    patch._STATE["method"] = "baseline"


# ---------------------------------------------------------------------------
# Inference helpers — called by eval.py via method_get_logits / method_generate
# ---------------------------------------------------------------------------

def get_contrastive_logits(model, inp: dict, **kwargs) -> torch.Tensor:
    """Single forward pass with VHR o_proj hooks active."""
    _VHR["mode"] = "inference"
    patch._STATE["method"] = "baseline"
    with torch.inference_mode():
        out = model(**inp)
    return out.logits[:, -1, :].float()


def generate_contrastive(
    model,
    inp: dict,
    processor,
    max_new_tokens: int = 20,
    content_offset: int = 0,
    **kwargs,
) -> list[int]:
    """Standard greedy generation with VHR o_proj hooks active."""
    _VHR["mode"] = "inference"
    patch._STATE["method"] = "baseline"
    with torch.inference_mode():
        out_ids = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return out_ids[0, inp["input_ids"].shape[1]:].tolist()
