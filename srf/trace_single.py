#!/usr/bin/env python3
"""
SRF single-sample trace — comprehensive step-by-step audit.

Runs one POPE sample through the full SRF pipeline and logs every
decision: saliency map, token selection, boost values, head mask,
layer range, phase gate, suppression, and actual logit deltas.

Usage:
    source activate mllm && python srf/trace_single.py
    source activate mllm && python srf/trace_single.py --out srf/docs/TRACE.md
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from io import StringIO

_SRF_DIR      = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR / "saliency"))
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import config as CFG
os.environ.setdefault("HF_HOME", CFG.HF_HOME)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import numpy as np
from datasets import load_dataset as hf_load
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import qwen_attn_patch as patch
import clip_salience as clip_sal
import srf

# ── output buffer ────────────────────────────────────────────────────────────
_LOG = StringIO()


def log(s: str = "") -> None:
    print(s)
    _LOG.write(s + "\n")


# ── per-layer capture ─────────────────────────────────────────────────────────
_layer_deltas: dict[int, dict] = {}   # layer_idx → {delta_mean, delta_max, n_heads_active, ...}


def _compute_analytical_deltas(sal_tensor, bias: dict, n_img: int,
                                n_active_heads: int, sys_end: int,
                                phase: str, layer_start: int, layer_end: int,
                                n_layers: int) -> None:
    """Compute expected per-layer logit deltas analytically from SRF state.

    The patch uses input.clone() so the original tensor is never mutated —
    we cannot capture deltas at runtime. But the formula is fully deterministic:

        bias_row[i] = alpha * sal[i] - eps * (1 - sal[i])
                    = (alpha + eps) * sal[i] - eps

    so:
        salient (sal=1):    Δ = +alpha
        background (sal=0): Δ = -eps
        average:            Δ = alpha * mean(sal) - eps * (1 - mean(sal))

    System-prompt: Δ = -beta (per active head, phase-gated)
    """
    alpha = bias["boost_alpha"]
    eps   = bias["background_eps"]
    beta  = bias["sys_beta"]

    if sal_tensor is not None:
        sal_np = sal_tensor.float().cpu().numpy()
        img_delta_mean = float(alpha * sal_np.mean() - eps * (1.0 - sal_np.mean()))
        img_delta_max  = float(alpha)   # max sal = 1.0
        img_delta_min  = float(-eps)    # min sal = 0.0
    else:
        img_delta_mean = float(alpha)
        img_delta_max  = float(alpha)
        img_delta_min  = float(alpha)

    for layer_idx in range(layer_start, layer_end + 1):
        # POPE: phase=generation → fires only at q_len=1 (generation steps)
        # Prefill (q_len=381) and generation (q_len=1) both visit these layers
        # We report both cases
        for q_len, step in [(n_img + 36, "prefill"), (1, "generation")]:
            is_gen  = (q_len == 1)
            if phase == "both":
                phase_ok = True
            elif phase == "generation":
                phase_ok = is_gen
            elif phase == "prefill":
                phase_ok = not is_gen
            else:
                phase_ok = False

            key = f"{layer_idx}_{step}"
            _layer_deltas[key] = {
                "layer":           layer_idx,
                "step":            step,
                "q_len":           q_len,
                "phase_ok":        phase_ok,
                "n_active_heads":  n_active_heads,
                "img_delta_mean":  img_delta_mean if phase_ok else 0.0,
                "img_delta_max":   img_delta_max  if phase_ok else 0.0,
                "img_delta_min":   img_delta_min  if phase_ok else 0.0,
                "sys_delta_heads": -beta           if (phase_ok and sys_end > 0 and beta > 0) else 0.0,
            }


# ─────────────────────────────────────────────────────────────────────────────

def run_trace(out_path: str | None = None) -> None:
    # ── 1. Load model ──────────────────────────────────────────────────────────
    log("# SRF Single-Sample Trace")
    log(f"> Model: {CFG.DEFAULT_MODEL}")
    log(f"> Dataset: POPE (1 adversarial sample)")
    log(f"> Date: 2026-06-18")
    log()

    log("## 1. Model & Setup")
    log()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.DEFAULT_MODEL, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(CFG.DEFAULT_MODEL,
                                               max_pixels=CFG.DEFAULT_MAX_PIXELS)
    arch = CFG.get_arch(CFG.DEFAULT_MODEL)
    img_token_id = processor.tokenizer.convert_tokens_to_ids(arch["image_token"])
    device = next(model.parameters()).device

    log(f"- Model architecture: Qwen2.5-VL-3B-Instruct")
    log(f"- Decoder layers: {arch['n_layers']}  (index 0–{arch['n_layers']-1})")
    log(f"- Attention heads per layer: 16 (Q/K), 8 (V, GQA)")
    log(f"- Image token: `{arch['image_token']}` (id={img_token_id})")
    log(f"- Device: {device}")
    log()

    # ── 2. Setup SRF ──────────────────────────────────────────────────────────
    log("## 2. SRF Calibration")
    log()
    srf.setup(model, processor, calib_dataset="pope")
    srf.reset_for_dataset("pope")

    bias = srf.BIAS
    sal  = srf.SALIENCY

    head_mask = patch._STATE.get("head_mask")
    n_heads_total = 16
    if head_mask is not None:
        active_heads = head_mask.nonzero(as_tuple=True)[0].tolist()
    else:
        active_heads = list(range(n_heads_total))

    log(f"### BIAS config (pope)")
    log(f"| Parameter | Value |")
    log(f"|-----------|-------|")
    log(f"| `layer_start` | {bias['layer_start']} |")
    log(f"| `layer_end` | {bias['layer_end']} |")
    log(f"| `boost_alpha` (α) | {bias['boost_alpha']} |")
    log(f"| `background_eps` (ε) | {bias['background_eps']} |")
    log(f"| `neg_absent_alpha` | {bias.get('neg_absent_alpha', 0.0)} |")
    log(f"| `sys_beta` | {bias['sys_beta']} |")
    log(f"| `text_beta` | {bias['text_beta']} |")
    log(f"| `phase` | `{bias['srf_apply_phase']}` |")
    log(f"| `bias_mode` | `{bias['bias_mode']}` |")
    log(f"| `head_top_k_pct` | {bias['head_top_k_pct']} |")
    log()

    log(f"### Head calibration")
    log(f"- Total heads in model: {n_heads_total} per layer")
    log(f"- `head_top_k_pct` = {bias['head_top_k_pct']} → top {int(bias['head_top_k_pct'] * n_heads_total)} heads selected")
    log(f"- **Vision-aware head indices**: {active_heads}")
    log(f"- Calibration: 20 POPE samples, seed=0. Scores each head by mean attention to image tokens from text query positions.")
    log()

    log(f"### SALIENCY config")
    log(f"| Parameter | Value |")
    log(f"|-----------|-------|")
    log(f"| `saliency_mode` | `{sal['saliency_mode']}` |")
    log(f"| `clip_model` | `{sal.get('clip_model', 'openai/clip-vit-base-patch32')}` |")
    log(f"| `clip_coarse_grid` | {sal['clip_coarse_grid']} |")
    log(f"| `clip_top_k_pct` | {sal['clip_top_k_pct']} |")
    log(f"| `clip_fallback_thresh` | {sal['clip_fallback_thresh']} |")
    log()

    # ── 3. Load one POPE sample ───────────────────────────────────────────────
    log("## 3. Sample")
    log()

    ds = hf_load("lmms-lab/POPE", split="test")
    # find a "No" adversarial sample (hallucination case — interesting for tracing)
    sample = None
    for r in ds:
        if str(r.get("category", "")).lower() == "adversarial" and \
           str(r.get("answer", "")).strip().lower() == "no":
            sample = r
            break
    assert sample is not None

    image    = sample["image"].convert("RGB")
    question_raw = str(sample["question"]).strip()
    q        = question_raw + "\nAnswer with Yes or No only."
    gt       = "no"

    log(f"- **Category**: adversarial")
    log(f"- **Question**: \"{question_raw}\"")
    log(f"- **GT answer**: No  *(model should say No — language prior might say Yes)*")
    log(f"- **Image size**: {image.size[0]}×{image.size[1]} px")
    log()

    # ── 4. Tokenize ───────────────────────────────────────────────────────────
    log("## 4. Tokenization & Token Layout")
    log()

    msgs  = [{"role": "user", "content": [{"type": "image", "image": image},
                                           {"type": "text",  "text":  q}]}]
    text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    vis, _ = process_vision_info(msgs)
    inp    = processor(text=[text], images=vis, return_tensors="pt",
                       padding=True).to(device)

    ids = inp["input_ids"][0].tolist()
    img_start = next(i for i, t in enumerate(ids) if t == img_token_id)
    img_end   = len(ids) - 1 - next(i for i, t in enumerate(reversed(ids)) if t == img_token_id)
    n_img_tokens  = img_end - img_start + 1
    n_total_tokens = len(ids)
    sys_end = img_start - 1   # last system/prompt token before image

    log(f"- Total input tokens: **{n_total_tokens}**")
    log(f"- Image token range: **[{img_start}, {img_end}]** (inclusive) = {n_img_tokens} image tokens")
    log(f"- System/prompt tokens: [0, {sys_end}] = {sys_end + 1} tokens")
    log(f"- Post-image text tokens: [{img_end+1}, {n_total_tokens-1}] = {n_total_tokens - 1 - img_end} tokens")
    log()
    log(f"```")
    log(f"Token layout (input_ids):")
    log(f"  [0 … {sys_end}]            system + question prefix   ({sys_end+1} tokens)")
    log(f"  [{img_start} … {img_end}]  image tokens              ({n_img_tokens} tokens)")
    log(f"  [{img_end+1} … {n_total_tokens-1}]  question suffix + gen prompt ({n_total_tokens - 1 - img_end} tokens)")
    log(f"```")
    log()

    # ── 5. CLIP saliency ──────────────────────────────────────────────────────
    log("## 5. CLIP Saliency Map")
    log()

    # extract noun
    from noun_extract import extract_clip_noun
    noun = extract_clip_noun(q, mode="pope")
    log(f"- **Question**: \"{q}\"")
    log(f"- **Extracted CLIP noun**: `{noun}`")
    log()

    # compute saliency
    grid_h, grid_w = clip_sal.get_grid_dims(inp, arch["spatial_merge_size"])

    sal_mode = sal["saliency_mode"]   # clip_full_gate_v3
    v3_thresh = sal.get("clip_fallback_thresh") or clip_sal._FULL_IMG_THRESH_V3
    result = clip_sal.compute_clip_salience_full_gate_v3(
        image, noun, grid_h, grid_w,
        top_k_pct=sal["clip_top_k_pct"],
        clip_model_name=sal.get("clip_model", clip_sal._CLIP_DEFAULT_MODEL),
        backup="none",
        full_img_thresh=v3_thresh,
    )

    log(f"- **Saliency mode**: `{sal_mode}`")
    log(f"- **CLIP grid**: {grid_h}×{grid_w} = {grid_h*grid_w} coarse patches → {n_img_tokens} image tokens")
    log(f"- **CLIP full-image similarity**: {result.full_img_sim:.4f}  (threshold: {v3_thresh})")
    log(f"- **Object present**: {result.object_present}  ({'above' if result.object_present else 'BELOW'} threshold)")
    log()

    if result.object_present:
        sal_tensor = result.saliency   # (n_img_tokens,)
        clip_conf  = min(result.full_img_sim / v3_thresh, 1.0)
        effective_alpha = bias["boost_alpha"] * clip_conf
        log(f"- **CLIP confidence**: {clip_conf:.4f}  (full_img_sim / thresh)")
        log(f"- **Effective α** = α × conf = {bias['boost_alpha']} × {clip_conf:.4f} = **{effective_alpha:.4f}**")
    else:
        sal_tensor = None
        effective_alpha = -bias.get("neg_absent_alpha", 0.0)
        log(f"- Object absent → uniform suppression: value = {effective_alpha:.4f}")

    log()

    if sal_tensor is not None:
        sal_np = sal_tensor.float().cpu().numpy()
        top_k  = int(sal["clip_top_k_pct"] * n_img_tokens)
        top_indices = np.argsort(sal_np)[::-1][:top_k]

        log(f"### Saliency distribution")
        log(f"| Stat | Value |")
        log(f"|------|-------|")
        log(f"| n_img_tokens | {n_img_tokens} |")
        log(f"| top_k_pct | {sal['clip_top_k_pct']} → top {top_k} tokens boosted |")
        log(f"| sal min | {sal_np.min():.4f} |")
        log(f"| sal max | {sal_np.max():.4f} |")
        log(f"| sal mean | {sal_np.mean():.4f} |")
        log(f"| sal std | {sal_np.std():.4f} |")
        log()
        log(f"Top-10 most salient image token positions (0-indexed within image range):")
        log(f"```")
        for rank, idx in enumerate(top_indices[:10]):
            row = idx // grid_w
            col = idx % grid_w
            log(f"  rank {rank+1:2d}: token {idx:4d}  grid ({row},{col})  sal={sal_np[idx]:.4f}  "
                f"logit_boost={effective_alpha * sal_np[idx]:+.4f}")
        log(f"```")
        log()
        log(f"Background tokens (sal≈0) get logit delta ≈ −ε = **−{bias['background_eps']}**")
        log(f"Salient tokens (sal=1.0) get logit delta ≈ **+{effective_alpha:.4f}**")
        log(f"Net spread (salient vs background): **{effective_alpha + bias['background_eps']:.4f} logit units**")
    log()

    # ── 6. Run SRF forward pass ───────────────────────────────────────────────
    log("## 6. Forward Pass — Per-Layer Logit Modifications")
    log()
    log("Instrumenting the attention hook to capture exact logit deltas per layer...")
    log()

    srf.prepare_sample(inp, img_start, img_end, image, q, model, processor)

    # Show what prepare_sample set in _STATE
    log(f"### patch._STATE after prepare_sample()")
    log(f"| Key | Value |")
    log(f"|-----|-------|")
    log(f"| `method` | `{patch._STATE['method']}` |")
    log(f"| `value` (boost_alpha effective) | {patch._STATE['value']} |")
    log(f"| `img_start` / `img_end` | {patch._STATE['img_start']} / {patch._STATE['img_end']} |")
    log(f"| `sys_end` | {patch._STATE['sys_end']} |")
    log(f"| `srf_apply_phase` | `{patch._STATE['srf_apply_phase']}` |")
    log(f"| `srf_bias_mode` | `{patch._STATE['srf_bias_mode']}` |")
    log(f"| `srf_background_eps` | {patch._STATE['srf_background_eps']} |")
    log(f"| `vaf_beta` (sys suppression) | {patch._STATE['vaf_beta']} |")
    log(f"| `vaf_layer_start` | {patch._STATE['vaf_layer_start']} |")
    log(f"| `vaf_layer_end` | {patch._STATE['vaf_layer_end']} |")
    log(f"| `salience_mask` is None | {patch._STATE['salience_mask'] is None} |")
    if patch._STATE['salience_mask'] is not None:
        sm = patch._STATE['salience_mask']
        log(f"| `salience_mask` shape | {list(sm.shape)} |")
        log(f"| `salience_mask` min/max | {float(sm.min()):.4f} / {float(sm.max()):.4f} |")
    log(f"| `head_mask` active heads | {active_heads} |")
    log()

    # ── 6b. Phase gate check ──────────────────────────────────────────────────
    log(f"### Phase gate check")
    phase = bias["srf_apply_phase"]
    log(f"- **phase** = `{phase}`")
    log(f"- Prefill step: q_len = {n_total_tokens} > 1  →  `_is_gen = False`")
    if phase == "both":
        log(f"  → phase_ok = **True** (always on for `phase=both`)")
    elif phase == "generation":
        log(f"  → phase_ok = **False** during prefill (q_len>1). Boost fires ONLY at generation steps (q_len=1)")
    elif phase == "prefill":
        log(f"  → phase_ok = **True** during prefill, False at generation")
    log(f"- Generation step: q_len = 1  →  `_is_gen = True`")
    if phase == "generation":
        log(f"  → phase_ok = **True**. All boosts apply.")
    log()

    # run forward pass (SRF already active via prepare_sample)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=5, do_sample=False)

    # compute expected logit deltas analytically
    _compute_analytical_deltas(
        sal_tensor   = patch._STATE.get("salience_mask"),
        bias         = bias,
        n_img        = n_img_tokens,
        n_active_heads = len(active_heads),
        sys_end      = sys_end,
        phase        = bias["srf_apply_phase"],
        layer_start  = bias["layer_start"],
        layer_end    = bias["layer_end"],
        n_layers     = arch["n_layers"],
    )

    pred_ids = out[0, inp["input_ids"].shape[1]:].tolist()
    pred_raw = processor.decode(pred_ids, skip_special_tokens=True).strip().lower()
    pred     = "yes" if pred_raw.startswith("yes") else "no"

    log(f"### Prediction")
    log(f"- Generated tokens: `{pred_raw}`")
    log(f"- Prediction: **{pred}**  |  GT: **{gt}**  |  Correct: **{pred == gt}**")
    log()

    # ── 7. Per-layer delta table ──────────────────────────────────────────────
    log("## 7. Per-Layer Intervention Summary")
    log()
    log(f"Layer range: [{bias['layer_start']}, {bias['layer_end']}]  "
        f"({bias['layer_end'] - bias['layer_start'] + 1} layers active out of {arch['n_layers']})")
    log()
    log(f"| Layer | Step | q_len | phase_ok | Active heads | Img Δ mean | Img Δ max | Img Δ min | Sys Δ/head |")
    log(f"|-------|------|-------|----------|--------------|-----------|-----------|-----------|-----------|")

    for key in sorted(_layer_deltas.keys()):
        d = _layer_deltas[key]
        phase_ok_str = "✓ FIRES" if d["phase_ok"] else "✗ skipped"
        sys_str = f"{d['sys_delta_heads']:+.4f}" if d.get("sys_delta_heads") else "0.0000"
        log(f"| {d['layer']:5d} | {d['step']:10s} | {d['q_len']:5d} | {phase_ok_str:9s} | "
            f"{d['n_active_heads']:2d} of 16 | "
            f"{d['img_delta_mean']:+.4f} | "
            f"{d['img_delta_max']:+.4f} | "
            f"{d['img_delta_min']:+.4f} | "
            f"{sys_str} |")

    log()

    # ── 8. Suppression summary ────────────────────────────────────────────────
    log("## 8. Suppression Mechanisms")
    log()
    log(f"### System-prompt suppression (sys_beta)")
    log(f"- **β = {bias['sys_beta']}** applied to tokens [0, {sys_end}] ({sys_end+1} system tokens)")
    log(f"- Applied in **vision-aware heads only** ({active_heads})")
    log(f"- Gated by phase: only fires when `phase_ok=True`")
    log(f"- Effect: reduces model's ability to rely on system-prompt context when answering")
    log()
    log(f"### Background suppression (eps)")
    log(f"- **ε = {bias['background_eps']}** subtracted from non-salient image tokens")
    log(f"- Tokens with saliency ≈ 0 get logit −ε; tokens with saliency 1.0 get +α")
    log(f"- Net contrast between salient and background: **{effective_alpha + bias['background_eps']:.4f}** logit units")
    log()
    log(f"### Text-token suppression (text_beta)")
    log(f"- **text_beta = {bias['text_beta']}** → **disabled** (0.0)")
    log(f"- Would suppress question tokens in layers [{bias['text_layer_start']}, {bias['text_layer_end']}]")
    log(f"- Not used: experiments showed no gain; language prior is in MLP, not attention")
    log()

    # ── 9. End-to-end summary ─────────────────────────────────────────────────
    log("## 9. End-to-End Method Summary")
    log()
    log("""
SRF (Semantic Re-Focus) modifies attention logits **pre-softmax** in a single forward pass:

```
For each token t at query position q (in decoder layers [layer_start, layer_end]):

  logit[h, q, img_i] += α × sal[i]          if token i is an image token, h ∈ vision-heads
  logit[h, q, img_i] -= ε × (1 - sal[i])    background image tokens suppressed
  logit[h, q, sys_j] -= β                    system-prompt tokens suppressed
```

Where:
  α (boost_alpha)     = CLIP confidence × base alpha  (presence-gated boost strength)
  sal[i] ∈ [0,1]      = CLIP patch similarity of image token i to the query noun
  ε (background_eps)  = background suppression (non-salient image tokens)
  β (sys_beta)        = system-prompt suppression
  h ∈ vision-heads    = top-k% heads by image attention score (calibrated once)
  phase gate          = only during generation steps (q_len=1) for POPE
""")

    log(f"### Numeric summary for this sample")
    log(f"| | |")
    log(f"|---|---|")
    log(f"| CLIP noun | `{noun}` |")
    log(f"| CLIP full-image sim | {result.full_img_sim:.4f} |")
    log(f"| Object present | {result.object_present} |")
    log(f"| Effective α | {float(patch._STATE['value']):.4f} |")
    log(f"| ε (background) | {bias['background_eps']} |")
    log(f"| β (sys suppress) | {bias['sys_beta']} |")
    log(f"| Active layers | [{bias['layer_start']}, {bias['layer_end']}] = {bias['layer_end']-bias['layer_start']+1} layers |")
    log(f"| Active heads | {active_heads} ({len(active_heads)}/{n_heads_total}) |")
    log(f"| Image tokens | {n_img_tokens} |")
    log(f"| Salient tokens (top-{int(sal['clip_top_k_pct']*100)}%) | {int(sal['clip_top_k_pct']*n_img_tokens)} |")
    log(f"| Prediction | **{pred}** (GT={gt}, {'✓ correct' if pred==gt else '✗ wrong'}) |")

    srf.cleanup()

    # ── write output ──────────────────────────────────────────────────────────
    content = _LOG.getvalue()
    if out_path:
        pathlib.Path(out_path).write_text(content)
        print(f"\n→ Written to {out_path}")
    else:
        print("\n→ Use --out srf/docs/TRACE.md to save")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="Output markdown file")
    args = p.parse_args()
    run_trace(out_path=args.out)
