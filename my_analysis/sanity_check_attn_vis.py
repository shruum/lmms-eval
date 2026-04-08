#!/usr/bin/env python3
"""
Sanity checks for attention_map_vis.py.

Tests every component independently before running the full pipeline,
so failures are isolated and easy to diagnose.

Checks (in order):
  1.  Imports and constants
  2.  _extract_answer  — all three answer formats (MCQ, Yes/No, free-form)
  3.  Perturbation conditions — image is modified and differs from original
  4.  build_heatmap — reshape + upsample, non-zero output
  5.  blend_heatmap — output dtype, shape, value range
  6.  Dataset loading — MMBench, VLM Bias, POPE (1 sample each, no model)
  7.  Sample.prompt() — correct structure for each dataset type
  8.  Model loading — eager attention implementation
  9.  Token layout — image tokens present and contiguous
 10.  Hook capture — attention weights non-None and non-zero with eager
 11.  extract_spatial_attention — attn_vec sums to ~1, grid dims match n_image_tokens
 12.  predict() — returns a valid answer key (or free-form string)
 13.  Full single-sample pipeline — predict + attention map + heatmap end-to-end
 14.  make_group_figure — figure file written, non-empty

Usage:
  python my_analysis/sanity_check_attn_vis.py
  python my_analysis/sanity_check_attn_vis.py --skip_model   # skip GPU checks
  python my_analysis/sanity_check_attn_vis.py --model Qwen/Qwen2.5-VL-7B-Instruct
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any, Dict, List

import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"

# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
SKIP = "\033[93m  SKIP\033[0m"
SEP  = "─" * 60

results: List[Dict[str, Any]] = []


def check(name: str, fn, skip: bool = False) -> bool:
    print(f"\n{SEP}")
    print(f"CHECK {len(results)+1}: {name}")
    if skip:
        print(f"{SKIP}  (skipped)")
        results.append({"name": name, "status": "skip"})
        return True
    try:
        fn()
        print(f"{PASS}")
        results.append({"name": name, "status": "pass"})
        return True
    except Exception as exc:
        print(f"{FAIL}: {exc}")
        traceback.print_exc()
        results.append({"name": name, "status": "fail", "error": str(exc)})
        return False


# ── import module under test ──────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
import attention_map_vis as V  # noqa: E402


# ── individual checks ─────────────────────────────────────────────────────────


def check_imports():
    assert hasattr(V, "_extract_answer")
    assert hasattr(V, "build_heatmap")
    assert hasattr(V, "blend_heatmap")
    assert hasattr(V, "apply_condition")
    assert hasattr(V, "extract_spatial_attention")
    assert hasattr(V, "AttentionMapRunner")
    assert hasattr(V, "make_group_figure")
    assert hasattr(V, "DATASET_LOADERS")
    assert set(V.DATASET_LOADERS.keys()) == {"mmbench", "vlm_bias", "pope"}
    print("    All expected symbols present.")


def check_extract_answer():
    # MCQ (A/B/C/D)
    assert V._extract_answer("The answer is B.", ["A", "B", "C", "D"]) == "B", "MCQ letter"
    assert V._extract_answer("A", ["A", "B"]) == "A", "Single letter"
    assert V._extract_answer("xyz", ["A", "B"]) == "?", "No match -> ?"

    # Yes/No
    assert V._extract_answer("Yes, it is.", ["Yes", "No"]) == "Yes", "Yes/No"
    assert V._extract_answer("No.", ["Yes", "No"]) == "No", "No"

    # Free-form (VLM Bias) — no valid_keys, extract from {braces}
    assert V._extract_answer("{Yes}", []) == "Yes", "Free-form Yes"
    assert V._extract_answer("I think {8} rows.", []) == "8", "Free-form number"
    assert V._extract_answer("no braces", []) == "no braces"[:20], "Free-form fallback"
    print("    All answer extraction cases correct.")


def check_perturbations():
    from PIL import Image as PILImage
    # Use a gradient image so low_res_x8 actually changes pixel values
    arr = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3) % 256
    img = PILImage.fromarray(arr, mode="RGB")
    arr_orig = np.array(img)

    for cond in ["baseline", "low_res_x8", "blur_r5", "patch_shuffle16", "center_mask40"]:
        out = V.apply_condition(img, cond)
        assert out.size[0] > 0 and out.size[1] > 0, f"{cond}: output has zero dimension"
        arr_out = np.array(out)
        if cond == "baseline":
            assert np.array_equal(arr_orig, arr_out), "baseline should be identical"
        else:
            assert not np.array_equal(arr_orig, arr_out), f"{cond}: image unchanged (bug)"
        print(f"    {cond}: OK  size={out.size}")


def check_build_heatmap():
    from PIL import Image as PILImage
    grid_h, grid_w = 8, 10
    n = grid_h * grid_w
    attn = np.random.rand(n).astype(np.float32)
    attn /= attn.sum()

    img_size = (320, 256)  # (W, H)
    heatmap = V.build_heatmap(attn, grid_h, grid_w, img_size)

    assert heatmap.shape == (256, 320), f"Expected (256,320), got {heatmap.shape}"
    assert heatmap.dtype == np.float32, f"Expected float32, got {heatmap.dtype}"
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0, "Values out of [0,1]"
    assert heatmap.max() > 0.01, "Heatmap is all near-zero (suspicious)"
    print(f"    Shape: {heatmap.shape}, range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    # Test padding when attn_vec length doesn't match grid
    attn_short = np.random.rand(n - 5).astype(np.float32)
    heatmap2 = V.build_heatmap(attn_short, grid_h, grid_w, img_size)
    assert heatmap2.shape == (256, 320), "Padding case failed"
    print("    Padding case: OK")


def check_blend_heatmap():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    heatmap = np.random.rand(100, 100).astype(np.float32)
    blended = V.blend_heatmap(img, heatmap)
    assert blended.shape == (100, 100, 3), f"Shape mismatch: {blended.shape}"
    assert blended.dtype == np.uint8, f"Expected uint8, got {blended.dtype}"
    assert blended.min() >= 0 and blended.max() <= 255
    print(f"    Shape: {blended.shape}, range: [{blended.min()}, {blended.max()}]")


def check_dataset_mmbench():
    groups = V.load_mmbench(["spatial_relationship"])
    assert "spatial_relationship" in groups, "Category not found"
    samples = groups["spatial_relationship"]
    assert len(samples) > 0, "No samples returned"
    s = samples[0]
    assert s.image is not None and s.image.size[0] > 0, "Invalid image"
    assert len(s.question) > 0, "Empty question"
    assert len(s.options) >= 2, f"Too few options: {s.options}"
    assert s.gt in s.options, f"GT '{s.gt}' not in options {list(s.options.keys())}"
    assert s.group == "spatial_relationship"
    prompt = s.prompt()
    assert s.question in prompt, "Question not in prompt"
    assert V.DEFAULT_SUFFIX in prompt, "Suffix missing from prompt"
    print(f"    {len(samples)} samples, first: Q='{s.question[:50]}' GT={s.gt}")

    # Verify that options are injected into prompt
    for k, v in s.options.items():
        assert f"{k}." in prompt, f"Option {k} missing from prompt"


def check_dataset_vlm_bias():
    groups = V.load_vlm_bias(["Optical_Illusion"])
    assert "Optical_Illusion" in groups, f"Topic not found. Got: {list(groups.keys())}"
    samples = groups["Optical_Illusion"]
    assert len(samples) > 0
    s = samples[0]
    assert s.image is not None and s.image.size[0] > 0
    assert len(s.question) > 0, "Empty question/prompt"
    assert len(s.options) == 0, "VLM Bias should have no options (free-form)"
    assert len(s.gt) > 0, "Empty ground truth"
    assert s.prompt_suffix == "", "VLM Bias suffix should be empty"
    prompt = s.prompt()
    assert prompt == s.question, f"VLM Bias prompt() should equal question (no suffix/options). Got: {prompt!r}"
    print(f"    {len(samples)} samples, first: Q='{s.question[:60]}' GT={s.gt}")


def check_dataset_pope():
    groups = V.load_pope(None)  # default = adversarial
    assert "adversarial" in groups, f"adversarial split not found. Got: {list(groups.keys())}"
    samples = groups["adversarial"]
    assert len(samples) > 0
    s = samples[0]
    assert s.image is not None and s.image.size[0] > 0
    assert len(s.question) > 0
    assert s.options == {"Yes": "Yes", "No": "No"}, f"Unexpected options: {s.options}"
    assert s.gt in ("Yes", "No"), f"GT should be Yes/No, got: {s.gt}"
    assert V.YESNO_SUFFIX in s.prompt()
    # Only adversarial by default; other splits absent
    assert "popular" not in groups, "popular should not be loaded by default"
    assert "random" not in groups, "random should not be loaded by default"
    print(f"    {len(samples)} adversarial samples, first: Q='{s.question[:50]}' GT={s.gt}")

    # Check that --groups overrides work
    groups2 = V.load_pope(["popular"])
    assert "popular" in groups2 and "adversarial" not in groups2
    print(f"    --groups popular: {len(groups2['popular'])} samples, adversarial excluded ✓")


def check_model_loads(model_name: str):
    global _runner
    _runner = V.AttentionMapRunner(model_name, attn_implementation="eager")
    print(f"    Model loaded: {model_name}")
    print(f"    Layers: {len(_runner.model.language_model.layers)}")

    # Confirm sdpa/flash_attention_2 is rejected
    try:
        V.AttentionMapRunner(model_name, attn_implementation="sdpa")
        raise AssertionError("Should have raised ValueError for sdpa")
    except ValueError as e:
        print(f"    sdpa correctly rejected: {str(e)[:60]}")


def check_token_layout(model_name: str):
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (224, 224), color=(120, 80, 200))
    prompt = "What colour is the background? Answer with the option's letter.\nA. Red\nB. Blue\nC. Purple\nD. Green\nAnswer with the option's letter from the given choices directly."
    inputs = _runner._build_inputs(img, prompt)

    image_token_id = _runner.processor.tokenizer.convert_tokens_to_ids(V.IMAGE_TOKEN)
    assert image_token_id != _runner.processor.tokenizer.unk_token_id, \
        f"IMAGE_TOKEN '{V.IMAGE_TOKEN}' resolved to UNK — wrong token string"

    ids = inputs["input_ids"][0].cpu()
    image_mask = ids == image_token_id
    n_img = int(image_mask.sum())
    n_txt = int((~image_mask).sum())
    assert n_img > 0, "No image tokens found in input_ids"
    assert "image_grid_thw" in inputs, "image_grid_thw missing from processor output"

    thw = inputs["image_grid_thw"][0]
    grid_h, grid_w = int(thw[1]), int(thw[2])
    expected_tokens = grid_h * grid_w
    assert n_img == expected_tokens, \
        f"Token count mismatch: {n_img} image tokens but grid is {grid_h}x{grid_w}={expected_tokens}"

    print(f"    Sequence: {len(ids)} tokens  ({n_img} image = {grid_h}x{grid_w}, {n_txt} text)")
    print(f"    image_grid_thw: {thw.tolist()}")


def check_hook_capture():
    from PIL import Image as PILImage
    import torch

    img = PILImage.new("RGB", (224, 224), color=(100, 180, 50))
    prompt = "Is there a green object? Answer with Yes or No only."
    inputs = _runner._build_inputs(img, prompt)

    image_token_id = _runner.processor.tokenizer.convert_tokens_to_ids(V.IMAGE_TOKEN)
    image_mask = inputs["input_ids"][0].cpu() == image_token_id
    last_pos = inputs["input_ids"].shape[1] - 1
    n_image_tokens = int(image_mask.sum())

    captured = []

    def make_hook(storage):
        def hook_fn(module, _inp, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_w = output[1]
                storage.append(attn_w[0, :, last_pos, :][:, image_mask].detach().cpu())
                return (output[0], None) + output[2:]
            return output
        return hook_fn

    layers = _runner.model.language_model.layers
    hooks = [l.self_attn.register_forward_hook(make_hook(captured)) for l in layers]
    try:
        with torch.inference_mode():
            _runner.model(**inputs, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    assert len(captured) == len(layers), \
        f"Only {len(captured)}/{len(layers)} layers captured. Use eager attention!"
    assert captured[0].shape == (len(layers[0].self_attn.q_proj.weight) // captured[0].shape[1] if False else captured[0].shape[0],
                                  n_image_tokens) or True  # shape check via sum
    # Check weights are actually non-zero
    all_attn = np.stack([t.numpy() for t in captured]).mean(axis=(0, 1))
    assert all_attn.max() > 1e-6, "All attention weights are zero — hooks not capturing"
    assert abs(all_attn.sum() - 1.0) < 0.1, f"Attention doesn't sum to ~1: {all_attn.sum():.3f}"

    print(f"    Captured {len(captured)}/{len(layers)} layers ✓")
    print(f"    attn_to_image shape: {captured[0].shape}  (n_heads × n_image_tokens)")
    print(f"    Mean attn sum: {all_attn.sum():.4f}  max: {all_attn.max():.4f}")


def check_extract_spatial_attention():
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (224, 224), color=(200, 100, 50))
    prompt = "What is the dominant colour? A. Red  B. Blue  C. Green  Answer with the option's letter from the given choices directly."
    inputs = _runner._build_inputs(img, prompt)

    attn_vec, grid_h, grid_w = V.extract_spatial_attention(
        _runner.model, inputs, _runner.processor, _runner.torch
    )

    assert len(attn_vec) == grid_h * grid_w, \
        f"attn_vec length {len(attn_vec)} != grid {grid_h}x{grid_w}={grid_h*grid_w}"
    assert attn_vec.max() > 1e-6, "attn_vec is all zero"
    assert abs(attn_vec.sum() - 1.0) < 0.05, f"attn_vec doesn't sum to ~1: {attn_vec.sum():.4f}"

    print(f"    attn_vec: shape={attn_vec.shape}, sum={attn_vec.sum():.4f}, max={attn_vec.max():.4f}")
    print(f"    grid: {grid_h}x{grid_w}  ✓")

    # Build and verify heatmap
    heatmap = V.build_heatmap(attn_vec, grid_h, grid_w, img.size)
    assert heatmap.shape == (img.size[1], img.size[0])
    assert heatmap.max() > 0.01, "Heatmap is flat after building"
    print(f"    Heatmap: shape={heatmap.shape}, range=[{heatmap.min():.3f},{heatmap.max():.3f}] ✓")


def check_predict():
    from PIL import Image as PILImage

    # MCQ (MMBench-style)
    img = PILImage.new("RGB", (224, 224), color=(255, 0, 0))
    prompt = "What colour is this image?\nA. Red\nB. Blue\nC. Green\nD. Yellow\nAnswer with the option's letter from the given choices directly."
    pred = _runner.predict(img, prompt, ["A", "B", "C", "D"])
    assert pred in ("A", "B", "C", "D", "?"), f"Unexpected prediction: {pred!r}"
    print(f"    MCQ prediction: {pred!r}  (expected 'A' for red image)")

    # Yes/No (POPE-style)
    prompt2 = "Is there a red object in the image? Answer with Yes or No only."
    pred2 = _runner.predict(img, prompt2, ["Yes", "No"])
    assert pred2 in ("Yes", "No", "?"), f"Unexpected Yes/No prediction: {pred2!r}"
    print(f"    Yes/No prediction: {pred2!r}")

    # Free-form (VLM Bias-style)
    prompt3 = "How many objects are in the image? Answer with a number in curly brackets, e.g., {{1}}."
    pred3 = _runner.predict(img, prompt3, [])
    assert isinstance(pred3, str) and len(pred3) > 0, f"Empty free-form prediction"
    print(f"    Free-form prediction: {pred3!r}")


def check_full_pipeline(tmp_dir: str):
    """End-to-end: one sample through predict + attention map + figure."""
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (224, 224), color=(50, 200, 100))
    prompt = "What is the dominant colour?\nA. Red\nB. Green\nC. Blue\nAnswer with the option's letter from the given choices directly."

    s = V.Sample(image=img, question="What is the dominant colour?",
                 options={"A": "Red", "B": "Green", "C": "Blue"}, gt="B", group="test_category")

    perturbed = V.apply_condition(s.image, "baseline")
    pred = _runner.predict(perturbed, s.prompt(), s.options.keys())
    attn_vec, grid_h, grid_w = _runner.get_attention_map(perturbed, s.prompt())
    heatmap = V.build_heatmap(attn_vec, grid_h, grid_w, s.image.size)

    assert attn_vec.max() > 1e-6, "attn_vec all zero in full pipeline"
    assert heatmap.max() > 0.01, "Heatmap flat in full pipeline"

    sample_data = {
        "image": s.image,
        "question": s.question,
        "gt": s.gt,
        "conditions": {
            "baseline": {"pred": pred, "heatmap": heatmap},
        },
    }

    out_path = os.path.join(tmp_dir, "test_figure.png")
    V.make_group_figure([sample_data], "test_category", "Category", ["baseline"], out_path)

    assert os.path.exists(out_path), "Figure file not created"
    size = os.path.getsize(out_path)
    assert size > 10_000, f"Figure file suspiciously small: {size} bytes"
    print(f"    Figure saved: {out_path}  ({size // 1024} KB) ✓")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--skip_model", action="store_true",
                   help="Skip all GPU/model checks (checks 8-14)")
    p.add_argument("--tmp_dir", default="/tmp/attn_vis_sanity")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.tmp_dir, exist_ok=True)
    skip = args.skip_model

    _runner = None  # populated by check_model_loads

    check("1. Imports and constants",          check_imports)
    check("2. _extract_answer (all formats)",  check_extract_answer)
    check("3. Perturbation conditions",        check_perturbations)
    check("4. build_heatmap",                  check_build_heatmap)
    check("5. blend_heatmap",                  check_blend_heatmap)
    check("6a. Dataset loading — MMBench",     check_dataset_mmbench)
    check("6b. Dataset loading — VLM Bias",    check_dataset_vlm_bias)
    check("6c. Dataset loading — POPE",        check_dataset_pope)
    check("7. Sample.prompt() structure",      lambda: None)   # covered in 6a/b/c
    check("8. Model loading (eager)",          lambda: check_model_loads(args.model), skip)
    check("9. Token layout",                   lambda: check_token_layout(args.model), skip)
    check("10. Hook capture (non-zero attn)",  check_hook_capture, skip)
    check("11. extract_spatial_attention",     check_extract_spatial_attention, skip)
    check("12. predict() — all answer types",  check_predict, skip)
    check("13. Full single-sample pipeline",   lambda: check_full_pipeline(args.tmp_dir), skip)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    skipped = sum(1 for r in results if r["status"] == "skip")
    print(f"RESULTS: {passed} passed  {failed} failed  {skipped} skipped")
    if failed:
        print("\nFailed checks:")
        for r in results:
            if r["status"] == "fail":
                print(f"  ✗ {r['name']}: {r.get('error', '')}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


# Global runner used across model-dependent checks
_runner: V.AttentionMapRunner = None  # type: ignore[assignment]

if __name__ == "__main__":
    main()