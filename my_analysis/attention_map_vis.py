#!/usr/bin/env python3
"""
Attention map visualization for Qwen2.5-VL — MMBench, VLM Bias, POPE.

Visualizes which image regions the decoder LLM attends to when generating
its answer, grouped by dataset-specific categories/topics/splits.

Unlike encoder-rollout approaches (e.g. AdaptVis which hooks into CLIP ViT),
we analyze decoder self-attention to image token positions directly. This shows
what the LLM "looks at" when making its decision -- directly tied to modality
reliance.

Method:
  1. Forward pass with hooks on decoder self-attention layers.
  2. Capture attn[last_input_pos, image_token_positions] per layer.
     shape per layer: (n_heads, n_image_tokens)
  3. Mean over layers and heads -> (n_image_tokens,).
  4. Reshape using image_grid_thw (exact patch grid from Qwen processor).
  5. Bilinear upsample to image resolution -> smooth heatmap overlay.

Datasets
--------
  mmbench   -- grouped by fine_category (20 cats), options A/B/C/D
  vlm_bias  -- grouped by topic (7 topics: Animals, Flags, ...), options A/B/C/D
  pope      -- grouped by split (random/popular/adversarial), binary Yes/No

Usage
-----
  # MMBench: all categories, 3 samples each, baseline only
  python attention_map_vis.py --dataset mmbench --n_per_group 3

  # MMBench: specific categories, compare baseline vs low_res
  python attention_map_vis.py --dataset mmbench \\
    --groups spatial_relationship ocr physical_relation \\
    --conditions baseline low_res_x8

  # VLM Bias: all topics, 4 samples each
  python attention_map_vis.py --dataset vlm_bias --n_per_group 4

  # VLM Bias: specific topics
  python attention_map_vis.py --dataset vlm_bias \\
    --groups Animals Flags Optical_Illusion

  # POPE: adversarial split, 5 samples, compare baseline vs patch_shuffle
  python attention_map_vis.py --dataset pope \\
    --groups adversarial \\
    --conditions baseline patch_shuffle16 \\
    --n_per_group 5
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

# Pillow ≥9.1 moved resampling filters to Image.Resampling; keep backward compat
try:
    _BILINEAR: int = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
except AttributeError:
    _BILINEAR = 2  # BILINEAR integer value

DEFAULT_SUFFIX = "Answer with the option's letter from the given choices directly."
YESNO_SUFFIX = "Answer with Yes or No only."
IMAGE_TOKEN = "<|image_pad|>"


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------


class Sample:
    """Unified sample representation across datasets."""

    def __init__(
        self,
        image: Image.Image,
        question: str,
        options: Dict[str, str],  # e.g. {"A": "cat", "B": "dog"} or {"Yes": "Yes", "No": "No"}
        gt: str,                  # e.g. "A" or "Yes"
        group: str,               # category / topic / split name
        prompt_suffix: str = DEFAULT_SUFFIX,
    ) -> None:
        self.image = image
        self.question = question
        self.options = options
        self.gt = gt
        self.group = group
        self.prompt_suffix = prompt_suffix

    def prompt(self) -> str:
        lines = [self.question]
        for k, v in self.options.items():
            lines.append(f"{k}. {v}")
        if self.prompt_suffix:
            lines.append(self.prompt_suffix)
        return "\n".join(lines)


def _to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB") if image.mode != "RGB" else image


# --- MMBench ---

def load_mmbench(groups_filter: Optional[List[str]]) -> Dict[str, List[Sample]]:
    from datasets import load_dataset

    print("Loading MMBench-Dev (EN)...")
    # Columns: question, answer, A, B, C, D, category, hint, L2-category, image, ...
    ds = load_dataset("lmms-lab/MMBench", "en", split="dev")

    by_group: Dict[str, List[Sample]] = defaultdict(list)
    for row in ds:
        cat = str(row.get("category", "unknown")).strip()
        if groups_filter and cat not in groups_filter:
            continue
        options = _mmbench_options(row)
        if not options:
            continue
        gt = str(row.get("answer", "")).strip().upper()
        question = str(row.get("question", "")).strip()
        hint = str(row.get("hint", "") or "").strip()
        if hint and hint.lower() != "nan":
            question = f"{hint}\n{question}"
        s = Sample(
            image=_to_rgb(row["image"]),
            question=question,
            options=options,
            gt=gt,
            group=cat,
        )
        by_group[cat].append(s)

    return dict(by_group)


def _mmbench_category(row: Dict[str, Any]) -> str:
    return str(row.get("category", "unknown")).strip()


def _mmbench_options(row: Dict[str, Any]) -> Dict[str, str]:
    opts: Dict[str, str] = {}
    for letter in ["A", "B", "C", "D"]:
        v = row.get(letter)
        if v is None:
            continue
        text = str(v).strip()
        if text and text.lower() != "nan":
            opts[letter] = text
    return opts


# --- VLM Bias ---

def load_vlm_bias(groups_filter: Optional[List[str]]) -> Dict[str, List[Sample]]:
    from datasets import load_dataset

    print("Loading VLM Bias (anvo25/vlms-are-biased)...")
    # Dataset columns: image, topic, prompt, ground_truth, expected_bias, ...
    # Answers are free-form (e.g. "Yes", "8"), NOT A/B/C/D multiple-choice.
    # Prompts already include answer format instructions (e.g. "Answer in {Yes} or {No}").
    ds = load_dataset("anvo25/vlms-are-biased", split="main")

    by_group: Dict[str, List[Sample]] = defaultdict(list)
    for row in ds:
        topic = str(row.get("topic", "unknown")).strip().replace(" ", "_")
        if groups_filter and topic not in groups_filter:
            continue
        prompt = str(row.get("prompt", "")).strip()
        gt = str(row.get("ground_truth", "")).strip()
        if not prompt or not gt:
            continue
        s = Sample(
            image=_to_rgb(row["image"]),
            question=prompt,       # prompt already complete, used verbatim
            options={},            # free-form — no A/B/C/D options
            gt=gt,
            group=topic,
            prompt_suffix="",      # suffix already embedded in prompt
        )
        by_group[topic].append(s)

    return dict(by_group)


# --- POPE ---

def load_pope(groups_filter: Optional[List[str]]) -> Dict[str, List[Sample]]:
    from datasets import load_dataset

    print("Loading POPE (lmms-lab/POPE)...")
    # Single split='test', grouped by the 'category' column: adversarial/popular/random
    # Columns: id, question_id, question, answer (yes/no), image_source, image, category
    ds = load_dataset("lmms-lab/POPE", split="test")
    # Default to adversarial only; allow override via --groups
    targets = [g.lower() for g in groups_filter] if groups_filter else ["adversarial"]

    by_group: Dict[str, List[Sample]] = defaultdict(list)
    for row in ds:
        cat = str(row.get("category", "unknown")).strip().lower()
        if cat not in targets:
            continue
        question = str(row.get("question", "")).strip()
        label = str(row.get("answer", "")).strip().lower()
        gt = "Yes" if label == "yes" else "No"
        s = Sample(
            image=_to_rgb(row["image"]),
            question=question,
            options={"Yes": "Yes", "No": "No"},
            gt=gt,
            group=cat,
            prompt_suffix=YESNO_SUFFIX,
        )
        by_group[cat].append(s)

    return dict(by_group)


DATASET_LOADERS = {
    "mmbench": load_mmbench,
    "vlm_bias": load_vlm_bias,
    "pope": load_pope,
}

DATASET_GROUP_LABEL = {
    "mmbench": "Category",
    "vlm_bias": "Topic",
    "pope": "Split",
}


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------


def apply_condition(image: Image.Image, condition: str) -> Image.Image:
    import PIL.ImageFilter as F

    if condition == "baseline":
        return image
    elif condition == "low_res_x8":
        w, h = image.size
        small = image.resize((max(1, w // 8), max(1, h // 8)), _BILINEAR)
        return small.resize((w, h), _BILINEAR)
    elif condition == "blur_r5":
        return image.filter(F.GaussianBlur(radius=5))
    elif condition == "patch_shuffle16":
        return _patch_shuffle(image, patch_size=16)
    elif condition == "center_mask40":
        return _center_mask(image, frac=0.4)
    else:
        raise ValueError(f"Unknown condition: {condition!r}")


def _patch_shuffle(image: Image.Image, patch_size: int = 16) -> Image.Image:
    arr = np.array(image)
    h, w = arr.shape[:2]
    patches, coords = [], []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patches.append(arr[y: y + patch_size, x: x + patch_size].copy())
            coords.append((y, x))
    random.shuffle(patches)
    out = arr.copy()
    for (y, x), patch in zip(coords, patches):
        out[y: y + patch_size, x: x + patch_size] = patch
    return Image.fromarray(out)


def _center_mask(image: Image.Image, frac: float = 0.4) -> Image.Image:
    arr = np.array(image).copy()
    h, w = arr.shape[:2]
    mh, mw = int(h * frac), int(w * frac)
    y0, x0 = (h - mh) // 2, (w - mw) // 2
    arr[y0: y0 + mh, x0: x0 + mw] = 128
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------


def extract_spatial_attention(
    model: Any,
    inputs: Any,
    processor: Any,
    torch: Any,
    spatial_merge_size: int = 2,
) -> Tuple[np.ndarray, int, int]:
    """
    Run one forward pass and return (attn_vec, grid_h, grid_w).

    attn_vec: (n_image_tokens,) normalised mean attention from the last input
              position to each image patch token, averaged over all layers+heads.
    grid_h/w: MERGED patch grid (after Qwen2.5-VL spatial_merge_size reduction).

    IMPORTANT: image_grid_thw contains the grid BEFORE spatial merging.
    Qwen2.5-VL uses spatial_merge_size=2 by default, so the actual token grid
    is grid_h//2 × grid_w//2. Using the pre-merge grid caused zero-padded
    heatmaps (tokens filled only the top-left corner of the larger grid).
    """
    image_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    image_mask = inputs["input_ids"][0].cpu() == image_token_id
    last_pos = inputs["input_ids"].shape[1] - 1
    n_image_tokens = int(image_mask.sum().item())

    # image_grid_thw is (T, H, W) BEFORE spatial merge.
    # After merge with spatial_merge_size=2: merged_h = H//2, merged_w = W//2.
    thw = inputs["image_grid_thw"][0]
    grid_h = int(thw[1].item()) // spatial_merge_size
    grid_w = int(thw[2].item()) // spatial_merge_size

    captured: List[Any] = []  # (n_heads, n_image_tokens) per layer

    def make_hook(storage: List[Any]) -> Any:
        def hook_fn(module: Any, _inp: Any, output: Any) -> Any:
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_w = output[1]  # (1, n_heads, seq_len, seq_len)
                # Attention from last input position -> image token positions only
                attn_to_image = attn_w[0, :, last_pos, :][:, image_mask].detach().cpu()
                storage.append(attn_to_image)  # (n_heads, n_image_tokens)
                return (output[0], None) + output[2:]  # free GPU memory
            return output
        return hook_fn

    try:
        layers = model.language_model.layers
    except AttributeError as exc:
        raise RuntimeError("Cannot access model.language_model.layers") from exc

    hooks = [layer.self_attn.register_forward_hook(make_hook(captured)) for layer in layers]
    try:
        with torch.inference_mode():
            model(**inputs, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    if not captured:
        return np.zeros(n_image_tokens), grid_h, grid_w

    # (n_layers, n_heads, n_image_tokens) -> mean over layers and heads
    stacked = np.stack([t.float().numpy() for t in captured], axis=0)
    mean_attn = stacked.mean(axis=(0, 1))  # (n_image_tokens,)
    total = mean_attn.sum()
    if total > 1e-9:
        mean_attn /= total

    return mean_attn, grid_h, grid_w


# ---------------------------------------------------------------------------
# Heatmap construction
# ---------------------------------------------------------------------------


def build_heatmap(
    attn_vec: np.ndarray,
    grid_h: int,
    grid_w: int,
    image_size: Tuple[int, int],  # PIL (W, H)
) -> np.ndarray:
    """
    Reshape and bilinearly upsample attention to image resolution.

    Unlike AdaptVis (nearest-neighbour on hardcoded 24x24), we use the exact
    Qwen grid dims and bilinear interpolation for smooth paper-ready maps.

    Returns:
        (H, W) float32 array in [0, 1].
    """
    expected = grid_h * grid_w
    if len(attn_vec) != expected:
        padded = np.zeros(expected, dtype=np.float32)
        padded[: min(len(attn_vec), expected)] = attn_vec[:expected]
        attn_vec = padded

    patch_grid = attn_vec.reshape(grid_h, grid_w).astype(np.float32)

    vmin, vmax = patch_grid.min(), patch_grid.max()
    if vmax - vmin > 1e-9:
        patch_grid = (patch_grid - vmin) / (vmax - vmin)

    img_w, img_h = image_size
    pil_grid = Image.fromarray((patch_grid * 255).astype(np.uint8), mode="L")
    pil_resized = pil_grid.resize((img_w, img_h), resample=_BILINEAR)
    return np.array(pil_resized).astype(np.float32) / 255.0


def blend_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    colored = (plt.get_cmap("plasma")(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return (image * (1 - alpha) + colored * alpha).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------


class AttentionMapRunner:
    def __init__(self, model_name: str, attn_implementation: str = "sdpa") -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError("Install qwen-vl-utils: uv add qwen-vl-utils") from exc

        if attn_implementation != "eager":
            raise ValueError(
                f"attn_implementation must be 'eager' for output_attentions=True to work. "
                f"SDPA and flash_attention_2 both return None for attention weights. Got: {attn_implementation!r}"
            )

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        ).eval()
        # max_pixels caps the ViT input resolution to avoid OOM on large images.
        # 512 * 28 * 28 = 401408 pixels ~ 634x634 effective resolution.
        self.processor = AutoProcessor.from_pretrained(
            model_name, max_pixels=512 * 28 * 28
        )
        # Qwen2.5-VL merges vision patches spatially before feeding to the LLM.
        # spatial_merge_size=2 means actual token grid is (H//2) × (W//2).
        vision_cfg = getattr(self.model.config, "vision_config", None)
        self.spatial_merge_size: int = int(getattr(vision_cfg, "spatial_merge_size", 2))

    def _build_inputs(self, image: Image.Image, prompt: str) -> Any:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        return inputs.to(next(self.model.parameters()).device)

    def predict(self, image: Image.Image, prompt: str, valid_keys: Iterable[str]) -> str:
        inputs = self._build_inputs(image, prompt)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                      temperature=None, top_p=None, num_beams=1)
        raw = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        self.torch.cuda.empty_cache()
        return _extract_answer(raw, valid_keys)

    def get_attention_map(self, image: Image.Image, prompt: str) -> Tuple[np.ndarray, int, int]:
        inputs = self._build_inputs(image, prompt)
        result = extract_spatial_attention(self.model, inputs, self.processor, self.torch, self.spatial_merge_size)
        self.torch.cuda.empty_cache()
        return result


def _extract_answer(text: str, valid_keys: Iterable[str]) -> str:
    raw = (text or "").strip()
    keys = list(valid_keys)

    # Free-form dataset (e.g. VLM Bias): extract content inside {curly braces}
    if not keys:
        m = re.search(r"\{([^{]+)\}", raw)
        return m.group(1).strip() if m else raw[:20]

    # Multiple-choice or Yes/No: word match against valid keys
    upper = raw.upper()
    for k in keys:
        if re.search(rf"\b{re.escape(k.upper())}\b", upper):
            return k
    return "?"


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def make_group_figure(
    samples: List[Dict[str, Any]],
    group_name: str,
    group_label: str,  # "Category" / "Topic" / "Split"
    conditions: List[str],
    output_path: str,
) -> None:
    """
    Save one figure per group (category/topic/split).

    Layout:
      Columns: n_samples × n_conditions
      Row 0:   original image (with condition label if >1 condition)
      Row 1:   attention overlay  (plasma, bilinear smooth, coloured border ✓/✗)
      Row 2:   question snippet + GT + prediction
    """
    n_samples = len(samples)
    if not n_samples:
        return

    n_conds = len(conditions)
    n_cols = n_samples * n_conds

    fig_w = max(4 * n_cols, 8)
    fig = plt.figure(figsize=(fig_w, 11.5), facecolor="white")
    gs = gridspec.GridSpec(
        3, n_cols, figure=fig,
        hspace=0.06, wspace=0.05,
        height_ratios=[3, 3, 1.1],
    )

    col = 0
    for s_idx, sample_data in enumerate(samples):
        image_rgb = np.array(_to_rgb(sample_data["image"]))
        question = sample_data["question"]
        gt = sample_data["gt"]

        for cond in conditions:
            cond_data = sample_data["conditions"][cond]
            pred = cond_data["pred"]
            heatmap = cond_data["heatmap"]
            blended = blend_heatmap(image_rgb, heatmap)
            correct = pred.upper() == gt.upper()

            # Row 0 — original image
            ax_img = fig.add_subplot(gs[0, col])
            ax_img.imshow(image_rgb)
            ax_img.axis("off")
            title = f"Cond: {cond}" if n_conds > 1 else f"Sample {s_idx + 1}"
            ax_img.set_title(title, fontsize=8, pad=3, color="#444444")

            # Row 1 — attention overlay
            ax_heat = fig.add_subplot(gs[1, col])
            ax_heat.imshow(blended)
            ax_heat.axis("off")
            border_color = "#2ecc71" if correct else "#e74c3c"
            for spine in ax_heat.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)

            # Row 2 — text annotation
            ax_txt = fig.add_subplot(gs[2, col])
            ax_txt.axis("off")
            q_short = (question[:65] + "…") if len(question) > 65 else question
            verdict = "✓" if correct else "✗"
            verdict_color = "#27ae60" if correct else "#c0392b"
            ax_txt.text(0.5, 0.78, q_short, transform=ax_txt.transAxes,
                        fontsize=6.5, ha="center", va="top", color="#222222")
            ax_txt.text(0.5, 0.28, f"GT: {gt}   Pred: {pred} {verdict}",
                        transform=ax_txt.transAxes, fontsize=7.5, ha="center",
                        va="top", color=verdict_color, fontweight="bold")
            col += 1

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.35, 0.012, 0.3])
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Attention", fontsize=8, labelpad=4)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["low", "mid", "high"], fontsize=7)

    fig.suptitle(
        f"Decoder Attention Maps — {group_label}: {group_name.replace('_', ' ').title()}",
        fontsize=11, fontweight="bold", y=0.995, color="#111111",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    loader = DATASET_LOADERS[args.dataset]
    group_label = DATASET_GROUP_LABEL[args.dataset]

    by_group = loader(args.groups if args.groups else None)

    if not by_group:
        print("No groups found. Check --groups values or dataset availability.")
        sys.exit(1)

    print(f"\nGroups available: {sorted(by_group.keys())}")
    if args.list_groups:
        return

    print(f"Groups to process : {sorted(by_group.keys())}")
    print(f"Samples per group : {args.n_per_group}")
    print(f"Conditions        : {args.conditions}")
    print(f"Output dir        : {args.output_dir}\n")

    print(f"Loading model: {args.model}")
    runner = AttentionMapRunner(args.model, attn_implementation=args.attn_implementation)

    for group_name, group_samples in sorted(by_group.items()):
        print(f"\n{'='*60}")
        print(f"{group_label}: {group_name}  ({len(group_samples)} total samples)")

        random.seed(args.seed)
        selected = (
            random.sample(group_samples, args.n_per_group)
            if len(group_samples) > args.n_per_group
            else group_samples[: args.n_per_group]
        )

        samples_data: List[Dict[str, Any]] = []
        for s in tqdm(selected, desc=f"  {group_name}", leave=False):
            sample_data: Dict[str, Any] = {
                "image": s.image,
                "question": s.question,
                "gt": s.gt,
                "conditions": {},
            }
            for cond in args.conditions:
                perturbed = apply_condition(s.image, cond)
                pred = runner.predict(perturbed, s.prompt(), s.options.keys())
                attn_vec, grid_h, grid_w = runner.get_attention_map(perturbed, s.prompt())
                heatmap = build_heatmap(attn_vec, grid_h, grid_w, image_size=s.image.size)
                sample_data["conditions"][cond] = {"pred": pred, "heatmap": heatmap}
            samples_data.append(sample_data)

        safe_name = group_name.replace(" ", "_").replace("/", "-")
        out_path = os.path.join(args.output_dir, args.dataset, f"{safe_name}.png")
        make_group_figure(samples_data, group_name, group_label, args.conditions, out_path)

    print(f"\nDone. All figures saved to: {args.output_dir}/{args.dataset}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decoder attention map visualisation — MMBench / VLM Bias / POPE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True, choices=["mmbench", "vlm_bias", "pope"],
                        help="Dataset to run on.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--"
                        ""
                        ""
                        "", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument(
        "--n_per_group", type=int, default=3,
                        help="Number of samples to visualise per group.")
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help=(
            "Which groups to process. Default: all.\n"
            "  mmbench  : fine_category names, e.g. --groups spatial_relationship ocr physical_relation\n"
            "  vlm_bias : topic names,         e.g. --groups Animals Flags Optical_Illusion\n"
            "  pope     : split names,         e.g. --groups adversarial popular\n"
            "Run with --list_groups to see all available names for a dataset."
        ),
    )
    parser.add_argument("--list_groups", action="store_true",
                        help="Print all available group names for the dataset and exit.")
    parser.add_argument(
        "--conditions", nargs="+", default=["baseline"],
        choices=["baseline", "low_res_x8", "blur_r5", "patch_shuffle16", "center_mask40"],
        help="Conditions to show. Use multiple for side-by-side comparison.",
    )
    parser.add_argument("--output_dir", default="results/attention_maps")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
