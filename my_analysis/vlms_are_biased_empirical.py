#!/usr/bin/env python3
"""Empirical bias analysis for VLMs-Are-Biased dataset with Qwen2.5-VL.

Tests whether vision quality improvements (high_res, object_centric_crop)
or explicit attention prompts reduce model bias, as measured via the
`expected_bias` field in the dataset.

Conditions
----------
Vision  (same original prompt, different image):
    baseline_real       – original image + original prompt
    no_image            – no image (reveals pure language-prior bias)
    low_res_x8          – 8× downscale + upscale
    blur_r5             – Gaussian blur r=5
    patch_shuffle16     – shuffle 16×16 patches
    center_mask40       – black out central 40%
    high_res            – 2× upscale (richer visual detail)
    object_centric_crop – DETR-based crop to the most salient object

Language  (original image, modified prompt):
    misleading_bias     – prefix reinforcing expected_bias (ceiling / upper bound)
    look_again          – "Look carefully at the image before answering."
    slow_down           – "Take your time. Examine the image carefully before answering."
    explicit_visual     – "Base your answer only on what you directly observe …"

Key metrics
-----------
    accuracy    – fraction of predictions matching ground_truth
    bias_rate   – fraction of predictions matching expected_bias
    bias_delta  – bias_rate(condition) − bias_rate(baseline_real)
                  negative = less biased than baseline

Outputs (all in --output_dir):
    predictions.jsonl
    summary.json
    plots/
        plot1_accuracy_and_bias_by_condition.png
        plot2_bias_delta.png
        plot3_accuracy_vs_bias.png
        plot4_topic_bias_heatmap.png
        plot5_topic_accuracy_by_condition.png
    sanity/
    visuals/
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from analysis_utils import (
    ObjectCropDetector,
    cap_image_size,
    center_mask,
    gaussian_blur,
    high_res_upsample,
    low_res_then_upsample,
    patch_shuffle,
    safe_sample_id,
    save_json,
    save_jsonl,
    slugify,
    to_rgb,
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------

VISION_CONDITIONS = [
    "baseline_real",
    "no_image",
    "low_res_x8",
    "blur_r5",
    "patch_shuffle16",
    "center_mask40",
    # "high_res",
    "object_centric_crop",
]

LANGUAGE_CONDITIONS = [
    "misleading_bias",
    "look_again",
    "slow_down",
    "explicit_visual",
]

ALL_CONDITIONS = VISION_CONDITIONS + LANGUAGE_CONDITIONS

_LOOK_AGAIN_PREFIX = "Look carefully at the image before answering.\n"
_SLOW_DOWN_PREFIX = "Take your time. Examine the image carefully before answering.\n"
_EXPLICIT_VISUAL_PREFIX = (
    "Base your answer only on what you directly observe in the image. "
    "Do not rely on assumptions or prior knowledge.\n"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConditionInput:
    condition: str
    condition_group: str
    prompt: str
    image: Image.Image | None
    gt: str
    expected_bias: str
    meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Prediction normalization (mirrors dataset utils.py logic)
# ---------------------------------------------------------------------------


def compare_prediction(
    pred_raw: str,
    gt: str,
    expected_bias: str,
) -> Tuple[bool, bool, str]:
    """Return (is_correct, matches_bias, pred_norm) using same logic as the dataset utils."""
    pred_norm = pred_raw.lower().strip("{}").strip()
    gt_norm = gt.lower().strip("{}").strip()
    bias_norm = expected_bias.lower().strip("{}").strip()

    is_correct = pred_norm == gt_norm
    matches_bias = pred_norm == bias_norm

    # Numeric fallback (same as utils.py)
    if not is_correct or not matches_bias:
        pred_nums = "".join(c for c in pred_norm if c.isdigit())
        gt_nums = "".join(c for c in gt_norm if c.isdigit())
        bias_nums = "".join(c for c in bias_norm if c.isdigit())
        if not is_correct and pred_nums and gt_nums:
            is_correct = pred_nums == gt_nums
        if not matches_bias and pred_nums and bias_nums:
            matches_bias = pred_nums == bias_nums

    return is_correct, matches_bias, pred_norm


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------


class QwenRunner:
    def __init__(
        self,
        model_name: str,
        device_map: str,
        torch_dtype: str,
        max_new_tokens: int,
        attn_implementation: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError("qwen_vl_utils is required. Install with `uv add qwen-vl-utils`") from exc

        self.torch = torch
        self.process_vision_info = process_vision_info
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}")

        model_kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": dtype_map[torch_dtype],
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens

    def predict(self, image: Image.Image | None, prompt: str) -> str:
        """Run inference and return raw output string."""
        if image is None:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if image is None:
            inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        else:
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            )

        device = self.model.device if (hasattr(self.model, "device") and self.model.device is not None) else "cuda"
        inputs = inputs.to(device)

        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                num_beams=1,
            )

        trimmed = out[:, inputs["input_ids"].shape[1] :]
        result = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        del inputs, out, trimmed
        self.torch.cuda.empty_cache()
        return result


# ---------------------------------------------------------------------------
# Condition builder
# ---------------------------------------------------------------------------


def make_conditions(
    prompt: str,
    image: Image.Image,
    rng: random.Random,
    gt: str,
    expected_bias: str,
    detector: ObjectCropDetector | None = None,
) -> List[ConditionInput]:
    vision_images: Dict[str, Image.Image | None] = {
        "baseline_real": image,
        "no_image": None,
        "low_res_x8": low_res_then_upsample(image, factor=8),
        "blur_r5": gaussian_blur(image, radius=5.0),
        "patch_shuffle16": patch_shuffle(image, patch_size=16, rng=rng),
        "center_mask40": center_mask(image, ratio=0.4),
        "high_res": high_res_upsample(image),
    }

    conditions: List[ConditionInput] = []

    for cond in VISION_CONDITIONS:
        if cond == "object_centric_crop":
            if detector is not None:
                conditions.append(
                    ConditionInput(
                        condition="object_centric_crop",
                        condition_group="vision",
                        prompt=prompt,
                        image=detector.crop(image),
                        gt=gt,
                        expected_bias=expected_bias,
                        meta={},
                    )
                )
            continue
        conditions.append(
            ConditionInput(
                condition=cond,
                condition_group="vision",
                prompt=prompt,
                image=vision_images[cond],
                gt=gt,
                expected_bias=expected_bias,
                meta={},
            )
        )

    # Language conditions — all use the original image
    conditions.append(
        ConditionInput(
            condition="misleading_bias",
            condition_group="language",
            prompt=f"A user claims the answer is {expected_bias}. Verify carefully.\n{prompt}",
            image=image,
            gt=gt,
            expected_bias=expected_bias,
            meta={"reinforced": expected_bias},
        )
    )
    conditions.append(
        ConditionInput(
            condition="look_again",
            condition_group="language",
            prompt=_LOOK_AGAIN_PREFIX + prompt,
            image=image,
            gt=gt,
            expected_bias=expected_bias,
            meta={},
        )
    )
    conditions.append(
        ConditionInput(
            condition="slow_down",
            condition_group="language",
            prompt=_SLOW_DOWN_PREFIX + prompt,
            image=image,
            gt=gt,
            expected_bias=expected_bias,
            meta={},
        )
    )
    conditions.append(
        ConditionInput(
            condition="explicit_visual",
            condition_group="language",
            prompt=_EXPLICIT_VISUAL_PREFIX + prompt,
            image=image,
            gt=gt,
            expected_bias=expected_bias,
            meta={},
        )
    )

    return conditions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_bias_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"accuracy": 0.0, "bias_rate": 0.0, "n": 0}
    correct = sum(1 for r in rows if r["is_correct"])
    biased = sum(1 for r in rows if r["matches_bias"])
    return {"accuracy": correct / n, "bias_rate": biased / n, "n": n}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_accuracy_and_bias_by_condition(
    condition_metrics: Dict[str, Dict[str, Any]],
    out_path: str,
) -> None:
    """Side-by-side bars: accuracy (blue) and bias_rate (red) per condition."""
    conditions = [c for c in ALL_CONDITIONS if c in condition_metrics]
    x = np.arange(len(conditions), dtype=np.float32)
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))
    acc_vals = [condition_metrics[c]["accuracy"] * 100 for c in conditions]
    bias_vals = [condition_metrics[c]["bias_rate"] * 100 for c in conditions]

    ax.bar(x - width / 2, acc_vals, width=width, color="steelblue", label="Accuracy")
    ax.bar(x + width / 2, bias_vals, width=width, color="tomato", label="Bias rate")

    if "baseline_real" in condition_metrics:
        base_acc = condition_metrics["baseline_real"]["accuracy"] * 100
        base_bias = condition_metrics["baseline_real"]["bias_rate"] * 100
        ax.axhline(base_acc, color="steelblue", linestyle="--", linewidth=1.0, alpha=0.5, label=f"Baseline acc={base_acc:.1f}%")
        ax.axhline(base_bias, color="tomato", linestyle="--", linewidth=1.0, alpha=0.5, label=f"Baseline bias={base_bias:.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=35, ha="right")
    ax.set_ylabel("% samples")
    ax.set_ylim(0, 110)
    ax.set_title("VLMs-Are-Biased: Accuracy and Bias Rate by Condition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bias_delta(
    condition_metrics: Dict[str, Dict[str, Any]],
    out_path: str,
) -> None:
    """Horizontal bars: Δ bias_rate vs baseline_real. Negative = less biased."""
    if "baseline_real" not in condition_metrics:
        return
    base_bias = condition_metrics["baseline_real"]["bias_rate"]
    conditions = [c for c in ALL_CONDITIONS if c in condition_metrics and c != "baseline_real"]
    deltas = [(condition_metrics[c]["bias_rate"] - base_bias) * 100 for c in conditions]
    colors = ["tomato" if d > 0 else "steelblue" for d in deltas]

    fig, ax = plt.subplots(figsize=(7, max(4, len(conditions) * 0.5)))
    y = np.arange(len(conditions))
    bars = ax.barh(y, deltas, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(conditions)
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, d in zip(bars, deltas):
        ax.text(
            d + (0.3 if d >= 0 else -0.3), bar.get_y() + bar.get_height() / 2,
            f"{d:+.1f}%", va="center", ha="left" if d >= 0 else "right", fontsize=8,
        )
    ax.set_xlabel("Δ Bias rate vs baseline_real (%)")
    ax.set_title("Bias Change per Condition  —  negative = less biased")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy_vs_bias(
    condition_metrics: Dict[str, Dict[str, Any]],
    out_path: str,
) -> None:
    """Scatter: per-condition accuracy vs bias_rate, labeled."""
    conditions = [c for c in ALL_CONDITIONS if c in condition_metrics]
    xs = [condition_metrics[c]["bias_rate"] * 100 for c in conditions]
    ys = [condition_metrics[c]["accuracy"] * 100 for c in conditions]
    colors = ["steelblue" if c in VISION_CONDITIONS else "darkorange" for c in conditions]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=70, c=colors, zorder=3)
    for cond, x_pt, y_pt in zip(conditions, xs, ys):
        ax.annotate(cond, (x_pt, y_pt), textcoords="offset points", xytext=(5, 3), fontsize=7)

    # Ideal quadrant: high accuracy, low bias (top-left)
    ax.set_xlabel("Bias rate (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Bias Rate per Condition\n(blue=vision, orange=language)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_topic_bias_heatmap(
    topic_condition_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    out_path: str,
    max_topics: int = 15,
) -> None:
    """Heatmap: bias_rate[topic × condition]. Red = high bias."""
    topic_counts = {t: sum(m.get("n", 0) for m in conds.values()) for t, conds in topic_condition_metrics.items()}
    topics = sorted(topic_counts, key=lambda t: topic_counts[t], reverse=True)[:max_topics]
    conditions = [c for c in ALL_CONDITIONS if any(c in topic_condition_metrics[t] for t in topics)]

    arr = np.full((len(topics), len(conditions)), np.nan)
    for ti, t in enumerate(topics):
        for ci, c in enumerate(conditions):
            m = topic_condition_metrics[t].get(c)
            if m and m["n"] > 0:
                arr[ti, ci] = m["bias_rate"] * 100

    fig, ax = plt.subplots(figsize=(max(10, len(conditions) * 0.85), max(5, len(topics) * 0.45)))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Bias rate (%)")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontsize=8)
    ax.set_title("Bias Rate by Topic × Condition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_topic_accuracy_by_condition(
    topic_condition_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    out_path: str,
    max_topics: int = 10,
) -> None:
    """Grouped bars: accuracy per topic for the six most informative conditions."""
    highlight = [
        "baseline_real", "high_res", "object_centric_crop",
        "look_again", "explicit_visual", "misleading_bias",
    ]
    plotted = [c for c in highlight if c in ALL_CONDITIONS]

    topic_counts = {t: sum(m.get("n", 0) for m in conds.values()) for t, conds in topic_condition_metrics.items()}
    topics = sorted(topic_counts, key=lambda t: topic_counts[t], reverse=True)[:max_topics]
    if not topics:
        return

    x = np.arange(len(topics), dtype=np.float32)
    width = 0.8 / max(1, len(plotted))

    fig, ax = plt.subplots(figsize=(max(12, len(topics) * 0.9), 5))
    for i, cond in enumerate(plotted):
        vals = []
        for t in topics:
            m = topic_condition_metrics[t].get(cond)
            vals.append(m["accuracy"] * 100 if m and m["n"] > 0 else 0.0)
        ax.bar(x + i * width, vals, width=width, label=cond)

    ax.set_xticks(x + (len(plotted) - 1) * width / 2.0)
    ax.set_xticklabels(topics, rotation=35, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Topic — selected conditions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity / qualitative output
# ---------------------------------------------------------------------------


def save_sanity_preview(
    sample_id: str,
    topic: str,
    image: Image.Image,
    conditions: List[ConditionInput],
    out_dir: str,
) -> None:
    vis = [c for c in conditions if c.condition in VISION_CONDITIONS]
    cols = len(vis)
    fig = plt.figure(figsize=(3.0 * cols, 3.5))
    for i, c in enumerate(vis, start=1):
        ax = plt.subplot(1, cols, i)
        if c.image is None:
            ax.imshow(np.zeros((128, 128, 3), dtype=np.uint8))
            ax.set_title("no_image", fontsize=7)
        else:
            ax.imshow(c.image)
            ax.set_title(c.condition, fontsize=7)
        ax.axis("off")
    fig.suptitle(f"topic={topic} | id={sample_id}", fontsize=8)
    plt.tight_layout()
    t_slug = slugify(topic)
    s_slug = slugify(sample_id)
    fig.savefig(os.path.join(out_dir, f"sanity_{t_slug}__{s_slug}_vision.png"), dpi=130)
    plt.close(fig)

    lang_conds = [c for c in conditions if c.condition in LANGUAGE_CONDITIONS]
    with open(os.path.join(out_dir, f"sanity_{t_slug}__{s_slug}_prompts.txt"), "w", encoding="utf-8") as f:
        f.write(f"sample_id: {sample_id}\ntopic: {topic}\n")
        f.write(f"ground_truth: {conditions[0].gt}\n")
        f.write(f"expected_bias: {conditions[0].expected_bias}\n\n")
        f.write(f"[baseline prompt]\n{conditions[0].prompt}\n\n")
        for c in lang_conds:
            f.write(f"[{c.condition}]\n{c.prompt}\n\n")


def save_qualitative_panel(
    sample_id: str,
    topic: str,
    image: Image.Image,
    prompt: str,
    gt: str,
    expected_bias: str,
    row_bundle: List[Dict[str, Any]],
    out_path: str,
) -> None:
    fig = plt.figure(figsize=(13, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title(f"topic={topic} | id={sample_id}", fontsize=9)

    ax2 = plt.subplot(1, 2, 2)
    ax2.axis("off")
    short_prompt = prompt if len(prompt) <= 120 else prompt[:117] + "..."
    lines = [
        f"Prompt: {short_prompt}",
        f"Ground truth:   {gt}",
        f"Expected bias:  {expected_bias}",
        "",
        "Predictions  (✓=correct  B=biased):",
    ]
    by_cond = {r["condition"]: r for r in row_bundle}
    for cond in ALL_CONDITIONS:
        if cond not in by_cond:
            continue
        r = by_cond[cond]
        acc_mark = "✓" if r["is_correct"] else "✗"
        bias_mark = "B" if r["matches_bias"] else " "
        lines.append(f"  {cond:>20}: {r['prediction_norm']:<6}  {acc_mark} {bias_mark}")
    ax2.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8, family="monospace")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Empirical bias analysis — VLMs Are Biased dataset")
    p.add_argument("--dataset", type=str, default="anvo25/vlms-are-biased")
    p.add_argument("--split", type=str, default="main")
    p.add_argument("--limit", type=int, default=None, help="Limit samples for pilot runs")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device_map", type=str, default="cuda:0")
    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    p.add_argument("--max_new_tokens", type=int, default=16)

    p.add_argument("--output_dir", type=str, default="results/vlms_are_biased_empirical")
    p.add_argument("--n_visual_per_topic", type=int, default=10, help="Qualitative panels saved per topic")
    p.add_argument("--sanity_count", type=int, default=5, help="Number of sanity grid previews to save")
    p.add_argument("--dry_run", action="store_true", help="Skip model inference; emit sanity outputs only")

    p.add_argument("--max_image_size", type=int, default=840,
                   help="Cap the longer side of each source image before conditions are applied. "
                        "high_res will then 2x this capped size. Set 0 to disable.")
    p.add_argument("--no_object_crop", action="store_true", help="Skip object_centric_crop (avoids loading DETR)")
    p.add_argument("--object_crop_model", type=str, default="facebook/detr-resnet-50")
    p.add_argument("--object_crop_threshold", type=float, default=0.5)
    p.add_argument("--max_topics_plot", type=int, default=15)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ("plots", "sanity", "visuals"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    rng = random.Random(args.seed)
    # Load full split, then shuffle + limit so all topics are represented
    # (dataset is ordered by topic: Optical Illusion 792, Animals 546, … a sequential
    #  slice would only hit one topic for small --limit values)
    dataset = load_dataset(args.dataset, split=args.split)
    all_indices = list(range(len(dataset)))
    rng.shuffle(all_indices)
    if args.limit is not None:
        all_indices = all_indices[: args.limit]
    dataset = dataset.select(all_indices)
    dataset_size = len(dataset)

    detector: ObjectCropDetector | None = None
    if not args.no_object_crop and not args.dry_run:
        print(f"Loading object crop detector ({args.object_crop_model}) on CPU …")
        detector = ObjectCropDetector(model_name=args.object_crop_model, threshold=args.object_crop_threshold)

    runner: QwenRunner | None = None
    if not args.dry_run:
        runner = QwenRunner(
            model_name=args.model_name,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=(args.attn_implementation or None),
        )

    rows: List[Dict[str, Any]] = []
    sample_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    visual_payload: Dict[str, Dict[str, Any]] = {}
    visual_counts: Dict[str, int] = defaultdict(int)
    sanity_counts_by_topic: Dict[str, int] = defaultdict(int)

    condition_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    topic_condition_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    pbar = tqdm(enumerate(dataset), desc="VLMs-Are-Biased empirical", total=dataset_size)
    for idx, sample in pbar:
        image = to_rgb(sample["image"])
        if args.max_image_size > 0:
            image = cap_image_size(image, args.max_image_size)
        sample_id = safe_sample_id(sample, idx)
        prompt = str(sample.get("prompt", "")).strip()
        gt = str(sample.get("ground_truth", "")).strip()
        expected_bias = str(sample.get("expected_bias", "")).strip()
        topic = str(sample.get("topic", "unknown")).strip()
        pbar.set_postfix({"topic": topic[:20], "id": sample_id}, refresh=False)

        if not prompt or not gt or not expected_bias:
            continue

        conditions = make_conditions(prompt, image, rng, gt, expected_bias, detector=detector)
        unique_key = f"{topic}::{sample_id}"

        if sanity_counts_by_topic[topic] < args.sanity_count:
            sanity_counts_by_topic[topic] += 1
            save_sanity_preview(sample_id, topic, image, conditions, os.path.join(args.output_dir, "sanity"))

        if visual_counts[topic] < args.n_visual_per_topic:
            visual_counts[topic] += 1
            visual_payload[unique_key] = {
                "sample_id": sample_id, "image": image, "prompt": prompt,
                "gt": gt, "expected_bias": expected_bias, "topic": topic,
            }

        for cond in conditions:
            if args.dry_run:
                raw_pred = "DRY_RUN"
                is_correct, matches_bias, pred_norm = False, False, "dry_run"
            else:
                assert runner is not None
                raw_pred = runner.predict(cond.image, cond.prompt)
                is_correct, matches_bias, pred_norm = compare_prediction(raw_pred, gt, expected_bias)

            row: Dict[str, Any] = {
                "sample_id": sample_id,
                "topic": topic,
                "condition": cond.condition,
                "condition_group": cond.condition_group,
                "ground_truth": gt,
                "expected_bias": expected_bias,
                "prediction_raw": raw_pred,
                "prediction_norm": pred_norm,
                "is_correct": is_correct,
                "matches_bias": matches_bias,
                "condition_meta": cond.meta,
            }
            rows.append(row)
            condition_rows[cond.condition].append(row)
            topic_condition_rows[topic][cond.condition].append(row)
            sample_to_rows[unique_key].append(row)

    # -- Save raw records immediately (crash-safe) --
    jsonl_path = os.path.join(args.output_dir, "predictions.jsonl")
    save_jsonl(jsonl_path, rows)
    print(f"Saved predictions: {jsonl_path}")

    if not rows:
        print("No records collected. Exiting.")
        return

    # -- Aggregate metrics --
    condition_metrics = {
        c: compute_bias_metrics(condition_rows[c])
        for c in ALL_CONDITIONS if condition_rows[c]
    }
    topic_condition_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {
        t: {c: compute_bias_metrics(topic_condition_rows[t][c]) for c in ALL_CONDITIONS if topic_condition_rows[t][c]}
        for t in topic_condition_rows
    }

    summary: Dict[str, Any] = {
        "dataset": args.dataset,
        "split": args.split,
        "model_name": args.model_name,
        "dry_run": args.dry_run,
        "n_samples": len({r["sample_id"] for r in rows}),
        "n_rows": len(rows),
        "condition_metrics": condition_metrics,
        "topic_condition_metrics": topic_condition_metrics,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    save_json(summary_path, summary)
    print(f"Saved summary:     {summary_path}")

    # -- Plots --
    if condition_metrics:
        plot_accuracy_and_bias_by_condition(
            condition_metrics,
            os.path.join(args.output_dir, "plots", "plot1_accuracy_and_bias_by_condition.png"),
        )
        plot_bias_delta(
            condition_metrics,
            os.path.join(args.output_dir, "plots", "plot2_bias_delta.png"),
        )
        plot_accuracy_vs_bias(
            condition_metrics,
            os.path.join(args.output_dir, "plots", "plot3_accuracy_vs_bias.png"),
        )
    if topic_condition_metrics:
        plot_topic_bias_heatmap(
            topic_condition_metrics,
            os.path.join(args.output_dir, "plots", "plot4_topic_bias_heatmap.png"),
            max_topics=args.max_topics_plot,
        )
        plot_topic_accuracy_by_condition(
            topic_condition_metrics,
            os.path.join(args.output_dir, "plots", "plot5_topic_accuracy_by_condition.png"),
        )
    print(f"Saved plots:       {os.path.join(args.output_dir, 'plots')}")

    # -- Qualitative panels --
    for unique_key, payload in visual_payload.items():
        if unique_key not in sample_to_rows:
            continue
        t_slug = slugify(payload["topic"])
        s_slug = slugify(payload["sample_id"])
        save_qualitative_panel(
            sample_id=payload["sample_id"],
            topic=payload["topic"],
            image=payload["image"],
            prompt=payload["prompt"],
            gt=payload["gt"],
            expected_bias=payload["expected_bias"],
            row_bundle=sample_to_rows[unique_key],
            out_path=os.path.join(args.output_dir, "visuals", f"{t_slug}__sample_{s_slug}.png"),
        )
    print(f"Saved visuals:     {os.path.join(args.output_dir, 'visuals')}")
    print(f"Saved sanity:      {os.path.join(args.output_dir, 'sanity')}")

    # -- Console summary --
    if "baseline_real" in condition_metrics:
        base = condition_metrics["baseline_real"]
        base_bias = base["bias_rate"]
        print(f"\n=== Baseline  accuracy={base['accuracy']*100:.1f}%  bias_rate={base_bias*100:.1f}% ===")
        print("\nBias delta vs baseline_real (negative = less biased):")
        for cond in ALL_CONDITIONS:
            if cond == "baseline_real" or cond not in condition_metrics:
                continue
            m = condition_metrics[cond]
            delta = (m["bias_rate"] - base_bias) * 100
            arrow = "↓ less biased" if delta < -0.5 else ("↑ more biased" if delta > 0.5 else "≈ no change")
            print(f"  {cond:>20}:  {delta:+.1f}%  {arrow}")


if __name__ == "__main__":
    main()