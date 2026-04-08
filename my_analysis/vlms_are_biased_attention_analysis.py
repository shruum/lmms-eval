#!/usr/bin/env python3
"""Attention analysis for VLMs-Are-Biased dataset — across visual conditions.

For each sample we run get_attention_stats() under multiple visual conditions
(baseline_real, low_res_x8, blur_r5, patch_shuffle16, high_res,
object_centric_crop) and record:

  • n_image_tokens   — how many image tokens the processor produced
  • n_text_tokens    — text token count
  • overall_vtar     — fraction of attention going to image tokens
  • per_layer_vtar   — same, per transformer layer

Prediction (baseline_real only) gives is_correct and matches_bias.

Key question: does visual degradation / enrichment change how much attention
the model pays to the image tokens?

NOTE: requires --attn_implementation eager (SDPA returns None for attn weights).

Outputs (all in --output_dir):
  attention_records.jsonl
  summary.json
  run_log.txt
  plots/
    plot1_tokens_by_condition.png       mean image / text token count per condition
    plot2_vtar_by_condition.png         mean VTAR per condition (bar + scatter)
    plot3_input_vs_output.png           input token % vs output attention % per condition
    plot4_layerwise_by_condition.png    layer-wise VTAR per condition (line plot)
    plot5_vtar_by_topic.png             per-topic VTAR (baseline_real only)
    plot6_head_layer_heatmap.png        head × layer heatmap (baseline_real only)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from analysis_utils import (
    ObjectCropDetector,
    cap_image_size,
    gaussian_blur,
    high_res_upsample,
    low_res_then_upsample,
    patch_shuffle,
    safe_sample_id,
    save_json,
    save_jsonl,
    to_rgb,
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

ALL_CONDITIONS = [
    "baseline_real",
    "low_res_x8",
    "blur_r5",
    "patch_shuffle16",
    "high_res",
    "object_centric_crop",
]

CONDITION_COLORS = {
    "baseline_real":      "steelblue",
    "low_res_x8":         "darkorange",
    "blur_r5":            "mediumpurple",
    "patch_shuffle16":    "tomato",
    "high_res":           "seagreen",
    "object_centric_crop": "saddlebrown",
}


# ---------------------------------------------------------------------------
# Tee — write every print() to both stdout and a log file
# ---------------------------------------------------------------------------


class _Tee:
    def __init__(self, log_path: str) -> None:
        self._terminal = sys.stdout
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, message: str) -> int:
        self._terminal.write(message)
        self._log.write(message)
        return len(message)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def close(self) -> None:
        if not self._log.closed:
            self._log.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._terminal, name)


# ---------------------------------------------------------------------------
# Prediction normalisation (mirrors vlms_are_biased_empirical.py)
# ---------------------------------------------------------------------------


def compare_prediction(pred_raw: str, gt: str, expected_bias: str) -> Tuple[bool, bool, str]:
    pred_norm = pred_raw.lower().strip("{}").strip()
    gt_norm = gt.lower().strip("{}").strip()
    bias_norm = expected_bias.lower().strip("{}").strip()

    is_correct = pred_norm == gt_norm
    matches_bias = pred_norm == bias_norm

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
# Image condition builder
# ---------------------------------------------------------------------------


def build_condition_images(
    image: Image.Image,
    rng: random.Random,
    conditions: List[str],
    detector: Optional[ObjectCropDetector] = None,
) -> Dict[str, Optional[Image.Image]]:
    """Return a dict mapping condition name → transformed PIL image (or None if unavailable)."""
    result: Dict[str, Optional[Image.Image]] = {}
    for cond in conditions:
        if cond == "baseline_real":
            result[cond] = image
        elif cond == "low_res_x8":
            result[cond] = low_res_then_upsample(image, factor=8)
        elif cond == "blur_r5":
            result[cond] = gaussian_blur(image, radius=5.0)
        elif cond == "patch_shuffle16":
            result[cond] = patch_shuffle(image, patch_size=16, rng=rng)
        elif cond == "high_res":
            result[cond] = high_res_upsample(image, scale=2)
        elif cond == "object_centric_crop":
            result[cond] = detector.crop(image) if detector is not None else None
        else:
            result[cond] = None
    return result


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------


class QwenAttentionRunner:
    """Loads Qwen2.5-VL and runs greedy inference + per-layer VTAR extraction."""

    def __init__(
        self,
        model_name: str,
        device_map: str,
        torch_dtype: str,
        max_new_tokens: int,
        attn_implementation: str = "eager",
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError("qwen_vl_utils is required. Install with `uv add qwen-vl-utils`") from exc

        if attn_implementation == "flash_attention_2":
            raise ValueError("flash_attention_2 does not support output_attentions=True. Use eager.")

        dtype_map = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        if torch_dtype not in dtype_map:
            raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}")

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.max_new_tokens = max_new_tokens

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype_map[torch_dtype],
            attn_implementation=attn_implementation,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _build_inputs(self, image: Image.Image, prompt: str) -> Any:
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        device = self.model.device if (hasattr(self.model, "device") and self.model.device is not None) else "cuda"
        return inputs.to(device)

    def predict(self, image: Image.Image, prompt: str) -> str:
        inputs = self._build_inputs(image, prompt)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None, num_beams=1,
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def get_attention_stats(
        self,
        image: Image.Image,
        prompt: str,
        image_token: str = "<|image_pad|>",
    ) -> Dict[str, Any]:
        """Extract per-layer VTAR via forward hooks. Returns stats dict."""
        inputs = self._build_inputs(image, prompt)

        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)
        input_ids_cpu = inputs["input_ids"][0].cpu()
        image_mask = input_ids_cpu == image_token_id
        n_image_tokens = int(image_mask.sum().item())
        n_text_tokens = int((~image_mask).sum().item())
        last_pos = inputs["input_ids"].shape[1] - 1

        captured: List[Any] = []

        def make_hook(storage: List[Any]) -> Any:
            def hook_fn(module: Any, _inp: Any, output: Any) -> Any:
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    storage.append(output[1][0, :, last_pos, :].detach().cpu())
                    return (output[0], None) + output[2:]
                return output
            return hook_fn

        try:
            layers = self.model.language_model.layers
        except AttributeError as exc:
            raise RuntimeError("Cannot access model.language_model.layers") from exc

        hooks = [layer.self_attn.register_forward_hook(make_hook(captured)) for layer in layers]
        try:
            with self.torch.inference_mode():
                self.model(**inputs, output_attentions=True)
        finally:
            for h in hooks:
                h.remove()

        assert len(captured) == len(layers), (
            f"Expected {len(layers)} captured, got {len(captured)}. Use --attn_implementation eager."
        )

        per_layer_per_head_vtar: List[List[float]] = []
        per_layer_vtar: List[float] = []
        for attn_slice in captured:
            vision_sum = attn_slice[:, image_mask].sum(dim=-1)
            total_sum = attn_slice.sum(dim=-1).clamp(min=1e-9)
            ratio_per_head = (vision_sum / total_sum).tolist()
            per_layer_per_head_vtar.append(ratio_per_head)
            per_layer_vtar.append(float(sum(ratio_per_head) / len(ratio_per_head)))

        return {
            "per_layer_vtar": per_layer_vtar,
            "per_layer_per_head_vtar": per_layer_per_head_vtar,
            "overall_vtar": float(sum(per_layer_vtar) / len(per_layer_vtar)) if per_layer_vtar else 0.0,
            "n_image_tokens": n_image_tokens,
            "n_text_tokens": n_text_tokens,
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_tokens_by_condition(
    cond_stats: Dict[str, Dict[str, float]],
    out_path: str,
) -> None:
    """Grouped bar: mean image tokens and text tokens per condition."""
    conditions = [c for c in ALL_CONDITIONS if c in cond_stats]
    x = np.arange(len(conditions))
    width = 0.38

    img_vals = [cond_stats[c]["mean_n_image_tokens"] for c in conditions]
    txt_vals = [cond_stats[c]["mean_n_text_tokens"] for c in conditions]

    fig, ax = plt.subplots(figsize=(max(9, len(conditions) * 1.4), 5))
    bars_i = ax.bar(x - width / 2, img_vals, width=width, color="steelblue", label="Image tokens")
    bars_t = ax.bar(x + width / 2, txt_vals, width=width, color="tomato", label="Text tokens")
    for bar, val in list(zip(bars_i, img_vals)) + list(zip(bars_t, txt_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel("Mean token count")
    ax.set_title("Image vs Text Token Count per Visual Condition — VLMs-Are-Biased")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vtar_by_condition(
    cond_records: Dict[str, List[Dict[str, Any]]],
    out_path: str,
) -> None:
    """Bar chart of mean VTAR per condition with std error bars and per-sample scatter."""
    conditions = [c for c in ALL_CONDITIONS if c in cond_records]
    means = [float(np.mean([r["overall_vtar"] for r in cond_records[c]])) * 100 for c in conditions]
    stds = [float(np.std([r["overall_vtar"] for r in cond_records[c]])) * 100 for c in conditions]
    colors = [CONDITION_COLORS.get(c, "grey") for c in conditions]

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(max(9, len(conditions) * 1.4), 5))
    ax.bar(conditions, means, yerr=stds, capsize=4, color=colors, alpha=0.7, label="Mean ± std")

    # Overlay per-sample scatter
    for i, cond in enumerate(conditions):
        vals = [r["overall_vtar"] * 100 for r in cond_records[cond]]
        jitter = rng.uniform(-0.25, 0.25, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=6, alpha=0.3, color=colors[i])

    for i, (m, c) in enumerate(zip(means, conditions)):
        ax.text(i, m + stds[i] + 0.3, f"{m:.2f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel("Mean VTAR (%)")
    ax.set_title("Vision Token Attention Ratio by Visual Condition — VLMs-Are-Biased")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_input_vs_output(
    cond_stats: Dict[str, Dict[str, float]],
    out_path: str,
) -> None:
    """Grouped bars per condition: input token % vs output attention % for vision and text."""
    conditions = [c for c in ALL_CONDITIONS if c in cond_stats]
    x = np.arange(len(conditions))
    width = 0.22

    vision_input  = [cond_stats[c]["mean_image_token_ratio"] * 100 for c in conditions]
    vision_output = [cond_stats[c]["mean_vtar"] * 100 for c in conditions]
    text_input    = [(1 - cond_stats[c]["mean_image_token_ratio"]) * 100 for c in conditions]
    text_output   = [(1 - cond_stats[c]["mean_vtar"]) * 100 for c in conditions]

    fig, ax = plt.subplots(figsize=(max(10, len(conditions) * 1.6), 5))
    b1 = ax.bar(x - 1.5 * width, vision_input,  width=width, color="steelblue",  label="Vision input %")
    b2 = ax.bar(x - 0.5 * width, vision_output, width=width, color="deepskyblue", label="Vision attn %")
    b3 = ax.bar(x + 0.5 * width, text_input,    width=width, color="tomato",      label="Text input %")
    b4 = ax.bar(x + 1.5 * width, text_output,   width=width, color="lightsalmon", label="Text attn %")

    for bars, vals in [(b1, vision_input), (b2, vision_output), (b3, text_input), (b4, text_output)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel("% tokens / % attention")
    ax.set_title("Input Token % vs Output Attention % — Vision and Text by Condition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_layerwise_by_condition(
    cond_records: Dict[str, List[Dict[str, Any]]],
    out_path: str,
) -> None:
    """Line plot: mean VTAR per layer for each visual condition."""
    conditions = [c for c in ALL_CONDITIONS if c in cond_records]
    valid = {c: [r for r in cond_records[c] if r["per_layer_vtar"]] for c in conditions}
    valid = {c: recs for c, recs in valid.items() if recs}
    if not valid:
        return

    n_layers = len(next(iter(valid.values()))[0]["per_layer_vtar"])
    xs = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(12, 5))
    for cond, recs in valid.items():
        means = [float(np.mean([r["per_layer_vtar"][l] for r in recs])) * 100 for l in xs]
        ax.plot(xs, means, marker="o", markersize=2.5, linewidth=1.5,
                color=CONDITION_COLORS.get(cond, "grey"), label=f"{cond} (n={len(recs)})")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean VTAR (%)")
    ax.set_title("Layer-wise Vision Token Attention Ratio by Visual Condition — VLMs-Are-Biased")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vtar_by_topic(
    topic_records: Dict[str, List[Dict[str, Any]]],
    out_path: str,
    max_topics: int = 20,
) -> None:
    """Bar chart: mean VTAR per topic (baseline_real only)."""
    topic_means = {t: float(np.mean([r["overall_vtar"] for r in recs])) for t, recs in topic_records.items()}
    topic_stds = {t: float(np.std([r["overall_vtar"] for r in recs])) for t, recs in topic_records.items()}

    sorted_topics = sorted(topic_means, key=lambda t: topic_means[t], reverse=True)[:max_topics]
    vals = [topic_means[t] * 100 for t in sorted_topics]
    errs = [topic_stds[t] * 100 for t in sorted_topics]
    counts = [len(topic_records[t]) for t in sorted_topics]

    fig, ax = plt.subplots(figsize=(max(9, len(sorted_topics) * 0.9), 5))
    bars = ax.bar(sorted_topics, vals, yerr=errs, capsize=3, color="steelblue")
    ax.set_xticks(range(len(sorted_topics)))
    ax.set_xticklabels(sorted_topics, rotation=35, ha="right")
    ax.set_ylabel("Mean VTAR (%)")
    ax.set_title("VTAR by Topic (baseline_real) — VLMs-Are-Biased")
    for bar, val, cnt in zip(bars, vals, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}\n(n={cnt})", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_head_layer_heatmap(
    per_sample_layer_head: List[List[List[float]]],
    out_path: str,
    title_suffix: str = "",
) -> None:
    """Heatmap: mean VTAR[layer, head]."""
    arr = np.mean(np.array(per_sample_layer_head), axis=0) * 100
    fig, ax = plt.subplots(figsize=(max(8, arr.shape[1] * 0.45), max(6, arr.shape[0] * 0.3)))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="VTAR (%)")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_title(f"Mean VTAR per Head × Layer{title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention analysis (multi-condition) for VLMs-Are-Biased")
    parser.add_argument("--dataset", type=str, default="anvo25/vlms-are-biased")
    parser.add_argument("--split", type=str, default="main")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device_map", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa"],
                        help="eager required for output_attentions=True")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--image_token", type=str, default="<|image_pad|>")

    parser.add_argument("--max_image_size", type=int, default=840,
                        help="Cap longer side of source image before applying conditions. "
                             "high_res will 2× this, so keep it ≤840 to avoid OOM. Set 0 to disable.")
    parser.add_argument("--conditions", type=str,
                        default="baseline_real,low_res_x8,blur_r5,patch_shuffle16,high_res",
                        help="Comma-separated list of conditions to run. "
                             "Add 'object_centric_crop' if DETR is available.")
    parser.add_argument("--no_object_crop", action="store_true",
                        help="Skip object_centric_crop even if listed in --conditions")
    parser.add_argument("--object_crop_model", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--object_crop_threshold", type=float, default=0.5)

    parser.add_argument("--output_dir", type=str, default="results/vlms_are_biased_attention")
    parser.add_argument("--max_topics_plot", type=int, default=20)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    log_path = os.path.join(args.output_dir, "run_log.txt")
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"Logging all output to: {log_path}")
    print(f"Args: {vars(args)}\n")

    try:
        _main(args)
    finally:
        sys.stdout = tee._terminal
        tee.close()


def _main(args: argparse.Namespace) -> None:
    conditions: List[str] = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if args.no_object_crop and "object_centric_crop" in conditions:
        conditions.remove("object_centric_crop")
    print(f"Running conditions: {conditions}")

    # Load and shuffle dataset so a small --limit covers many topics
    rng_shuffle = random.Random(args.seed)
    rng_patch = random.Random(args.seed + 1)
    dataset = load_dataset(args.dataset, split=args.split)
    indices = list(range(len(dataset)))
    rng_shuffle.shuffle(indices)
    if args.limit is not None:
        indices = indices[: args.limit]
    dataset = dataset.select(indices)
    print(f"Dataset: {args.dataset} | split={args.split} | samples={len(dataset)}")

    # Optional DETR detector
    detector: Optional[ObjectCropDetector] = None
    if "object_centric_crop" in conditions and not args.dry_run:
        print(f"Loading DETR crop detector ({args.object_crop_model}) on CPU …")
        detector = ObjectCropDetector(
            model_name=args.object_crop_model,
            threshold=args.object_crop_threshold,
        )

    runner: Optional[QwenAttentionRunner] = None
    if not args.dry_run:
        runner = QwenAttentionRunner(
            model_name=args.model_name,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation,
        )

    # records: one entry per (sample, condition)
    records: List[Dict[str, Any]] = []
    # For per-topic plot — baseline_real only
    topic_baseline_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    pbar = tqdm(enumerate(dataset), desc="Attention analysis", total=len(dataset))
    for idx, sample in pbar:
        base_image = to_rgb(sample["image"])
        if args.max_image_size > 0:
            base_image = cap_image_size(base_image, args.max_image_size)

        sample_id = safe_sample_id(sample, idx)
        prompt = str(sample.get("prompt", "")).strip()
        gt = str(sample.get("ground_truth", "")).strip()
        expected_bias = str(sample.get("expected_bias", "")).strip()
        topic = str(sample.get("topic", "unknown")).strip()
        pbar.set_postfix({"topic": topic[:18]}, refresh=False)

        if not prompt or not gt or not expected_bias:
            continue

        # Build all condition images
        cond_images = build_condition_images(base_image, rng_patch, conditions, detector=detector)

        # Predict on baseline_real only (for accuracy / bias label)
        is_correct, matches_bias, pred_norm = False, False, "dry_run"
        if not args.dry_run and "baseline_real" in cond_images and cond_images["baseline_real"] is not None:
            assert runner is not None
            raw_pred = runner.predict(base_image, prompt)
            is_correct, matches_bias, pred_norm = compare_prediction(raw_pred, gt, expected_bias)

        # Attention stats per condition
        for cond in conditions:
            cond_img = cond_images.get(cond)
            if cond_img is None:
                continue  # object_centric_crop unavailable (no detector)

            if args.dry_run:
                record: Dict[str, Any] = {
                    "sample_id": sample_id, "topic": topic, "condition": cond,
                    "ground_truth": gt, "expected_bias": expected_bias,
                    "prediction_norm": pred_norm, "is_correct": is_correct, "matches_bias": matches_bias,
                    "n_image_tokens": 0, "n_text_tokens": 0, "image_token_ratio": 0.0,
                    "overall_vtar": 0.0, "per_layer_vtar": [], "per_layer_per_head_vtar": [],
                }
            else:
                assert runner is not None
                try:
                    attn = runner.get_attention_stats(cond_img, prompt, image_token=args.image_token)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\nOOM: sample={sample_id} cond={cond} — skipping this condition.")
                        runner.torch.cuda.empty_cache()
                        continue
                    raise

                if attn["n_image_tokens"] == 0:
                    print(f"\nWarning: no image tokens for sample={sample_id} cond={cond} — skipping.")
                    continue

                n_total = attn["n_image_tokens"] + attn["n_text_tokens"]
                record = {
                    "sample_id": sample_id, "topic": topic, "condition": cond,
                    "ground_truth": gt, "expected_bias": expected_bias,
                    "prediction_norm": pred_norm, "is_correct": is_correct, "matches_bias": matches_bias,
                    "n_image_tokens": attn["n_image_tokens"],
                    "n_text_tokens": attn["n_text_tokens"],
                    "image_token_ratio": attn["n_image_tokens"] / n_total if n_total > 0 else 0.0,
                    "overall_vtar": attn["overall_vtar"],
                    "per_layer_vtar": attn["per_layer_vtar"],
                    "per_layer_per_head_vtar": attn["per_layer_per_head_vtar"],
                }

            records.append(record)
            if cond == "baseline_real":
                topic_baseline_records[topic].append(record)

    # -----------------------------------------------------------------------
    # Save raw records (crash-safe first)
    # -----------------------------------------------------------------------
    jsonl_path = os.path.join(args.output_dir, "attention_records.jsonl")
    save_jsonl(jsonl_path, records)
    print(f"Saved records: {jsonl_path}  ({len(records)} rows)")

    if not records:
        print("No records collected. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Aggregate per condition
    # -----------------------------------------------------------------------
    cond_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        cond_records[r["condition"]].append(r)

    cond_stats: Dict[str, Dict[str, float]] = {}
    for cond, recs in cond_records.items():
        vtars = [r["overall_vtar"] for r in recs]
        n_img = [r["n_image_tokens"] for r in recs]
        n_txt = [r["n_text_tokens"] for r in recs]
        img_ratio = [r["image_token_ratio"] for r in recs]
        mean_vtar = float(np.mean(vtars))
        mean_img_ratio = float(np.mean(img_ratio))
        cond_stats[cond] = {
            "n_samples": len(recs),
            "mean_n_image_tokens": float(np.mean(n_img)),
            "std_n_image_tokens": float(np.std(n_img)),
            "mean_n_text_tokens": float(np.mean(n_txt)),
            "mean_image_token_ratio": mean_img_ratio,
            "mean_vtar": mean_vtar,
            "std_vtar": float(np.std(vtars)),
        }

    # -----------------------------------------------------------------------
    # Summary JSON
    # -----------------------------------------------------------------------
    baseline_recs = cond_records.get("baseline_real", [])
    summary: Dict[str, Any] = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "split": args.split,
        "attn_implementation": args.attn_implementation,
        "conditions_run": conditions,
        "dry_run": args.dry_run,
        "n_samples": len({r["sample_id"] for r in records}),
        "n_records_total": len(records),
        "baseline_accuracy": float(np.mean([r["is_correct"] for r in baseline_recs])) if baseline_recs else 0.0,
        "baseline_bias_rate": float(np.mean([r["matches_bias"] for r in baseline_recs])) if baseline_recs else 0.0,
        "per_condition": cond_stats,
    }
    save_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(f"Saved summary: {os.path.join(args.output_dir, 'summary.json')}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    plot_tokens_by_condition(cond_stats, os.path.join(args.output_dir, "plots", "plot1_tokens_by_condition.png"))
    print("Saved plot1: token counts by condition")

    plot_vtar_by_condition(cond_records, os.path.join(args.output_dir, "plots", "plot2_vtar_by_condition.png"))
    print("Saved plot2: VTAR by condition")

    plot_input_vs_output(cond_stats, os.path.join(args.output_dir, "plots", "plot3_input_vs_output.png"))
    print("Saved plot3: input token % vs output attention %")

    plot_layerwise_by_condition(cond_records, os.path.join(args.output_dir, "plots", "plot4_layerwise_by_condition.png"))
    print("Saved plot4: layer-wise VTAR by condition")

    if topic_baseline_records:
        plot_vtar_by_topic(
            topic_baseline_records,
            os.path.join(args.output_dir, "plots", "plot5_vtar_by_topic.png"),
            max_topics=args.max_topics_plot,
        )
        print("Saved plot5: VTAR by topic (baseline_real)")

    baseline_with_heads = [r for r in baseline_recs if r["per_layer_per_head_vtar"]]
    if baseline_with_heads:
        plot_head_layer_heatmap(
            [r["per_layer_per_head_vtar"] for r in baseline_with_heads],
            os.path.join(args.output_dir, "plots", "plot6_head_layer_heatmap.png"),
            title_suffix=" — baseline_real",
        )
        print("Saved plot6: head × layer heatmap (baseline_real)")

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print("\n=== Attention Analysis Summary — VLMs-Are-Biased (multi-condition) ===")
    if baseline_recs:
        print(f"  Samples    : {len({r['sample_id'] for r in records})}")
        print(f"  Accuracy   : {summary['baseline_accuracy']:.3f}  (baseline_real prediction)")
        print(f"  Bias rate  : {summary['baseline_bias_rate']:.3f}")
    print()
    print(f"  {'condition':<22}  {'img_tok':>7}  {'txt_tok':>7}  {'vision_attn%':>13}  {'text_attn%':>11}")
    print(f"  {'-'*68}")
    for cond in conditions:
        if cond not in cond_stats:
            continue
        s = cond_stats[cond]
        text_attn = (1.0 - s["mean_vtar"]) * 100
        print(
            f"  {cond:<22}  {s['mean_n_image_tokens']:>7.1f}  {s['mean_n_text_tokens']:>7.1f}"
            f"  {s['mean_vtar']*100:>12.2f}%  {text_attn:>10.2f}%"
        )


if __name__ == "__main__":
    main()
