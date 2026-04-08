#!/usr/bin/env python3
"""Empirical ablation analysis for MMBench-Dev (EN) with Qwen2.5-VL.

Protocol defaults are chosen to match common MMBench evaluation practice:
- deterministic decoding (temperature=0, do_sample=False, num_beams=1)
- option-letter answer instruction
- one-factor-at-a-time perturbations
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from analysis_utils import (
    DEFAULT_MC_SUFFIX as DEFAULT_SUFFIX,
    ObjectCropDetector,
    accuracy,
    cap_image_size,
    build_mc_prompt as build_prompt,
    center_mask,
    extract_categories,
    extract_option_letter,
    gaussian_blur,
    get_ground_truth,
    get_hint,
    get_question,
    get_valid_options,
    high_res_upsample,
    low_res_then_upsample,
    patch_shuffle,
    safe_sample_id,
    save_json,
    save_jsonl,
    slugify as slugify_for_filename,
    to_rgb,
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # RTX 2080 Ti (sm_75)
os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"


MISLEADING_PREFIX_TEMPLATES = [
    "A user claims the answer is likely option {letter}. Verify carefully.",
    "Most people answer option {letter}. Verify carefully.",
]

VISION_CONDITIONS = [
    "baseline_real",
    "no_image",
    "low_res_x8",
    "blur_r2",
    "blur_r5",
    "patch_shuffle16",
    "center_mask40",
    "high_res",
    "counterfactual_image",
    "object_centric_crop",
]

LANGUAGE_CONDITIONS = [
    "baseline_real",
    "no_text",
    "misleading_prefix",
]

PLOT_CONDITIONS_DEFAULT = [
    "baseline_real",
    "no_image",
    "low_res_x8",
    "blur_r5",
    "patch_shuffle16",
    "center_mask40",
    "high_res",
    "counterfactual_image",
    "object_centric_crop",
    "no_text",
    "misleading_prefix",
]


@dataclass
class ConditionInput:
    condition: str
    condition_group: str
    prompt: str
    options: Dict[str, str]
    image: Image.Image | None
    gt_for_condition: str
    meta: Dict[str, Any]


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
            raise ValueError(f"Unsupported torch_dtype={torch_dtype}")

        model_kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": dtype_map[torch_dtype],
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens

    def predict_letter(self, image: Image.Image | None, prompt: str, valid_letters: Iterable[str]) -> Tuple[str, str]:
        messages: List[Dict[str, Any]]
        if image is None:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if image is None:
            inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        else:
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        if hasattr(self.model, "device") and self.model.device is not None:
            inputs = inputs.to(self.model.device)
        else:
            inputs = inputs.to("cuda")

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
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        letter = extract_option_letter(raw, valid_letters)
        del inputs, out, trimmed
        self.torch.cuda.empty_cache()
        return raw, letter


# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------


def build_no_text_prompt(valid_letters: Iterable[str]) -> str:
    joined = ", ".join(valid_letters)
    return f"Look at the image and choose the best option. Answer with one letter only: {joined}."


def choose_wrong_letter(gt: str, valid_letters: Iterable[str], rng: random.Random) -> str:
    candidates = [x for x in valid_letters if x != gt]
    return rng.choice(candidates) if candidates else gt


def make_vision_inputs(base_image: Image.Image, rng: random.Random) -> Dict[str, Image.Image | None]:
    return {
        "baseline_real": base_image,
        "no_image": None,
        "low_res_x8": low_res_then_upsample(base_image, factor=8),
        "blur_r2": gaussian_blur(base_image, radius=2.0),
        "blur_r5": gaussian_blur(base_image, radius=5.0),
        "patch_shuffle16": patch_shuffle(base_image, patch_size=16, rng=rng),
        "center_mask40": center_mask(base_image, ratio=0.4),
        "high_res": high_res_upsample(base_image),
    }


def prepare_conditions(
    sample: Dict[str, Any],
    image: Image.Image,
    rng: random.Random,
    counterfactual_img: Image.Image | None = None,
    detector: ObjectCropDetector | None = None,
) -> List[ConditionInput]:
    options = get_valid_options(sample)
    valid_letters = list(options.keys())
    question = get_question(sample)
    hint = get_hint(sample)
    gt = get_ground_truth(sample)

    base_prompt = build_prompt(question=question, options=options, hint=hint)
    vision_images = make_vision_inputs(image, rng)

    conditions: List[ConditionInput] = []

    for cond in VISION_CONDITIONS:
        if cond in ("counterfactual_image", "object_centric_crop"):
            continue
        conditions.append(ConditionInput(condition=cond, condition_group="vision", prompt=base_prompt,
                                         options=options, image=vision_images[cond], gt_for_condition=gt, meta={}))

    if counterfactual_img is not None:
        conditions.append(ConditionInput(condition="counterfactual_image", condition_group="vision",
                                         prompt=base_prompt, options=options, image=counterfactual_img,
                                         gt_for_condition=gt, meta={}))

    if detector is not None:
        conditions.append(ConditionInput(condition="object_centric_crop", condition_group="vision",
                                         prompt=base_prompt, options=options, image=detector.crop(image),
                                         gt_for_condition=gt, meta={}))

    conditions.append(ConditionInput(condition="no_text", condition_group="language",
                                     prompt=build_no_text_prompt(valid_letters), options=options,
                                     image=image, gt_for_condition=gt, meta={}))

    wrong_letter = choose_wrong_letter(gt, valid_letters, rng)
    misleading_prefix = rng.choice(MISLEADING_PREFIX_TEMPLATES).format(letter=wrong_letter)
    conditions.append(ConditionInput(condition="misleading_prefix", condition_group="language",
                                     prompt=build_prompt(question=question, options=options, hint=hint, prefix=misleading_prefix),
                                     options=options, image=image, gt_for_condition=gt,
                                     meta={"misleading_letter": wrong_letter}))

    return conditions


def make_serializable_options(options: Dict[str, str]) -> Dict[str, str]:
    return {k: str(v) for k, v in options.items()}


def make_unique_sample_key(sample_id: str, category: str) -> str:
    return f"{category}::{sample_id}"


def safe_image_path(sample: Dict[str, Any]) -> str:
    for key in ["image_path", "img_path", "path", "filename"]:
        if key in sample and sample[key] is not None:
            return str(sample[key])
    return ""


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_accuracy_by_condition(condition_acc: Dict[str, float], out_path: str) -> None:
    ordered = [c for c in PLOT_CONDITIONS_DEFAULT if c in condition_acc]
    vals = [condition_acc[c] * 100.0 for c in ordered]

    plt.figure(figsize=(11, 5))
    bars = plt.bar(ordered, vals)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("MMBench-EN-Dev Accuracy by Condition")
    for bar, value in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value + 0.3, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_vision_gain_hist(vision_gain: List[int], out_path: str) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.hist(vision_gain, bins=[-1.5, -0.5, 0.5, 1.5], rwidth=0.7)
    plt.xticks([-1, 0, 1])
    plt.xlabel("vision_gain = correct(baseline_real) - correct(no_image)")
    plt.ylabel("Sample count")
    plt.title("Vision Necessity Histogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_category_grouped_accuracy(
    rows: List[Dict[str, Any]],
    out_path: str,
    max_categories: int,
    plotted_conditions: List[str],
) -> None:
    category_counts: Dict[str, int] = defaultdict(int)
    grouped: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        category = row["category"]
        category_counts[category] += 1 if row["condition"] == "baseline_real" else 0
        grouped[category][row["condition"]].append(bool(row["is_correct"]))

    categories = sorted(category_counts.keys(), key=lambda k: category_counts[k], reverse=True)[:max_categories]
    conditions = [c for c in plotted_conditions if c]
    if not categories:
        return

    x = np.arange(len(categories), dtype=np.float32)
    width = 0.8 / max(1, len(conditions))

    plt.figure(figsize=(max(12, len(categories) * 0.8), 6))
    for idx, condition in enumerate(conditions):
        vals = [100.0 * accuracy(grouped[cat].get(condition, [])) for cat in categories]
        plt.bar(x + idx * width, vals, width=width, label=condition)

    plt.xticks(x + (len(conditions) - 1) * width / 2.0, categories, rotation=35, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy by Category and Condition")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_qualitative_panel(
    sample_id: str,
    image: Image.Image,
    question: str,
    options: Dict[str, str],
    gt: str,
    row_bundle: List[Dict[str, Any]],
    out_path: str,
) -> None:
    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title(f"Sample {sample_id}")

    ax2 = plt.subplot(1, 2, 2)
    ax2.axis("off")
    lines = [f"Question: {question}", "", "Options:"]
    for letter, text in options.items():
        lines.append(f"{letter}. {text}")
    lines.extend(["", f"Ground truth: {gt}", "", "Predictions:"])

    order = ["baseline_real", "no_image", "low_res_x8", "blur_r2", "blur_r5",
             "patch_shuffle16", "center_mask40", "high_res", "counterfactual_image",
             "object_centric_crop", "no_text", "misleading_prefix"]
    by_cond = {row["condition"]: row for row in row_bundle}
    for cond in order:
        if cond not in by_cond:
            continue
        row = by_cond[cond]
        marker = "✓" if row["is_correct"] else "✗"
        lines.append(f"{cond:>20}: {row['prediction_letter']} ({marker})")

    ax2.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def save_sanity_preview(
    sample_id: str,
    category: str,
    image: Image.Image,
    conditions: List[ConditionInput],
    out_dir: str,
) -> None:
    vis_conditions = [c for c in conditions if c.condition in VISION_CONDITIONS]
    cols = len(vis_conditions)
    plt.figure(figsize=(3.0 * cols, 3.2))
    for i, cond in enumerate(vis_conditions, start=1):
        ax = plt.subplot(1, cols, i)
        if cond.image is None:
            ax.imshow(np.zeros((128, 128, 3), dtype=np.uint8))
            ax.set_title("no_image")
        else:
            ax.imshow(cond.image)
            ax.set_title(cond.condition)
        ax.axis("off")
    plt.tight_layout()

    cat_slug = slugify_for_filename(category)
    sid_slug = slugify_for_filename(sample_id)
    plt.savefig(os.path.join(out_dir, f"sanity_{cat_slug}__{sid_slug}_vision.png"), dpi=140)
    plt.close()

    language_prompts = [c for c in conditions if c.condition in ["no_text", "misleading_prefix"]]
    with open(os.path.join(out_dir, f"sanity_{cat_slug}__{sid_slug}_language_prompts.txt"), "w", encoding="utf-8") as f:
        for cond in language_prompts:
            f.write(f"[{cond.condition}]\n{cond.prompt}\n\n")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empirical MMBench ablation study for Qwen2.5-VL")
    parser.add_argument("--dataset", type=str, default="lmms-lab/MMBench")
    parser.add_argument("--dataset_config", type=str, default="en")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for pilot runs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device_map", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--max_new_tokens", type=int, default=16)

    parser.add_argument("--output_dir", type=str, default="results/mmbench_empirical")
    parser.add_argument("--n_visual_per_category", type=int, default=5)
    parser.add_argument("--sanity_preview_count", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true", help="Skip model loading/inference, only emit sanity previews")

    parser.add_argument("--max_categories_plot", type=int, default=15)
    parser.add_argument("--category_plot_conditions", type=str, default=",".join(PLOT_CONDITIONS_DEFAULT))

    parser.add_argument("--cf_pool_size", type=int, default=200, help="Number of images pre-sampled for the counterfactual image pool")
    parser.add_argument("--no_object_crop", action="store_true", help="Skip object-centric crop condition (avoids loading DETR)")
    parser.add_argument("--object_crop_model", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--object_crop_threshold", type=float, default=0.5, help="DETR confidence threshold")
    parser.add_argument("--max_image_size", type=int, default=448,
                        help="Cap the longer side of each source image before conditions are applied. "
                             "high_res will then 2x this capped size. Set 0 to disable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visuals"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "sanity"), exist_ok=True)

    rng = random.Random(args.seed)

    split = args.split if args.limit is None else f"{args.split}[:{args.limit}]"
    dataset = load_dataset(args.dataset, args.dataset_config, split=split)
    dataset_size = len(dataset)

    # Pre-build counterfactual image pool via random index access
    if not args.dry_run:
        pool_size = min(args.cf_pool_size, dataset_size)
        pool_idxs = random.Random(args.seed + 99).sample(range(dataset_size), pool_size)
        cf_pool: List[Tuple[int, Image.Image]] = [
            (i, cap_image_size(to_rgb(dataset[i]["image"]), args.max_image_size) if args.max_image_size > 0 else to_rgb(dataset[i]["image"]))
            for i in tqdm(pool_idxs, desc="Building CF image pool")
        ]
    else:
        cf_pool = []

    # Initialize object crop detector (CPU, to avoid GPU memory conflict with Qwen)
    detector: ObjectCropDetector | None = None
    if not args.no_object_crop and not args.dry_run:
        print(f"Loading object crop detector ({args.object_crop_model}) on CPU ...")
        detector = ObjectCropDetector(model_name=args.object_crop_model, threshold=args.object_crop_threshold)

    runner: QwenRunner | None = None
    if not args.dry_run:
        runner = QwenRunner(
            model_name=args.model_name,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation,
        )

    rows: List[Dict[str, Any]] = []
    sample_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    visual_sample_payload: Dict[str, Dict[str, Any]] = {}
    selected_visual_counts: Dict[str, int] = defaultdict(int)
    condition_correct: Dict[str, List[bool]] = defaultdict(list)
    category_condition_correct: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
    baseline_by_sample: Dict[str, bool] = {}
    no_image_by_sample: Dict[str, bool] = {}

    pbar = tqdm(enumerate(dataset), desc="MMBench empirical eval", total=dataset_size)
    for idx, sample in pbar:
        image = to_rgb(sample["image"])
        if args.max_image_size > 0:
            image = cap_image_size(image, args.max_image_size)
        sample_id = safe_sample_id(sample, idx)
        question = get_question(sample)
        hint = get_hint(sample)
        gt = get_ground_truth(sample)
        options = get_valid_options(sample)
        valid_letters = list(options.keys())
        category, l2_category = extract_categories(sample)
        pbar.set_postfix({"category": category, "sample_id": sample_id}, refresh=False)

        if gt not in valid_letters:
            continue

        unique_key = make_unique_sample_key(sample_id, category)
        cf_candidates = [(pi, pimg) for pi, pimg in cf_pool if pi != idx]
        counterfactual_img: Image.Image | None = rng.choice(cf_candidates)[1] if cf_candidates else None
        conditions = prepare_conditions(sample, image, rng, counterfactual_img=counterfactual_img, detector=detector)

        if selected_visual_counts[category] < args.n_visual_per_category:
            selected_visual_counts[category] += 1
            visual_sample_payload[unique_key] = {
                "sample_id": sample_id, "image": image, "question": question,
                "options": options, "gt": gt, "category": category,
            }
            save_sanity_preview(sample_id, category, image, conditions, os.path.join(args.output_dir, "sanity"))

        for cond in conditions:
            if args.dry_run:
                raw_pred, pred_letter = "DRY_RUN", "?"
            else:
                assert runner is not None
                raw_pred, pred_letter = runner.predict_letter(cond.image, cond.prompt, cond.options.keys())

            is_correct = pred_letter == cond.gt_for_condition
            condition_correct[cond.condition].append(is_correct)
            category_condition_correct[category][cond.condition].append(is_correct)

            if cond.condition == "baseline_real":
                baseline_by_sample[unique_key] = is_correct
            if cond.condition == "no_image":
                no_image_by_sample[unique_key] = is_correct

            rows.append({
                "sample_id": sample_id, "category": category, "l2_category": l2_category,
                "condition_group": cond.condition_group, "condition": cond.condition,
                "question": question, "hint": hint, "prompt": cond.prompt,
                "options": make_serializable_options(cond.options),
                "image_path": safe_image_path(sample),
                "ground_truth": gt, "ground_truth_for_condition": cond.gt_for_condition,
                "prediction_raw": raw_pred, "prediction_letter": pred_letter,
                "is_correct": is_correct, "condition_meta": cond.meta,
            })
            sample_to_rows[unique_key].append(rows[-1])

    jsonl_path = os.path.join(args.output_dir, "predictions.jsonl")
    save_jsonl(jsonl_path, rows)

    condition_acc = {cond: accuracy(flags) for cond, flags in condition_correct.items()}
    vision_gain = [
        int(base_correct) - int(no_image_by_sample[uid])
        for uid, base_correct in baseline_by_sample.items()
        if uid in no_image_by_sample
    ]
    per_category_accuracy = {
        cat: {cond: accuracy(flags) for cond, flags in cdict.items()}
        for cat, cdict in category_condition_correct.items()
    }

    summary = {
        "dataset": args.dataset, "dataset_config": args.dataset_config, "split": args.split,
        "n_samples": len({(row["sample_id"], row["category"]) for row in rows}),
        "n_rows": len(rows), "model_name": args.model_name, "dry_run": args.dry_run,
        "condition_accuracy": condition_acc,
        "vision_gain_counts": {
            "-1": sum(1 for x in vision_gain if x == -1),
            "0": sum(1 for x in vision_gain if x == 0),
            "+1": sum(1 for x in vision_gain if x == 1),
        },
        "per_category_accuracy": per_category_accuracy,
        "files": {
            "predictions_jsonl": jsonl_path,
            "plot1_accuracy_by_condition": os.path.join(args.output_dir, "plots", "plot1_accuracy_by_condition.png"),
            "plot2_vision_gain_hist": os.path.join(args.output_dir, "plots", "plot2_vision_gain_hist.png"),
            "plot3_category_grouped": os.path.join(args.output_dir, "plots", "plot3_category_grouped.png"),
        },
    }
    save_json(os.path.join(args.output_dir, "summary.json"), summary)

    if rows:
        plot_accuracy_by_condition(condition_acc, os.path.join(args.output_dir, "plots", "plot1_accuracy_by_condition.png"))
        plot_vision_gain_hist(vision_gain, os.path.join(args.output_dir, "plots", "plot2_vision_gain_hist.png"))
        plot_conditions = [c.strip() for c in args.category_plot_conditions.split(",") if c.strip()]
        plot_category_grouped_accuracy(
            rows, os.path.join(args.output_dir, "plots", "plot3_category_grouped.png"),
            max_categories=args.max_categories_plot, plotted_conditions=plot_conditions,
        )

    for unique_key, payload in visual_sample_payload.items():
        if unique_key not in sample_to_rows:
            continue
        cat_slug = slugify_for_filename(payload["category"])
        sid_slug = slugify_for_filename(payload["sample_id"])
        save_qualitative_panel(
            sample_id=payload["sample_id"], image=payload["image"],
            question=payload["question"], options=payload["options"], gt=payload["gt"],
            row_bundle=sample_to_rows[unique_key],
            out_path=os.path.join(args.output_dir, "visuals", f"{cat_slug}__sample_{sid_slug}.png"),
        )

    print(f"Saved predictions: {jsonl_path}")
    print(f"Saved summary: {os.path.join(args.output_dir, 'summary.json')}")
    print(f"Saved plots in: {os.path.join(args.output_dir, 'plots')}")
    print(f"Saved qualitative samples in: {os.path.join(args.output_dir, 'visuals')}")
    print(f"Saved sanity previews in: {os.path.join(args.output_dir, 'sanity')}")
    if selected_visual_counts:
        print("Saved per-category visual/sanity counts:")
        for cat in sorted(selected_visual_counts):
            print(f"  {cat}: {selected_visual_counts[cat]}")


if __name__ == "__main__":
    main()