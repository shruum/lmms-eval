#!/usr/bin/env python3
"""Empirical ablation analysis for POPE with Qwen2.5-VL.

Evaluates split-wise behavior on:
- adversarial
- popular
- random
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image, ImageFilter
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"

POPE_SPLITS = ["adversarial", "popular", "random"]

VISION_CONDITIONS = [
    "baseline_real",
    "no_image",
    "low_res_x8",
    "blur_r3",
    "patch_shuffle16",
    "center_mask40",
    "high_res",
    "counterfactual_image",
    "object_centric_crop",
]

LANGUAGE_CONDITIONS = [
    "no_text",
    "misleading_yes",
    "misleading_no",
]

ALL_CONDITIONS = VISION_CONDITIONS + LANGUAGE_CONDITIONS


@dataclass
class ConditionInput:
    condition: str
    condition_group: str
    prompt: str
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

    def predict_yesno(self, image: Image.Image | None, prompt: str) -> Tuple[str, str]:
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
        pred = extract_yes_no(raw)
        return raw, pred

def choose_two_questions_per_image(ds, n_images: int, seed: int) -> list[int]:
    """Select 2 rows per unique image_source for n_images images (total 2*n_images rows)."""
    by_image: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(ds):
        by_image[str(item["image_source"])].append(i)

    image_keys = list(by_image.keys())
    rng = random.Random(seed)
    rng.shuffle(image_keys)
    picked_images = image_keys[: min(n_images, len(image_keys))]

    selected: list[int] = []
    for key in picked_images:
        idxs = by_image[key]
        # Keep deterministic order by question_id-ish numeric if possible
        def _qsort(ix: int):
            qid = str(ds[ix].get("question_id", ""))
            return int(qid) if qid.isdigit() else qid

        idxs = sorted(idxs, key=_qsort)
        if len(idxs) >= 2:
            selected.extend(idxs[:2])
        elif len(idxs) == 1:
            selected.extend(idxs)
    return selected

def extract_yes_no(text: str) -> str:
    t = (text or "").strip().lower()
    m = re.search(r"\b(yes|no)\b", t)
    if m:
        return m.group(1)
    return "unknown"


def slugify(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    out = out.strip("_")
    return out or "unknown"


def to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB") if image.mode != "RGB" else image


def build_prompt(question: str, prefix: str = "") -> str:
    lines: List[str] = []
    if prefix:
        lines.append(prefix)
    lines.append(question.strip())
    lines.append("Answer with one word only: yes or no.")
    return "\n".join(lines)


def build_no_text_prompt() -> str:
    return "Look at the image only. Answer with one word only: yes or no."


def low_res_then_upsample(image: Image.Image, factor: int = 8) -> Image.Image:
    image = to_rgb(image)
    w, h = image.size
    down = image.resize((max(1, w // factor), max(1, h // factor)), Image.Resampling.BILINEAR)
    return down.resize((w, h), Image.Resampling.BICUBIC)


def high_res_upsample(image: Image.Image, scale: int = 2) -> Image.Image:
    """Upscale image by `scale`x using bicubic interpolation."""
    image = to_rgb(image)
    w, h = image.size
    return image.resize((w * scale, h * scale), Image.Resampling.BICUBIC)


def gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    return to_rgb(image).filter(ImageFilter.GaussianBlur(radius=radius))


def patch_shuffle(image: Image.Image, patch_size: int, rng: random.Random) -> Image.Image:
    image = to_rgb(image)
    arr = np.array(image)
    h, w = arr.shape[:2]
    rows = h // patch_size
    cols = w // patch_size
    if rows == 0 or cols == 0:
        return image.copy()

    cropped = arr[: rows * patch_size, : cols * patch_size].copy()
    patches = []
    for r in range(rows):
        for c in range(cols):
            patches.append(cropped[r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size].copy())
    order = list(range(len(patches)))
    rng.shuffle(order)

    shuffled = np.zeros_like(cropped)
    for idx, patch_idx in enumerate(order):
        r = idx // cols
        c = idx % cols
        shuffled[r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size] = patches[patch_idx]

    out = arr.copy()
    out[: rows * patch_size, : cols * patch_size] = shuffled
    return Image.fromarray(out)


def center_mask(image: Image.Image, ratio: float = 0.4) -> Image.Image:
    image = to_rgb(image)
    arr = np.array(image)
    h, w = arr.shape[:2]
    mw = int(w * ratio)
    mh = int(h * ratio)
    x0 = (w - mw) // 2
    y0 = (h - mh) // 2
    arr[y0 : y0 + mh, x0 : x0 + mw] = (0, 0, 0)
    return Image.fromarray(arr)


class ObjectCropDetector:
    """DETR-based object detector for object-centric cropping. Runs on CPU to avoid GPU memory conflicts."""

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        threshold: float = 0.5,
        padding_ratio: float = 0.15,
    ) -> None:
        import torch
        from transformers import DetrForObjectDetection, DetrImageProcessor

        self._torch = torch
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name).eval().to("cpu")
        self.threshold = threshold
        self.padding_ratio = padding_ratio

    def crop(self, image: Image.Image) -> Image.Image:
        image = to_rgb(image)
        inputs = self.processor(images=image, return_tensors="pt")
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = self._torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, threshold=self.threshold, target_sizes=target_sizes)[0]

        boxes = results["boxes"].cpu().numpy()
        if len(boxes) == 0:
            return image  # fallback: no objects detected

        x0, y0 = float(boxes[:, 0].min()), float(boxes[:, 1].min())
        x1, y1 = float(boxes[:, 2].max()), float(boxes[:, 3].max())
        pad_x = (x1 - x0) * self.padding_ratio
        pad_y = (y1 - y0) * self.padding_ratio
        w, h = image.size
        x0 = max(0.0, x0 - pad_x)
        y0 = max(0.0, y0 - pad_y)
        x1 = min(float(w), x1 + pad_x)
        y1 = min(float(h), y1 + pad_y)
        if x1 <= x0 or y1 <= y0:
            return image
        cropped = image.crop((int(x0), int(y0), int(x1), int(y1)))
        # Upsample crop to near-original resolution (bicubic), capped at original dims
        cw, ch = cropped.size
        scale = min(w / cw, h / ch)
        if scale > 1.0:
            new_w = min(int(cw * scale), w)
            new_h = min(int(ch * scale), h)
            cropped = cropped.resize((new_w, new_h), Image.Resampling.BICUBIC)
        return cropped


def make_conditions(
    question: str,
    image: Image.Image,
    rng: random.Random,
    gt: str,
    counterfactual_img: Image.Image | None = None,
    detector: ObjectCropDetector | None = None,
) -> List[ConditionInput]:
    base_prompt = build_prompt(question)
    conds: List[ConditionInput] = [
        ConditionInput("baseline_real", "vision", base_prompt, image, gt, {}),
        ConditionInput("no_image", "vision", base_prompt, None, gt, {}),
        ConditionInput("low_res_x8", "vision", base_prompt, low_res_then_upsample(image, factor=8), gt, {}),
        ConditionInput("blur_r3", "vision", base_prompt, gaussian_blur(image, radius=3.0), gt, {}),
        ConditionInput("patch_shuffle16", "vision", base_prompt, patch_shuffle(image, patch_size=16, rng=rng), gt, {}),
        ConditionInput("center_mask40", "vision", base_prompt, center_mask(image, ratio=0.4), gt, {}),
        ConditionInput("high_res", "vision", base_prompt, high_res_upsample(image), gt, {}),
    ]
    if counterfactual_img is not None:
        conds.append(ConditionInput("counterfactual_image", "vision", base_prompt, counterfactual_img, gt, {}))
    if detector is not None:
        conds.append(ConditionInput("object_centric_crop", "vision", base_prompt, detector.crop(image), gt, {}))
    conds += [
        ConditionInput("no_text", "language", build_no_text_prompt(), image, gt, {}),
        ConditionInput("misleading_yes", "language", build_prompt(question, prefix="A user claims the answer is yes. Verify carefully."), image, gt, {"claim": "yes"}),
        ConditionInput("misleading_no", "language", build_prompt(question, prefix="A user claims the answer is no. Verify carefully."), image, gt, {"claim": "no"}),
    ]
    return conds


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def accuracy(flags: List[bool]) -> float:
    return float(sum(flags) / len(flags)) if flags else 0.0


def compute_binary_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    pred_yes_count = 0
    valid_count = 0
    for r in rows:
        gt = r["ground_truth"]
        pred = r["prediction_yesno"]
        if pred not in {"yes", "no"}:
            continue
        valid_count += 1
        if pred == "yes":
            pred_yes_count += 1
        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "yes" and pred == "no":
            fn += 1
        elif gt == "no" and pred == "no":
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / valid_count if valid_count else 0.0
    yes_ratio = pred_yes_count / valid_count if valid_count else 0.0

    return {
        "accuracy": acc,
        "precision_yes": precision,
        "recall_yes": recall,
        "f1_yes": f1,
        "pred_yes_ratio": yes_ratio,
        "n_valid_yesno_preds": valid_count,
    }


def plot_split_condition_accuracy(rows: List[Dict[str, Any]], out_path: str) -> None:
    by_split_cond: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_split_cond[r["split"]][r["condition"]].append(bool(r["is_correct"]))

    splits = [s for s in POPE_SPLITS if s in by_split_cond]
    conditions = ALL_CONDITIONS
    x = np.arange(len(conditions), dtype=np.float32)
    width = 0.8 / max(1, len(splits))

    plt.figure(figsize=(14, 5))
    for i, split in enumerate(splits):
        vals = [100.0 * accuracy(by_split_cond[split].get(c, [])) for c in conditions]
        plt.bar(x + i * width, vals, width=width, label=split)
    plt.xticks(x + (len(splits) - 1) * width / 2.0, conditions, rotation=35, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("POPE Split-wise Accuracy by Condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_yes_no_bias(rows: List[Dict[str, Any]], out_path: str) -> None:
    by_split_cond_pred: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    by_split_gt: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        by_split_cond_pred[r["split"]][r["condition"]].append(r["prediction_yesno"])
        if r["condition"] == "baseline_real":
            by_split_gt[r["split"]].append(r["ground_truth"])

    splits = [s for s in POPE_SPLITS if s in by_split_cond_pred]
    conditions = ALL_CONDITIONS
    x = np.arange(len(conditions), dtype=np.float32)
    width = 0.8 / max(1, len(splits))

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, split in enumerate(splits):
        pred_vals = []
        for c in conditions:
            preds = [p for p in by_split_cond_pred[split].get(c, []) if p in {"yes", "no"}]
            pred_vals.append(100.0 * sum(1 for p in preds if p == "yes") / len(preds) if preds else 0.0)
        ax.bar(x + i * width, pred_vals, width=width, color=f"C{i}", label=f"{split} pred yes%")
        gt = by_split_gt.get(split, [])
        if gt:
            gt_ratio = 100.0 * sum(1 for g in gt if g == "yes") / len(gt)
            ax.axhline(gt_ratio, color=f"C{i}", linestyle="--", linewidth=1.2, alpha=0.7, label=f"{split} GT yes%={gt_ratio:.0f}%")

    ax.axhline(50.0, color="black", linestyle=":", linewidth=1.0, alpha=0.5, label="50% balanced")
    ax.set_xticks(x + (len(splits) - 1) * width / 2.0)
    ax.set_xticklabels(conditions, rotation=35, ha="right")
    ax.set_ylabel("Predicted 'Yes' (%)")
    ax.set_ylim(0, 110)
    ax.set_title("POPE Yes-Bias by Condition and Split")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_sanity_grid(
    sample_id: str,
    split: str,
    image_source: str,
    question: str,
    gt: str,
    conds: list[ConditionInput],
    out_dir: str,
) -> None:
    vis = [c for c in conds if c.condition in VISION_CONDITIONS]
    cols = len(vis)

    fig = plt.figure(figsize=(3.0 * cols, 4.4))
    for i, c in enumerate(vis, start=1):
        ax = plt.subplot(1, cols, i)
        if c.image is None:
            ax.imshow(np.zeros((128, 128, 3), dtype=np.uint8))
            ax.set_title("no_image")
        else:
            ax.imshow(c.image)
            ax.set_title(c.condition)
        ax.axis("off")

    q_short = question if len(question) <= 160 else question[:157] + "..."
    fig.suptitle(
        f"split={split} | image_source={image_source} | sample_id={sample_id}\nQ: {q_short}\nGT: {gt}",
        fontsize=9,
        y=1.02,
    )
    plt.tight_layout()

    s = slugify(split)
    sid = slugify(sample_id)
    src = slugify(str(image_source))
    png_path = os.path.join(out_dir, f"sanity_{s}__img_{src}__q_{sid}_vision.png")
    plt.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close()

    # also save text sidecar
    txt_path = os.path.join(out_dir, f"sanity_{s}__img_{src}__q_{sid}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"split: {split}\n")
        f.write(f"image_source: {image_source}\n")
        f.write(f"sample_id: {sample_id}\n")
        f.write(f"ground_truth: {gt}\n")
        f.write(f"question: {question}\n\n")
        for c in conds:
            if c.condition in LANGUAGE_CONDITIONS:
                f.write(f"[{c.condition}] prompt:\n{c.prompt}\n\n")


def save_qual_panel(
    sample_id: str,
    split: str,
    image_source: str,
    question: str,
    gt: str,
    row_bundle: list[dict[str, Any]],
    image: Image.Image,
    out_path: str,
) -> None:
    by_cond = {r["condition"]: r for r in row_bundle}
    plt.figure(figsize=(13, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title(f"{split} | img={image_source} | q={sample_id}")

    ax2 = plt.subplot(1, 2, 2)
    ax2.axis("off")
    lines = [f"Question: {question}", "", f"Ground truth: {gt}", "", "Predictions:"]
    for c in ALL_CONDITIONS:
        if c not in by_cond:
            continue
        r = by_cond[c]
        mark = "✓" if r["is_correct"] else "✗"
        lines.append(f"{c:>16}: {r['prediction_yesno']} ({mark})")
    ax2.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="POPE empirical ablation analysis")
    p.add_argument("--dataset", type=str, default="lmms-lab/POPE")
    p.add_argument("--dataset_config", type=str, default="Full")
    p.add_argument("--splits", type=str, default="adversarial") #"popular" "random"
    p.add_argument("--limit_per_split", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--device_map", type=str, default="cuda:0")
    p.add_argument("--torch_dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default="")
    p.add_argument("--max_new_tokens", type=int, default=8)

    p.add_argument("--output_dir", type=str, default="results/pope_empirical")
    p.add_argument("--n_visual_per_split", type=int, default=5)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--cf_pool_size", type=int, default=200, help="Images pre-sampled per split for the counterfactual pool")
    p.add_argument("--no_object_crop", action="store_true", help="Skip object_centric_crop condition (avoids loading DETR)")
    p.add_argument("--object_crop_model", type=str, default="facebook/detr-resnet-50")
    p.add_argument("--object_crop_threshold", type=float, default=0.5)
    p.add_argument("--n_sanity_images_per_split", type=int, default=6,
                   help="Unique images per split for sanity/visuals.")
    p.add_argument("--n_questions_per_sanity_image", type=int, default=2,
                   help="Questions per selected sanity image (currently supports 2).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visuals"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "sanity"), exist_ok=True)

    rng = random.Random(args.seed)
    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]

    runner: QwenRunner | None = None
    if not args.dry_run:
        runner = QwenRunner(
            model_name=args.model_name,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=(args.attn_implementation or None),
        )

    detector: ObjectCropDetector | None = None
    if not args.no_object_crop and not args.dry_run:
        print(f"Loading object crop detector ({args.object_crop_model}) on CPU ...")
        detector = ObjectCropDetector(model_name=args.object_crop_model, threshold=args.object_crop_threshold)

    rows: List[Dict[str, Any]] = []
    rows_by_unique: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    selected_visual_count: Dict[str, int] = defaultdict(int)
    visual_payload: Dict[str, Dict[str, Any]] = {}

    for split in split_list:
        ds = load_dataset(args.dataset, args.dataset_config, split=split)
        if args.limit_per_split is not None:
            ds = ds.select(range(min(args.limit_per_split, len(ds))))
        if args.n_questions_per_sanity_image != 2:
            raise ValueError("This script currently supports n_questions_per_sanity_image=2 only.")
        selected_indices = set(
            choose_two_questions_per_image(
                ds,
                n_images=args.n_sanity_images_per_split,
                seed=args.seed + hash(split) % 10_000,
            )
        )

        if not args.dry_run:
            pool_size = min(args.cf_pool_size, len(ds))
            pool_idxs = random.Random(args.seed + hash(split) % 10_000 + 1).sample(range(len(ds)), pool_size)
            cf_pool: list[tuple[int, Image.Image]] = [
                (i, to_rgb(ds[i]["image"])) for i in tqdm(pool_idxs, desc=f"Building CF pool ({split})")
            ]
        else:
            cf_pool = []

        pbar = tqdm(ds, desc=f"POPE {split}")
        for idx, sample in enumerate(pbar):
            image = to_rgb(sample["image"])
            sample_id = str(sample.get("question_id", sample.get("id", idx)))
            image_source = str(sample.get("image_source", "unknown"))
            question = str(sample["question"]).strip()
            gt = str(sample["answer"]).strip().lower()
            if gt not in {"yes", "no"}:
                continue

            pbar.set_postfix({"split": split, "sample_id": sample_id}, refresh=False)
            cf_candidates = [(pi, pimg) for pi, pimg in cf_pool if pi != idx]
            counterfactual_img: Image.Image | None = rng.choice(cf_candidates)[1] if cf_candidates else None
            conds = make_conditions(question, image, rng, gt, counterfactual_img=counterfactual_img, detector=detector)

            unique_key = f"{split}::{sample_id}"

            # only save sanity/visual payload for selected 10 rows per split (5 images * 2 questions)
            if idx in selected_indices:
                visual_payload[unique_key] = {"split": split,"sample_id": sample_id,"image_source": image_source,
                                              "question": question,"gt": gt,"image": image,}
                selected_visual_count[split] += 1
                save_sanity_grid(sample_id=sample_id,split=split,image_source=image_source,question=question,
                                 gt=gt,conds=conds,out_dir=os.path.join(args.output_dir, "sanity"),)
            for cond in conds:
                if args.dry_run:
                    raw, pred = "DRY_RUN", "unknown"
                else:
                    assert runner is not None
                    raw, pred = runner.predict_yesno(cond.image, cond.prompt)
                row = {
                    "split": split,
                    "sample_id": sample_id,
                    "question": question,
                    "ground_truth": gt,
                    "condition": cond.condition,
                    "condition_group": cond.condition_group,
                    "prompt": cond.prompt,
                    "prediction_raw": raw,
                    "prediction_yesno": pred,
                    "is_correct": pred == gt,
                    "condition_meta": cond.meta,
                    "category": sample.get("category", ""),
                    "image_source": sample.get("image_source", ""),
                }
                rows.append(row)
                rows_by_unique[unique_key].append(row)

    save_jsonl(os.path.join(args.output_dir, "predictions.jsonl"), rows)
    plot_split_condition_accuracy(rows, os.path.join(args.output_dir, "plots", "plot_split_condition_accuracy.png"))
    plot_yes_no_bias(rows, os.path.join(args.output_dir, "plots", "plot_yes_no_bias.png"))

    summary: Dict[str, Any] = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "splits": split_list,
        "n_rows": len(rows),
        "n_samples": len({(r["split"], r["sample_id"]) for r in rows}),
        "model_name": args.model_name,
        "dry_run": args.dry_run,
        "split_condition_metrics": {},
    }

    for split in split_list:
        summary["split_condition_metrics"][split] = {}
        for cond in ALL_CONDITIONS:
            subset = [r for r in rows if r["split"] == split and r["condition"] == cond]
            summary["split_condition_metrics"][split][cond] = compute_binary_metrics(subset)

    for unique_key, payload in visual_payload.items():
        row_bundle = rows_by_unique.get(unique_key, [])
        if not row_bundle:
            continue
        split_slug = slugify(payload["split"])
        sid_slug = slugify(payload["sample_id"])
        src_slug = slugify(payload["image_source"])
        out_path = os.path.join(args.output_dir, "visuals", f"{split_slug}__img_{src_slug}__q_{sid_slug}.png")
        save_qual_panel(
            sample_id=payload["sample_id"],
            split=payload["split"],
            image_source=payload["image_source"],
            question=payload["question"],
            gt=payload["gt"],
            row_bundle=row_bundle,
            image=payload["image"],
            out_path=out_path,
        )

    save_json(os.path.join(args.output_dir, "summary.json"), summary)

    print(f"Saved predictions: {os.path.join(args.output_dir, 'predictions.jsonl')}")
    print(f"Saved summary: {os.path.join(args.output_dir, 'summary.json')}")
    print(f"Saved plots: {os.path.join(args.output_dir, 'plots')}")
    print(f"Saved visuals: {os.path.join(args.output_dir, 'visuals')}")
    print(f"Saved sanity: {os.path.join(args.output_dir, 'sanity')}")
    print("Saved visual/sanity rows per split (should be 10 each if enough data):")
    for split in split_list:
        print(f"  {split}: {selected_visual_count[split]}")


if __name__ == "__main__":
    main()