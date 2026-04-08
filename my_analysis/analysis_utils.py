"""Shared utilities for POPE and MMBench empirical/attention analysis scripts.

Exports
-------
Constants:
    DEFAULT_MC_SUFFIX

I/O:
    save_json, save_jsonl

Basic helpers:
    to_rgb, accuracy, slugify, safe_sample_id, extract_yes_no

Image transforms:
    low_res_then_upsample, high_res_upsample, gaussian_blur,
    patch_shuffle, center_mask, ObjectCropDetector

MMBench / multiple-choice dataset helpers:
    get_valid_options, get_question, get_hint, get_ground_truth,
    extract_categories, extract_option_letter, build_mc_prompt
"""

from __future__ import annotations

import json
import re
import random
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MC_SUFFIX = "Answer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB") if image.mode != "RGB" else image


def accuracy(flags: List[bool]) -> float:
    return float(sum(flags) / len(flags)) if flags else 0.0


def slugify(value: str) -> str:
    """Convert a string to a safe filename component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def safe_sample_id(sample: Dict[str, Any], idx: int) -> str:
    for key in ["index", "id", "sample_id", "question_id"]:
        if key in sample and str(sample[key]).strip() != "":
            return str(sample[key])
    return str(idx)


def extract_yes_no(text: str) -> str:
    t = (text or "").strip().lower()
    m = re.search(r"\b(yes|no)\b", t)
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------


def cap_image_size(image: Image.Image, max_size: int) -> Image.Image:
    """Resize image so its longer side is at most max_size, maintaining aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)


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
    rows, cols = h // patch_size, w // patch_size
    if rows == 0 or cols == 0:
        return image.copy()

    cropped = arr[: rows * patch_size, : cols * patch_size].copy()
    patches = [
        cropped[r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size].copy()
        for r in range(rows)
        for c in range(cols)
    ]
    order = list(range(len(patches)))
    rng.shuffle(order)

    shuffled = np.zeros_like(cropped)
    for idx, patch_idx in enumerate(order):
        r, c = idx // cols, idx % cols
        shuffled[r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size] = patches[patch_idx]

    out = arr.copy()
    out[: rows * patch_size, : cols * patch_size] = shuffled
    return Image.fromarray(out)


def center_mask(image: Image.Image, ratio: float = 0.5, fill: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    image = to_rgb(image)
    arr = np.array(image.copy())
    h, w = arr.shape[:2]
    mw, mh = int(w * ratio), int(h * ratio)
    x0, y0 = (w - mw) // 2, (h - mh) // 2
    arr[y0 : y0 + mh, x0 : x0 + mw] = fill
    return Image.fromarray(arr)


class ObjectCropDetector:
    """DETR-based object detector for object-centric cropping. Runs on CPU."""

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        threshold: float = 0.5,
        padding_ratio: float = 0.15,
    ) -> None:
        import warnings
        import torch
        from transformers import DetrForObjectDetection, DetrImageProcessor

        self._torch = torch
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="copying from a non-meta parameter")
            self.model = DetrForObjectDetection.from_pretrained(model_name).eval().to("cpu")
        self.threshold = threshold
        self.padding_ratio = padding_ratio

    def crop(self, image: Image.Image) -> Image.Image:
        image = to_rgb(image)
        inputs = self.processor(images=image, return_tensors="pt")
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = self._torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_sizes
        )[0]

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


# ---------------------------------------------------------------------------
# MMBench / multiple-choice dataset helpers
# ---------------------------------------------------------------------------


def get_valid_options(sample: Dict[str, Any]) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for letter in ["A", "B", "C", "D"]:
        value = sample.get(letter)
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        options[letter] = text
    return options


def get_question(sample: Dict[str, Any]) -> str:
    return str(sample.get("question", "")).strip()


def get_hint(sample: Dict[str, Any]) -> str:
    hint = sample.get("hint", "")
    if hint is None:
        return ""
    text = str(hint).strip()
    return "" if text.lower() == "nan" else text


def get_ground_truth(sample: Dict[str, Any]) -> str:
    return str(sample.get("answer", "")).strip().upper()


def extract_categories(sample: Dict[str, Any]) -> Tuple[str, str]:
    fine_keys = ["category", "fine_category", "fine-grained_category", "fine_grained_category"]
    l2_keys = ["l2-category", "l2_category", "category_l2", "coarse_category"]
    fine = next((str(sample[k]).strip() for k in fine_keys if k in sample and str(sample[k]).strip()), "unknown")
    l2 = next((str(sample[k]).strip() for k in l2_keys if k in sample and str(sample[k]).strip()), "unknown")
    return fine, l2


def extract_option_letter(text: str, valid_letters: Iterable[str]) -> str:
    upper = (text or "").upper().strip()
    letters = "".join(sorted(set(valid_letters)))
    if not letters:
        return "?"
    explicit = re.search(rf"\b(?:OPTION|ANSWER)\s*[:\-]?\s*\(?([{letters}])\)?\b", upper)
    if explicit:
        return explicit.group(1)
    match = re.search(rf"\b([{letters}])\b", upper)
    if match:
        return match.group(1)
    return "?"


def build_mc_prompt(
    question: str,
    options: Dict[str, str],
    hint: str = "",
    prefix: str = "",
    suffix: str = DEFAULT_MC_SUFFIX,
) -> str:
    """Build a multiple-choice prompt (MMBench style)."""
    lines: List[str] = []
    if prefix:
        lines.append(prefix)
    if hint:
        lines.append(f"Context: {hint}")
    lines.append(question)
    for letter, option_text in options.items():
        lines.append(f"{letter}. {option_text}")
    lines.append(suffix)
    return "\n".join(lines)