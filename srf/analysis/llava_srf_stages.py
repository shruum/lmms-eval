"""
LLaVA SRF stages figure — 4-row × 3-col portrait.

Rows 1–2: MME samples (count, color subtasks)
Rows 3–4: MMHal-Bench samples

Each row: original image | Semantic relevance map | Foveated image
Column headers shown on top row only (matching srf_stages_qwen.png style).

Usage:
  cd /home/sgowda/workspace/SRF/lmms-eval
  conda activate mllm
  python srf/analysis/llava_srf_stages.py
"""
from __future__ import annotations

import os, sys, pathlib, json, textwrap

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/home/sgowda/.cache/huggingface"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "srf"))
sys.path.insert(0, str(_ROOT / "srf" / "saliency"))

import clip_salience as clip_sal

# ── Config ────────────────────────────────────────────────────────────────────
MME_DIR    = "/home/sgowda/workspace/ILVAD/data/MME/MME_Benchmark_release_version/MME_Benchmark"
MMHAL_JSON = "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json"
MMHAL_DIR  = "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/images"

SIGMA     = 20.0
GH, GW    = 6, 6
OUT_PATH  = pathlib.Path("/home/sgowda/workspace/SRF/paper/images/llava_srf_stages.png")

# ── Sample definitions ────────────────────────────────────────────────────────
# (image_path, question, noun_for_clip)
def _mme(subdir: str, stem: str, noun: str) -> tuple:
    img = os.path.join(MME_DIR, subdir, f"{stem}.jpg")
    txt = os.path.join(MME_DIR, subdir, f"{stem}.txt")
    with open(txt) as f:
        question = f.readline().split("\t")[0].strip()
    return img, question, noun

def _mmhal(idx: int, noun: str) -> tuple:
    with open(MMHAL_JSON) as f:
        data = json.load(f)
    r = data[idx]
    fname = r["image_src"].rstrip("/").split("/")[-1]
    img = os.path.join(MMHAL_DIR, fname)
    return img, r["question"], noun

SAMPLES = [
    _mmhal(1,  "bench"),                         # MMHal — who is sitting on the bench
    _mme("color", "000000492362", "skateboard"), # MME color — skateboard (portrait)
    _mme("color", "000000532761", "yellow"),     # MME color — living room painted yellow
    _mmhal(27, "zebra"),                         # MMHal — how many zebras
]


# ══════════════════════════════════════════════════════════════════════════════
# Processing helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_saliency(image: Image.Image, noun: str) -> np.ndarray:
    """Returns raw saliency array (GH*GW,)."""
    result = clip_sal.compute_clip_salience_full_gate_v3(
        image, noun, GH, GW, top_k_pct=0.30, backup="none")
    sal = result.saliency.cpu().float().numpy()
    if not result.object_present:
        raw = clip_sal.compute_clip_salience(image, noun, GH, GW, top_k_pct=1.0)
        sal = raw.saliency.cpu().float().numpy()
    return sal


def sal_to_heatmap(sal: np.ndarray, image: Image.Image) -> np.ndarray:
    """Normalized saliency upsampled to image size, in [0, 1]."""
    sal_2d = sal.reshape(GH, GW).astype(np.float32)
    sal_2d = (sal_2d - sal_2d.min()) / (sal_2d.max() - sal_2d.min() + 1e-8)
    sal_img = Image.fromarray((sal_2d * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR)
    return np.array(sal_img) / 255.0


def apply_foveal_blur(image: Image.Image, sal: np.ndarray) -> Image.Image:
    import torch, torch.nn.functional as F
    sal_t  = torch.tensor(sal).reshape(1, 1, GH, GW).float()
    W, H   = image.size
    weight = F.interpolate(sal_t, size=(H, W), mode="bilinear",
                           align_corners=False).squeeze().numpy()
    blurred = image.filter(ImageFilter.GaussianBlur(radius=SIGMA))
    img_arr = np.array(image).astype(np.float32)
    blr_arr = np.array(blurred).astype(np.float32)
    w = weight[:, :, np.newaxis]
    return Image.fromarray((w * img_arr + (1 - w) * blr_arr).clip(0, 255).astype(np.uint8))


# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

def make_figure(rows: list[tuple]) -> None:
    """
    rows: list of (image, heatmap, fovea, question) per sample
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.04},
                             facecolor="white")

    col_titles = ["Semantic relevance map", "Foveated image"]

    for row_idx, (image, heatmap, fovea, question) in enumerate(rows):
        ax_orig, ax_sal, ax_fov = axes[row_idx]

        # Original image
        ax_orig.imshow(np.array(image))
        ax_orig.axis("off")
        wrapped = textwrap.fill(question, width=38)
        ax_orig.set_title(wrapped, fontsize=9, fontweight="bold",
                          loc="left", pad=4, wrap=False)

        # Saliency overlay
        ax_sal.imshow(np.array(image))
        ax_sal.imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax_sal.axis("off")
        if row_idx == 0:
            ax_sal.set_title(col_titles[0], fontsize=13, pad=6)

        # Foveated image
        ax_fov.imshow(np.array(fovea))
        ax_fov.axis("off")
        if row_idx == 0:
            ax_fov.set_title(col_titles[1], fontsize=13, pad=6)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PATH), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUT_PATH}")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    rows = []
    for img_path, question, noun in SAMPLES:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        print(f"  Q: {question}")
        print(f"  noun: '{noun}'")

        image = Image.open(img_path).convert("RGB")
        print(f"  size: {image.size}")

        sal     = compute_saliency(image, noun)
        heatmap = sal_to_heatmap(sal, image)
        fovea   = apply_foveal_blur(image, sal)
        rows.append((image, heatmap, fovea, question))

    print("\nGenerating figure…")
    make_figure(rows)


if __name__ == "__main__":
    main()
