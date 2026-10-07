"""
Run CLIP saliency on top LLaVA figure candidates and save comparison strips.
Output: analysis/candidates_llava/<idx>_<noun>_strip.png
"""
from __future__ import annotations
import os, sys, pathlib
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

MMHAL_DIR = "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/images"
OUT_DIR   = pathlib.Path("/home/sgowda/workspace/SRF/lmms-eval/analysis/candidates_llava")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GH, GW  = 6, 6
SIGMA   = 20.0

CANDIDATES = [
    # (idx, filename, question, noun)
    (15, "6555470659_4c69a30b73_o.jpg",
         "Which company owns the airplane displayed in the back of the image?",
         "airplane"),
    (47, "13407081714_0375f7b3e0_o.jpg",
         "Which tournament is this tennis competition?",
         "text"),
    (39, "6097499605_d06c51eed9_o.jpg",
         "From this photo, how much does each jerk chicken dumpling cost?",
         "sign"),
    (87, "9019477149_f7d04cdb65_o.jpg",
         "What is the name of the book?",
         "book"),
    (12, "5014730631_9c2701e063_o.jpg",
         "How is the yellow boat positioned in relation to the white yacht?",
         "yellow boat"),
    (16, "14901778703_09a206232c_o.jpg",
         "What are the colors of the shirts worn by the three men from left to right?",
         "shirt"),
]


def sal_to_heatmap(sal, image):
    sal_2d = sal.reshape(GH, GW).astype(np.float32)
    sal_2d = (sal_2d - sal_2d.min()) / (sal_2d.max() - sal_2d.min() + 1e-8)
    sal_img = Image.fromarray((sal_2d * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    return np.array(sal_img) / 255.0


def foveal_blur(image, sal):
    import torch, torch.nn.functional as F
    sal_t  = torch.tensor(sal).reshape(1, 1, GH, GW).float()
    W, H   = image.size
    weight = F.interpolate(sal_t, size=(H, W), mode="bilinear",
                           align_corners=False).squeeze().numpy()
    blurred = image.filter(ImageFilter.GaussianBlur(radius=SIGMA))
    img_arr = np.array(image).astype(np.float32)
    blr_arr = np.array(blurred).astype(np.float32)
    w = weight[:, :, np.newaxis]
    return Image.fromarray((w * img_arr + (1-w) * blr_arr).clip(0,255).astype(np.uint8))


def save_strip(idx, image, heatmap, fovea, question, noun, max_sim, present):
    import textwrap
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
    axes[0].imshow(np.array(image)); axes[0].axis("off")
    axes[0].set_title(textwrap.fill(question, 40), fontsize=8, fontweight="bold", loc="left")
    axes[1].imshow(np.array(image)); axes[1].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1].axis("off")
    axes[1].set_title(f"noun='{noun}'  sim={max_sim:.3f}  present={present}", fontsize=9)
    axes[2].imshow(np.array(fovea)); axes[2].axis("off")
    axes[2].set_title("Foveated (σ=20)", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / f"{idx:02d}_{noun.replace(' ','_')}_strip.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")


def main():
    for idx, fname, question, noun in CANDIDATES:
        img_path = os.path.join(MMHAL_DIR, fname)
        print(f"\n[{idx}] {fname}  noun='{noun}'")
        image = Image.open(img_path).convert("RGB")

        result = clip_sal.compute_clip_salience_full_gate_v3(
            image, noun, GH, GW, top_k_pct=0.30, backup="none")
        sal = result.saliency.cpu().float().numpy()
        if not result.object_present:
            raw = clip_sal.compute_clip_salience(image, noun, GH, GW, top_k_pct=1.0)
            sal = raw.saliency.cpu().float().numpy()

        heatmap = sal_to_heatmap(sal, image)
        fovea   = foveal_blur(image, sal)
        save_strip(idx, image, heatmap, fovea, question, noun,
                   result.max_sim, result.object_present)

    print(f"\nDone. All strips in {OUT_DIR}")


if __name__ == "__main__":
    main()
