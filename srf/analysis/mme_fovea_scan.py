"""
Scan MME existence-category positives for good foveation candidates.

Ranks by sim × size_bonus (same formula as MMHal scan).
Good range: 0.05 < cov < 0.55.

Output: analysis/mme_candidates/NNN_sim_cov_<question>.png
        Each panel: Image | SRF Saliency | Foveated image

Usage:
  python srf/analysis/mme_fovea_scan.py
"""
from __future__ import annotations

import os, sys, pathlib, re

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
from noun_extract import extract_clip_noun

CAND_DIR = pathlib.Path("/home/sgowda/workspace/SRF/lmms-eval/analysis/mme_candidates")
SIGMA    = 20.0
CATEGORIES = ["existence", "color", "count", "position"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def saliency_overlay(sal: np.ndarray, gh: int, gw: int,
                     image: Image.Image, alpha: float = 0.55) -> np.ndarray:
    sal_2d  = sal.reshape(gh, gw)
    sal_2d  = (sal_2d - sal_2d.min()) / (sal_2d.max() - sal_2d.min() + 1e-8)
    sal_img = Image.fromarray((sal_2d * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    sal_arr = np.array(sal_img) / 255.0
    heat    = plt.colormaps["jet"](sal_arr)[..., :3]
    img_arr = np.array(image) / 255.0
    overlay = (1 - alpha) * img_arr + alpha * heat
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def apply_foveal_blur(image: Image.Image, sal: np.ndarray,
                      gh: int, gw: int, sigma: float) -> Image.Image:
    import torch, torch.nn.functional as F
    sal_2d = torch.tensor(sal).reshape(1, 1, gh, gw).float()
    W, H   = image.size
    weight = F.interpolate(sal_2d, size=(H, W), mode="bilinear",
                           align_corners=False).squeeze().numpy()
    blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    img_arr = np.array(image).astype(np.float32)
    blr_arr = np.array(blurred).astype(np.float32)
    w       = weight[:, :, np.newaxis]
    result  = w * img_arr + (1.0 - w) * blr_arr
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def render_candidate(image, sal_ov, fovea_img, question, category,
                     sim, cov, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), facecolor="white")
    fig.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.86, bottom=0.04)

    axes[0].imshow(np.array(image))
    axes[1].imshow(sal_ov)
    axes[2].imshow(np.array(fovea_img))

    for ax, ttl in zip(axes, ["Image", "SRF Saliency", "Foveated"]):
        ax.axis("off")
        ax.set_title(ttl, fontsize=9, fontweight="bold", pad=3)

    q_short = question[:75] + "…" if len(question) > 75 else question
    fig.suptitle(
        f'"{q_short}"\n[{category}  sim={sim:.3f}  cov={cov:.2f}]',
        fontsize=8, color="#444", y=0.99,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out_path.name}")


# ── Main scan ─────────────────────────────────────────────────────────────────

def main():
    from datasets import load_dataset
    print("Loading MME dataset…")
    ds = load_dataset("lmms-lab/MME", split="test")

    # Keep only selected categories, positive examples (answer = "Yes")
    samples = [
        {"image": r["image"].convert("RGB"),
         "question": r["question"],
         "category": r["category"]}
        for r in ds
        if r.get("category") in CATEGORIES
        and str(r.get("answer", "")).strip().lower() == "yes"
    ]
    print(f"  {len(samples)} positive samples across {CATEGORIES}")

    # Pass 1: CLIP saliency scoring (fast)
    scored = []
    for i, s in enumerate(samples):
        image = s["image"]
        noun  = extract_clip_noun(s["question"], mode="pope")
        try:
            result = clip_sal.compute_clip_salience_full_gate_v3(
                image, noun, 6, 6, top_k_pct=0.30, backup="none")
            sim = float(result.max_sim)
            sal = result.saliency.cpu().float().numpy()
            sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            cov = float((sal_norm > 0.4).mean())
            obj_present = result.object_present
        except Exception as e:
            print(f"  [WARN] {e}")
            sim, cov, obj_present = 0.0, 0.0, False

        size_bonus  = float(np.exp(-((cov - 0.25) ** 2) / (2 * 0.15 ** 2)))
        rank_score  = sim * size_bonus
        scored.append((rank_score, sim, cov, obj_present, s, noun))

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}]", flush=True)

    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"\nTop candidates (ranked by sim × size_bonus):")
    print(f"  {'rank':>4}  {'score':>6}  {'sim':>6}  {'cov':>5}  {'ok':2}  {'cat':10}  {'noun':12}  question")
    for rank, (score, sim, cov, ok, s, noun) in enumerate(scored[:30]):
        flag = "✓" if 0.05 < cov < 0.55 and ok else " "
        print(f"  {rank:03d}  {score:.3f}  {sim:.3f}  {cov:.2f} {flag}  "
              f"[{s['category']:10s}]  {noun:12s}  {s['question'][:55]}")

    # Pass 2: render top candidates (only object_present=True, good cov)
    good = [(r, sc, si, cv, ok, s, n)
            for r, (sc, si, cv, ok, s, n) in enumerate(scored)
            if ok and 0.05 < cv < 0.55][:20]

    print(f"\nRendering {len(good)} good candidates…")
    for rank, score, sim, cov, _, s, noun in good:
        image = s["image"]
        result = clip_sal.compute_clip_salience_full_gate_v3(
            image, noun, 6, 6, top_k_pct=0.30, backup="none")
        sal    = result.saliency.cpu().float().numpy()
        sal_ov = saliency_overlay(sal, 6, 6, image)
        fovea  = apply_foveal_blur(image, sal, 6, 6, SIGMA)
        safe_q = re.sub(r"[^\w]", "_", s["question"])[:35]
        out    = CAND_DIR / f"{rank:03d}_sim{sim:.3f}_cov{cov:.2f}_{s['category']}_{safe_q}.png"
        render_candidate(image, sal_ov, fovea, s["question"],
                         s["category"], sim, cov, out)

    print(f"\nCandidates → {CAND_DIR}/")
    print("Pick one, then update llava_srf_demo.py with the question text.")


if __name__ == "__main__":
    main()
