"""
LLaVA SRF pipeline demo — single image figure.

4-panel figure for one MMHal sample:
  (a) Original image
  (b) SRF saliency map (CLIP spatial map overlaid)
  (c) Foveated image   (W·img + (1-W)·blur, sigma=20)
  (d) Vision attention ratio per (layer, head) with selected SRF slots marked

Outputs:
  analysis/llava_srf_demo.png            — working copy
  paper/images/llava_srf_comp.png        — paper figure

Usage:
  cd /home/sgowda/workspace/SRF/lmms-eval
  conda activate mllm
  python srf/analysis/llava_srf_demo.py
"""
from __future__ import annotations

import os, sys, pathlib, json

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

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "llava-hf/llava-1.5-7b-hf"
MMHAL_JSON    = "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json"
MMHAL_IMG_DIR = "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/images"
QUESTION_KEY  = "Which company owns the airplane"   # substring to identify the sample
SIGMA         = 20.0
FUSE_START    = 8
FUSE_END      = 20          # inclusive — SRF patches layers [FUSE_START, FUSE_END]
HEAD_TOP_K_PCT = 0.20
OUT_ANALYSIS  = pathlib.Path("/home/sgowda/workspace/SRF/lmms-eval/analysis/llava_srf_demo.png")
OUT_PAPER     = pathlib.Path("/home/sgowda/workspace/SRF/paper/images/llava_srf_comp.png")

LLAVA_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: <image>\n{question}\nASSISTANT:"
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_lm(model):
    if hasattr(model, "language_model"):
        return model.language_model
    raise AttributeError(f"Cannot find language_model in {type(model)}")


def find_sample() -> tuple[Image.Image, str, str]:
    """Returns (image, question, gt_answer)."""
    with open(MMHAL_JSON) as f:
        data = json.load(f)
    for r in data:
        if QUESTION_KEY in r["question"]:
            fname = r["image_src"].rstrip("/").split("/")[-1]
            img_path = os.path.join(MMHAL_IMG_DIR, fname)
            gt = r.get("gt_answer", r.get("answer", ""))
            return Image.open(img_path).convert("RGB"), r["question"], gt
    raise RuntimeError(f"Sample with '{QUESTION_KEY}' not found in MMHal")


def sal_to_heatmap(sal: np.ndarray, gh: int, gw: int,
                   image: Image.Image) -> np.ndarray:
    """Normalize saliency and upsample to image resolution in [0, 1]."""
    sal_2d  = sal.reshape(gh, gw).astype(np.float32)
    sal_2d  = (sal_2d - sal_2d.min()) / (sal_2d.max() - sal_2d.min() + 1e-8)
    sal_img = Image.fromarray((sal_2d * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    return np.array(sal_img) / 255.0   # (H, W) in [0, 1]


def apply_foveal_blur(image: Image.Image, sal: np.ndarray,
                      gh: int, gw: int, sigma: float) -> Image.Image:
    import torch, torch.nn.functional as F
    sal_2d   = torch.tensor(sal).reshape(1, 1, gh, gw).float()
    W, H     = image.size
    weight   = F.interpolate(sal_2d, size=(H, W), mode="bilinear",
                             align_corners=False).squeeze().numpy()  # (H, W) in [0, 1]
    blurred  = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    img_arr  = np.array(image).astype(np.float32)
    blr_arr  = np.array(blurred).astype(np.float32)
    w        = weight[:, :, np.newaxis]
    result   = w * img_arr + (1.0 - w) * blr_arr
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


# ══════════════════════════════════════════════════════════════════════════════
# GPU: per-(layer, head) vision attention for one image
# ══════════════════════════════════════════════════════════════════════════════

def get_per_head_attn(image: Image.Image, question: str, model, processor) -> np.ndarray:
    """
    Returns (n_layers, n_heads) array — fraction of attention FROM query
    positions TO image token positions, per head.
    """
    import torch

    prompt = LLAVA_PROMPT.format(question=question)
    inp    = processor(text=prompt, images=[image],
                       return_tensors="pt", padding=True).to("cuda")

    ids        = inp["input_ids"][0].cpu()
    seq_len    = ids.shape[0]
    img_tok_id = model.config.image_token_index
    vis_cfg    = model.config.vision_config
    n_img_tok  = (vis_cfg.image_size // vis_cfg.patch_size) ** 2  # 576

    img_start  = (ids == img_tok_id).nonzero(as_tuple=True)[0][0].item()
    img_end    = img_start + n_img_tok
    vis_mask   = torch.zeros(seq_len, dtype=torch.bool)
    vis_mask[img_start:img_end] = True
    query_idx  = list(range(img_end, seq_len - 1)) or [seq_len - 1]

    with torch.inference_mode():
        out = model(**inp, output_attentions=True)

    n_layers = len(out.attentions)
    n_heads  = out.attentions[0].shape[1]
    rho      = np.zeros((n_layers, n_heads), dtype=np.float32)

    for li, attn_layer in enumerate(out.attentions):
        a = attn_layer[0].float().cpu().numpy()   # (n_heads, seq, seq)
        a_q = a[:, query_idx, :]                  # (n_heads, n_query, seq)
        rho[li] = a_q[:, :, vis_mask.numpy()].sum(axis=(1, 2)) / len(query_idx)

    return rho   # (32, 32)


# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

def make_figure(image, sal_heatmap, fovea_img, rho, question, gt):
    n_layers, n_heads = rho.shape   # 32 × 32 for LLaVA

    fig = plt.figure(figsize=(20, 5.2), facecolor="white")
    gs  = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.18], wspace=0.10)

    titles = [
        "(a) Original image",
        "(b) SRF saliency",
        "(c) Foveated image",
        "(d) Vision attention per (layer, head)",
    ]

    # ── Panels (a)–(c): image panels ─────────────────────────────────────────
    for i, ttl in enumerate(titles[:3]):
        ax = fig.add_subplot(gs[i])
        if i == 0:
            ax.imshow(np.array(image))
        elif i == 1:
            ax.imshow(np.array(image))
            ax.imshow(sal_heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        else:
            ax.imshow(np.array(fovea_img))
        ax.axis("off")
        ax.set_title(ttl, fontsize=16, fontweight="bold", pad=4)

    # ── Panel (d): VTAR heatmap + selected slots ──────────────────────────────
    ax_h = fig.add_subplot(gs[3])

    _W, _H = image.size
    _A = (rho.shape[1] * _H) / (rho.shape[0] * _W)
    im = ax_h.imshow(rho, aspect=_A, cmap="viridis", vmin=0,
                     interpolation="nearest", origin="upper")

    # Selected slots: top HEAD_TOP_K_PCT heads per layer in [FUSE_START, FUSE_END]
    k = max(1, int(HEAD_TOP_K_PCT * n_heads))
    sel_x, sel_y = [], []
    for l in range(FUSE_START, FUSE_END + 1):
        for h in np.argsort(rho[l])[::-1][:k]:
            sel_x.append(h)
            sel_y.append(l)
    ax_h.scatter(sel_x, sel_y, s=10, c="red", marker="s", linewidths=0, zorder=5)

    ax_h.set_xlabel("head", fontsize=13)
    ax_h.set_ylabel("layer", fontsize=13)
    step = max(1, n_heads // 8)
    ax_h.set_xticks(range(0, n_heads, step))
    ax_h.set_xticklabels([str(h) for h in range(0, n_heads, step)], fontsize=11)
    ytick_step = max(1, n_layers // 8)
    ax_h.set_yticks(range(0, n_layers, ytick_step))
    ax_h.set_yticklabels([str(l) for l in range(0, n_layers, ytick_step)], fontsize=11)
    ax_h.set_title(titles[3], fontsize=16, fontweight="bold", pad=4)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax_h)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("vision attention ratio", fontsize=11)
    cbar.ax.tick_params(labelsize=11)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(f'"{question}"', fontsize=19, fontweight="bold", y=1.02)

    for out in (OUT_ANALYSIS, OUT_PAPER):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    print("Finding MMHal sample…")
    image, question, gt = find_sample()
    noun = extract_clip_noun(question, mode="pope")
    print(f"  Q: {question}")
    print(f"  GT: {gt}")
    print(f"  noun: '{noun}'  image: {image.size}")

    # ── CLIP saliency ────────────────────────────────────────────────────────
    print("Running CLIP saliency…")
    gh, gw = 6, 6
    result = clip_sal.compute_clip_salience_full_gate_v3(
        image, noun, gh, gw, top_k_pct=0.30, backup="none")
    sal = result.saliency.cpu().float().numpy()
    print(f"  max_sim={result.max_sim:.3f}  object_present={result.object_present}")

    if not result.object_present:
        raw = clip_sal.compute_clip_salience(image, noun, gh, gw, top_k_pct=1.0)
        sal = raw.saliency.cpu().float().numpy()
        print("  Gate fired — using raw CLIP map for visualization")

    sal_heatmap = sal_to_heatmap(sal, gh, gw, image)
    fovea_img   = apply_foveal_blur(image, sal, gh, gw, SIGMA)

    # ── LLaVA attention (GPU, cached) ────────────────────────────────────────
    rho_path = OUT_ANALYSIS.parent / "llava_srf_demo_rho.npy"
    if rho_path.exists():
        print(f"Loading saved rho from {rho_path}…")
        rho = np.load(str(rho_path))
    else:
        print("Loading LLaVA-1.5-7B…")
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager",
        ).eval().cuda()
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        print("Extracting per-(layer, head) vision attention…")
        rho = get_per_head_attn(image, question, model, processor)
        np.save(str(rho_path), rho)
        print(f"  Saved rho → {rho_path}")
    print(f"  rho shape: {rho.shape}  mean={rho.mean()*100:.1f}%  max={rho.max()*100:.1f}%")

    # ── Figure ───────────────────────────────────────────────────────────────
    print("Generating figure…")
    make_figure(image, sal_heatmap, fovea_img, rho, question, gt)


if __name__ == "__main__":
    main()
