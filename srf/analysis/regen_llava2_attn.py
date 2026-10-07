"""
Regenerate llava2_attn.png (boat image) with fixed title:
- Full question, big font
- No [sim/cov] metadata line
"""
from __future__ import annotations
import os, sys, pathlib, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/home/sgowda/.cache/huggingface"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "srf"))
sys.path.insert(0, str(_ROOT / "srf" / "saliency"))

import clip_salience as clip_sal

MMHAL_JSON = "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json"
MMHAL_DIR  = "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/images"
QUESTION_KEY = "yellow boat"
GH, GW = 6, 6

OUT_ANALYSIS = pathlib.Path("/home/sgowda/workspace/SRF/lmms-eval/analysis/llava2_attn.png")
OUT_PAPER    = pathlib.Path("/home/sgowda/workspace/SRF/paper/images/llava2_attn.png")

LLAVA_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: <image>\n{question}\nASSISTANT:"
)


def find_sample():
    with open(MMHAL_JSON) as f:
        data = json.load(f)
    for r in data:
        if QUESTION_KEY in r["question"].lower():
            fname = r["image_src"].rstrip("/").split("/")[-1]
            return Image.open(os.path.join(MMHAL_DIR, fname)).convert("RGB"), r["question"]
    raise RuntimeError(f"Sample with '{QUESTION_KEY}' not found")


FUSE_START = 8   # match SRF fusion zone (same as Qwen script)
FUSE_END   = 20


def baseline_attn_overlay(image, question, model, processor):
    """Mean attention from query positions to image tokens, fusion layers only (8-20),
    normalized by sum — matches Qwen script approach for visual consistency."""
    import torch
    prompt = LLAVA_PROMPT.format(question=question)
    inp    = processor(text=prompt, images=[image], return_tensors="pt", padding=True).to("cuda")
    ids    = inp["input_ids"][0].cpu()
    img_tok_id = model.config.image_token_index
    vis_cfg    = model.config.vision_config
    n_img_tok  = (vis_cfg.image_size // vis_cfg.patch_size) ** 2
    img_start  = (ids == img_tok_id).nonzero(as_tuple=True)[0][0].item()
    img_end    = img_start + n_img_tok
    query_idx  = list(range(img_end, len(ids) - 1)) or [len(ids) - 1]

    with torch.inference_mode():
        out = model(**inp, output_attentions=True)

    # Average only fusion layers (8-20), normalize by sum — matches Qwen approach
    n_patch_side = vis_cfg.image_size // vis_cfg.patch_size  # 24
    attn_map = np.zeros(n_img_tok, dtype=np.float32)
    n_fuse = 0
    for li, attn_layer in enumerate(out.attentions):
        if FUSE_START <= li <= FUSE_END:
            a = attn_layer[0].float().cpu().numpy()  # (n_heads, seq, seq)
            a_q = a[:, query_idx, img_start:img_end]  # (n_heads, n_query, n_img)
            attn_map += a_q.mean(axis=(0, 1))
            n_fuse += 1
    if n_fuse > 0:
        attn_map /= n_fuse
    s = attn_map.sum()
    if s > 1e-9:
        attn_map /= s  # normalize by sum → diffuse distribution like Qwen

    attn_map = attn_map.reshape(n_patch_side, n_patch_side)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    attn_img = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    attn_np  = np.array(attn_img) / 255.0

    import matplotlib.cm as cm
    cmap  = cm.get_cmap("jet")
    color = (cmap(attn_np)[:, :, :3] * 255).astype(np.uint8)
    img_arr = np.array(image)
    overlay = (img_arr * (1 - attn_np[:, :, None] * 0.75) +
               color * attn_np[:, :, None] * 0.75).clip(0, 255).astype(np.uint8)
    return overlay


def srf_overlay(image, noun):
    result = clip_sal.compute_clip_salience_full_gate_v3(
        image, noun, GH, GW, top_k_pct=0.30, backup="none")
    sal = result.saliency.cpu().float().numpy()
    if not result.object_present:
        raw = clip_sal.compute_clip_salience(image, noun, GH, GW, top_k_pct=1.0)
        sal = raw.saliency.cpu().float().numpy()

    sal_2d  = sal.reshape(GH, GW).astype(np.float32)
    sal_2d  = (sal_2d - sal_2d.min()) / (sal_2d.max() - sal_2d.min() + 1e-8)
    sal_img = Image.fromarray((sal_2d * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    sal_np  = np.array(sal_img) / 255.0

    import matplotlib.cm as cm
    cmap    = cm.get_cmap("jet")
    color   = (cmap(sal_np)[:, :, :3] * 255).astype(np.uint8)
    img_arr = np.array(image)
    overlay = (img_arr * (1 - sal_np[:, :, None] * 0.6) +
               color * sal_np[:, :, None] * 0.6).clip(0, 255).astype(np.uint8)
    return overlay


def make_figure(image, base_ov, srf_ov, question):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), facecolor="white")
    fig.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.85, bottom=0.02)

    panels = [np.array(image), base_ov, srf_ov]
    titles = ["Image", "VLM Attention", "SRF Saliency"]
    borders = [None, "#e74c3c", "#27ae60"]

    for ax, panel, ttl, bc in zip(axes, panels, titles, borders):
        ax.imshow(panel)
        ax.axis("off")
        ax.set_title(ttl, fontsize=9, fontweight="bold", pad=3, color="#333")
        if bc:
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(bc)
                sp.set_linewidth(2.0)

    fig.suptitle(f'"{question}"', fontsize=12, fontweight="bold", color="#222", y=0.99)

    for out in (OUT_ANALYSIS, OUT_PAPER):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out}")
    plt.close(fig)


def main():
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    print("Finding boat sample…")
    image, question = find_sample()
    print(f"  Q: {question}  size: {image.size}")

    print("Loading LLaVA-1.5-7B…")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.bfloat16,
        attn_implementation="eager").eval().cuda()
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    print("Computing baseline attention overlay…")
    base_ov = baseline_attn_overlay(image, question, model, processor)

    print("Computing SRF saliency overlay…")
    noun = "yellow boat"
    srf_ov = srf_overlay(image, noun)

    print("Saving figure…")
    make_figure(image, base_ov, srf_ov, question)


if __name__ == "__main__":
    main()
