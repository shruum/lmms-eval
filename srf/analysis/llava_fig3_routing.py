"""
LLaVA-1.5-7B Fig 3 equivalent — Image | VLM Attention | SRF Saliency.

Two modes:
  --scan          Scan all MMHal-Bench samples → save candidate panels to
                  analysis/candidates/NNN_<type>_<qid>.png
                  (ranked by CLIP similarity so interesting ones bubble up)

  (default)       Render final Fig 3 using two chosen candidates:
                  --idx_a NNN --idx_b NNN
                  Output: analysis/llava_fig3_attn.png

Usage:
  cd /home/sgowda/workspace/SRF/lmms-eval
  conda activate mllm
  python srf/analysis/llava_fig3_routing.py --scan
  # inspect analysis/candidates/, pick two
  python srf/analysis/llava_fig3_routing.py --idx_a 003 --idx_b 011
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

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
from noun_extract import extract_clip_noun

MMHAL_JSON    = "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json"
MMHAL_IMG_DIR = "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/images"
CAND_DIR      = pathlib.Path("/home/sgowda/workspace/SRF/lmms-eval/analysis/candidates")
FINAL_OUT     = pathlib.Path("/home/sgowda/workspace/SRF/lmms-eval/analysis/llava_fig3_attn.png")

MODEL_ID     = "llava-hf/llava-1.5-7b-hf"
LLAVA_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: <image>\n{question}\nASSISTANT:"
)
FUSE_START = 8
FUSE_END   = 20   # layers used for attention extraction


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_lm(model):
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    raise AttributeError(f"Cannot find language_model in {type(model)}")


def load_mmhal() -> list[dict]:
    with open(MMHAL_JSON) as f:
        data = json.load(f)
    rows = []
    for i, r in enumerate(data):
        fname = r["image_src"].rstrip("/").split("/")[-1]
        img_path = os.path.join(MMHAL_IMG_DIR, fname)
        if os.path.exists(img_path):
            rows.append({
                "idx": i,
                "img_path": img_path,
                "question": r["question"],
                "question_type": r.get("question_type", "other"),
            })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Saliency overlay (same as Qwen paper scripts)
# ══════════════════════════════════════════════════════════════════════════════

def saliency_overlay(sal: np.ndarray, grid_h: int, grid_w: int,
                     image: Image.Image, alpha: float = 0.55) -> np.ndarray:
    sal_2d  = sal.reshape(grid_h, grid_w)
    sal_2d  = (sal_2d - sal_2d.min()) / (sal_2d.max() - sal_2d.min() + 1e-8)
    sal_img = Image.fromarray((sal_2d * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR)
    sal_arr = np.array(sal_img) / 255.0
    heat    = plt.colormaps["jet"](sal_arr)[..., :3]
    img_arr = np.array(image.convert("RGB")) / 255.0
    overlay = (1 - alpha) * img_arr + alpha * heat
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# CLIP saliency
# ══════════════════════════════════════════════════════════════════════════════

def run_srf_saliency(image: Image.Image, noun: str) -> tuple[np.ndarray, float, float]:
    """Run CLIP saliency and return (overlay, max_sim, coverage).

    coverage = fraction of 6×6 patches with normalised saliency > 0.4,
    a proxy for object size in the image.
    Good candidates: 0.05 < coverage < 0.55 (object neither tiny nor filling frame).
    """
    grid_h, grid_w = 6, 6
    try:
        result = clip_sal.compute_clip_salience_full_gate_v3(
            image, noun, grid_h, grid_w,
            top_k_pct=0.30, backup="none",
        )
        sal = result.saliency.cpu().float().numpy()
        sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        coverage = float((sal_norm > 0.4).mean())
        return saliency_overlay(sal, grid_h, grid_w, image), float(result.max_sim), coverage
    except Exception as e:
        print(f"  [WARN] CLIP error: {e}")
        return np.array(image.convert("RGB")), 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# VLM baseline decoder attention
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    print("Loading LLaVA-1.5-7B (eager)…")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).eval().cuda()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def baseline_overlay(image: Image.Image, question: str, model, processor) -> np.ndarray:
    """
    Mean attention FROM query-text positions TO image tokens,
    averaged over fusion layers 8-20 and all heads.
    """
    import torch

    prompt = LLAVA_PROMPT.format(question=question)
    inp    = processor(text=prompt, images=[image],
                       return_tensors="pt", padding=True).to("cuda")

    ids        = inp["input_ids"][0].cpu()
    seq_len    = ids.shape[0]
    img_tok_id = model.config.image_token_index
    vis_cfg    = model.config.vision_config
    n_img_tok  = (vis_cfg.image_size // vis_cfg.patch_size) ** 2   # 576

    img_start = (ids == img_tok_id).nonzero(as_tuple=True)[0][0].item()
    img_end   = img_start + n_img_tok

    img_mask   = torch.zeros(seq_len, dtype=torch.bool)
    img_mask[img_start:img_end] = True
    query_idx  = list(range(img_end, seq_len - 1)) or [seq_len - 1]

    captured: list = []

    def make_hook(li: int):
        def fn(module, _inp, out):
            if FUSE_START <= li <= FUSE_END:
                if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                    a   = out[1][0].float().cpu()             # (n_heads, seq, seq)
                    a_q = a[:, query_idx, :][:, :, img_mask]  # (n_heads, n_query, n_vis)
                    captured.append(a_q.detach())
                    return (out[0], None) + out[2:]
            return out
        return fn

    lm    = get_lm(model)
    hooks = [lay.self_attn.register_forward_hook(make_hook(i))
             for i, lay in enumerate(lm.layers)]
    try:
        with torch.inference_mode():
            model(**inp, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    if not captured:
        arr = np.zeros(n_img_tok, dtype=np.float32)
    else:
        arr = np.stack([t.numpy() for t in captured]).mean(axis=(0, 1, 2))  # (n_vis,)
        s = arr.sum()
        if s > 1e-9:
            arr /= s

    # LLaVA image tokens are 24×24 — upsample overlay from that grid
    return saliency_overlay(arr, 24, 24, image)


# ══════════════════════════════════════════════════════════════════════════════
# Figure rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_candidate(image: Image.Image, base_ov: np.ndarray, srf_ov: np.ndarray,
                     question: str, qtype: str, clip_sim: float, coverage: float,
                     out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), facecolor="white")
    fig.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.88, bottom=0.12)

    axes[0].imshow(np.array(image.convert("RGB")))
    axes[1].imshow(base_ov)
    axes[2].imshow(srf_ov)

    for ax, ttl, bc in zip(axes,
                            ["Image", "VLM Attention", "SRF Saliency"],
                            [None, "#e74c3c", "#27ae60"]):
        ax.axis("off")
        ax.set_title(ttl, fontsize=9, fontweight="bold", pad=3, color="#333")
        if bc:
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(bc)
                sp.set_linewidth(2.0)

    fig.suptitle(
        f'"{question}"',
        fontsize=12, fontweight="bold", color="#222", y=0.99,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out_path.name}")


def render_final(examples: list[dict], out_path: pathlib.Path) -> None:
    """6-panel: [img|attn|sal] | [img|attn|sal] matching Qwen Fig 3 style."""
    fig = plt.figure(figsize=(13.5, 3.2), facecolor="white")
    gs  = gridspec.GridSpec(
        1, 8, figure=fig,
        width_ratios=[1, 1, 1, 0.09, 1, 1, 1, 0.05],
        wspace=0.03, left=0.01, right=0.99, top=0.99, bottom=0.04,
    )
    panel_start = [0, 4]
    borders     = [None, "#e74c3c", "#27ae60"]

    for ex_idx, ex in enumerate(examples):
        c0 = panel_start[ex_idx]
        panels = [np.array(ex["image"].convert("RGB")), ex["base_ov"], ex["srf_ov"]]
        for p in range(3):
            ax = fig.add_subplot(gs[0, c0 + p])
            ax.imshow(panels[p])
            ax.axis("off")
            if borders[p]:
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_edgecolor(borders[p])
                    sp.set_linewidth(2.0)
            if p == 1:
                q = ex["question"].strip()
                if len(q) > 60:
                    q = q[:58] + "…"
                ax.set_title(f'"{q}"', fontsize=9.5, fontweight="bold",
                             color="#222", style="italic", pad=4)

    ax_gap = fig.add_subplot(gs[0, 3])
    ax_gap.axis("off")
    ax_gap.plot([0.5, 0.5], [0.05, 0.95], color="#ccc", lw=1.2,
                ls="--", transform=ax_gap.transAxes)
    fig.add_subplot(gs[0, 7]).axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Scan mode
# ══════════════════════════════════════════════════════════════════════════════

def scan(model, processor, n: int = 0) -> None:
    """Scan MMHal samples (n=0 means all).

    Ranking key: sim * size_bonus, where size_bonus peaks at coverage ≈ 0.25
    (object occupies ~1/4 of patches — not too big, not too small).
    Good range: 0.05 < coverage < 0.55.
    """
    samples = load_mmhal()
    if n > 0:
        samples = samples[:n]
    print(f"\nScanning {len(samples)} MMHal samples…")

    # ── Pass 1: CLIP saliency (fast, CPU/GPU) ──────────────────────────────────
    scored = []
    for s in samples:
        image = Image.open(s["img_path"]).convert("RGB")
        noun  = extract_clip_noun(s["question"], mode="pope")
        try:
            result = clip_sal.compute_clip_salience_full_gate_v3(
                image, noun, 6, 6, top_k_pct=0.30, backup="none")
            sim = float(result.max_sim)
            sal = result.saliency.cpu().float().numpy()
            sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            coverage = float((sal_norm > 0.4).mean())
        except Exception:
            sim, coverage = 0.0, 0.0
        # size bonus: bell curve peaking at coverage=0.25, falls off for tiny/large
        size_bonus = float(np.exp(-((coverage - 0.25) ** 2) / (2 * 0.15 ** 2)))
        rank_score = sim * size_bonus
        scored.append((rank_score, sim, coverage, s, image, noun))

    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"\nTop candidates (ranked by sim × size_bonus):")
    print(f"  {'rank':>4}  {'score':>6}  {'sim':>6}  {'cov':>5}  {'type':12}  {'noun':15}  question")
    for rank, (score, sim, cov, s, _, noun) in enumerate(scored[:30]):
        flag = "✓" if 0.05 < cov < 0.55 else " "
        print(f"  {rank:03d}  {score:.3f}  {sim:.3f}  {cov:.2f} {flag}  "
              f"[{s['question_type']:12s}]  {noun:15s}  {s['question'][:55]}")

    print(f"\nRendering all candidates…")
    for rank, (score, sim, cov, s, image, noun) in enumerate(scored):
        base_ov         = baseline_overlay(image, s["question"], model, processor)
        srf_ov, _, _    = run_srf_saliency(image, noun)
        safe_q          = re.sub(r"[^\w]", "_", s["question"])[:30]
        out_path        = CAND_DIR / f"{rank:03d}_sim{sim:.3f}_cov{cov:.2f}_{s['question_type']}_{safe_q}.png"
        render_candidate(image, base_ov, srf_ov, s["question"],
                         s["question_type"], sim, cov, out_path)

    print(f"\nCandidates → {CAND_DIR}/")
    print("Good range: 0.05 < cov < 0.55 (object not too small, not filling frame)")
    print("Pick two, then run:")
    print("  python srf/analysis/llava_fig3_routing.py --idx_a 003 --idx_b 011")


# ══════════════════════════════════════════════════════════════════════════════
# Final render mode
# ══════════════════════════════════════════════════════════════════════════════

def load_candidate(idx: str, model, processor) -> dict:
    matches = sorted(CAND_DIR.glob(f"{idx}*.png"))
    if not matches:
        raise FileNotFoundError(f"No candidate '{idx}*.png' in {CAND_DIR}")

    # Recover metadata from filename: NNN_simX.XXX_<type>_<question>.png
    fname  = matches[0].stem
    parts  = fname.split("_")
    # rank=parts[0], sim=parts[1], type=parts[2], question from the rest
    qtype  = parts[2] if len(parts) > 2 else "unknown"

    samples = load_mmhal()
    rank    = int(parts[0])
    scored  = []
    for s in samples:
        image = Image.open(s["img_path"]).convert("RGB")
        noun  = extract_clip_noun(s["question"], mode="pope")
        try:
            result = clip_sal.compute_clip_salience_full_gate_v3(
                image, noun, 6, 6, top_k_pct=0.30, backup="none")
            sim = float(result.max_sim)
            sal = result.saliency.cpu().float().numpy()
            sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            cov = float((sal_norm > 0.4).mean())
        except Exception:
            sim, cov = 0.0, 0.0
        size_bonus = float(np.exp(-((cov - 0.25) ** 2) / (2 * 0.15 ** 2)))
        scored.append((sim * size_bonus, sim, cov, s, image, noun))
    scored.sort(key=lambda x: x[0], reverse=True)

    score, sim, cov, s, image, noun = scored[rank]
    print(f"  Candidate {idx}: [{s['question_type']}] noun='{noun}' sim={sim:.3f} cov={cov:.2f}")
    print(f"  Q: {s['question']}")
    print("  baseline attention…")
    base_ov = baseline_overlay(image, s["question"], model, processor)
    print("  SRF saliency…")
    srf_ov, _, _ = run_srf_saliency(image, noun)
    return dict(image=image, base_ov=base_ov, srf_ov=srf_ov, question=s["question"])


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scan",  action="store_true", help="Scan MMHal samples, save candidates")
    p.add_argument("--n",     type=int, default=0, help="Number of samples to scan (0 = all)")
    p.add_argument("--idx_a", default="000", help="Candidate index for left example")
    p.add_argument("--idx_b", default="001", help="Candidate index for right example")
    args = p.parse_args()

    model, processor = load_model()

    if args.scan:
        scan(model, processor, n=args.n)
        return

    if not CAND_DIR.exists() or not any(CAND_DIR.glob("*.png")):
        print("No candidates found. Run --scan first.")
        return

    ex_a = load_candidate(args.idx_a, model, processor)
    ex_b = load_candidate(args.idx_b, model, processor)
    render_final([ex_a, ex_b], FINAL_OUT)


if __name__ == "__main__":
    main()
