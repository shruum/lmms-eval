#!/usr/bin/env python3
"""
Attention heatmap + gradient attribution visualization for VLMs-Are-Biased.

Two complementary influence metrics per sample:
  1. Attn Map  — layer-averaged direct attention from last_pos to each image token
                 (spatial version of VTAR; same scalar, per-patch heatmap)
  2. Grad×Emb  — |embedding × gradient| saliency w.r.t. first output token logit

NOTE on attention rollout:
  Classic rollout (Abnar & Zuidema 2020) multiplies lower-triangular attention
  matrices across layers.  For causal decoders the product collapses to ~0 for
  early tokens (image patches) after ~36 layers.  It works in bidirectional ViTs
  but is the wrong tool here.  We will revisit with gradient-weighted rollout or
  attention flow if needed.

Per-sample output:
  viz_{idx}_{sample_id}.png   3-panel: original | attn-map overlay | grad overlay

Console:
  sample | VTAR% | AttnMap% | Grad%

Usage (from repo root):
    PYTHONPATH=. python my_analysis/test_attribution_viz.py \\
        --n_samples 5 --max_image_size 320 --output_dir results/attr_test
"""
from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image

from analysis_utils import cap_image_size, safe_sample_id, to_rgb

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------


class QwenAttributionRunner:
    """Qwen2.5-VL runner: VTAR scalar + per-token attention map + grad attribution."""

    def __init__(self, model_name: str, device: str = "cuda:0", torch_dtype: str = "bfloat16") -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError("Install qwen-vl-utils: uv add qwen-vl-utils") from exc

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        self.torch = torch
        self.device = torch.device(device)
        self.process_vision_info = process_vision_info

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=dtype_map[torch_dtype],
            attn_implementation="eager",  # required for output_attentions=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.image_token_id: int = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        # .language_model in some transformers versions, .model in others
        self._lm = getattr(self.model, "language_model", None) or self.model.model

    # -----------------------------------------------------------------------
    # Input preparation
    # -----------------------------------------------------------------------

    def _build_inputs(self, image: Image.Image, prompt: str) -> Any:
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        return inputs.to(self.device)

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def run_all(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Compute VTAR, per-token attention map, and gradient attribution."""
        torch = self.torch
        inputs = self._build_inputs(image, prompt)

        input_ids_cpu = inputs["input_ids"][0].cpu()          # [S]
        image_mask = input_ids_cpu == self.image_token_id      # bool [S]
        n_img = int(image_mask.sum())
        n_txt = int((~image_mask).sum())
        last_pos = input_ids_cpu.shape[0] - 1                  # last input token

        # ------------------------------------------------------------------
        # STEP 1: Hook — capture last_pos attention row at every layer.
        #
        # We only need last_pos's row: output[1][0, :, last_pos, :] → [n_heads, S].
        # Storing the full [n_heads, S, S] per layer would cost ~800 MB RAM
        # for S≈170, 36 layers, 16 heads and is not needed here.
        # ------------------------------------------------------------------
        captured: List[Any] = []  # each: [n_heads, S] float32 on CPU

        def _make_hook(storage: List[Any]) -> Any:
            def hook_fn(module: Any, _inp: Any, output: Any) -> Any:
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    # output[1]: [batch, n_heads, S, S]
                    storage.append(output[1][0, :, last_pos, :].detach().cpu())  # [n_heads, S]
                    return (output[0], None) + output[2:]  # drop attn from GPU immediately
                return output
            return hook_fn

        layers = self._lm.layers
        hooks = [layer.self_attn.register_forward_hook(_make_hook(captured)) for layer in layers]
        try:
            with torch.inference_mode():
                self.model(**inputs, output_attentions=True)
        finally:
            for h in hooks:
                h.remove()

        assert len(captured) == len(layers), (
            f"Expected {len(layers)} layers, got {len(captured)}. Use attn_implementation=eager."
        )

        # ------------------------------------------------------------------
        # STEP 2: VTAR scalar — fraction of last_pos attention going to image
        #         tokens, averaged over heads and layers.
        # ------------------------------------------------------------------
        vtar_per_layer: List[float] = []
        for row in captured:
            # row: [n_heads, S]
            vis = row[:, image_mask].sum(dim=-1)          # [n_heads]
            tot = row.sum(dim=-1).clamp(min=1e-9)         # [n_heads]
            vtar_per_layer.append(float((vis / tot).mean()))
        vtar = float(np.mean(vtar_per_layer))

        # Free GPU allocator fragmentation before the grad pass
        torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # STEP 3: Per-token attention map — same data as VTAR but kept per
        #         image token so we can build a spatial heatmap.
        #
        #   For each layer: normalize the last_pos row, then average per-token
        #   weights over heads.  Average across all layers.  Result [n_img]
        #   shows which image patches receive the most direct attention.
        #   The sum of img_attn_map == VTAR (sanity check).
        # ------------------------------------------------------------------
        img_attn_map, attn_map_vision_frac = self._compute_attn_map(captured, image_mask)

        # ------------------------------------------------------------------
        # STEP 4: Gradient × Embedding attribution.
        #
        #   Freeze model weights (avoids allocating ~6 GB of weight gradients),
        #   forward with merged inputs_embeds, backprop from argmax logit,
        #   saliency = |emb * grad| summed over hidden dim.
        # ------------------------------------------------------------------
        img_grad, grad_vision_frac = self._compute_grad_attribution(inputs, image_mask)

        # ------------------------------------------------------------------
        # STEP 5: Resolve image_grid_thw → token grid (t, h_tok, w_tok).
        #
        #   image_grid_thw is in patch units BEFORE Qwen's 2×2 spatial merge,
        #   so token grid = (t, h // merge_size, w // merge_size).
        # ------------------------------------------------------------------
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None and len(image_grid_thw) > 0:
            g = image_grid_thw[0]
            t_dim, h_dim, w_dim = int(g[0]), int(g[1]), int(g[2])
            if n_img == t_dim * h_dim * w_dim:
                grid_thw = (t_dim, h_dim, w_dim)           # no merge (unusual)
            else:
                ms: int = getattr(getattr(self.processor, "image_processor", None), "merge_size", 2)
                grid_thw = (t_dim, h_dim // ms, w_dim // ms)
        else:
            side = max(1, int(round(n_img ** 0.5)))
            grid_thw = (1, side, side)
            print(f"  Warning: image_grid_thw missing, using fallback grid {grid_thw}")

        t, h, w = grid_thw
        if n_img != t * h * w:
            print(f"  Warning: token count mismatch n_img={n_img} != t*h*w={t*h*w}")

        return {
            "n_image_tokens": n_img,
            "n_text_tokens": n_txt,
            # VTAR
            "vtar": vtar,
            "vtar_per_layer": vtar_per_layer,
            # Per-token attention map
            "img_attn_map": img_attn_map,           # [n_img] for heatmap
            "attn_map_vision_frac": attn_map_vision_frac,  # should ≈ vtar
            # Gradient attribution
            "img_grad": img_grad,                   # [n_img] for heatmap
            "grad_vision_frac": grad_vision_frac,
            # Grid
            "grid_thw": grid_thw,
        }

    # -----------------------------------------------------------------------
    # Per-token attention map
    # -----------------------------------------------------------------------

    def _compute_attn_map(
        self,
        captured: List[Any],   # list of [n_heads, S] tensors
        image_mask: Any,       # bool [S]
    ) -> Tuple[np.ndarray, float]:
        """
        Layer-averaged per-image-token attention from last_pos.

        For each layer: normalize the last_pos row → [S] sums to 1.
        Average over heads, then over layers.
        Extract image-token columns → [n_img].

        Returns:
          img_scores: [n_img] float32  (for heatmap)
          vision_frac: scalar == sum(img_scores)  (≈ VTAR)
        """
        img_mask_np = image_mask.numpy()
        accum = np.zeros(int(image_mask.sum()), dtype=np.float64)

        for row in captured:
            # row: [n_heads, S]
            row_f = row.float()
            # normalize each head's row so it sums to 1 (softmax already does,
            # but clamp guards against rare zero rows)
            row_norm = row_f / row_f.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            # mean over heads → [S]
            mean_row = row_norm.mean(dim=0).numpy()
            # accumulate only image-token positions
            accum += mean_row[img_mask_np]

        accum /= max(len(captured), 1)          # average over layers
        img_scores = accum.astype(np.float32)
        vision_frac = float(img_scores.sum())
        return img_scores, vision_frac

    # -----------------------------------------------------------------------
    # Gradient × Embedding attribution
    # -----------------------------------------------------------------------

    def _compute_grad_attribution(
        self,
        inputs: Any,
        image_mask: Any,       # bool [S]
    ) -> Tuple[np.ndarray, float]:
        """
        Gradient × Embedding saliency for the first output token.

        Freezes model weights so backward only needs to propagate to
        inputs_embeds — avoids allocating ~6 GB of parameter gradients.

        Returns:
          img_scores: [n_img] float32  (for heatmap)
          vision_frac: scalar  (fraction of total saliency in image tokens)
        """
        torch = self.torch
        img_mask_np = image_mask.numpy()

        # Build merged inputs_embeds: text embeddings + visual patch embeddings
        with torch.no_grad():
            text_embeds = self._lm.embed_tokens(inputs["input_ids"])  # [1, S, D]
            if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                vis_out = self.model.visual(
                    inputs["pixel_values"],
                    grid_thw=inputs.get("image_grid_thw"),
                )  # [n_img, D]
                img_mask_gpu = inputs["input_ids"][0] == self.image_token_id
                text_embeds[0, img_mask_gpu] = vis_out.to(text_embeds.dtype)

        inputs_embeds = text_embeds.detach().requires_grad_(True)

        fwd: Dict[str, Any] = {
            "input_ids": None,
            "inputs_embeds": inputs_embeds,
            "pixel_values": None,
            "attention_mask": inputs.get("attention_mask"),
            "position_ids": inputs.get("position_ids"),
        }
        if "rope_deltas" in inputs:
            fwd["rope_deltas"] = inputs["rope_deltas"]

        self.model.requires_grad_(False)
        try:
            with torch.enable_grad():
                logits = self.model(**fwd).logits          # [1, S, vocab]
                target = int(logits[0, -1].detach().argmax())
                logits[0, -1, target].backward()
        finally:
            self.model.requires_grad_(True)
            torch.cuda.empty_cache()

        grad = inputs_embeds.grad[0].detach().cpu().float()   # [S, D]
        emb  = inputs_embeds.detach()[0].cpu().float()        # [S, D]
        saliency = (emb * grad).sum(dim=-1).abs().numpy()     # [S]

        img_scores = saliency[img_mask_np].astype(np.float32)
        vision_frac = float(img_scores.sum() / max(saliency.sum(), 1e-9))
        return img_scores, vision_frac


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _token_scores_to_overlay(
    img_arr: np.ndarray,          # [H, W, 3] float32 in [0, 1]
    scores: np.ndarray,           # [n_img_tokens] float
    grid_thw: Tuple[int, int, int],
    alpha: float = 0.50,
) -> np.ndarray:
    """Reshape per-image-token scores to a spatial heatmap and overlay on image."""
    t, h, w = grid_thw
    H, W = img_arr.shape[:2]

    spatial = scores.reshape(t * h, w).astype(np.float32)
    vmin, vmax = spatial.min(), spatial.max()
    spatial_norm = (spatial - vmin) / max(vmax - vmin, 1e-9)

    heat_pil = Image.fromarray((spatial_norm * 255).astype(np.uint8), mode="L")
    heat_pil = heat_pil.resize((W, H), Image.Resampling.BILINEAR)
    heat_arr = np.array(heat_pil) / 255.0                    # [H, W]

    heatmap_rgb = plt.cm.jet(heat_arr)[:, :, :3]             # [H, W, 3]
    return np.clip((1.0 - alpha) * img_arr + alpha * heatmap_rgb, 0.0, 1.0)


def save_visualization(
    image: Image.Image,
    result: Dict[str, Any],
    out_path: str,
    title: str,
) -> None:
    """3-panel figure: original image | attention map overlay | gradient overlay."""
    img_arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    grid_thw: Tuple[int, int, int] = result["grid_thw"]

    attn_overlay = _token_scores_to_overlay(img_arr, result["img_attn_map"], grid_thw)
    grad_overlay = _token_scores_to_overlay(img_arr, result["img_grad"], grid_thw)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_arr)
    axes[0].set_title(
        f"Original\n"
        f"img tokens={result['n_image_tokens']}  txt tokens={result['n_text_tokens']}\n"
        f"VTAR  vision={result['vtar']*100:.1f}%  text={(1-result['vtar'])*100:.1f}%"
    )
    axes[0].axis("off")

    axes[1].imshow(attn_overlay)
    axes[1].set_title(
        f"Attention Map (layer avg)\n"
        f"vision={result['attn_map_vision_frac']*100:.1f}%  "
        f"text={(1-result['attn_map_vision_frac'])*100:.1f}%"
    )
    axes[1].axis("off")

    axes[2].imshow(grad_overlay)
    axes[2].set_title(
        f"Gradient × Embedding\n"
        f"vision={result['grad_vision_frac']*100:.1f}%  "
        f"text={(1-result['grad_vision_frac'])*100:.1f}%"
    )
    axes[2].axis("off")

    fig.suptitle(title, fontsize=11, y=1.01)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.015, pad=0.01,
                 label="Relative score (low → high)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attention map + gradient attribution viz")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--torch_dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--dataset", type=str, default="anvo25/vlms-are-biased")
    p.add_argument("--split", type=str, default="main")
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_image_size", type=int, default=320,
                   help="Cap longer side before processing (keeps token count manageable)")
    p.add_argument("--output_dir", type=str, default="results/attr_test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset {args.dataset} [{args.split}] ...")
    ds = load_dataset(args.dataset, split=args.split)
    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)
    ds = ds.select(indices[: args.n_samples])
    print(f"  {len(ds)} samples selected\n")

    print(f"Loading model {args.model_name} ...")
    runner = QwenAttributionRunner(args.model_name, device=args.device, torch_dtype=args.torch_dtype)
    print(f"  Loaded on {args.device}  (eager attention)\n")

    rows = []
    for idx, sample in enumerate(ds):
        sample_id = safe_sample_id(sample, idx)
        prompt = str(sample.get("prompt", "")).strip()
        topic  = str(sample.get("topic", "unknown")).strip()
        gt     = str(sample.get("ground_truth", "")).strip()

        if not prompt:
            print(f"[{idx+1}/{len(ds)}] {sample_id}: skipping (no prompt)")
            continue

        image = cap_image_size(to_rgb(sample["image"]), args.max_image_size)
        print(f"[{idx+1}/{len(ds)}] id={sample_id}  topic={topic}  gt={gt!r}  img={image.size}")

        result = runner.run_all(image, prompt)

        print(f"  VTAR     : vision={result['vtar']*100:5.2f}%  "
              f"(AttnMap cross-check: {result['attn_map_vision_frac']*100:5.2f}%)")
        print(f"  Grad×Emb : vision={result['grad_vision_frac']*100:5.2f}%")

        out_path = os.path.join(args.output_dir, f"viz_{idx:02d}_{sample_id}.png")
        save_visualization(image, result, out_path,
                           f"id={sample_id} | {topic} | gt={gt!r}")
        rows.append({"sample_id": sample_id, "topic": topic,
                     "vtar": result["vtar"], "grad": result["grad_vision_frac"]})
        print()

    # Summary
    print("=" * 64)
    print("  Vision influence % per sample")
    print("=" * 64)
    print(f"  {'id':<12}  {'topic':<22}  {'VTAR':>7}  {'Grad×Emb':>9}")
    print(f"  {'-'*55}")
    for r in rows:
        print(f"  {r['sample_id']:<12}  {r['topic'][:22]:<22}  "
              f"{r['vtar']*100:>6.1f}%  {r['grad']*100:>8.1f}%")
    if rows:
        print(f"  {'-'*55}")
        print(f"  {'mean':<12}  {'':22}  "
              f"{np.mean([r['vtar'] for r in rows])*100:>6.1f}%  "
              f"{np.mean([r['grad'] for r in rows])*100:>8.1f}%")
    print(f"\nSaved to: {args.output_dir}/")


if __name__ == "__main__":
    main()