#!/usr/bin/env python3
"""Attention analysis for MMBench-Dev (EN) with Qwen2.5-VL.

Measures Vision Token Attention Ratio (VTAR) — how much attention the first
generated token pays to image tokens vs text tokens at each layer.

Method (following common attention analysis practice, e.g. arXiv 2503.01773):
  1. Build baseline prompt: real image + question + options.
  2. Forward pass with output_attentions=True via per-layer hooks.
  3. Query position: last token of the input (= predicting the first output token).
  4. Per layer:
       VTAR[layer] = mean_over_heads(
           sum(attn[last_input_pos, image_positions]) /
           sum(attn[last_input_pos, all_positions])
       )
  5. Aggregate per sample, per category, and layer-wise.

NOTE: incompatible with flash_attention_2.
      Use --attn_implementation sdpa (default) or eager.

Memory note: hooks capture only the (n_heads, seq_len) slice at last_pos
per layer and immediately replace the full (B, H, S, S) tensor with None,
so only one layer's attention matrix lives in memory at a time.

Outputs (all in --output_dir):
  attention_records.jsonl              per-sample data: VTAR + prediction
  summary.json                         all aggregated numbers
  plots/
    plot1_layerwise_vtar.png           mean ± std VTAR per layer (all samples)
    plot2_vtar_by_category.png         per-category mean VTAR bar chart
    plot3_head_layer_heatmap.png       head × layer VTAR heatmap
    plot4_vtar_vs_accuracy.png         per-sample VTAR vs correctness
    plot5_layerwise_by_category.png    layer-wise VTAR for top-N categories
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"

DEFAULT_SUFFIX = "Answer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# Tee — write every print() to both stdout and a log file simultaneously
# ---------------------------------------------------------------------------


class _Tee:
    """Wraps sys.stdout so that every write goes to both the terminal and a file."""

    def __init__(self, log_path: str) -> None:
        self._terminal = sys.stdout
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, message: str) -> int:
        self._terminal.write(message)
        self._log.write(message)
        return len(message)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def close(self) -> None:
        if not self._log.closed:
            self._log.close()

    # Delegate anything else (isatty, fileno, …) to the real terminal
    def __getattr__(self, name: str) -> Any:
        return getattr(self._terminal, name)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB") if image.mode != "RGB" else image


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


def build_prompt(question: str, options: Dict[str, str], hint: str = "", suffix: str = DEFAULT_SUFFIX) -> str:
    lines: List[str] = []
    if hint:
        lines.append(f"Context: {hint}")
    lines.append(question)
    for letter, text in options.items():
        lines.append(f"{letter}. {text}")
    lines.append(suffix)
    return "\n".join(lines)


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


def safe_sample_id(sample: Dict[str, Any], idx: int) -> str:
    for key in ["index", "id", "sample_id", "question_id"]:
        if key in sample and str(sample[key]).strip() != "":
            return str(sample[key])
    return str(idx)


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------


class QwenAttentionRunner:
    """Loads Qwen2.5-VL and runs baseline inference + VTAR attention extraction."""

    def __init__(
        self,
        model_name: str,
        device_map: str,
        torch_dtype: str,
        max_new_tokens: int,
        attn_implementation: str = "sdpa",
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError("qwen_vl_utils is required. Install with `uv add qwen-vl-utils`") from exc

        if attn_implementation == "flash_attention_2":
            raise ValueError(
                "flash_attention_2 does not support output_attentions=True. "
                "Use --attn_implementation sdpa or eager."
            )

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}")

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.max_new_tokens = max_new_tokens

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype_map[torch_dtype],
            attn_implementation=attn_implementation,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _build_inputs(self, image: Image.Image, prompt: str) -> Any:
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        device = self.model.device if (hasattr(self.model, "device") and self.model.device is not None) else "cuda"
        return inputs.to(device)

    def predict_letter(self, image: Image.Image, prompt: str, valid_letters: Iterable[str]) -> Tuple[str, str]:
        inputs = self._build_inputs(image, prompt)
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
        return raw, extract_option_letter(raw, valid_letters)

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------

    def sanity_check_tokens(self, image: Image.Image, prompt: str, image_token: str = "<|image_pad|>") -> None:
        """Print a detailed breakdown of the token sequence to verify image_mask.

        Checks:
          1. Whether image_token resolves to a real vocab ID (not UNK).
          2. Counts and positions of every Qwen vision special token.
          3. Contiguous image-token block(s): start / end positions.
          4. Decoded context tokens immediately before and after each block.
          5. Last 10 tokens (= end of prompt / generation prefix).
          6. Row-sum sanity on a tiny attention slice (optional).
        """
        SEP = "=" * 64
        inputs = self._build_inputs(image, prompt)
        tok = self.processor.tokenizer
        input_ids = inputs["input_ids"][0].cpu()
        seq_len = len(input_ids)

        # --- [1] Image-token ID resolution ---
        image_token_id = tok.convert_tokens_to_ids(image_token)
        unk_id = tok.unk_token_id
        print(f"\n{SEP}")
        print("SANITY CHECK — TOKEN LAYOUT")
        print(SEP)
        print(f"\n[1] Image-token resolution")
        print(f"    token string : '{image_token}'")
        print(f"    token id     : {image_token_id}")
        if image_token_id == unk_id:
            print(f"    *** WARNING: resolves to UNK ({unk_id}). Token not in vocabulary! ***")
        else:
            print(f"    OK — found in vocabulary")

        # --- [2] All Qwen vision special tokens ---
        print(f"\n[2] Qwen vision special-token IDs (and counts in this sequence)")
        vision_tokens = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|video_pad|>"]
        for vt in vision_tokens:
            vid = tok.convert_tokens_to_ids(vt)
            cnt = int((input_ids == vid).sum())
            flag = " *** UNK ***" if vid == unk_id else ""
            print(f"    '{vt}': id={vid}, count={cnt}{flag}")

        # --- [3] Sequence-level counts ---
        image_mask = input_ids == image_token_id
        n_image = int(image_mask.sum())
        n_text = int((~image_mask).sum())
        print(f"\n[3] Sequence lengths")
        print(f"    total tokens : {seq_len}")
        print(f"    image tokens : {n_image}  ({n_image / seq_len * 100:.1f}%)")
        print(f"    text  tokens : {n_text}  ({n_text / seq_len * 100:.1f}%)")

        # --- [4] Contiguous image-token block(s) ---
        image_positions = image_mask.nonzero(as_tuple=True)[0].tolist()
        print(f"\n[4] Contiguous image-token block(s)")
        if not image_positions:
            print("    *** NO image tokens found — image_mask is all False! ***")
        else:
            runs: List[Tuple[int, int]] = []
            start = image_positions[0]
            prev = image_positions[0]
            for pos in image_positions[1:]:
                if pos != prev + 1:
                    runs.append((start, prev))
                    start = pos
                prev = pos
            runs.append((start, prev))
            for i, (s, e) in enumerate(runs):
                print(f"    block {i}: positions [{s} … {e}]  ({e - s + 1} tokens)")

        # --- [5] Decoded context around each image block boundary ---
        CONTEXT = 4
        if image_positions:
            runs_to_show = runs[:2]  # avoid flooding output for many blocks
            for i, (s, e) in enumerate(runs_to_show):
                print(f"\n[5.{i}] Context around block {i} (±{CONTEXT} tokens)")
                window_start = max(0, s - CONTEXT)
                window_end = min(seq_len - 1, e + CONTEXT)
                for pos in range(window_start, window_end + 1):
                    tid = input_ids[pos].item()
                    decoded = tok.decode([tid]).replace("\n", "\\n")
                    if pos == s:
                        tag = "  ← IMAGE START"
                    elif pos == e:
                        tag = "  ← IMAGE END"
                    elif image_mask[pos]:
                        tag = "  [img]"
                    else:
                        tag = ""
                    print(f"    [{pos:5d}] id={tid:7d}  '{decoded}'{tag}")

        # --- [6] Last 10 tokens (prompt tail / generation trigger) ---
        print(f"\n[6] Last 10 tokens (end of prompt)")
        for i, tid in enumerate(input_ids[-10:].tolist()):
            real_i = seq_len - 10 + i
            decoded = tok.decode([tid]).replace("\n", "\\n")
            tag = "  ← last_pos (query)" if real_i == seq_len - 1 else ""
            print(f"    [{real_i:5d}] id={tid:7d}  '{decoded}'{tag}")

        print(f"\n{SEP}\n")

    def sanity_check_attention(self, image: Image.Image, prompt: str, image_token: str = "<|image_pad|>") -> None:
        """Run one forward pass with output_attentions=True and verify hook captures.

        Checks:
          1. How many layers returned non-None attention weights.
          2. Attention shape matches expectations.
          3. Row-sums at last_pos are ~1.0 (proper softmax distribution).
          4. Per-layer VTAR for the first / middle / last layer.
        """
        SEP = "=" * 64
        inputs = self._build_inputs(image, prompt)
        tok = self.processor.tokenizer
        image_token_id = tok.convert_tokens_to_ids(image_token)
        input_ids_cpu = inputs["input_ids"][0].cpu()
        image_mask = input_ids_cpu == image_token_id
        last_pos = inputs["input_ids"].shape[1] - 1

        captured_ok: List[Any] = []   # (layer_idx, tensor) for non-None captures
        captured_none: List[int] = []  # layer indices where output[1] was None

        def make_hook(layer_idx: int) -> Any:
            def hook_fn(module: Any, _inp: Any, output: Any) -> Any:
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    attn_w = output[1]  # (1, n_heads, seq_len, seq_len)
                    captured_ok.append((layer_idx, attn_w[0, :, last_pos, :].detach().cpu()))
                    return (output[0], None) + output[2:]
                else:
                    captured_none.append(layer_idx)
                return output
            return hook_fn

        try:
            layers = self.model.language_model.layers
        except AttributeError as exc:
            raise RuntimeError("Cannot access model.language_model.layers") from exc

        hooks = []
        for li, layer in enumerate(layers):
            hooks.append(layer.self_attn.register_forward_hook(make_hook(li)))

        try:
            with self.torch.inference_mode():
                self.model(**inputs, output_attentions=True)
        finally:
            for h in hooks:
                h.remove()

        n_layers = len(layers)
        n_ok = len(captured_ok)
        n_none = len(captured_none)

        print(f"\n{SEP}")
        print("SANITY CHECK — ATTENTION CAPTURE")
        print(SEP)
        print(f"\n[1] Hook capture summary")
        print(f"    total layers              : {n_layers}")
        print(f"    layers with attn weights  : {n_ok}")
        print(f"    layers with None (SDPA?)  : {n_none}")
        if n_none > 0:
            print(f"    *** WARNING: {n_none} layers returned None for attention weights.")
            print(f"        This typically means output_attentions=True is ignored by SDPA.")
            print(f"        Switch to --attn_implementation eager to fix this. ***")
        if n_ok == 0:
            print(f"\n    Nothing to inspect — no attention weights captured.")
            print(SEP)
            return

        # Token-count breakdown (input composition)
        n_vision_tok = int(image_mask.sum())
        n_text_tok = int((~image_mask).sum())
        n_total_tok = len(image_mask)
        print(f"\n[2] Input token composition")
        print(f"    {'modality':<12}  {'count':>6}  {'share of input':>14}")
        print(f"    {'-'*36}")
        print(f"    {'vision':<12}  {n_vision_tok:>6}  {n_vision_tok / n_total_tok * 100:>13.1f}%")
        print(f"    {'text':<12}  {n_text_tok:>6}  {n_text_tok / n_total_tok * 100:>13.1f}%")
        print(f"    {'total':<12}  {n_total_tok:>6}")

        # Show first, middle, last captured layer
        sample_entries = [captured_ok[0], captured_ok[n_ok // 2], captured_ok[-1]]
        labels = ["first", "middle", "last"]
        print(f"\n[3] Per-layer attention checks (first / middle / last captured layer)")
        print(f"    (query = last input token → attending over full sequence)")
        for label, (li, attn_slice) in zip(labels, sample_entries):
            row_sums = attn_slice.sum(dim=-1)                           # (n_heads,)
            vision_sum = attn_slice[:, image_mask].sum(dim=-1)          # (n_heads,)
            text_sum = attn_slice[:, ~image_mask].sum(dim=-1)           # (n_heads,)
            total_sum = row_sums.clamp(min=1e-9)
            vision_frac = vision_sum / total_sum                        # (n_heads,)
            text_frac = text_sum / total_sum                            # (n_heads,)

            print(f"\n    --- layer {li} ({label}) ---")
            print(f"    attn_slice shape : {list(attn_slice.shape)}  (n_heads × seq_len)")
            print(f"    row-sum per head : min={float(row_sums.min()):.4f}  max={float(row_sums.max()):.4f}  mean={float(row_sums.mean()):.4f}  (should be ~1.0)")
            print(f"    {'modality':<10}  {'tokens':>6}  {'input %':>8}  {'attn mean':>10}  {'attn min':>9}  {'attn max':>9}")
            print(f"    {'-'*58}")
            print(f"    {'vision':<10}  {n_vision_tok:>6}  {n_vision_tok/n_total_tok*100:>7.1f}%  {float(vision_frac.mean())*100:>9.2f}%  {float(vision_frac.min())*100:>8.2f}%  {float(vision_frac.max())*100:>8.2f}%")
            print(f"    {'text':<10}  {n_text_tok:>6}  {n_text_tok/n_total_tok*100:>7.1f}%  {float(text_frac.mean())*100:>9.2f}%  {float(text_frac.min())*100:>8.2f}%  {float(text_frac.max())*100:>8.2f}%")

        # Overall VTAR / TTAR across all captured layers
        per_layer_vision_frac: List[float] = []
        per_layer_text_frac: List[float] = []
        for _, attn_slice in captured_ok:
            ts = attn_slice.sum(dim=-1).clamp(min=1e-9)
            per_layer_vision_frac.append(float((attn_slice[:, image_mask].sum(dim=-1) / ts).mean()))
            per_layer_text_frac.append(float((attn_slice[:, ~image_mask].sum(dim=-1) / ts).mean()))
        overall_vision = sum(per_layer_vision_frac) / len(per_layer_vision_frac)
        overall_text = sum(per_layer_text_frac) / len(per_layer_text_frac)

        print(f"\n[4] Overall attention allocation (averaged over {n_ok} layers)")
        print(f"    {'modality':<10}  {'tokens':>6}  {'input %':>8}  {'output attn %':>14}")
        print(f"    {'-'*44}")
        print(f"    {'vision':<10}  {n_vision_tok:>6}  {n_vision_tok/n_total_tok*100:>7.1f}%  {overall_vision*100:>13.2f}%")
        print(f"    {'text':<10}  {n_text_tok:>6}  {n_text_tok/n_total_tok*100:>7.1f}%  {overall_text*100:>13.2f}%")
        print(f"    vision attn: {overall_vision*100:.2f}%   text attn: {overall_text*100:.2f}%")
        print(SEP + "\n")

    def get_attention_stats(self, image: Image.Image, prompt: str, image_token: str = "<|image_pad|>") -> Dict[str, Any]:
        """Extract per-layer VTAR via forward hooks.

        Hooks capture attn[0, :, last_input_pos, :] (shape: n_heads × seq_len)
        for each layer and immediately replace the full (B, H, S, S) attention
        tensor with None — so only one layer's matrix lives in GPU memory at a time.

        Args:
            image:        PIL image (baseline, no modifications).
            prompt:       Formatted question + options prompt.
            image_token:  Special token string for image patches in input_ids.

        Returns dict with keys:
            per_layer_vtar:          List[float]         shape [n_layers]
            per_layer_per_head_vtar: List[List[float]]   shape [n_layers, n_heads]
            overall_vtar:            float   (mean over layers)
            n_image_tokens:          int
            n_text_tokens:           int
        """
        inputs = self._build_inputs(image, prompt)

        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)
        input_ids_cpu = inputs["input_ids"][0].cpu()
        image_mask = input_ids_cpu == image_token_id
        n_image_tokens = int(image_mask.sum().item())
        n_text_tokens = int((~image_mask).sum().item())
        last_pos = inputs["input_ids"].shape[1] - 1

        captured: List[Any] = []  # one (n_heads, seq_len) tensor per layer

        def make_hook(storage: List[Any]) -> Any:
            def hook_fn(module: Any, _inp: Any, output: Any) -> Any:
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    attn_w = output[1]  # (1, n_heads, seq_len, seq_len)
                    storage.append(attn_w[0, :, last_pos, :].detach().cpu())  # (n_heads, seq_len)
                    # Replace the full attention tensor with None to free GPU memory immediately
                    return (output[0], None) + output[2:]
                return output

            return hook_fn

        try:
            layers = self.model.language_model.layers
        except AttributeError as exc:
            raise RuntimeError("Cannot access model.model.layers — unsupported model architecture.") from exc

        hooks = []
        for layer in layers:
            hooks.append(layer.self_attn.register_forward_hook(make_hook(captured)))

        try:
            with self.torch.inference_mode():
                self.model(**inputs, output_attentions=True)
        finally:
            for h in hooks:
                h.remove()

        per_layer_per_head_vtar: List[List[float]] = []
        per_layer_vtar: List[float] = []

        assert len(captured) == len(layers)
        for attn_slice in captured:  # (n_heads, seq_len)
            vision_sum = attn_slice[:, image_mask].sum(dim=-1)  # (n_heads,)
            total_sum = attn_slice.sum(dim=-1).clamp(min=1e-9)
            ratio_per_head = (vision_sum / total_sum).tolist()
            per_layer_per_head_vtar.append(ratio_per_head)
            per_layer_vtar.append(float(sum(ratio_per_head) / len(ratio_per_head)))

        overall_vtar = float(sum(per_layer_vtar) / len(per_layer_vtar)) if per_layer_vtar else 0.0

        return {
            "per_layer_vtar": per_layer_vtar,
            "per_layer_per_head_vtar": per_layer_per_head_vtar,
            "overall_vtar": overall_vtar,
            "n_image_tokens": n_image_tokens,
            "n_text_tokens": n_text_tokens,
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_layerwise_vtar(per_sample_layer_vtar: List[List[float]], out_path: str) -> None:
    """Line plot: mean ± std VTAR per layer, averaged over all samples."""
    n_layers = len(per_sample_layer_vtar[0])
    xs = list(range(n_layers))
    means = [float(np.mean([s[l] for s in per_sample_layer_vtar])) * 100 for l in xs]
    stds = [float(np.std([s[l] for s in per_sample_layer_vtar])) * 100 for l in xs]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(xs, means, marker="o", markersize=3, linewidth=1.5, label="Mean VTAR")
    ax.fill_between(xs, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.25, label="±1 std")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("VTAR (%)")
    ax.set_title("Layer-wise Vision Token Attention Ratio — baseline, first generated token")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vtar_by_category(
    category_records: Dict[str, List[Dict[str, Any]]],
    out_path: str,
    max_categories: int = 20,
) -> None:
    """Bar chart: mean VTAR per category, sorted descending, with error bars and sample count."""
    cat_means = {cat: float(np.mean([r["overall_vtar"] for r in recs])) for cat, recs in category_records.items()}
    cat_stds = {cat: float(np.std([r["overall_vtar"] for r in recs])) for cat, recs in category_records.items()}
    cat_counts = {cat: len(recs) for cat, recs in category_records.items()}

    sorted_cats = sorted(cat_means, key=lambda c: cat_means[c], reverse=True)[:max_categories]
    vals = [cat_means[c] * 100 for c in sorted_cats]
    errs = [cat_stds[c] * 100 for c in sorted_cats]
    counts = [cat_counts[c] for c in sorted_cats]

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_cats) * 0.7), 5))
    bars = ax.bar(sorted_cats, vals, yerr=errs, capsize=3)
    ax.set_xticks(range(len(sorted_cats)))
    ax.set_xticklabels(sorted_cats, rotation=35, ha="right")
    ax.set_ylabel("Mean VTAR (%)")
    ax.set_title("Vision Token Attention Ratio by Category — baseline")
    for bar, val, cnt in zip(bars, vals, counts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.5, f"{val:.1f}\n(n={cnt})", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_head_layer_heatmap(per_sample_layer_head_vtar: List[List[List[float]]], out_path: str) -> None:
    """Heatmap: mean VTAR[layer, head] averaged over all samples."""
    arr = np.mean(np.array(per_sample_layer_head_vtar), axis=0) * 100  # (n_layers, n_heads)

    fig, ax = plt.subplots(figsize=(max(8, arr.shape[1] * 0.45), max(6, arr.shape[0] * 0.3)))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="VTAR (%)")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_title("Mean Vision Token Attention Ratio per Head × Layer — baseline")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vtar_vs_accuracy(records: List[Dict[str, Any]], out_path: str) -> None:
    """Scatter: per-sample overall VTAR colored by correctness, with mean lines."""
    correct_vtars = [r["overall_vtar"] * 100 for r in records if r["is_correct"]]
    wrong_vtars = [r["overall_vtar"] * 100 for r in records if not r["is_correct"]]

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        correct_vtars, rng.uniform(0.75, 1.25, len(correct_vtars)),
        alpha=0.35, s=8, color="steelblue", label=f"Correct (n={len(correct_vtars)})",
    )
    ax.scatter(
        wrong_vtars, rng.uniform(-0.25, 0.25, len(wrong_vtars)),
        alpha=0.35, s=8, color="tomato", label=f"Wrong (n={len(wrong_vtars)})",
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Wrong", "Correct"])
    ax.set_xlabel("Overall VTAR (%)")
    ax.set_title("Vision Attention vs Answer Correctness — baseline")
    if correct_vtars:
        ax.axvline(float(np.mean(correct_vtars)), color="steelblue", linestyle="--", linewidth=1, alpha=0.8, label=f"Correct mean={np.mean(correct_vtars):.1f}%")
    if wrong_vtars:
        ax.axvline(float(np.mean(wrong_vtars)), color="tomato", linestyle="--", linewidth=1, alpha=0.8, label=f"Wrong mean={np.mean(wrong_vtars):.1f}%")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_layerwise_vtar_by_category(
    category_records: Dict[str, List[Dict[str, Any]]],
    out_path: str,
    top_n: int = 6,
) -> None:
    """Layer-wise VTAR line plot for the top-N categories by sample count."""
    sorted_cats = sorted(
        (cat for cat, recs in category_records.items() if any(r["per_layer_vtar"] for r in recs)),
        key=lambda c: len(category_records[c]),
        reverse=True,
    )[:top_n]
    if not sorted_cats:
        return

    n_layers = len(next(r for r in category_records[sorted_cats[0]] if r["per_layer_vtar"])["per_layer_vtar"])
    xs = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("tab10")
    for i, cat in enumerate(sorted_cats):
        layer_recs = [r for r in category_records[cat] if r["per_layer_vtar"]]
        means = [float(np.mean([r["per_layer_vtar"][l] for r in layer_recs])) * 100 for l in xs]
        ax.plot(xs, means, marker="o", markersize=2.5, linewidth=1.2, color=cmap(i), label=f"{cat} (n={len(layer_recs)})")

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean VTAR (%)")
    ax.set_title(f"Layer-wise VTAR by Category — top {top_n} by sample count")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention analysis for MMBench with Qwen2.5-VL")
    parser.add_argument("--dataset", type=str, default="lmms-lab/MMBench")
    parser.add_argument("--dataset_config", type=str, default="en")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for quick test runs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device_map", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager"],
        help="flash_attention_2 is NOT supported here (incompatible with output_attentions=True)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument(
        "--image_token",
        type=str,
        default="<|image_pad|>",
        help="Special token string for image patches in Qwen2.5-VL input_ids",
    )

    parser.add_argument("--output_dir", type=str, default="results/mmbench_attention")
    parser.add_argument("--max_categories_plot", type=int, default=20)
    parser.add_argument("--top_n_categories_layerwise", type=int, default=6)
    parser.add_argument("--dry_run", action="store_true", help="Skip model inference, write dummy records (for pipeline testing)")
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run token-layout and attention-capture sanity checks on the first valid sample, then exit.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    log_path = os.path.join(args.output_dir, "run_log.txt")
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"Logging all output to: {log_path}")
    print(f"Args: {vars(args)}\n")

    try:
        _main(args)
    finally:
        sys.stdout = tee._terminal
        tee.close()


def _main(args: argparse.Namespace) -> None:
    # Load full split then shuffle + limit so all categories are represented.
    # (Sequential slicing would over-sample whichever categories appear first.)
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.limit is not None:
        indices = indices[: args.limit]
    dataset = dataset.select(indices)
    dataset_size = len(dataset)

    runner: QwenAttentionRunner | None = None
    if not args.dry_run:
        runner = QwenAttentionRunner(
            model_name=args.model_name,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation,
        )

    # -----------------------------------------------------------------------
    # Sanity-check mode: inspect token layout + attention capture, then exit.
    # -----------------------------------------------------------------------
    if args.sanity_check:
        if args.dry_run:
            print("--sanity_check is incompatible with --dry_run (model is required). Exiting.")
            return
        assert runner is not None
        # Find first valid sample
        sc_sample = None
        for _sc_idx, _sc_s in enumerate(dataset):
            _sc_gt = get_ground_truth(_sc_s)
            _sc_opts = get_valid_options(_sc_s)
            if _sc_gt in _sc_opts:
                sc_sample = (_sc_idx, _sc_s)
                break
        if sc_sample is None:
            print("No valid sample found for sanity check. Exiting.")
            return
        sc_idx, sc_s = sc_sample
        sc_image = to_rgb(sc_s["image"])
        sc_prompt = build_prompt(
            question=get_question(sc_s),
            options=get_valid_options(sc_s),
            hint=get_hint(sc_s),
        )
        print(f"\nRunning sanity checks on sample idx={sc_idx} …")
        print(f"Question : {get_question(sc_s)[:120]}")
        print(f"GT       : {get_ground_truth(sc_s)}")
        runner.sanity_check_tokens(sc_image, sc_prompt, image_token=args.image_token)
        runner.sanity_check_attention(sc_image, sc_prompt, image_token=args.image_token)
        return

    records: List[Dict[str, Any]] = []
    category_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    pbar = tqdm(enumerate(dataset), desc="Attention analysis", total=dataset_size)
    for idx, sample in pbar:
        image = to_rgb(sample["image"])
        sample_id = safe_sample_id(sample, idx)
        question = get_question(sample)
        hint = get_hint(sample)
        gt = get_ground_truth(sample)
        options = get_valid_options(sample)
        valid_letters = list(options.keys())
        category, l2_category = extract_categories(sample)
        pbar.set_postfix({"cat": category[:20]}, refresh=False)

        if gt not in valid_letters:
            continue

        prompt = build_prompt(question=question, options=options, hint=hint)

        if args.dry_run:
            record: Dict[str, Any] = {
                "sample_id": sample_id,
                "category": category,
                "l2_category": l2_category,
                "question": question,
                "options": {k: str(v) for k, v in options.items()},
                "ground_truth": gt,
                "prediction_raw": "DRY_RUN",
                "prediction_letter": "?",
                "is_correct": False,
                "n_image_tokens": 0,
                "n_text_tokens": 0,
                "image_token_ratio": 0.0,
                "overall_vtar": 0.0,
                "per_layer_vtar": [],
                "per_layer_per_head_vtar": [],
            }
        else:
            assert runner is not None
            raw_pred, pred_letter = runner.predict_letter(image, prompt, valid_letters)
            attn = runner.get_attention_stats(image, prompt, image_token=args.image_token)
            n_total = attn["n_image_tokens"] + attn["n_text_tokens"]
            if attn["n_image_tokens"] == 0:
                print(f"Warning: no image tokens found for sample {sample_id}, skipping.")
                continue
            record = {
                "sample_id": sample_id,
                "category": category,
                "l2_category": l2_category,
                "question": question,
                "options": {k: str(v) for k, v in options.items()},
                "ground_truth": gt,
                "prediction_raw": raw_pred,
                "prediction_letter": pred_letter,
                "is_correct": pred_letter == gt,
                "n_image_tokens": attn["n_image_tokens"],
                "n_text_tokens": attn["n_text_tokens"],
                "image_token_ratio": attn["n_image_tokens"] / n_total if n_total > 0 else 0.0,
                "overall_vtar": attn["overall_vtar"],
                "per_layer_vtar": attn["per_layer_vtar"],
                "per_layer_per_head_vtar": attn["per_layer_per_head_vtar"],
            }

        records.append(record)
        category_records[category].append(record)

    # -----------------------------------------------------------------------
    # Save raw per-sample records (always first — crash-safe)
    # -----------------------------------------------------------------------
    jsonl_path = os.path.join(args.output_dir, "attention_records.jsonl")
    save_jsonl(jsonl_path, records)
    print(f"Saved records: {jsonl_path}")

    if not records:
        print("No records collected. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Aggregate all statistics into summary.json
    # -----------------------------------------------------------------------
    all_vtars = [r["overall_vtar"] for r in records]
    all_correct = [r["is_correct"] for r in records]
    all_n_image = [r["n_image_tokens"] for r in records]
    all_n_text = [r["n_text_tokens"] for r in records]
    all_img_ratio = [r["image_token_ratio"] for r in records]

    # Layer-wise aggregation (only records that have attention data)
    records_with_layers = [r for r in records if r["per_layer_vtar"]]
    if records_with_layers:
        n_layers = len(records_with_layers[0]["per_layer_vtar"])
        layer_vtar_mean = [float(np.mean([r["per_layer_vtar"][l] for r in records_with_layers])) for l in range(n_layers)]
        layer_vtar_std = [float(np.std([r["per_layer_vtar"][l] for r in records_with_layers])) for l in range(n_layers)]
    else:
        n_layers = 0
        layer_vtar_mean = []
        layer_vtar_std = []

    # Head × layer mean (n_layers × n_heads)
    records_with_heads = [r for r in records if r["per_layer_per_head_vtar"]]
    head_layer_mean: List[List[float]] = (
        np.mean(np.array([r["per_layer_per_head_vtar"] for r in records_with_heads]), axis=0).tolist()
        if records_with_heads
        else []
    )

    # Per-category aggregation
    per_category: Dict[str, Any] = {}
    for cat, cat_recs in category_records.items():
        vtars = [r["overall_vtar"] for r in cat_recs]
        correct_flags = [r["is_correct"] for r in cat_recs]
        cat_layer_recs = [r for r in cat_recs if r["per_layer_vtar"]]
        cat_n_layers = len(cat_layer_recs[0]["per_layer_vtar"]) if cat_layer_recs else 0
        per_category[cat] = {
            "n_samples": len(cat_recs),
            "accuracy": float(np.mean(correct_flags)),
            "vtar_mean": float(np.mean(vtars)),
            "vtar_std": float(np.std(vtars)),
            "vtar_median": float(np.median(vtars)),
            "layer_vtar_mean": [float(np.mean([r["per_layer_vtar"][l] for r in cat_layer_recs])) for l in range(cat_n_layers)],
            "layer_vtar_std": [float(np.std([r["per_layer_vtar"][l] for r in cat_layer_recs])) for l in range(cat_n_layers)],
        }

    summary: Dict[str, Any] = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "attn_implementation": args.attn_implementation,
        "image_token": args.image_token,
        "dry_run": args.dry_run,
        "n_samples": len(records),
        "n_correct": int(sum(all_correct)),
        "overall_accuracy": float(np.mean(all_correct)),
        "image_token_stats": {
            "mean_n_image_tokens": float(np.mean(all_n_image)),
            "mean_n_text_tokens": float(np.mean(all_n_text)),
            "mean_image_token_ratio": float(np.mean(all_img_ratio)),
            "std_image_token_ratio": float(np.std(all_img_ratio)),
        },
        "overall_vtar": {
            "mean": float(np.mean(all_vtars)),
            "std": float(np.std(all_vtars)),
            "median": float(np.median(all_vtars)),
            "min": float(np.min(all_vtars)),
            "max": float(np.max(all_vtars)),
        },
        "layer_wise_vtar": {
            "n_layers": n_layers,
            "mean": layer_vtar_mean,   # list of n_layers floats
            "std": layer_vtar_std,     # list of n_layers floats
        },
        "head_layer_mean_vtar": head_layer_mean,  # n_layers × n_heads nested list
        "per_category": per_category,
        "files": {
            "records_jsonl": jsonl_path,
            "plot1_layerwise": os.path.join(args.output_dir, "plots", "plot1_layerwise_vtar.png"),
            "plot2_by_category": os.path.join(args.output_dir, "plots", "plot2_vtar_by_category.png"),
            "plot3_heatmap": os.path.join(args.output_dir, "plots", "plot3_head_layer_heatmap.png"),
            "plot4_vtar_vs_acc": os.path.join(args.output_dir, "plots", "plot4_vtar_vs_accuracy.png"),
            "plot5_layerwise_by_category": os.path.join(args.output_dir, "plots", "plot5_layerwise_by_category.png"),
        },
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    save_json(summary_path, summary)
    print(f"Saved summary: {summary_path}")

    # -----------------------------------------------------------------------
    # Plots (each backed by the numbers already stored in summary.json)
    # -----------------------------------------------------------------------
    if records_with_layers:
        plot_layerwise_vtar(
            [r["per_layer_vtar"] for r in records_with_layers],
            os.path.join(args.output_dir, "plots", "plot1_layerwise_vtar.png"),
        )
        print("Saved plot1: layerwise VTAR")

    plot_vtar_by_category(
        category_records,
        os.path.join(args.output_dir, "plots", "plot2_vtar_by_category.png"),
        max_categories=args.max_categories_plot,
    )
    print("Saved plot2: VTAR by category")

    if records_with_heads:
        plot_head_layer_heatmap(
            [r["per_layer_per_head_vtar"] for r in records_with_heads],
            os.path.join(args.output_dir, "plots", "plot3_head_layer_heatmap.png"),
        )
        print("Saved plot3: head × layer heatmap")

    plot_vtar_vs_accuracy(records, os.path.join(args.output_dir, "plots", "plot4_vtar_vs_accuracy.png"))
    print("Saved plot4: VTAR vs accuracy")

    if records_with_layers:
        plot_layerwise_vtar_by_category(
            {cat: [r for r in recs if r["per_layer_vtar"]] for cat, recs in category_records.items()},
            os.path.join(args.output_dir, "plots", "plot5_layerwise_by_category.png"),
            top_n=args.top_n_categories_layerwise,
        )
        print("Saved plot5: layerwise VTAR by category")

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    mean_n_img = summary["image_token_stats"]["mean_n_image_tokens"]
    mean_n_txt = summary["image_token_stats"]["mean_n_text_tokens"]
    mean_n_tot = mean_n_img + mean_n_txt
    mean_vision_attn = summary["overall_vtar"]["mean"]
    mean_text_attn = 1.0 - mean_vision_attn

    print("\n=== Attention Analysis Summary ===")
    print(f"  Samples  : {len(records)}")
    print(f"  Accuracy : {summary['overall_accuracy']:.3f}")
    print()
    print(f"  {'':10}  {'img_tok':>7}  {'txt_tok':>7}  {'vision_attn%':>13}  {'text_attn%':>11}")
    print(f"  {'-'*56}")
    print(f"  {'vision':<10}  {mean_n_img:>7.1f}  {mean_n_txt:>7.1f}  {mean_vision_attn*100:>12.2f}%  {mean_text_attn*100:>10.2f}%")
    print(f"  Mean VTAR: {mean_vision_attn * 100:.1f}% ± {summary['overall_vtar']['std'] * 100:.1f}%")
    if layer_vtar_mean:
        peak_layer = int(np.argmax(layer_vtar_mean))
        print(f"  Peak VTAR layer: {peak_layer} ({layer_vtar_mean[peak_layer] * 100:.1f}%)")


if __name__ == "__main__":
    main()
