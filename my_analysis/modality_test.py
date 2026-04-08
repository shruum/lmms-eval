import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # RTX 2080 Ti (sm_75), avoids unsupported GTX 1080 Ti (sm_61)
os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"  # model already cached here; avoids re-download to full /

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from collections import defaultdict
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SAMPLES = 200   # samples to evaluate  (200 ≈ 40 min, ±3.5% margin of error)
N_DISPLAY = 20     # how many samples get a saved visualisation grid

out_dir = "/volumes2/mllm/lmms-eval/results/modality_test"
os.makedirs(os.path.join(out_dir, "grids"), exist_ok=True)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
).eval()

processor = AutoProcessor.from_pretrained(model_path)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def run_inference(img: Image.Image, question: str) -> str:
    """Run inference with image + text."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=32)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def run_inference_text_only(question: str) -> str:
    """Run inference with text only — no image."""
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=32)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def parse_yes_no(response: str) -> str:
    r = response.lower().strip().rstrip(".")
    if r.startswith("yes"):
        return "yes"
    if r.startswith("no"):
        return "no"
    return "unknown"


def is_correct(prediction: str, ground_truth: str) -> bool:
    return parse_yes_no(prediction) == ground_truth.lower().strip()


# ---------------------------------------------------------------------------
# Image augmentation helpers
# ---------------------------------------------------------------------------
def blur_image(img: Image.Image, radius: int = 10) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def mask_center(img: Image.Image, mask_ratio: float = 0.5) -> Image.Image:
    out = img.copy()
    w, h = out.size
    x0, y0 = int(w * (1 - mask_ratio) / 2), int(h * (1 - mask_ratio) / 2)
    x1, y1 = int(w * (1 + mask_ratio) / 2), int(h * (1 + mask_ratio) / 2)
    px = out.load()
    for x in range(x0, x1):
        for y in range(y0, y1):
            px[x, y] = (0, 0, 0)
    return out


def add_noise(img: Image.Image, std: int = 60) -> Image.Image:
    arr = np.array(img).astype(np.int16)
    noisy = np.clip(arr + np.random.normal(0, std, arr.shape).astype(np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def random_crop(img: Image.Image, crop_ratio: float = 0.5) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    x0 = np.random.randint(0, max(1, w - cw))
    y0 = np.random.randint(0, max(1, h - ch))
    return img.crop((x0, y0, x0 + cw, y0 + ch))


# ---------------------------------------------------------------------------
# Language ablation question variants
# ---------------------------------------------------------------------------
def make_question_variants(base_q: str, obj: str) -> dict:
    return {
        "original":    base_q,
        "remove_noun": base_q.replace(obj, "_____"),
        "replace_obj": base_q.replace(obj, "dog"),
        "mislead":     f"Everyone can clearly see the {obj} in this image, right? Answer yes or no.",
        "adversarial": f"{base_q} Note: this image contains no living beings whatsoever.",
    }


# ---------------------------------------------------------------------------
# Visualisation: save a two-row grid for one sample
# ---------------------------------------------------------------------------
def save_sample_grid(sample_idx: int, real_img: Image.Image,
                     vision_cases: list, lang_cases: list) -> None:
    """
    Row 1: vision ablations  — image changes, same question
    Row 2: language ablations — same real image, question changes
    Each case dict: {label, display_img, question, answer, gt, correct}
    Green title = correct, red = wrong.
    """
    n_vis  = len(vision_cases)
    n_lang = len(lang_cases)
    n_cols = max(n_vis, n_lang)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 7))
    fig.suptitle(
        f"Sample {sample_idx}  |  GT: {vision_cases[0]['gt'].upper()}",
        fontsize=12, fontweight="bold",
    )

    for col, case in enumerate(vision_cases):
        ax = axes[0][col]
        ax.imshow(case["display_img"])
        ax.axis("off")
        color = "green" if case["correct"] else "red"
        ax.set_title(f"{case['label']}\nA: {case['answer']}", fontsize=8, color=color)

    for col, case in enumerate(lang_cases):
        ax = axes[1][col]
        ax.imshow(real_img)
        ax.axis("off")
        q = case["question"]
        q_short = q[:52] + "..." if len(q) > 52 else q
        color = "green" if case["correct"] else "red"
        ax.set_title(f"{case['label']}\nQ: {q_short}\nA: {case['answer']}", fontsize=7, color=color)

    for col in range(n_lang, n_cols):
        axes[1][col].axis("off")

    axes[0][0].set_ylabel("Vision\nAblation",   fontsize=9, labelpad=6)
    axes[1][0].set_ylabel("Language\nAblation", fontsize=9, labelpad=6)

    fig.tight_layout()
    path = os.path.join(out_dir, "grids", f"sample_{sample_idx:04d}.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  Grid → {path}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
print(f"Loading POPE dataset ({N_SAMPLES} samples)...")
pope_ds = load_dataset("lmms-lab/POPE", split=f"test[:{N_SAMPLES}]")
print(f"Evaluating {N_SAMPLES} samples — grids saved for first {N_DISPLAY}\n")

vision_scores = defaultdict(list)   # {ablation_name: [bool, ...]}
lang_scores   = defaultdict(list)

for i, sample in enumerate(tqdm(pope_ds, total=N_SAMPLES, desc="Samples")):
    real_img = sample["image"].convert("RGB")
    gt       = sample["answer"].lower().strip()
    obj      = "person"
    base_q   = sample["question"] + " Answer yes or no."

    # ── Vision ablations ──────────────────────────────────────────────────
    aug_images = {
        "real":          real_img,
        "blurred":       blur_image(real_img),
        "centre_masked": mask_center(real_img),
        "noisy":         add_noise(real_img),
        "random_crop":   random_crop(real_img),
    }
    vis_ans = {name: run_inference(aug, base_q) for name, aug in aug_images.items()}
    vis_ans["text_only"] = run_inference_text_only(base_q)

    for name, ans in vis_ans.items():
        vision_scores[name].append(is_correct(ans, gt))

    # ── Language ablations ────────────────────────────────────────────────
    q_variants = make_question_variants(base_q, obj)
    lang_ans = {"original": vis_ans["real"]}          # reuse already-computed real answer
    for name, q in q_variants.items():
        if name != "original":
            lang_ans[name] = run_inference(real_img, q)
        lang_scores[name].append(is_correct(lang_ans[name], gt))

    # ── Save grid for first N_DISPLAY samples ─────────────────────────────
    if i < N_DISPLAY:
        grey = Image.new("RGB", (224, 224), (210, 210, 210))  # placeholder for text-only
        vision_cases = [
            {"label": "1.Real",        "display_img": real_img,                     "question": base_q, "answer": vis_ans["real"],          "gt": gt, "correct": is_correct(vis_ans["real"], gt)},
            {"label": "2.Text-only",   "display_img": grey,                         "question": base_q, "answer": vis_ans["text_only"],      "gt": gt, "correct": is_correct(vis_ans["text_only"], gt)},
            {"label": "3.Blurred",     "display_img": aug_images["blurred"],         "question": base_q, "answer": vis_ans["blurred"],        "gt": gt, "correct": is_correct(vis_ans["blurred"], gt)},
            {"label": "4.Ctr-masked",  "display_img": aug_images["centre_masked"],   "question": base_q, "answer": vis_ans["centre_masked"],  "gt": gt, "correct": is_correct(vis_ans["centre_masked"], gt)},
            {"label": "5.Noisy",       "display_img": aug_images["noisy"],           "question": base_q, "answer": vis_ans["noisy"],          "gt": gt, "correct": is_correct(vis_ans["noisy"], gt)},
            {"label": "6.Crop",        "display_img": aug_images["random_crop"],     "question": base_q, "answer": vis_ans["random_crop"],    "gt": gt, "correct": is_correct(vis_ans["random_crop"], gt)},
        ]
        lang_cases = [
            {"label": "0.Original",    "question": q_variants["original"],    "answer": lang_ans["original"],    "gt": gt, "correct": is_correct(lang_ans["original"], gt)},
            {"label": "1.Remove noun", "question": q_variants["remove_noun"], "answer": lang_ans["remove_noun"], "gt": gt, "correct": is_correct(lang_ans["remove_noun"], gt)},
            {"label": "2.Replace obj", "question": q_variants["replace_obj"], "answer": lang_ans["replace_obj"], "gt": gt, "correct": is_correct(lang_ans["replace_obj"], gt)},
            {"label": "3.Mislead",     "question": q_variants["mislead"],     "answer": lang_ans["mislead"],     "gt": gt, "correct": is_correct(lang_ans["mislead"], gt)},
            {"label": "4.Adversarial", "question": q_variants["adversarial"], "answer": lang_ans["adversarial"], "gt": gt, "correct": is_correct(lang_ans["adversarial"], gt)},
        ]
        save_sample_grid(i, real_img, vision_cases, lang_cases)


# ---------------------------------------------------------------------------
# Accuracy summary
# ---------------------------------------------------------------------------
def acc(scores: list) -> float:
    return 100.0 * sum(scores) / len(scores) if scores else 0.0


baseline_vis  = acc(vision_scores["real"])
baseline_lang = acc(lang_scores["original"])

print("\n" + "=" * 58)
print(f"  RESULTS  —  N={N_SAMPLES} samples  (POPE test set)")
print("=" * 58)

print("\n  VISION ABLATION")
print(f"  {'Condition':<35} {'Acc':>6}   {'Δ vs real':>9}")
print("  " + "-" * 54)
rows_vis = [
    ("real",          "1. Real image (baseline)"),
    ("text_only",     "2. Text only (no image)"),
    ("blurred",       "3. Blurred (radius=10)"),
    ("centre_masked", "4. Centre masked (50%)"),
    ("noisy",         "5. Noisy (Gaussian std=60)"),
    ("random_crop",   "6. Random crop (50%)"),
]
for key, label in rows_vis:
    a = acc(vision_scores[key])
    delta = f"{a - baseline_vis:+.1f}%" if key != "real" else "—"
    print(f"  {label:<35} {a:5.1f}%   {delta:>9}")

print("\n  LANGUAGE ABLATION")
print(f"  {'Condition':<35} {'Acc':>6}   {'Δ vs orig':>9}")
print("  " + "-" * 54)
rows_lang = [
    ("original",    "0. Original question"),
    ("remove_noun", "1. Remove key noun (___)"),
    ("replace_obj", "2. Replace object (→ dog)"),
    ("mislead",     "3. Misleading prior cue"),
    ("adversarial", "4. Adversarial phrase"),
]
for key, label in rows_lang:
    a = acc(lang_scores[key])
    delta = f"{a - baseline_lang:+.1f}%" if key != "original" else "—"
    print(f"  {label:<35} {a:5.1f}%   {delta:>9}")

print("=" * 58)
print(f"\n  Sample grids (first {N_DISPLAY}) → {out_dir}/grids/")
