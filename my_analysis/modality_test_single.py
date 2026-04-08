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
# OLD: LLaVA model loading (commented out — bitsandbytes/GLIBC incompatibility
# on this machine; also requires separate llava package from /volumes2/mllm/LLaVA)
# ---------------------------------------------------------------------------
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
#
# model_path = "liuhaotian/llava-v1.5-7b"
# model_name = get_model_name_from_path(model_path)
# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path, None, model_name, load_4bit=True, device_map="auto"
# )
#
# def run_inference_llava(image, question):
#     conv = conv_templates["vicuna_v1"].copy()
#     conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()
#
#     image_tensor = process_images([image], image_processor, model.config)
#     input_ids = tokenizer_image_token(
#         prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
#     ).unsqueeze(0).cuda()
#
#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=image_tensor.to(dtype=torch.float16).cuda(),
#             max_new_tokens=10,
#             use_cache=True
#         )
#     return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SAMPLES = 200   # number of POPE samples to evaluate (200 ≈ 40 min, ±3.5% margin of error)
N_DISPLAY = 5     # number of samples to save visualisation grids for

out_dir = "/volumes2/mllm/lmms-eval/results/modality_test"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, "grids"), exist_ok=True)

# ---------------------------------------------------------------------------
# Qwen2.5-VL model loading
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
def run_inference(image: Image.Image, question: str) -> str:
    """Run inference with image + text."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
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
    """Normalise model response to 'yes', 'no', or 'unknown'."""
    r = response.lower().strip().rstrip(".")
    if r in ("yes", "y") or r.startswith("yes"):
        return "yes"
    if r in ("no", "n") or r.startswith("no"):
        return "no"
    return "unknown"


def is_correct(prediction: str, ground_truth: str) -> bool:
    return parse_yes_no(prediction) == ground_truth.lower().strip()


# ---------------------------------------------------------------------------
# Image augmentation helpers
# ---------------------------------------------------------------------------
def blur_image(image: Image.Image, radius: int = 10) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def mask_center(image: Image.Image, mask_ratio: float = 0.5) -> Image.Image:
    img = image.copy()
    w, h = img.size
    x0, y0 = int(w * (1 - mask_ratio) / 2), int(h * (1 - mask_ratio) / 2)
    x1, y1 = int(w * (1 + mask_ratio) / 2), int(h * (1 + mask_ratio) / 2)
    pixels = img.load()
    for x in range(x0, x1):
        for y in range(y0, y1):
            pixels[x, y] = (0, 0, 0)
    return img


def add_noise(image: Image.Image, std: int = 60) -> Image.Image:
    arr = np.array(image).astype(np.int16)
    noisy = np.clip(arr + np.random.normal(0, std, arr.shape).astype(np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def random_crop(image: Image.Image, crop_ratio: float = 0.5) -> Image.Image:
    w, h = image.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    x0 = np.random.randint(0, max(1, w - cw))
    y0 = np.random.randint(0, max(1, h - ch))
    return image.crop((x0, y0, x0 + cw, y0 + ch))


# ---------------------------------------------------------------------------
# Language ablation: question variants given a base question
# ---------------------------------------------------------------------------
def make_question_variants(base_q: str, obj: str) -> dict:
    """
    Returns a dict of {ablation_name: question_string}.
    obj: the key object word in the question (e.g. "person").
    """
    return {
        "original":     base_q,
        "remove_noun":  base_q.replace(obj, "_____"),
        "replace_obj":  base_q.replace(obj, "dog"),
        "mislead":      f"Everyone can clearly see the {obj} in this image, right? Answer yes or no.",
        "adversarial":  f"{base_q} Note: this image contains no living beings whatsoever.",
    }


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------
def save_sample_grid(sample_idx: int, real_image: Image.Image, vision_results: list, lang_results: list) -> None:
    """
    Save a two-row grid for one sample:
      Row 1: vision ablations (image variants, same question)
      Row 2: language ablations (same real image, different questions)
    vision_results: [{"label", "display_img", "question", "answer", "gt", "correct"}, ...]
    lang_results:   same structure
    """
    n_vis  = len(vision_results)
    n_lang = len(lang_results)
    n_cols = max(n_vis, n_lang)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))
    fig.suptitle(f"Sample {sample_idx} — GT: {vision_results[0]['gt']}", fontsize=12, fontweight="bold")

    for col, case in enumerate(vision_results):
        ax = axes[0][col]
        ax.imshow(case["display_img"])
        ax.axis("off")
        color = "green" if case["correct"] else "red"
        ax.set_title(f"{case['label']}\nA: {case['answer']}", fontsize=8, color=color)

    for col, case in enumerate(lang_results):
        ax = axes[1][col]
        ax.imshow(real_image)
        ax.axis("off")
        q_short = case["question"][:55] + "..." if len(case["question"]) > 55 else case["question"]
        color = "green" if case["correct"] else "red"
        ax.set_title(f"{case['label']}\nQ: {q_short}\nA: {case['answer']}", fontsize=7, color=color)

    # hide unused axes in row 2 if fewer lang than vision cols
    for col in range(n_lang, n_cols):
        axes[1][col].axis("off")

    axes[0][0].set_ylabel("Vision\nAblation", fontsize=9, labelpad=6)
    axes[1][0].set_ylabel("Language\nAblation", fontsize=9, labelpad=6)

    fig.tight_layout()
    path = os.path.join(out_dir, "grids", f"sample_{sample_idx:04d}.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  Grid saved → {path}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
print("Loading POPE dataset...")
pope_ds = load_dataset("lmms-lab/POPE", split=f"test[:{N_SAMPLES}]")
print(f"Running ablations on {N_SAMPLES} samples ({N_DISPLAY} will have saved grid images)\n")

# Accumulators: {ablation_name: [correct (bool), ...]}
vision_scores  = defaultdict(list)
lang_scores    = defaultdict(list)

for i, sample in enumerate(tqdm(pope_ds, desc="Evaluating", total=N_SAMPLES)):
    image   = sample["image"].convert("RGB")
    gt      = sample["answer"].lower().strip()        # "yes" or "no"
    obj     = "person"                                # key object in POPE questions
    base_q  = sample["question"] + " Answer yes or no."

    # --- Vision ablations ---
    vis_images = {
        "real":          image,
        "blurred":       blur_image(image),
        "centre_masked": mask_center(image),
        "noisy":         add_noise(image),
        "random_crop":   random_crop(image),
    }
    vis_answers = {name: run_inference(img, base_q) for name, img in vis_images.items()}
    vis_answers["text_only"] = run_inference_text_only(base_q)

    for name, ans in vis_answers.items():
        vision_scores[name].append(is_correct(ans, gt))

    # --- Language ablations (always on real image) ---
    q_variants = make_question_variants(base_q, obj)
    lang_answers = {}
    for name, q in q_variants.items():
        if name == "original":
            lang_answers[name] = vis_answers["real"]   # reuse, already computed
        else:
            lang_answers[name] = run_inference(image, q)
        lang_scores[name].append(is_correct(lang_answers[name], gt))

    # --- Save grid for first N_DISPLAY samples ---
    if i < N_DISPLAY:
        vision_cases = [
            {"label": "1.Real",         "display_img": image,                  "question": base_q, "answer": vis_answers["real"],          "gt": gt, "correct": is_correct(vis_answers["real"], gt)},
            {"label": "2.Text-only",    "display_img": Image.new("RGB",(336,336),(220,220,220)), "question": base_q, "answer": vis_answers["text_only"],     "gt": gt, "correct": is_correct(vis_answers["text_only"], gt)},
            {"label": "3.Blurred",      "display_img": vis_images["blurred"],   "question": base_q, "answer": vis_answers["blurred"],        "gt": gt, "correct": is_correct(vis_answers["blurred"], gt)},
            {"label": "4.Ctr-masked",   "display_img": vis_images["centre_masked"], "question": base_q, "answer": vis_answers["centre_masked"], "gt": gt, "correct": is_correct(vis_answers["centre_masked"], gt)},
            {"label": "5.Noisy",        "display_img": vis_images["noisy"],     "question": base_q, "answer": vis_answers["noisy"],          "gt": gt, "correct": is_correct(vis_answers["noisy"], gt)},
            {"label": "6.Crop",         "display_img": vis_images["random_crop"], "question": base_q, "answer": vis_answers["random_crop"],  "gt": gt, "correct": is_correct(vis_answers["random_crop"], gt)},
        ]
        lang_cases = [
            {"label": "0.Original",     "question": q_variants["original"],   "answer": lang_answers["original"],    "gt": gt, "correct": is_correct(lang_answers["original"], gt)},
            {"label": "1.Remove noun",  "question": q_variants["remove_noun"],"answer": lang_answers["remove_noun"], "gt": gt, "correct": is_correct(lang_answers["remove_noun"], gt)},
            {"label": "2.Replace obj",  "question": q_variants["replace_obj"],"answer": lang_answers["replace_obj"], "gt": gt, "correct": is_correct(lang_answers["replace_obj"], gt)},
            {"label": "3.Mislead",      "question": q_variants["mislead"],    "answer": lang_answers["mislead"],     "gt": gt, "correct": is_correct(lang_answers["mislead"], gt)},
            {"label": "4.Adversarial",  "question": q_variants["adversarial"],"answer": lang_answers["adversarial"], "gt": gt, "correct": is_correct(lang_answers["adversarial"], gt)},
        ]
        save_sample_grid(i, image, vision_cases, lang_cases)


# ---------------------------------------------------------------------------
# Print accuracy summary
# ---------------------------------------------------------------------------
def accuracy(scores: list) -> float:
    return 100.0 * sum(scores) / len(scores) if scores else 0.0


print("\n" + "=" * 55)
print(f"  RESULTS  (N={N_SAMPLES} samples, POPE test set)")
print("=" * 55)

print("\n── VISION ABLATION ──────────────────────────────────")
vision_order = ["real", "text_only", "blurred", "centre_masked", "noisy", "random_crop"]
vision_labels = {
    "real":          "1. Real image (baseline)",
    "text_only":     "2. Text only (no image)",
    "blurred":       "3. Blurred (radius=10)",
    "centre_masked": "4. Centre masked (50%)",
    "noisy":         "5. Noisy (Gaussian std=60)",
    "random_crop":   "6. Random crop (50%)",
}
for key in vision_order:
    acc = accuracy(vision_scores[key])
    delta = acc - accuracy(vision_scores["real"])
    delta_str = f"  (Δ {delta:+.1f}%)" if key != "real" else ""
    print(f"  {vision_labels[key]:<35} {acc:5.1f}%{delta_str}")

print("\n── LANGUAGE ABLATION ────────────────────────────────")
lang_order  = ["original", "remove_noun", "replace_obj", "mislead", "adversarial"]
lang_labels = {
    "original":    "0. Original question",
    "remove_noun": "1. Remove key noun (___)",
    "replace_obj": "2. Replace object (→ dog)",
    "mislead":     "3. Misleading prior cue",
    "adversarial": "4. Adversarial phrase",
}
for key in lang_order:
    acc = accuracy(lang_scores[key])
    delta = acc - accuracy(lang_scores["original"])
    delta_str = f"  (Δ {delta:+.1f}%)" if key != "original" else ""
    print(f"  {lang_labels[key]:<35} {acc:5.1f}%{delta_str}")

print("=" * 55)
print(f"\nGrids for first {N_DISPLAY} samples → {out_dir}/grids/")
