"""
Central config — single source of truth for paths, model IDs, and SRF hyperparameters.
All eval scripts import from here; nothing is hardcoded in eval scripts.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
HF_HOME      = os.environ.get("HF_HOME", "/volumes2/hugging_face_cache")
MMVP_CSV     = os.path.join(HF_HOME, "mmvp_questions/Questions.csv")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Model ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MAX_PIXELS = 512 * 28 * 28
IMAGE_TOKEN        = "<|image_pad|>"   # Qwen default; LLaVA uses model.config.image_token_index

# ── Dataset: full sizes ────────────────────────────────────────────────────────
# POPE — run all three splits by default (adversarial + popular + random = 9000 total)
POPE_SPLITS        = ["adversarial", "popular", "random"]
POPE_N_FULL        = -1    # -1 = all available samples across selected splits
POPE_SEED          = 42

VLM_BIAS_CATEGORIES = [
    "Animals", "Chess Pieces", "Flags", "Game Boards",
    "Logos", "Optical Illusion", "Patterned Grid",
]
VLM_BIAS_N_PER_CAT  = None   # None = use all available per category
VLM_BIAS_SEED       = 42

MMVP_GT_CORRECTIONS = {99: "A", 279: "A"}   # known GT errors in original CSV

# ── SRF shared defaults (not arch-specific) ────────────────────────────────────
# These do NOT vary with model size. Arch-specific tunables live in SRF_ARCH_PARAMS.
SRF_DEFAULTS = {
    "sys_beta":         0.10,   # system-prompt attention suppression
    "text_beta":        0.0,    # text-token suppression (disabled)
    "text_layer_start": 20,     # text suppression zone (Qwen 3B proportions)
    "text_layer_end":   27,
    "bias_mode":        "additive_logit",
    "interp_lambda":    1.0,
    "prob_floor":       0.005,
    "img_scale":        1.5,
    "calib_n":          20,     # calibration samples for head identification
    "calib_seed":       0,
}

# ── SRF per-dataset params ─────────────────────────────────────────────────────
# Tunable per dataset, arch-agnostic.
# layer_start / layer_end live in SRF_ARCH_PARAMS (they scale with model depth).
SRF_DATASET_PARAMS = {
    "mmvp":    {"phase": "both",       "alpha": 4.0, "eps": 0.2},
    "pope":    {"phase": "both",       "alpha": 4.0, "eps": 0.2},
    "vlmbias": {"phase": "generation", "alpha": 8.0, "eps": 0.5},
    "mme":     {"phase": "both",       "alpha": 4.0, "eps": 0.2},
}

# ── SRF-E (evidence amplification) defaults ────────────────────────────────────
SRFE_DEFAULT_BETA = 2.0    # best on MMVP; sweep to confirm on POPE
SRFE_BETA_SWEEP   = [0.5, 1.0, 2.0]

# ── Architecture-specific hyperparameters ──────────────────────────────────────
# These all VARY with model size/architecture and must be tuned per model.
#
# layer_start / layer_end: vision-language fusion zone.
#   Scales with depth: use ~(8/28)*n_layers and ~(15/28)*n_layers as starting points.
#   dataset_layer_end: per-dataset fine-tuning (overrides the shared layer_end).
#
# head_top_k_pct: fraction of heads selected as vision-aware.
#   Typically 0.20 works across models; re-tune if accuracy drops.
#
# clip_coarse_grid: CLIP patch grid size. Larger = finer spatial resolution.
#   Adjust if image resolution changes significantly (e.g. LLaVA uses 336px vs Qwen's 448px).
#
# clip_top_k_pct: fraction of image tokens boosted by CLIP saliency.
#   Lower = more focused; higher = more context.
#
# clip_fallback_thresh: below this CLIP max_sim, object is likely absent → uniform boost.
#   0.20 is a reasonable default; lower for datasets with unusual visual content.
#
# image_token: token string used to find image token range in input_ids.
#   None = use model.config.image_token_index (LLaVA-style).
#
# All values marked "NOT tuned" are proportional starting points — sweep before paper.

SRF_ARCH_PARAMS = {
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "n_layers":             28,
        "spatial_merge_size":   2,
        "image_token":          "<|image_pad|>",
        # ── tuned by autoresearch ──
        "layer_start":          8,
        "layer_end":            15,
        "head_top_k_pct":       0.20,
        "clip_coarse_grid":     7,
        "clip_top_k_pct":       0.30,
        "clip_fallback_thresh": 0.20,
        # per-dataset layer_end fine-tuning (overrides layer_end above)
        "dataset_layer_end":    {"mmvp": 15, "pope": 15, "vlmbias": 14, "mme": 15},
    },
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "n_layers":             32,
        "spatial_merge_size":   2,
        "image_token":          "<|image_pad|>",
        # ── NOT tuned — proportionally scaled from 3B ──
        "layer_start":          9,    # round(8/28 * 32)
        "layer_end":            17,   # round(15/28 * 32)
        "head_top_k_pct":       0.20,
        "clip_coarse_grid":     7,
        "clip_top_k_pct":       0.30,
        "clip_fallback_thresh": 0.20,
        "dataset_layer_end":    {"mmvp": 17, "pope": 17, "vlmbias": 16, "mme": 17},
    },
    "llava-hf/llava-1.5-7b-hf": {
        "n_layers":             32,
        "spatial_merge_size":   1,
        "image_token":          None,   # use model.config.image_token_index
        # ── NOT tuned — ClearSight paper starting point ──
        "layer_start":          8,
        "layer_end":            20,
        "head_top_k_pct":       0.20,
        "clip_coarse_grid":     6,      # LLaVA uses 336px images → slightly smaller grid
        "clip_top_k_pct":       0.30,
        "clip_fallback_thresh": 0.20,
        "dataset_layer_end":    {"mmvp": 20, "pope": 20, "vlmbias": 19, "mme": 20},
    },
}

# Fallback arch params for unknown model IDs (conservative starting points)
SRF_ARCH_FALLBACK = {
    "n_layers":             32,
    "spatial_merge_size":   2,
    "image_token":          "<|image_pad|>",
    "layer_start":          9,
    "layer_end":            17,
    "head_top_k_pct":       0.20,
    "clip_coarse_grid":     7,
    "clip_top_k_pct":       0.30,
    "clip_fallback_thresh": 0.20,
    "dataset_layer_end":    {},
}


def get_arch(model_id: str) -> dict:
    """Return arch params for model_id, falling back to SRF_ARCH_FALLBACK."""
    return SRF_ARCH_PARAMS.get(model_id, SRF_ARCH_FALLBACK)
