# MMVP SRF Autoresearch — Loop Program

## Setup

```bash
cd /volumes2/mllm/lmms-eval
git checkout autoresearch/mmvp-srf 2>/dev/null || git checkout -b autoresearch/mmvp-srf
```

The model is **Qwen2.5-VL-7B-Instruct** split across GPU 0 (RTX 2080 Ti) + GPU 1 (GTX 1080 Ti).
CLIP runs on GPU 0. GPU usage when loaded: 7.6/11.5 GB (GPU 0) + 9.0/11.7 GB (GPU 1).
Do not run VLM Bias or other jobs in parallel — 7B needs both GPUs fully.

## The Objective

Maximise **MMVP pair accuracy** (primary) and image accuracy (secondary).

- **Baseline to beat**: VAR (ClearSight) achieves ~9.33% pair accuracy on MMVP.
- **Target**: >12% pair accuracy (CLIP-guided query-conditional boosting should outperform VAR).
- MMVP tests 9 fine-grained visual pattern categories (orientation, state, color, shape, count, etc.)
- Pair accuracy = both images in a pair correct. Random = 0%, language-prior-only ≈ 0%.

## One Experiment Cycle

```bash
# 1. Edit srf.py — one change at a time
nano my_analysis/autoresearch_mmvp/srf.py

# 2. Commit
git add my_analysis/autoresearch_mmvp/srf.py
git commit -m "experiment: <description>"

# 3. Run (both GPUs, ~20-25 min for 7B × 300 samples)
conda run -n mllm python my_analysis/autoresearch_mmvp/mmvp_eval.py \
    > my_analysis/autoresearch_mmvp/run.log 2>&1

# 4. Read results
grep "MMVP.*accuracy:" my_analysis/autoresearch_mmvp/run.log

# 5. Log result in results.tsv
# 6. Keep or discard
```

**Keep threshold**: pair_acc improved by ≥ 0.010 (1 percentage point).
**Discard threshold**: pair_acc worse or improved < 0.005.
**Gray zone** (0.005–0.010): keep if trend is clear, discard otherwise.

To discard: `git reset --hard HEAD~1`
**IMPORTANT**: After `git reset --hard`, srf.py is gone — restore from the last kept commit.

## What to Sweep (in order of expected impact)

### Stage 1: Core boost parameters
- `boost_alpha`: {2.0, 4.0, 8.0, 12.0, 16.0} (warmstart: 8.0)
- `background_eps`: {0.0, 0.3, 0.5, 0.7, 1.0} (warmstart: 0.5)
- `srf_apply_phase`: "both" vs "generation" vs "prefill" (warmstart: "generation")

### Stage 2: Layer range
- `layer_start` / `layer_end`: sweep {6-13, 8-14, 8-15, 8-19, 10-17} (model has 28 layers, same as 3B)
  - VLM Bias best was 8-14; warmstart here is 8-15
- `head_top_k_pct`: {0.10, 0.15, 0.20, 0.30, 0.50} (warmstart: 0.20)

### Stage 3: Saliency quality
- `clip_coarse_grid`: {3, 5, 7, 9} (7×7 default; finer grid = more precise but noisier)
- `clip_top_k_pct`: {0.15, 0.20, 0.30, 0.40} (warmstart: 0.30)
- `clip_fallback_thresh`: {0.15, 0.20, 0.25, 0.30} (warmstart: 0.20)
  - MMVP images are clear photos — may want higher threshold to only boost when CLIP is confident

### Stage 4: Bias mode
- Try `prob_interp` (redistributes existing img budget by saliency)
- Try `global_redistribute` with `img_scale` ∈ {1.5, 2.0, 3.0}

## Key Differences from VLM Bias Loop

| Aspect | VLM Bias | MMVP |
|--------|----------|------|
| Task | Counting (how many X) | Fine-grained attribute (A or B) |
| Model | 3B | 7B (32 layers) |
| Answer format | {number} | A or B |
| Noun extraction | Counted object | Subject visual object |
| CLIP utility | High for Logos/Flags, none for Animals/Chess | High for all (clear photos) |
| Baseline | 17.1% | ~40% image acc, ~5% pair acc |
| Prior work best | vhr_boost 22.6% | VAR 9.33% pair acc |

## Noun Extraction Notes

MMVP questions name the subject explicitly:
- "butterfly's wings" → extract "butterfly" (possessive gives the object)
- "flame of the match" → extract "match" (container gives object)
- "Is this escalator..." → "escalator"

Test noun extraction by printing in a small script before committing if unsure.

## Crash Protocol

If eval crashes mid-run:
1. Check run.log for the error
2. Fix the issue in srf.py or qwen_attn_patch.py
3. Do NOT modify mmvp_eval.py
4. Recommit and re-run

If OOM (7B on 2×11GB):
- Reduce `max_pixels` in mmvp_eval.py processor line (try 256*28*28)
- Or try `torch_dtype=torch.float16` — but only as last resort (update model load line)
- DO NOT modify the metric logic or eval loop

## Results Log Format

Columns: `commit  pair_acc  img_acc  status  description`
Status: keep / discard / gray

Example:
```
abc12345  0.0933  0.4500  keep  baseline: warmstart from VLM Bias best
def67890  0.1067  0.4700  keep  boost_alpha=12.0 +0.013
```
