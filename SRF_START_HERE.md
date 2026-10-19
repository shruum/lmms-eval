# SRF — start here

Read this first, in any session and on any machine. It records which branch,
environment and data to use, and what has been verified. Last updated 2026-10-19.

SRF (Semantic Re-Focus) is an inference-time, training-free attention-steering
method for VLM hallucination. It has four parts, all always on:
1. a CLIP v3 relevance map
2. foveation (Gaussian blur of the non-relevant regions)
3. a pre-softmax attention boost on the selected heads
4. background and system-prompt suppression

## 1. Branch

| Branch | Use |
|---|---|
| **`srf-final`** | **Use this.** One codebase for Qwen2.5-VL-3B and LLaVA-1.5-7B, transformers 4.x and 5.x. |
| `autoresearch/mmvp-srf` | Frozen Qwen history, including the dead variants `srf_re` and `srf_v`. |
| `srf-llava` | Frozen LLaVA history, including the dead variants `srfc2` and `srfc3`. |

`srf-final` reproduces both frozen branches sample by sample (Qwen MMVP and POPE,
LLaVA MME and MMHal). Develop only on `srf-final`. The dead variants were
deliberately not carried over.

## 2. Environment

```bash
conda activate srf          # Snellius; transformers 4.57.6, torch 2.10.0+cu128
# rebuild: conda create -n srf python=3.10 && conda activate srf
#          pip install -r requirements-srf.txt --extra-index-url https://download.pytorch.org/whl/cu128
```
- `requirements-srf.txt` pins the key versions. `requirements-srf.lock.txt` is the full freeze.
- `srf` matches the local (RTX 2080 Ti) Qwen machine. The older Snellius env `mllm`
  (transformers 5.9, torch 2.12) still runs, but produces different numbers.
- Offline compute nodes: `export HF_HOME=/home/sgowda/.cache/huggingface HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- The model is loaded with `--dtype`, default `bfloat16`.

**Results depend on GPU type and dtype** because of bf16 rounding (see §6).
Always report the GPU, dtype and env together with any number.

## 3. Data (Snellius)

| Dataset | Path | Config key / flag |
|---|---|---|
| MMVP | `/home/sgowda/datasets/MMVP` (Questions.csv, `MMVP Images/1..300.jpg`) | `config.MMVP_DIR`, env `SRF_MMVP_DIR` |
| MME | `/home/sgowda/workspace/ILVAD/data/MME/MME_Benchmark_release_version/MME_Benchmark` | `config.MME_DIR` / `--mme_data_dir` |
| MMHal-Bench | `/home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json` | `config.MMHAL_JSON` / `--mmhalbench_json` |
| POPE | HF `lmms-lab/POPE` (in the HF cache) | — |

- New datasets go under `/home/sgowda/datasets/`. The `/gpfs/work4/0/prjs1231` project space is over quota.
- Models live in the HF hub cache.
- MMHal GPT scoring needs `OPENAI_API_KEY`, inherited from `~/.bashrc`. Never write the key into a file.

## 4. Running

`srf/eval.py` is the single entry point. It dispatches on the model's `model_type`.

```bash
cd lmms-eval
# Qwen — MMVP (shipped config)
python -u srf/eval.py --method srffovea --model Qwen/Qwen2.5-VL-3B-Instruct --datasets mmvp \
    --head_mode ratio_topk --head_top_k_pct 0.2 --layer_start 6 --layer_end 31
# LLaVA — MME
python -u srf/eval.py --method srffovea --model llava-hf/llava-1.5-7b-hf --datasets mme \
    --mme_subtasks existence count position color \
    --alpha 0.5 --fovea_sigma 20 --layer_start 8 --layer_end 20 --head_top_k_pct 0.20 \
    --llava_boost_mode additive --clip_fallback_thresh 0.25 --clip_patch_thresh 0.27 --sys_beta 0.30
# LLaVA — MMHal: same as MME but --datasets mmhalbench --alpha 0.3 --openai_model gpt-4o
# Baselines: --method baseline --model <id> --datasets <ds>
```

Rules:
- Always use `--method srffovea`.
- Never use `--method srf` (plain) or `srfe` for LLaVA.
- Don't use `neg_absent_alpha` on LLaVA.
- Use only MME and MMHal for LLaVA, not POPE.
- `--head_mode` other than `global` is Qwen-only. It raises an error on LLaVA, because head calibration isn't ported yet.
- All hyperparameters live in `srf/config.py` and are overridden by CLI flags. Never hardcode them in logic.
- Per-model dataset differences go in `SRF_ARCH_PARAMS[...]["dataset_overrides"]`. Example: MME uses phase `both` for LLaVA and `generation` for Qwen.

SLURM on Snellius:
- Use partition `gpu_a100` or `gpu_h100`, 1 GPU, and `#SBATCH --exclude=gcn58`.
- Logs go to `srf_exp_runs/logs/`.
- Check a job within 5 minutes of submitting it.
- A job pending for more than 20 minutes means something is wrong.

## 5. Reproduction check (run after any change to core code)

```bash
sbatch -J chk --export=ALL,ENV=srf srf_exp_runs/repro_check.sh <code_dir> qwen_core  <tag>   # MMVP base+SRF
sbatch -J chk --export=ALL,ENV=srf srf_exp_runs/repro_check.sh <code_dir> llava_core <tag>   # MME  base+SRF
# full suites: qwen (MMVP + POPE n=500), llava (MME + MMHal; SCORE=1 for GPT scoring); DTYPE=float32 etc.
```

Outputs go to `results/repro/<tag>/`. Compare per sample with the stored `results/repro/*` runs. Any difference on the same GPU, env and dtype means a regression.

## 6. Verified numbers and caveats (2026-10-19, `srf-final`)

**LLaVA-1.5-7B, MME (4 subtasks).** The SRF gain is robust.

| GPU | env | dtype | baseline | SRF | gain |
|---|---|---|---|---|---|
| A100 | mllm | bf16 | 670.00 | 693.33 | +23.33 |
| A100 | srf | bf16 | 670.00 | 690.00 | +20.00 |
| A100 | srf | fp32 | 663.33 | 676.67 | +13.33 |
| H100 | srf | bf16 | 673.33 | 690.00 | +16.67 |

The gain is positive in all 8 GPU × env × dtype cells (+10 to +23).
MMHal SRF (α=0.3): 2.458 on A100 `mllm` and 2.354 on A100 `srf`.

**Qwen2.5-VL-3B, MMVP pair accuracy.** The SRF gain is **not** robust.

| GPU | env | dtype | baseline | SRF | gain |
|---|---|---|---|---|---|
| RTX 2080 Ti (local) | local | bf16 | 40.00 | 45.33 | +5.33 |
| A100 | srf | bf16 | 40.67 | 43.33 | +2.67 |
| A100 | srf | fp32 | 42.00 | 40.67 | −1.33 |
| H100 | srf | bf16 | 42.00 | 44.67 | +2.67 |

- bf16 gains range from +0.67 to +2.67. fp32 gains are −0.67 to −2.00 in every cell.
- McNemar p ≥ 0.57 everywhere: SRF fixes about 15 pairs and breaks about 14.
- The 2080 Ti (Turing) has no native bf16, which is why its numbers differ.
- **Do not cite Qwen MMVP +5.33 as an SRF result.** Next step: Qwen full POPE and MME, in fp32 and bf16, with significance tests.

**MMHal judge noise:** gpt-4o re-scoring byte-identical responses changed 12 of 96 ratings (2.458 → 2.427). Treat MMHal differences below about 0.05 as noise.

## 7. Key files

| File | Purpose |
|---|---|
| `srf/eval.py` | Entry point: `load_model` (Qwen/LLaVA dispatch, `--dtype`), all `run_*` benchmarks |
| `srf/srf.py` | SRF core: relevance map → attention state per sample |
| `srf/srf_fovea.py` | Foveation (`--method srffovea`) |
| `srf/config.py` | All hyperparameters, per-arch params, data paths |
| `srf/head_calibration.py` | VTAR head selection (`--head_mode`, Qwen only) |
| `my_analysis/qwen_attn_patch.py` / `llava_attn_patch.py` | Attention patches |
| `srf/saliency/clip_salience.py`, `srf/noun_extract.py` | CLIP relevance map, query noun |
| `srf_exp_runs/repro_check.sh` | Reproduction driver |

Older docs (history, may be outdated):
- `LLAVA_CONTEXT.md`: LLaVA sweep history
- `SRF_METHOD_CARD.md`, `SRF_details.md`, `SRF_extension.md`: Qwen method history
- `BRANCH_DIFF.md`: pre-merge analysis
- `SRF.md` → `srf/docs/`: Qwen research docs

Where these disagree with this file, this file wins.
