# LLaVA-1.5-7B-HF POPE Final Execution Status
**Started**: 2026-04-26 10:38 CEST
**Configuration**: Parallel execution on GPUs 0, 1, 2

---

## ✅ **Setup Verified**

### Sanity Tests Passed (50 samples):
- **Baseline**: 74.00% accuracy, 66.67% F1
- **SRF (α=4.0, ε=0.2)**: 76.00% accuracy (+2%), 68.42% F1 (+1.75%)
- **SRF-E (β=1.0)**: 74.00% accuracy, 72.34% F1 (+5.67%)

### Model Fixed:
- **Working**: `llava-hf/llava-1.5-7b-hf`
- **Broken**: `liuhaotian/llava-v1.5-7b` (missing preprocessor config)

---

## 🚀 **Parallel Execution Configured**

### GPU Assignment:
- **GPU 0**: Baseline + overflow
- **GPU 1**: SRF/SRF-E runs
- **GPU 2**: SRF/SRF-E runs

### Run Queue:
1. **Baseline** (GPU 0) - 1 run
2. **SRF Sweep** (GPUs 1,2) - 9 runs
   - α ∈ {2.0, 4.0, 8.0} × ε ∈ {0.1, 0.2, 0.3}
3. **SRF-E Sweep** (GPUs 1,2) - 4 runs
   - β ∈ {0.5, 1.0, 1.5, 2.0}

**Total**: 14 runs across 3 GPUs

---

## 📊 **Monitoring Commands**

### Quick status:
```bash
cd /home/anna2/shruthi/lmms-eval
./srf_exp_runs/final_monitor.sh
```

### Check specific run logs:
```bash
# Baseline
tail -f srf_exp_runs/results/llava15_7b_pope_full/baseline/run.log

# Specific SRF run
tail -f srf_exp_runs/results/llava15_7b_pope_full/srf_alpha4.0_eps0.2/run.log
```

### GPU usage:
```bash
nvidia-smi
watch -n 10 nvidia-smi
```

### Process status:
```bash
ps aux | grep "llava.*eval.py" | grep -v grep
```

---

## ⏱️ **Estimated Timeline**

With 3 GPUs running in parallel:
- **Per run**: ~40-60 minutes
- **Total time**: ~4-5 hours
- **Completion**: ~15:00-16:00 CEST

Sequential would have been: ~12+ hours

---

## 📁 **Results Location**

All results in: `srf_exp_runs/results/llava15_7b_pope_full/`

Structure:
```
llava15_7b_pope_full/
├── baseline/summary.json         # Baseline metrics
├── srf_alpha2.0_eps0.1/summary.json
├── srf_alpha2.0_eps0.2/summary.json
├── ... (7 more SRF combinations)
├── srfe_beta0.5/summary.json
├── srfe_beta1.0/summary.json
├── srfe_beta1.5/summary.json
└── srfe_beta2.0/summary.json
```

Each contains:
- `pope.json` - Detailed predictions
- `summary.json` - Aggregate metrics (accuracy, F1, precision, recall)
- `run.log` - Full execution log

---

## 🎯 **What to Check When Complete**

### 1. Baseline Performance
Expected: ~85% accuracy, ~84% F1 (based on prior run)

### 2. SRF Improvements
Look for α/ε combinations that give +2-5% over baseline

### 3. SRF-E Benefits
Check if higher β values improve F1 (as seen in sanity test)

### 4. Best Configuration
Compare all 14 runs to find optimal parameters

---

## 🔧 **Troubleshooting**

If processes stopped:
```bash
# Check what's running
ps aux | grep "llava.*eval.py" | grep -v grep

# Check logs for errors
tail -50 srf_exp_runs/results/llava15_7b_pope_full/*/run.log

# Restart if needed
bash srf_exp_runs/run_llava_parallel_fixed.sh
```

---

## ✅ **Verification**

### Methods Working:
- ✅ Calibration (640 softmax calls captured)
- ✅ 6 vision-aware heads identified
- ✅ CLIP salience computation
- ✅ Attention intervention (SRF)
- ✅ Contrastive decoding (SRF-E)

### GPUs Working:
- ✅ GPU 0: Baseline running (14.8GB)
- ✅ GPU 1: SRF running (27.7GB)
- ✅ GPU 2: SRF running (27.7GB)

---

**Status**: All systems operational, running in parallel! 🚀

**Next update**: When baseline completes (~40 min)
