#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --job-name=llava_fig3_scan
#SBATCH --output=srf_exp_runs/logs/llava_fig3_scan_%j.out
#SBATCH --error=srf_exp_runs/logs/llava_fig3_scan_%j.err
#SBATCH --exclude=gcn58

# Scan all 96 MMHal samples and save candidate panels.
# Candidates ranked by sim × size_bonus (object coverage ~0.25 is ideal).
# Output: analysis/candidates/NNN_simX.XXX_covX.XX_<type>_<question>.png
#
# After this job: inspect candidates, pick two by index, then run locally:
#   python srf/analysis/llava_fig3_routing.py --idx_a 003 --idx_b 011

set -eo pipefail
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "======================================================"

python -u srf/analysis/llava_fig3_routing.py --scan --n 20

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Candidates: analysis/candidates/"
echo "Next: pick two indices, then run locally:"
echo "  python srf/analysis/llava_fig3_routing.py --idx_a 003 --idx_b 011"
echo "======================================================"
