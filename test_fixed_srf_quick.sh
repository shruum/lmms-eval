#!/bin/bash
# Quick test of FIXED SRF on small RePOPE sample

echo "🔍 Testing FIXED SRF with dimension bug fix..."
echo "Running on 10 samples to verify upsampling works"

CUDA_VISIBLE_DEVICES=0 /home/anna2/miniconda3/envs/mllm/bin/python /home/anna2/shruthi/lmms-eval/srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope_vcd \
  --calib_dataset pope \
  --pope_vcd_file /home/anna2/shruthi/RePOPE/annotations/coco_repoe_adversarial.json \
  --pope_vcd_name "RePOPE adversarial quick test" \
  --pope_image_dir /home/anna2/shruthi/dataset/POPE_images/images/val2014 \
  --alpha 0.25 \
  --eps 0.1 \
  --sys_beta 0.15 \
  --layer_start 10 \
  --layer_end 15 \
  --head_top_k_pct 0.50 \
  --clip_top_k_pct 0.30 \
  --n_pope 10 \
  --do_sample --temperature 0.7 --top_p 0.9 \
  --output results/test_fixed_srf_dimension/