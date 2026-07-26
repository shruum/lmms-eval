#!/bin/bash
# Launch RePOPE-COCO SRF sweep - sequential per GPU to avoid OOM

echo "🚀 Launching REPOPE-COCO SRF Sweep (Sequential)"
echo "================================================"

mkdir -p results/repope_sequential_sweep

# Use all 8 GPUs but run 1 experiment per GPU at a time
GPUS=(0 1 2 3 4 5 6 7)
CATEGORIES=("coco_random" "coco_popular" "coco_adversarial")
ALPHAS=(0.15 0.25 0.50)
LAYERS_END=(15 18)

IMAGE_DIR="/home/anna2/shruthi/dataset/POPE_images/images/val2014"
OUTPUT_BASE="results/repope_sequential_sweep"

JOB_ID=0
PIDS=()

for category in "${CATEGORIES[@]}"; do
    case $category in
        "coco_random")
            REPOPE_FILE="/home/anna2/shruthi/RePOPE/annotations/coco_repope_random.json"
            ;;
        "coco_popular")
            REPOPE_FILE="/home/anna2/shruthi/RePOPE/annotations/coco_repope_popular.json"
            ;;
        "coco_adversarial")
            REPOPE_FILE="/home/anna2/shruthi/RePOPE/annotations/coco_repope_adversarial.json"
            ;;
    esac

    for alpha in "${ALPHAS[@]}"; do
        for layers_end in "${LAYERS_END[@]}"; do

            # Wait for available GPU
            while [ ${#PIDS[@]} -ge ${#GPUS[@]} ]; do
                # Check which PIDs are still running
                NEW_PIDS=()
                for pid in "${PIDS[@]}"; do
                    if ps -p "$pid" > /dev/null 2>&1; then
                        NEW_PIDS+=("$pid")
                    fi
                done
                PIDS=("${NEW_PIDS[@]}")

                if [ ${#PIDS[@]} -ge ${#GPUS[@]} ]; then
                    echo "⏳ Waiting for GPU to be available... (${#PIDS[@]}/${#GPUS[@]} in use)"
                    sleep 30
                fi
            done

            # Get next available GPU
            GPU=${GPUS[${#PIDS[@]}]}

            EXP_DIR="${OUTPUT_BASE}/${category}_alpha${alpha}_layers10-${layers_end}"
            mkdir -p "$EXP_DIR"

            echo "🚀 Job $JOB_ID: $category | α=$alpha | layers=10-$layers_end | GPU=$GPU"

            CUDA_VISIBLE_DEVICES=$GPU /home/anna2/miniconda3/envs/mllm/bin/python /home/anna2/shruthi/lmms-eval/srf/eval.py \
                --method srf \
                --model llava-hf/llava-1.5-7b-hf \
                --datasets pope_vcd \
                --calib_dataset pope \
                --pope_vcd_file "$REPOPE_FILE" \
                --pope_vcd_name "$category" \
                --pope_image_dir "$IMAGE_DIR" \
                --alpha "$alpha" \
                --eps 0.1 \
                --sys_beta 0.15 \
                --layer_start 10 \
                --layer_end "$layers_end" \
                --head_top_k_pct 0.50 \
                --clip_top_k_pct 0.30 \
                --do_sample \
                --temperature 0.7 \
                --top_p 0.9 \
                --output "$EXP_DIR/" \
                > "$EXP_DIR/run.log" 2>&1 &

            PIDS+=($!)
            JOB_ID=$((JOB_ID + 1))
            sleep 5

        done
    done
done

echo "================================================"
echo "🎯 Launched $JOB_ID experiments"
echo "📊 Results: $OUTPUT_BASE"

# Wait for all remaining jobs to complete
echo "⏳ Waiting for all experiments to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null
done

echo "✅ All experiments completed!"