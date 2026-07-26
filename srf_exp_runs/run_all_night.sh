#!/bin/bash
# =============================================================================
# Master overnight experiment launcher
# Keeps launching waves of experiments throughout the night
# =============================================================================

set -euo pipefail

WAVES_DIR="srf_exp_runs/results"
LOGFILE="${WAVES_DIR}/overnight_master.log"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

launch_wave() {
    local wave_num=$1
    local configs=$2

    log_msg "=== LAUNCHING WAVE ${wave_num} ==="

    OUT_DIR="${WAVES_DIR}/wave_${wave_num}"
    mkdir -p "${OUT_DIR}"

    # Parse configs and launch experiments
    # Config format: "gpu_id:layer_start:layer_end:head_pct:alpha:eps"
    for i in $(seq 1 8); do
        config=$(echo "$configs" | cut -d',' -f$i)
        IFS=':' read -r gpu_id layer_start layer_end head_pct alpha eps <<< "$config"

        if [ -z "$layer_start" ]; then
            continue
        fi

        log_msg "GPU ${gpu_id}: layers ${layer_start}-${layer_end}, heads ${head_pct}, alpha ${alpha}, eps ${eps}"

        CUDA_VISIBLE_DEVICES=${gpu_id} conda run -n mllm nohup python -u srf/eval.py \
            --method srf \
            --model "llava-hf/llava-1.5-7b-hf" \
            --datasets pope \
            --pope_splits adversarial \
            --n_pope 100 \
            --alpha ${alpha} \
            --eps ${eps} \
            --layer_start ${layer_start} \
            --layer_end ${layer_end} \
            --head_top_k_pct ${head_pct} \
            --output "${OUT_DIR}/gpu${gpu_id}_l${layer_start}_${layer_end}_h${head_pct}_a${alpha}_e${eps}/" \
            > "${OUT_DIR}/gpu${gpu_id}.log" 2>&1 &

        log_msg "Launched GPU ${gpu_id} PID: $!"
    done
}

wait_for_completion() {
    local wave_num=$1
    local max_wait=3600  # 1 hour max per wave

    log_msg "Waiting for wave ${wave_num} completion (max ${max_wait}s)..."

    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        completed=$(ls -1 "${WAVES_DIR}/wave_${wave_num}"/*/pope.json 2>/dev/null | wc -l)
        if [ "$completed" -ge 8 ]; then
            log_msg "Wave ${wave_num} completed! ($completed/8 experiments)"
            return 0
        fi

        sleep 60
        elapsed=$((elapsed + 60))

        # Log progress every 10 minutes
        if [ $((elapsed % 600)) -eq 0 ]; then
            log_msg "Wave ${wave_num} progress: $completed/8 completed (${elapsed}s elapsed)"
        fi
    done

    log_msg "Wave ${wave_num} timed out! Starting next wave anyway..."
}

analyze_results() {
    local wave_num=$1

    log_msg "=== ANALYZING WAVE ${wave_num} RESULTS ==="

    python3 << PYTHON_EOF
import json
import glob
import os

wave_dir = f"../../results/wave_{wave_num}"
baseline_acc = None

print(f"Wave {wave_num} Results:")
print("-" * 50)

for result_file in sorted(glob.glob(f"{wave_dir}/*/pope.json")):
    if not os.path.exists(result_file):
        continue

    dir_name = os.path.basename(os.path.dirname(result_file))

    try:
        with open(result_file) as f:
            data = json.load(f)

        if 'baseline' in data and 'method' not in data:
            baseline_acc = data['baseline']['accuracy']
            print(f"{dir_name:30s}: Baseline={baseline_acc:.4f}")
        else:
            method_data = data.get('method', {})
            if method_data:
                acc = list(method_data.values())[0]['accuracy']
                delta = acc - baseline_acc if baseline_acc else 0
                status = "✓" if delta > 0.01 else "✗"
                print(f"{dir_name:30s}: SRF={acc:.4f} (Δ={delta:+.4f}) {status}")
    except Exception as e:
        print(f"{dir_name:30s}: ERROR - {e}")

print()
PYTHON_EOF
}

# Main overnight loop
log_msg "=== OVERNIGHT EXPERIMENT LOOP STARTED ==="

wave_count=0

while true; do
    wave_count=$((wave_count + 1))

    # Generate random configs for this wave
    # This is a simple example - in production would use more sophisticated config generation
    case $wave_count in
        1)
            # Wave 1: Focus on early layers
            configs="0:2:7:0.5:0.5:0.0,1:2:8:0.5:1.0:0.0,2:3:9:0.5:1.5:0.0,3:2:10:0.3:2.0:0.0,4:3:11:0.5:0.5:0.1,5:2:12:0.5:1.0:0.1,6:3:13:0.5:1.5:0.1,7:2:14:0.3:2.0:0.1"
            ;;
        2)
            # Wave 2: Focus on mid layers
            configs="0:8:12:0.5:0.5:0.0,1:9:13:0.5:1.0:0.0,2:10:14:0.5:1.5:0.0,3:11:15:0.3:2.0:0.0,4:12:16:0.5:0.5:0.1,5:13:17:0.5:1.0:0.1,6:14:18:0.5:1.5:0.1,7:15:19:0.3:2.0:0.1"
            ;;
        3)
            # Wave 3: Focus on late layers
            configs="0:15:20:0.5:0.5:0.0,1:16:21:0.5:1.0:0.0,2:17:22:0.5:1.5:0.0,3:18:23:0.3:2.0:0.0,4:19:24:0.5:0.5:0.1,5:20:25:0.5:1.0:0.1,6:21:26:0.5:1.5:0.1,7:22:27:0.3:2.0:0.1"
            ;;
        4)
            # Wave 4: Very weak boosts
            configs="0:8:15:0.5:0.1:0.0,1:8:15:0.3:0.15:0.0,2:8:15:0.8:0.2:0.0,3:10:20:0.5:0.25:0.0,4:8:15:0.5:0.1:0.2,5:8:15:0.3:0.15:0.2,6:10:20:0.8:0.2:0.2,7:8:15:0.5:0.25:0.2"
            ;;
        5)
            # Wave 5: Very strong boosts
            configs="0:8:15:0.5:3.0:0.0,1:8:15:0.5:4.0:0.0,2:8:15:0.3:5.0:0.0,3:10:20:0.5:6.0:0.0,4:8:15:0.5:3.0:0.5,5:8:15:0.3:4.0:0.5,6:10:20:0.8:5.0:0.5,7:8:15:0.5:6.0:0.5"
            ;;
        *)
            # Wave 6+: Random exploration
            configs="0:8:15:0.5:1.0:0.0,1:5:20:0.5:1.5:0.0,2:10:25:0.3:2.0:0.0,3:8:15:0.8:0.5:0.1,4:12:18:0.5:1.0:0.1,5:6:22:0.5:1.5:0.2,6:9:16:0.3:2.0:0.0,7:11:21:0.5:0.8:0.1"
            ;;
    esac

    launch_wave $wave_count "$configs"
    wait_for_completion $wave_count
    analyze_results $wave_count

    # Check if we should continue (stop after 20 waves or if morning)
    current_hour=$(date +%H)
    if [ $wave_count -ge 20 ] || [ "$current_hour" -ge 8 ]; then
        log_msg "=== OVERNIGHT EXPERIMENTS COMPLETE ==="
        log_msg "Completed $wave_count waves"
        log_msg "Generating final summary..."

        # Generate comprehensive summary
        python3 << PYTHON_EOF
import json
import glob

print("=" * 70)
print("FINAL OVERNIGHT SUMMARY")
print("=" * 70)

all_results = []
for wave_file in glob.glob("../../results/wave_*/gpu*/pope.json"):
    try:
        with open(wave_file) as f:
            data = json.load(f)

        wave_num = wave_file.split('/')[2].split('_')[1]
        dir_name = os.path.basename(os.path.dirname(wave_file))

        if 'baseline' in data and 'method' not in data:
            baseline_acc = data['baseline']['accuracy']
            all_results.append(('baseline', wave_num, dir_name, baseline_acc, 0.0))
        else:
            method_data = data.get('method', {})
            if method_data:
                acc = list(method_data.values())[0]['accuracy']
                all_results.append(('srf', wave_num, dir_name, acc, 0.0))
    except:
        pass

# Sort by accuracy
all_results.sort(key=lambda x: x[3], reverse=True)

print(f"\nTop 20 configurations:")
print("-" * 70)
for i, (method, wave, name, acc, delta) in enumerate(all_results[:20]):
    print(f"{i+1:2d}. {method:8s} Wave {wave:2s} {name:20s} = {acc:.4f}")

print()
print("Total experiments run:", len(all_results))
print("=" * 70)
PYTHON_EOF

        break
    fi

    # Small delay between waves
    sleep 30
done

log_msg "Master overnight script completed. Good morning!"
