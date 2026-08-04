# Snellius Cluster Guide

## SSH Config

Add to `~/.ssh/config` on your local machine:

```
Host snellius
    HostName snellius.surf.nl
    User sgowda

Host snellius_gpu
    HostName int6.local.snellius.surf.nl   # current interactive node
    User sgowda
    ProxyJump snellius
```

To SSH directly into an allocated compute node (after `salloc`):
```
Host gcn42                                 # replace with your node name
    HostName gcn42
    User sgowda
    ProxyCommand ssh -Y sgowda@snellius -W %h:%p
```

---

## Filesystem

`/home/sgowda` is a symlink to `/gpfs/home1/sgowda` — same path, both work.

```bash
df -h          # disk usage
accinfo        # check compute budget / quota
```

---

## Partitions (relevant ones)

| Partition   | Hardware         | GPUs/node | Notes |
|-------------|-----------------|-----------|-------|
| `gpu_a100`  | A100 40GB        | 4         | Main GPU partition |
| `gpu_h100`  | H100 80GB        | 4         | Faster, more VRAM |
| `gpu_mig`   | A100 MIG slices  | 8 slices  | Shared GPU, ~20GB each |
| `rome`      | CPU only         | —         | 128 cores/node |
| `genoa`     | CPU only         | —         | 192 cores/node |

Check live node states:
```bash
sinfo -o "%P %G %N %t" | grep gpu
squeue -u $USER
```

---

## Allocating a GPU node

### Interactive session (recommended for testing/development)

```bash
# A100 — single GPU, 1 hour
salloc -N 1 -p gpu_a100 --gpus=1 --cpus-per-task=9 --mem=40G --time=01:00:00

# H100 — for large models or faster runs
salloc -N 1 -p gpu_h100 --gpus=1 --cpus-per-task=9 --mem=60G --time=01:00:00

# MIG slice — lighter workloads (~20GB VRAM)
salloc -N 1 -p gpu_mig --gpus=1 --cpus-per-task=9 --time=01:00:00
```

After allocation, verify:
```bash
nvidia-smi -L                            # list GPU(s)
echo $SLURM_JOB_ID                       # your job ID
echo $CUDA_VISIBLE_DEVICES               # which GPU index is yours
python -c "import torch; print(torch.cuda.get_device_name(0))"
taskset -pc $$                           # CPU cores assigned
```

### MIG slices — pick a specific slice

```bash
# Get all MIG UUIDs on this node
mig=($(nvidia-smi -L | sed -nr "s|^.*UUID:\s*(MIG-[^)]+)\)|\1|p"))
echo ${mig[0]}                           # e.g. MIG-bdc1d762-d094-5868-b40a-...

# Point CUDA at one slice
export CUDA_VISIBLE_DEVICES=${mig[0]}
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Batch job

```bash
sbatch job_script.sh
squeue -u $USER        # monitor
scancel <job_id>       # cancel
```

---

## VS Code Remote Development

### Connecting to an allocated GPU node

1. Allocate the node in your terminal:
   ```bash
   salloc -N 1 -p gpu_a100 --gpus=1 --cpus-per-task=9 --mem=40G --time=02:00:00
   ```

2. Note the node name:
   ```bash
   squeue -u $USER     # look for NODELIST column, e.g. gcn42
   echo $SLURM_JOB_ID
   ```

3. Add the node to `~/.ssh/config` (as above), then connect in VS Code via **Remote-SSH**.

4. In VS Code's terminal, attach to the job:
   ```bash
   echo $SLURM_JOB_ID               # confirm same job
   srun --jobid=$SLURM_JOB_ID --pty bash -l
   taskset -pc $$                   # verify CPU affinity matches
   echo $CUDA_VISIBLE_DEVICES       # verify GPU visible
   ```

### VS Code debugger with MIG

In `.vscode/launch.json`, set the MIG UUID so the debugger sees the right GPU:
```json
{
    "configurations": [
        {
            "name": "Python: SRF eval",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/srf/eval.py",
            "env": {
                "CUDA_VISIBLE_DEVICES": "MIG-xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
            }
        }
    ]
}
```

---

## Getting onto a GPU node and running code

### Step 1 — Allocate from the login node (int6)

```bash
salloc -N 1 -p gpu_a100 --gpus=1 --cpus-per-task=9 --mem=40G --time=01:00:00
```

This drops you directly into a shell on the GPU node (prompt changes to e.g. `[sgowda@gcn42 ~]$`).

### Step 2 — Set up environment on the GPU node

`salloc` spawns a non-login shell so `.bashrc` is not sourced automatically — conda won't be in PATH. Fix:

```bash
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/sgowda/.cache/huggingface   # model cache lives here, not /volumes2
nvidia-smi                # confirm GPU is visible
```

Conda is at `/home/sgowda/miniconda3`. HF model cache is at `~/.cache/huggingface/hub` — must set `HF_HOME` explicitly or HF will try `/volumes2` which has no permissions on compute nodes.

### Step 3 — Run code

```bash
python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope --pope_splits adversarial \
  --n_pope 10 \
  --output results/test_llava_merge/srf
```

Results are saved as JSON in the output dir. Quick check:
```bash
cat results/test_llava_merge/srf/pope.json
```

### Step 4 — Run Claude Code on the GPU node

`claude` is at `~/.local/bin/claude` and available on all nodes (GPFS is shared).

```bash
# One-time setup — add API key to ~/.bashrc so it persists across nodes
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
source ~/.bashrc

# Launch Claude in the project directory
cd /home/sgowda/workspace/SRF/lmms-eval
claude
```

Claude sees the same files on all nodes. You can debug crashes, read results, and edit scripts live from the GPU node while a run is in progress.

Check if API key is already set:
```bash
grep ANTHROPIC ~/.bashrc
```

---

## Running SRF experiments

### Environment

```bash
conda activate mllm
export HF_HOME=/gpfs/home1/sgowda/.cache/huggingface   # or wherever your HF cache is
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval
```

### Quick smoke test — LLaVA + POPE (10 samples)

```bash
# SRF base
conda run -n mllm python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope \
  --pope_splits adversarial \
  --n_pope 10 \
  --output results/test_llava/srf

# SRF-E (contrastive, gamma=3.0)
conda run -n mllm python srf/eval.py \
  --method srfe \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope \
  --pope_splits adversarial \
  --n_pope 10 \
  --gamma 3.0 \
  --output results/test_llava/srfe
```

### Quick smoke test — Qwen + POPE (10 samples)

```bash
conda run -n mllm python srf/eval.py \
  --method srf \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --datasets pope \
  --pope_splits adversarial \
  --n_pope 10 \
  --output results/test_qwen/srf
```

### Full POPE run (all 9 splits × 3000 samples)

```bash
conda run -n mllm python srf/eval.py \
  --method srfe \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --datasets pope \
  --gamma 3.0 \
  --output results/srfe_qwen3b/
```

---

## Using Claude Code on a compute node

`claude` is installed at `~/.local/bin/claude` and available in your PATH.

```bash
# Check it's there
which claude
claude --version

# Make sure ANTHROPIC_API_KEY is set (add to ~/.bashrc to persist)
export ANTHROPIC_API_KEY=sk-ant-...

# Launch Claude Code in the project directory
cd /home/sgowda/workspace/SRF/lmms-eval
claude
```

On a GPU compute node (after `salloc` + `srun --pty bash -l`), Claude Code works the same way — it runs on the CPU and can read/write files and run shell commands. It will see GPU output (nvidia-smi, training logs, etc.) through the Bash tool.

Useful for: debugging crash logs live, editing experiment scripts, reading results while a run is in progress.

---

## Useful commands

```bash
accinfo              # compute budget / quota
sinfo                # partition status
squeue -u $USER      # your running/pending jobs
scancel <job_id>     # cancel a job
htop                 # CPU + memory
nvtop                # GPU usage (run on compute node)
nvidia-smi           # GPU status
df -h                # disk usage
```

---

## Links

- [Snellius partitions and accounting](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions+and+accounting)
- [Creating and running jobs](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660217/Creating+and+running+jobs)
- [VS Code for remote development](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660616/Visual+Studio+Code+for+remote+development)
- [Interactive GPU node](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/74225195/Interactive+development+GPU+node)
- [Efficient CPU jobs](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660233/How+to+run+efficient+CPU+jobs)
- [Filesystem](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/92668034/Interacting+with+the+filesystem)
- [LLM finetuning on Snellius](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/121077820/LLM+finetuning+on+Snellius)
- [LLM inference with vLLM](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/232851290/LLM+inference+on+Snellius+with+vLLM)
