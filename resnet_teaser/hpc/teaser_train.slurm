#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=1
#SBATCH --job-name=teaser_train
#SBATCH --output=%x-out-train.%A_%a
#SBATCH --error=%x-err-train.%A_%a
#SBATCH --time=02:30:00
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:4
#SBATCH --array=1-8
# NOTE (public release): a template for a Slurm cluster with NVIDIA GPUs (the production
# runs used GH200 nodes, one GPU per task unless stated).  Replace <your-account>,
# <your-partition> and the GPU request with your site's values, and set
# BTB_VENV_ACTIVATE to the activate script of a Python environment that has this
# repository installed (pip install -e .).
# ==========================================================================
# ResNet teaser training: one run of hpc/teaser_runs.txt per array
# task, 4 GPUs (DDP via torchrun), 700 epochs on the
# 30k-map file (24k train / 6k val) ~ 1 h.
#   sbatch hpc/teaser_train.slurm                 # all six runs
#   sbatch --array=1,2,4,5 hpc/teaser_train.slurm # only the memo's T1a/b, T2a/b
#   TEASER_SMOKE=1 sbatch --array=1 --time=00:20:00 hpc/teaser_train.slurm   # 3 epochs
# Checkpoints -> checkpoints/<name>.pth ; logs -> logs/ (printlog_* via --hpc).
# ==========================================================================
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
# ----- locations (edit here only) -----
# TEASER_HOME : the resnet_teaser/ directory (scripts, hpc/, checkpoints/, eval_out/, logs/)
# TEASER_DATA : where the HDF5 datasets live (a project/scratch data area); both can also be exported before sbatch
TEASER_HOME=${TEASER_HOME:-$SLURM_SUBMIT_DIR}          # the resnet_teaser/ directory of this repository
DATA=${TEASER_DATA:-$TEASER_HOME/teaser_data}
cd "$TEASER_HOME" || { echo "TEASER_HOME=$TEASER_HOME not found"; exit 1; }
mkdir -p logs checkpoints eval_out
if [ -n "${BTB_VENV_ACTIVATE:-}" ]; then source "$BTB_VENV_ACTIVATE"; fi

# ----- pick this task's run -----
RUNS=${TEASER_RUNS:-hpc/teaser_runs.txt}
LINE=$(grep -v '^\s*#' "$RUNS" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$LINE" ]; then echo "no run for array index $SLURM_ARRAY_TASK_ID in $RUNS"; exit 1; fi
read -r LABEL MODEL TRAIN_H5 EVAL_H5 LOSS LAMBDA CKPT <<< "$LINE"
H5_FILE="$DATA/$TRAIN_H5"
[ -f "$H5_FILE" ] || { echo "missing $H5_FILE (run teaser_gen.slurm first)"; exit 1; }
LAMBDA_ARG=""; [ "$LAMBDA" != "-" ] && LAMBDA_ARG="--lambda-cond $LAMBDA"
EPOCHS=${TEASER_EPOCHS:-700}
if [ "${TEASER_SMOKE:-0}" = "1" ]; then EPOCHS=3; CKPT="smoke_${CKPT}"; echo "*** SMOKE: 3 epochs ***"; fi

# ===== HDF5 OPTIMIZATIONS =====
export HDF5_USE_FILE_LOCKING=TRUE
export HDF5_CACHE_SIZE=16777216
export HDF5_CHUNK_CACHE_SIZE=16777216
export HDF5_CHUNK_CACHE_NSLOTS=10007

# ===== NETWORK CONFIGURATION (site-specific: InfiniBand interface ib0; adjust or remove) =====
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_P2P_DISABLE=1
export NCCL_TREE_THRESHOLD=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_ALGO=Ring
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "========================================="
echo "run $LABEL: $MODEL  loss=$LOSS lambda=$LAMBDA  epochs=$EPOCHS"
echo "H5=$H5_FILE -> checkpoints/$CKPT"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  GPUs: $(($SLURM_JOB_NUM_NODES * 4))"
python -c "import torch, platform; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), platform.machine())"
echo "========================================="

# ===== LAUNCH WITH TORCHRUN =====
srun env -u CUDA_VISIBLE_DEVICES torchrun \
       --nproc_per_node=gpu \
       --nnodes="$SLURM_JOB_NUM_NODES" \
       --rdzv_id="$SLURM_JOB_ID" \
       --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
       --rdzv_backend=c10d \
       train_teaser_ddp.py \
           --h5 "$H5_FILE" \
           --checkpoint "checkpoints/$CKPT" \
           --loss-function "$LOSS" $LAMBDA_ARG \
           --seed 40 \
           --epochs "$EPOCHS" \
           --batch-size 50 \
           --lr 0.01 \
           --relative-threshold 0.2 \
           --absolute-threshold 0.4 \
           --log-dir logs \
           --hpc
echo "finished $(date)"
