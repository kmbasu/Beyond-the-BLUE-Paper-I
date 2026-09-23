#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --threads-per-core=1
#SBATCH --job-name=eta_ladder
#SBATCH --output=logs/out-eta.%A_%a
#SBATCH --error=logs/err-eta.%A_%a
#SBATCH --time=12:00:00
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --array=1-19

# NOTE (public release): a template for a Slurm cluster with NVIDIA GPUs (the production
# runs used GH200 nodes, one GPU per task unless stated).  Replace <your-account>,
# <your-partition> and the GPU request with your site's values, and set
# BTB_VENV_ACTIVATE to the activate script of a Python environment that has this
# repository installed (pip install -e .).
# ==========================================================================
# η-campaign variational2 ladder on a Slurm cluster — one (model, template) cell per
# array task, ONE GPU per task, no DDP (the rungs have ≤ 2.3k parameters;
# the cost is the double-backward through 128² convolutions, which a single
# GH200 handles at ~1 s/epoch).  Cells are listed in cells.txt.
#
# Submit:            sbatch eta_ladder_array.slurm
# Smoke test first:  ETA_SMOKE=1 sbatch --array=1 --time=00:20:00 eta_ladder_array.slurm
# Subset:            sbatch --array=1-4 eta_ladder_array.slurm
#
# Results land in eta_results_hpc/ (never in eta_results/), one JSON per
# cell, so array tasks cannot collide.  Bring them home with rsync and
# merge with tools/merge_remote_results.py.
# ==========================================================================

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"     # numpy FFTs in the ensemble generation

cd "${BTB_HOME:-$SLURM_SUBMIT_DIR}"      # repository root; submit from there or export BTB_HOME
mkdir -p logs results/eta_results_hpc
if [ -n "${BTB_VENV_ACTIVATE:-}" ]; then source "$BTB_VENV_ACTIVATE"; fi

# ----- pick this task's cell from cells.txt (skip comment / blank lines) -----
CELLS=scripts/hpc/${ETA_CELLS:-cells.txt}   # e.g. ETA_CELLS=cells_pool.txt sbatch --array=1-8 ...
CELL=$(grep -v '^\s*#' "$CELLS" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$CELL" ]; then
    echo "no cell for array index $SLURM_ARRAY_TASK_ID in $CELLS"; exit 1
fi
# optional 6th/7th columns: KMAX POOL (pooling basis of the quadratic/cubic/cnn rungs)
read -r MODEL TEMPLATE RUNGS EPOCHS RESTARTS KMAX POOL <<< "$CELL"
KMAX=${KMAX:-3}; POOL=${POOL:-fourier}

OUT=results/eta_results_hpc
if [ "$KMAX" != "3" ] || [ "$POOL" != "fourier" ]; then
    # basis-sweep runs never overwrite the production set, and DIFFERENT bases on the
    # same cell never share a JSON (2026-09-05: kmax=8 and template tasks on the _ATM
    # cells wrote the same file; the kmax=8 values survived only in the logs)
    OUT=results/eta_results_hpc_pool_${POOL}_k${KMAX}
fi
if [ -n "${ETA_N_ENS:-}" ]; then
    NENS_ARG="--n-ens $ETA_N_ENS"      # e.g. ETA_N_ENS=30000 -> n_eval 7500 (confusion production runs)
    OUT=${OUT}_n${ETA_N_ENS}
    echo "*** ETA_N_ENS=$ETA_N_ENS honoured: n_eval = $((ETA_N_ENS / 4)), output -> $OUT ***"
else
    NENS_ARG=""
fi
echo "script version: 2026-09-05c (per-basis output dirs, ETA_N_ENS, ETA_CELLS)"
if [ "${ETA_MAPS_FLOAT64:-0}" = "1" ]; then
    OUT=results/eta_results_hpc_f64          # float64 A/B test of the rounding channel
    echo "*** FLOAT64 MAPS: ETA_MAPS_FLOAT64=1 -> $OUT ***"
fi
if [ "${ETA_SMOKE:-0}" = "1" ]; then
    EPOCHS=2; RESTARTS=1; OUT=results/eta_results_smoke
    echo "*** SMOKE MODE: epochs=2 restarts=1 -> $OUT ***"
fi

echo "========================================="
echo "cell $SLURM_ARRAY_TASK_ID: $MODEL / $TEMPLATE / rungs=$RUNGS / epochs=$EPOCHS / restarts=$RESTARTS / kmax=$KMAX pool=$POOL -> $OUT"
echo "node $(hostname)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
python -c "import torch, platform; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), platform.machine())"
echo "========================================="

# patience / min-epochs use the 2026-09-02 defaults: max(15, epochs//3) and min(30, epochs)
srun python -u scripts/run_eta_campaign.py \
        --mode variational2 \
        --models "$MODEL" \
        --templates "$TEMPLATE" \
        --rungs "$RUNGS" \
        --epochs "$EPOCHS" \
        --restarts "$RESTARTS" \
        --kmax "$KMAX" --pool "$POOL" $NENS_ARG \
        --out "$OUT" \
        --force

echo "cell $SLURM_ARRAY_TASK_ID done: $(date)"
