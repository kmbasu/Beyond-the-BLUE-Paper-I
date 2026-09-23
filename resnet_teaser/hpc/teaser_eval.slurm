#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --threads-per-core=1
#SBATCH --job-name=teaser_eval
#SBATCH --output=%x-out-eval.%A_%a
#SBATCH --error=%x-err-eval.%A_%a
#SBATCH --time=00:30:00
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --array=1-8
# NOTE (public release): a template for a Slurm cluster with NVIDIA GPUs (the production
# runs used GH200 nodes, one GPU per task unless stated).  Replace <your-account>,
# <your-partition> and the GPU request with your site's values, and set
# BTB_VENV_ACTIVATE to the activate script of a Python environment that has this
# repository installed (pip install -e .).
# ==========================================================================
# A-grid evaluation of the teaser checkpoints (teaser_eval.py): 41 amplitudes x
# 4000 EVAL noise maps per checkpoint, one GPU, a few minutes each.
#   sbatch hpc/teaser_eval.slurm            # all runs of teaser_runs.txt, FINAL checkpoints -> eval_out/
#   sbatch hpc/teaser_eval.slurm best       # the same for the BEST-VALIDATION checkpoints -> eval_out_best/
#   sbatch --array=1 hpc/teaser_eval.slurm  # one run
# Results -> eval_out/<checkpoint stem>_{eval.json,grid.npz,diag.png,teaser.png,runlog.md}
# ==========================================================================
# ----- locations (edit here only) -----
# TEASER_HOME : the resnet_teaser/ directory (scripts, hpc/, checkpoints/, eval_out/, logs/)
# TEASER_DATA : where the HDF5 datasets live (a project/scratch data area); both can also be exported before sbatch
TEASER_HOME=${TEASER_HOME:-$SLURM_SUBMIT_DIR}          # the resnet_teaser/ directory of this repository
DATA=${TEASER_DATA:-$TEASER_HOME/teaser_data}
cd "$TEASER_HOME" || { echo "TEASER_HOME=$TEASER_HOME not found"; exit 1; }
mkdir -p logs checkpoints eval_out
if [ -n "${BTB_VENV_ACTIVATE:-}" ]; then source "$BTB_VENV_ACTIVATE"; fi
RUNS=${TEASER_RUNS:-hpc/teaser_runs.txt}
LINE=$(grep -v '^\s*#' "$RUNS" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "$LINE" ] || { echo "no run for index $SLURM_ARRAY_TASK_ID"; exit 1; }
read -r LABEL MODEL TRAIN_H5 EVAL_H5 LOSS LAMBDA CKPT <<< "$LINE"
OUT=eval_out
# best-validation checkpoints: EITHER  sbatch hpc/teaser_eval.slurm best
#                              OR      sbatch --export=ALL,TEASER_BEST=1 hpc/teaser_eval.slurm
# (a bare "TEASER_BEST=1 sbatch ..." depends on the site's SBATCH_EXPORT default and may not propagate)
MODE="${1:-}"; [ "$MODE" = "best" ] && TEASER_BEST=1
echo "mode: TEASER_BEST=${TEASER_BEST:-0} (arg '$MODE')"
if [ "${TEASER_BEST:-0}" = "1" ]; then       # -> the best-validation checkpoints
    CKPT="$CKPT.best.pth"; OUT=eval_out_best; LABEL="${LABEL}-best"
fi
[ -f "checkpoints/$CKPT" ] || { echo "missing checkpoints/$CKPT"; exit 1; }
mkdir -p "$OUT"
echo "eval $LABEL: checkpoints/$CKPT  vs  $DATA/$EVAL_H5  -> $OUT"
srun python -u teaser_eval.py --checkpoint "checkpoints/$CKPT" --eval-h5 "$DATA/$EVAL_H5" \
        --outdir "$OUT" --label "$LABEL" --n-grid 41 --batch 256 --n-boot 500 \
        --method-e-json r3_channels_wn.json
