#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --threads-per-core=1
#SBATCH --job-name=eta_confusion
#SBATCH --output=logs/out-conf.%A_%a
#SBATCH --error=logs/err-conf.%A_%a
#SBATCH --time=12:00:00
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --array=1-6

# NOTE (public release): a template for a Slurm cluster with NVIDIA GPUs (the production
# runs used GH200 nodes, one GPU per task unless stated).  Replace <your-account>,
# <your-partition> and the GPU request with your site's values, and set
# BTB_VENV_ACTIVATE to the activate script of a Python environment that has this
# repository installed (pip install -e .).
# ==========================================================================
# T2_CONFUSION variational2 ladder on a Slurm cluster — the confusion rung only.
#
# Derived from eta_ladder_array.slurm (2026-09-02).  Four differences, each
# forced by something specific to this rung:
#
#  1. CELL FILE is cells_confusion.txt, with SEVEN fields instead of five:
#     N_ENS and N_EXTRA are per cell.  The compact template's EVAL error is
#     tail-limited (MF-amplitude excess kurtosis 8.8 floored / 91.0
#     floorless), so this rung needs ~30k images where the rest of the
#     campaign uses 6k.  See the header of cells_confusion.txt.
#
#  2. RESULTS go to eta_results_hpc_confusion/, not eta_results_hpc/, so this
#     job cannot collide with a running eta_ladder_array job and the merge
#     back home can be done independently.
#
#  3. NO ETA_MAPS_FLOAT64 BRANCH.  The confusion generator ignores
#     map_dtype() and always returns float64 (noise_lib/confusion.py, "Band
#     regularisation"): float32 storage moves the floorless compact sigma_MF
#     by 21%.  The A/B test that variable exists for is therefore not
#     applicable here, and offering it would only invite a meaningless run.
#     The job asserts the dtype instead.
#
#  4. TWO-STAGE LADDER, so a wallclock kill is survivable.  run_eta_campaign
#     writes the JSON only after a cell's ENTIRE rung list completes, so a job
#     that dies at 11:59 loses every rung it computed.  This script therefore
#     runs the shallow rungs first (they save at ~1.2 h) and the deep rungs
#     second, merging at rung level into the same file from the same task -- no
#     race, since the two stages are sequential.  Costs one extra ensemble
#     regeneration (~2 min) and buys a checkpoint.
#
#  5. WALLCLOCK 11:45, not 18:00.  The 18 h inherited from
#     eta_ladder_array.slurm is over the 12 h limit on this system.
#
#  6. A PRE-FLIGHT GATE.  tools/validate_confusion.py must pass on the
#     machine that produces the numbers, exactly as run_null_tests.py gates
#     the rest of the campaign.  Task 1 runs it; the others skip it (it is
#     the same ensemble on the same node image).  Set CONF_SKIP_GATE=1 to
#     bypass.
#
# Submit:            sbatch eta_rung_confusion.slurm            # cells 1-4
# Smoke test first:  ETA_SMOKE=1 sbatch --array=1 --time=00:30:00 eta_rung_confusion.slurm
# Deep repeats:      sbatch --array=5-6 eta_rung_confusion.slurm
# ==========================================================================

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"     # numpy FFTs in the ensemble generation

cd "${BTB_HOME:-$SLURM_SUBMIT_DIR}"      # repository root; submit from there or export BTB_HOME
mkdir -p logs results/eta_results_hpc_confusion results/tests_and_results
if [ -n "${BTB_VENV_ACTIVATE:-}" ]; then source "$BTB_VENV_ACTIVATE"; fi

# ----- pick this task's cell (skip comment / blank lines) -------------------
CELL=$(grep -v '^\s*#' scripts/hpc/cells_confusion.txt | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$CELL" ]; then
    echo "no cell for array index $SLURM_ARRAY_TASK_ID"; exit 1
fi
read -r MODEL TEMPLATE RUNGS EPOCHS RESTARTS N_ENS N_EXTRA <<< "$CELL"

OUT=results/eta_results_hpc_confusion
if [ "${ETA_SMOKE:-0}" = "1" ]; then
    EPOCHS=2; RESTARTS=1; N_ENS=6000; N_EXTRA=6000; OUT=results/eta_results_smoke_confusion
    echo "*** SMOKE MODE: epochs=2 restarts=1 n_ens=6000 -> $OUT ***"
fi

echo "========================================="
echo "cell $SLURM_ARRAY_TASK_ID: $MODEL / $TEMPLATE"
echo "  rungs=$RUNGS  epochs=$EPOCHS  restarts=$RESTARTS"
echo "  n_ens=$N_ENS  n_extra=$N_EXTRA  (n_eval = n_ens/4 = $((N_ENS / 4)))"
echo "node $(hostname)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
python -c "import torch, platform; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), platform.machine())"

# ----- dtype assertion: this rung is only meaningful in float64 -------------
python - <<'PY'
import sys
sys.path.insert(0, '.')
from eta_pipeline import models as em
import numpy as np
m, _ = em.REGISTRY['T2_CONFUSION']['gen'](2, 1)
assert m.dtype == np.float64, f'confusion maps came back as {m.dtype}, expected float64'
print(f'dtype check OK: {m.dtype};  floor(T2_CONFUSION_WN) = '
      f'{em.REGISTRY["T2_CONFUSION_WN"]["floor"]:.4f} mJy/beam '
      f'(f_N = {em.CONFUSION_F_N})')
PY
if [ $? -ne 0 ]; then echo "dtype assertion FAILED — aborting"; exit 2; fi

# ----- pre-flight gate (task 1 only) ---------------------------------------
if [ "$SLURM_ARRAY_TASK_ID" = "1" ] && [ "${CONF_SKIP_GATE:-0}" != "1" ] \
   && [ "${ETA_SMOKE:-0}" != "1" ]; then
    echo "--- pre-flight: scripts/tools/validate_confusion.py ---"
    python -u scripts/tools/validate_confusion.py --n-ens 8000 --out results/tests_and_results
    if [ $? -ne 0 ]; then
        echo "VALIDATION GATE FAILED — not running the ladder"; exit 3
    fi
    echo "--- gate passed ---"
fi
echo "========================================="

# patience / min-epochs use the 2026-09-02 defaults: max(15, epochs//3), min(30, epochs)
SHALLOW=$(echo "$RUNGS" | tr ',' '\n' | grep -E '^(linear|reweight|quadratic)$' | paste -sd, -)
DEEP=$(echo    "$RUNGS" | tr ',' '\n' | grep -E '^(cubic|cnn)$'                 | paste -sd, -)

run_stage () {   # $1 = comma-separated rungs, $2 = stage label
    [ -z "$1" ] && { echo "stage $2: nothing to do"; return 0; }
    echo "----- stage $2: rungs=$1   $(date) -----"
    srun python -u scripts/run_eta_campaign.py \
            --mode variational2 \
            --models "$MODEL" \
            --templates "$TEMPLATE" \
            --rungs "$1" \
            --epochs "$EPOCHS" \
            --restarts "$RESTARTS" \
            --n-ens "$N_ENS" \
            --n-extra "$N_EXTRA" \
            --out "$OUT" \
            --force
    echo "----- stage $2 saved: $(date) -----"
}

run_stage "$SHALLOW" "shallow"     # ~1.2 h at n_fit 10800 x 6 restarts; CHECKPOINT
run_stage "$DEEP"    "deep"        # ~4.1 h; merges into the same JSON at rung level

echo "cell $SLURM_ARRAY_TASK_ID done: $(date)"
