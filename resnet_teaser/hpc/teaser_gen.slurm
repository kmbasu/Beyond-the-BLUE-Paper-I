#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --threads-per-core=1
#SBATCH --job-name=teaser_gen
#SBATCH --output=%x-out-gen.%j
#SBATCH --error=%x-err-gen.%j
#SBATCH --time=01:30:00
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
# NOTE (public release): a template for a Slurm cluster with NVIDIA GPUs (the production
# runs used GH200 nodes, one GPU per task unless stated).  Replace <your-account>,
# <your-partition> and the GPU request with your site's values, and set
# BTB_VENV_ACTIVATE to the activate script of a Python environment that has this
# repository installed (pip install -e .).
# ==========================================================================
# Generate the campaign-exact teaser datasets (make_teaser_dataset.py) on the cluster.
# CPU-only numpy work (~100 maps/s single-threaded: 36k maps per model ~ 6-8 min
# each), but submitted to a GPU partition because that is what the account had;
# it also runs fine on a login node if the CPU-time limit allows:
#     cd resnet_teaser && TEASER_HOME=$PWD bash hpc/teaser_gen.slurm
# Output: $TEASER_DATA/TEASER_<model>_train30k_L<L>_s<seed>.h5 (~2 GB)
#         $TEASER_DATA/TEASER_<model>_evalnoise6000_s<seed>.h5 (~0.4 GB)
#         + a *_manifest.json with the sigma_MF checks.
# ==========================================================================
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# ----- locations (edit here only) -----
# TEASER_HOME : the resnet_teaser/ directory (scripts, hpc/, checkpoints/, eval_out/, logs/)
# TEASER_DATA : where the HDF5 datasets live (a project/scratch data area); both can also be exported before sbatch
TEASER_HOME=${TEASER_HOME:-$SLURM_SUBMIT_DIR}          # the resnet_teaser/ directory of this repository
DATA=${TEASER_DATA:-$TEASER_HOME/teaser_data}
cd "$TEASER_HOME" || { echo "TEASER_HOME=$TEASER_HOME not found"; exit 1; }
mkdir -p logs
if [ -n "${BTB_VENV_ACTIVATE:-}" ]; then source "$BTB_VENV_ACTIVATE"; fi
mkdir -p "$DATA"
echo "node $(hostname)  $(date)"; python -c "import numpy, h5py; print('numpy', numpy.__version__, 'h5py', h5py.__version__)"

set -e
# T1: red + white (theorem case), L = 5  (sigma_MF = 0.863 -> r = 0.60)
python -u make_teaser_dataset.py --model T0_RED_REAL  --L 5 --n-train 30000 --n-eval 6000 --n-psd 2000 --outdir "$DATA"
# T2: spectral-tilt mixture + floor, L = 7 (sigma_MF exact = 1.329 -> r = 0.66)
python -u make_teaser_dataset.py --model T1_PSRAND_WN --L 7 --n-train 30000 --n-eval 6000 --n-psd 2000 --outdir "$DATA"
# optional T0: white noise at sigma_w = 9 (r = 0.75; the registry's sigma_w = 1 gives r = 0.08)
# python -u make_teaser_dataset.py --model T0_WHITE --white-sigma 9 --L 5 --n-train 30000 --n-eval 6000 --n-psd 2000 --outdir "$DATA"
ls -la "$DATA"
echo "done $(date)"
