## ############################################################## ##
## ResNet Training for HDF5 Data: DDP Version -- TEASER build      ##
## (Paper I Sec. 6.1 teaser runs)                                  ##
## ############################################################## ##
"""
train_teaser_ddp.py -- derived 2026-09-05 from train_emaloss_ddp.py.

Purpose
-------
Train the ResNet amplitude regressor on the campaign-exact datasets written by
make_teaser_dataset.py (T0_RED_REAL, L = 5; T1_PSRAND_WN, L = 7), with either
the plain MSE loss or the EMA bias-corrected MSE loss, so that the result can be
compared with the eta-campaign reference lines (sigma_MF, [1+g']^2 envelope,
reweight rung, exact ceiling, Method-E).

Changes relative to train_emaloss_ddp.py (everything else is identical):
  1. imports utils_teaser_ddp: float32 data path (no uint8/PIL quantisation),
     exact D4 augmentation (flips + rot90) instead of RandomRotation(180);
  2. the bias-correction bins (bias_corrected_mse AND ema_bias_corrected_mse) span
     [0, L] with L read from the HDF5 attrs (override with --ema-ymax; --ema-nbins
     for the count); defaults for bias_corrected_mse follow train_biascorr_ddp.py
     (lambda 15, 8 bins), for the EMA loss train_emaloss_ddp.py (lambda 10, 10 bins);
     --lambda-cond overrides either;
  3. the checkpoint additionally stores the normalisation constants
     (min/max), L, the dataset attributes (params_json) and the generator seeds,
     so teaser_eval.py can reproduce the input map exactly;
  4. --no-augment switch.

Usage (single GPU / Apple-silicon mps / CPU; for multi-GPU DDP on a Slurm
cluster see resnet_teaser/hpc/teaser_train.slurm):
    python train_teaser_ddp.py --h5 TEASER_T0_RED_REAL_train30k_L5_s880034.h5 \
        --checkpoint TEASER_T0_RED_REAL_mse.pth --loss-function mse \
        --epochs 700 --batch-size 50 --lr 0.01 --seed 40 --hpc
"""

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import datetime

import pandas as pd
import numpy as np
import pickle
import copy
from timeit import default_timer as timer
from scipy import stats
import sys
from pathlib import Path
import os
import h5py

# ===== DDP IMPORTS =====
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ==== UTILITIES IMPORT ====
from utils_teaser_ddp import *


# ===== PARSE ARGUMENTS =====
args = parse_args()


# ===== DDP INITIALIZATION =====
def init_ddp():
    """Initialize DDP (torchrun environment variables) or fall back to one device."""
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        dist.init_process_group(backend='cpu:gloo,cuda:nccl')

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        print(f"[Rank {rank}/{world_size}] Initialized on {device} (local_rank={local_rank})")
        return True, rank, local_rank, world_size, device
    else:
        print("Running in single-device mode (no DDP)")
        device = get_default_device()
        return False, 0, 0, 1, device


is_distributed, rank, local_rank, world_size, device = init_ddp()


# ===== HELPER FUNCTIONS =====
def print0(*args, **kwargs):
    """Print only on rank 0."""
    if rank == 0:
        print(*args, **kwargs)


def save0(*args, **kwargs):
    """Save only on rank 0."""
    if rank == 0:
        torch.save(*args, **kwargs)


# ===== EXTRACT ARGUMENTS =====
batch_size = args.batch_size
num_epochs = args.epochs
initial_lr = args.lr
h5_file_path = args.h5
saved_checkpoint = args.checkpoint
absolute_threshold = args.absolute_threshold
relative_threshold = args.relative_threshold
val_split = args.val_split
HPC = args.hpc
log_dir = args.log_dir
load_to_memory = args.load_to_memory
leaky_relu_slope = args.leaky_relu_slope

# This is kept separate from the generator seed so it can be used in filenames
# and log output regardless of whether --seed was explicitly passed.
seed_value = args.seed

use_softplus_variance = args.use_softplus_variance
log_var_min = args.log_var_min   # default -8.0 from parse_args
log_var_max = args.log_var_max   # default  8.0 from parse_args

# CHECKPOINT RESUME
resume_checkpoint = args.resume

# EMA BIAS SUMMARY
bias_summary_freq = args.bias_summary_freq

# DIAGNOSTIC SETTINGS
diagnostic_mode = args.diagnostic_mode
diagnostic_frequency = args.diagnostic_frequency
enable_health_monitor = not args.disable_health_monitor

# ===== DIAGNOSTIC MODE ANNOUNCEMENT =====
if diagnostic_mode:
    print0("\n" + "!"*70)
    print0("!!! DIAGNOSTIC MODE ENABLED !!!")
    print0(f"!!! Full diagnostics will run every {diagnostic_frequency} batches")
    print0("!!! Quick checks will run on EVERY batch")
    print0("!"*70 + "\n")

# ===== LOSS FUNCTION CONFIGURATION =====
loss_function = args.loss_function
if args.use_weighted_loss:
    print0("WARNING: --use-weighted-loss is deprecated. Use --loss-function weighted_mse instead.")
    if loss_function == "mse":
        loss_function = "weighted_mse"

# "seitzer_beta_nll" has been replaced with "original_beta_nll" throughout
is_heteroscedastic = loss_function in [
    "beta_nll", "gaussian_nll", "original_beta_nll", "slope_penalized_nll"
]
beta_param = args.beta if loss_function in [
    "beta_nll", "original_beta_nll", "slope_penalized_nll"
] else None


print0(f"Using device: {device}")
print0(f"World size: {world_size}, Rank: {rank}")

# ===== CUDA OPTIMIZATIONS =====
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print0("Enabled TF32 acceleration for matrix operations")
    except Exception:
        print0("TF32 not available")
    print0("CUDA optimizations enabled")


# ===== LOAD DATASET =====
print0(f"Loading dataset from: {h5_file_path}")
print0(f"Memory loading mode: {'ENABLED' if load_to_memory else 'DISABLED (lazy loading)'}")

dataset = HDF5ImageDataset(
    h5_path=h5_file_path,
    image_key="noisy",
    target_key="I0",
    transform=None,
    load_to_memory=load_to_memory
)
print0(f"Dataset loaded: {len(dataset)} images")


# ===== HPC LOG REDIRECTION =====
def redirect_output(log_dir: Path | None = None) -> None:
    """Send stdout/err to a timestamped log file (only on rank 0)."""
    if rank != 0:
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path.cwd() if log_dir is None else log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # TEASER: array tasks can start within the same second -> include the checkpoint
    # stem and the Slurm job id so two ranks-0 never share a log file
    jid = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    log_file = log_dir / f"printlog_{Path(saved_checkpoint).stem}_{jid}_{ts}.log"
    sys.stderr = sys.stdout = open(log_file, "w", buffering=1)
    print(f"[INFO] HPC flag active – logging to {log_file}\n", flush=True)


if HPC:
    redirect_output(Path(log_dir) if log_dir else None)


# ===== CONFIGURATION SUMMARY =====
print0("\n" + "="*60)
print0("TRAINING CONFIGURATION")
print0("="*60)
print0(f"HDF5 file:          {h5_file_path}")
print0(f"Load to memory:     {load_to_memory}")
print0(f"Batch size per GPU: {batch_size}")
if is_distributed:
    print0(f"Total effective batch size: {batch_size * world_size}")
    print0(f"Number of GPUs:     {world_size}")
print0(f"Epochs:             {num_epochs}")
print0(f"Initial LR:         {initial_lr}")
print0(f"Validation split:   {val_split}")
print0(f"Absolute threshold: {absolute_threshold}")
print0(f"Relative threshold: {relative_threshold} ({relative_threshold*100:.0f}%)")
print0(f"Random seed:        {seed_value}")
print0(f"Loss function:      {loss_function}")
if is_heteroscedastic:
    print0(f"  → Heteroscedastic regression (predicting mean + uncertainty)")
    if loss_function in ("beta_nll", "original_beta_nll", "slope_penalized_nll"):
        print0(f"  → β parameter: {beta_param}")
print0(f"Output checkpoint:  {saved_checkpoint}")
if resume_checkpoint:
    print0(f"Resume from:        {resume_checkpoint}")
if loss_function == "ema_bias_corrected_mse":
    print0(f"Bias summary freq:  every {bias_summary_freq} epochs")
print0("="*60)

# [FIX S3] Stability settings — now includes log-var bounds and softplus status
print0(f"\n--- STABILITY SETTINGS ---")
print0(f"LeakyReLU negative slope: {leaky_relu_slope}")
print0(f"Health monitoring:        {'ENABLED' if enable_health_monitor else 'DISABLED'}")
print0(f"Gradient clipping:        max_norm=200 (emergency clipping)") 
if is_heteroscedastic:
    print0(f"Log-variance bounds:      [{log_var_min}, {log_var_max}]")
    print0(f"Softplus variance:        {'ENABLED' if use_softplus_variance else 'DISABLED'}")
if diagnostic_mode:
    print0(f"Diagnostic mode:          ENABLED (full diagnostics every {diagnostic_frequency} batches)")
else:
    print0("Diagnostic mode:          DISABLED")
print0("="*60 + "\n")


# ===== IMAGE TRANSFORMS =====
# Exact dihedral-group augmentation (flips + 90-degree rotations) for the
# ISOTROPIC teaser models; no resampling, no corner fill, DC untouched.
# For ANISOTROPIC noise pass --no-augment (or restrict the group).
train_transform = D4Augment(enabled=not args.no_augment)
test_transform = NoAugment()
print0(f"Augmentation: {'D4 (flips + rot90)' if not args.no_augment else 'none'}")


# ===== DATA VALIDATION =====
print0("Validating dataset for NaN/inf values...")
sample_image, sample_target = dataset[0]
if sample_image is not None:
    if torch.isnan(sample_image).any() or torch.isinf(sample_image).any():
        print0("WARNING: Dataset contains NaN or inf values in images!")
    print0(f"Sample tensor: shape {tuple(sample_image.shape)}, dtype {sample_image.dtype}")
if torch.isnan(sample_target).any() or torch.isinf(sample_target).any():
    print0("WARNING: Dataset contains NaN or inf values in targets!")


# ===== REPRODUCIBLE TRAIN/VAL SPLIT =====
seed = getattr(args, "seed", None)
if seed is not None:
    generator = torch.Generator().manual_seed(seed)
    print0(f"Using random seed: {seed}")
else:
    generator = torch.Generator()
    print0("No seed specified — using a random split")

train_size = int((1 - val_split) * len(dataset))
test_size = len(dataset) - train_size

indices = torch.randperm(len(dataset), generator=generator)
train_indices, test_indices = indices[:train_size], indices[train_size:]


# ===== DATASET ATTRIBUTES (TEASER) =====
with h5py.File(h5_file_path, 'r') as _f:
    h5_attrs = {k: (v.item() if hasattr(v, 'item') else v) for k, v in _f.attrs.items()}
    h5_attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h5_attrs.items()}
dataset_L = float(h5_attrs.get('L', float('nan')))
dataset_model = str(h5_attrs.get('model_name', h5_attrs.get('model', 'unknown')))
print0(f"Dataset model: {dataset_model};  L (prior upper edge) = {dataset_L};  "
       f"master_seed = {h5_attrs.get('master_seed', 'n/a')}")


# ===== NORMALIZATION FROM RAW DATA =====
print0("Computing normalization parameters from RAW training data...")
try:
    with h5py.File(h5_file_path, 'r') as f:
        if 'min' in f['noisy'].attrs and 'max' in f['noisy'].attrs:
            min_val = float(f['noisy'].attrs['min'])
            max_val = float(f['noisy'].attrs['max'])
            print0("✓ Using normalization from HDF5 file attributes")
        else:
            print0("HDF5 attributes not found, computing from raw training data...")
            train_images_raw = dataset.get_images(train_indices.numpy())
            min_val = float(train_images_raw.min())
            max_val = float(train_images_raw.max())
            del train_images_raw
except Exception as e:
    print0(f"Could not read attributes ({e}), computing from raw training data...")
    train_images_raw = dataset.get_images(train_indices.numpy())
    min_val = float(train_images_raw.min())
    max_val = float(train_images_raw.max())
    del train_images_raw

print0(f"Normalization range (RAW data): [{min_val:.6f}, {max_val:.6f}]")
dataset.set_normalization(min_val, max_val)


# ===== TARGET STATISTICS =====
print0("Computing target statistics for model initialization...")
train_targets = dataset.get_targets(train_indices.numpy())
target_mean = float(train_targets.mean())
target_std = float(train_targets.std())
print0(f"Target mean: {target_mean:.4f}, std: {target_std:.4f}")

# Always-on data diagnostics (regardless of --diagnostic-mode)
print_data_diagnostics(train_targets, rank=rank)


# ===== APPLY IMAGE TRANSFORMS =====
train_base = copy.copy(dataset)
train_base.transform = train_transform
train_dataset = Subset(train_base, train_indices)

test_base = copy.copy(dataset)
test_base.transform = test_transform
test_dataset = Subset(test_base, test_indices)


# ===== NOISE RMS FROM BOTTOM QUARTILE =====
print0("Computing noise RMS from training set bottom quartile...")
train_targets = dataset.get_targets(train_indices.numpy())

pct25 = np.percentile(train_targets, 25)
noise_idx = train_indices.numpy()[train_targets < pct25]

noisy_imgs = dataset.get_images(noise_idx)
if noisy_imgs.ndim == 4:
    noisy_imgs = noisy_imgs[:, 0]

noise_pixels = noisy_imgs.ravel()
noise_rms = 1.48 * stats.median_abs_deviation(noise_pixels)
print0(f"Noise RMS from MAD of all pixels: {noise_rms:.4f}")


# ===== DATA LOADERS =====
if is_distributed:
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed if hasattr(args, 'seed') else 42
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    shuffle_train = False
else:
    train_sampler = None
    test_sampler = None
    shuffle_train = True

train_batch_size = min(batch_size, len(train_dataset))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    shuffle=shuffle_train,
    sampler=train_sampler,
    num_workers=4 if not load_to_memory else 2,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=True,
    worker_init_fn=dataset.worker_init_fn if not load_to_memory else None
)

test_batch_size = min(32, len(test_dataset))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    sampler=test_sampler,
    num_workers=4 if not load_to_memory else 2,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=True,
    worker_init_fn=dataset.worker_init_fn if not load_to_memory else None
)
print0(f"Train batch size: {train_batch_size} per GPU;  "
       f"test batch size: {test_batch_size}")


# ===== MODEL INITIALIZATION =====

model = ResNet(
    ResidualBlock,
    [3, 4, 6, 3],
    width_multiplier=1.0,
    heteroscedastic=is_heteroscedastic,
    negative_slope=leaky_relu_slope,
    log_var_bounds=(log_var_min, log_var_max),
    use_softplus_var=use_softplus_variance,
).to(device)


# STABILITY: Initialize final layer with small weights and bias at target mean
print0(f"Initializing final layer: bias={target_mean:.4f}, weight_std=0.01")
initialize_regression_head(model, target_mean=target_mean, weight_std=0.01)

if is_distributed:
    model = DDP(model, device_ids=[local_rank])
    print0("Model wrapped with DistributedDataParallel")

param_count = sum(p.numel() for p in (model.module if is_distributed else model).parameters())
model_type = "Heteroscedastic" if is_heteroscedastic else "Standard"
print0(f"{model_type} ResNet initialized with {param_count:,} parameters")


# ===== METRIC STORAGE =====
train_losses = []
train_accuracies = []
train_relative_accuracies = []
test_losses = []
test_accuracies = []
test_relative_accuracies = []
learning_rates = []

# Keep unweighted MSE for reference diagnostics (separate from criterion)
unweighted_mse = nn.MSELoss()


# ===== LOSS FUNCTION SELECTION =====
print0(f"\n{'='*60}")
print0("LOSS FUNCTION CONFIGURATION")
print0(f"{'='*60}")

if loss_function == "mse":
    criterion = nn.MSELoss()
    print0("Using standard MSELoss")

elif loss_function == "weighted_mse":
    low_thresh = 0.5
    high_thresh = 1.5
    w_low = 2.0
    w_high = 2.0
    over_weight_low = 5.0
    under_weight_high = 5.0
    criterion = ExtremesWeightedMSELoss(
        low_thresh=low_thresh,
        high_thresh=high_thresh,
        w_low=w_low,
        w_high=w_high,
        smooth=True,
        tau=0.1,
        directional=True,
        over_weight_low=over_weight_low,
        under_weight_high=under_weight_high,
        reduction="mean",
    )
    print0("Using ExtremesWeightedMSELoss")
    print0(f"  Parameters: low_thresh={low_thresh:.2f}, high_thresh={high_thresh:.2f}")
    print0(f"  Weights: w_low={w_low:.2f}, w_high={w_high:.2f}")
    print0(f"  Directional: over_weight_low={over_weight_low:.2f}, "
           f"under_weight_high={under_weight_high:.2f}")

elif loss_function == "slope_penalized_mse":
    lambda_slope = 30.0
    criterion = SlopePenalizedMSELoss(lambda_slope=lambda_slope)
    print0("Using slope-penalized MSE loss (penalizes deviation from slope=1)")
    print0("    Formula: L = MSE + λ * (1 − slope)²  (slope estimated in-batch)")
    print0(f"    Using λ = {lambda_slope:.1f}")

elif loss_function == "bias_corrected_mse":
    # defaults = train_biascorr_ddp.py (the production script of the CNN-vs-MF series): lambda 15, 8 bins;
    # train_emaloss_ddp.py had 1.0 / 8 bins over [0, 4] for this loss
    lambda_cond = 15.0 if args.lambda_cond is None else float(args.lambda_cond)
    n_bins      = 8 if args.ema_nbins is None else args.ema_nbins
    y_min       = 0.0
    # TEASER: bins span the prior [0, L] (legacy biascorr: 8 bins over [0, 5])
    if args.ema_ymax is not None:
        y_max = float(args.ema_ymax)
    elif np.isfinite(dataset_L):
        y_max = dataset_L
    else:
        y_max = 4.0
    criterion = ConditionalBiasPenalizedMSELoss(
        lambda_cond=lambda_cond,
        n_bins=n_bins,
        y_min=y_min,
        y_max=y_max,
    )
    print0("Using conditional-bias-penalized MSE loss")
    print0("    Formula: L = MSE + λ * mean_k[ (mean residual in bin k)² ]")
    print0(f"    λ_cond={lambda_cond:.1f},  {n_bins} bins over [{y_min}, {y_max}]")

elif loss_function == "ema_bias_corrected_mse":
    lambda_cond = 10.0 if args.lambda_cond is None else float(args.lambda_cond)
    n_bins      = 10 if args.ema_nbins is None else args.ema_nbins
    y_min       = 0.0
    # TEASER: bins must span the prior [0, L]; the legacy value 5.0 is the fallback
    if args.ema_ymax is not None:
        y_max = float(args.ema_ymax)
    elif np.isfinite(dataset_L):
        y_max = dataset_L
    else:
        y_max = 5.0
    ema_decay   = 0.99          # effective window ≈ 1/(1−0.99) = 100 steps
    criterion = EMABasedBiasPenalizedMSELoss(
        lambda_cond = lambda_cond,
        n_bins      = n_bins,
        y_min       = y_min,
        y_max       = y_max,
        ema_decay   = ema_decay,
        dist_sync   = True,     # all-reduce bin stats across GPUs before EMA update
    )
    print0("Using EMA-based conditional-bias-penalized MSE loss")
    print0("    Formula: L = MSE + (λ/K') Σ_k [ 2·μ_EMA·r̄_batch − μ_EMA² ]")
    print0("    Gradient direction ∝ μ_k^EMA (stable) rather than r̄_k^batch (noisy)")
    print0(f"    λ_cond={lambda_cond:.1f},  {n_bins} bins over [{y_min}, {y_max}],  "
           f"EMA decay β={ema_decay}")
    
#---------------------------------------------------------------------------------------------------

elif loss_function == "beta_nll":
    criterion = BetaNLLLoss(beta=beta_param, reduction='mean')
    print0(f"Using BetaNLLLoss (β-NLL) with β={beta_param}")
    print0("  Heteroscedastic regression: predicting both mean and uncertainty")
    print0("  β < 1: emphasize accuracy; β = 1: balanced; β > 1: emphasize calibration")

elif loss_function == "gaussian_nll":
    criterion = GaussianNLLLoss(reduction='mean')
    print0("Using GaussianNLLLoss (standard NLL, equivalent to β=1.0)")
    print0("  Heteroscedastic regression: predicting both mean and uncertainty")

elif loss_function == "original_beta_nll":
    criterion = OriginalbetaNLLLoss(beta=beta_param, reduction='mean')
    print0("Using Seitzer+ betaNLL Loss (with stop-gradient function)")
    print0("  Heteroscedastic regression: predicting both mean and uncertainty")

elif loss_function == "slope_penalized_nll":
    lambda_slope = 20.0
    criterion = SlopePenalizedBetaNLLLoss(
        beta=beta_param,
        lambda_slope=lambda_slope,
        lambda_bias=0.0,
        min_target_var=1e-3,
        verbose=False
    )
    print0("Using β-NLL with slope penalty")
    print0("  Loss = β-NLL + λ_slope * (1 − slope)² + λ_bias * bias²")
    print0(f"  β = {beta_param}, λ_slope = {lambda_slope}")
    print0("  Diagnostics will print every 50 forward calls or when loss is problematic")

else:
    raise ValueError(f"Unknown loss function: '{loss_function}'")

# ── Move criterion to the correct device ─────────────────────────────
# This is essential for loss functions with registered buffers (e.g.
# EMABasedBiasPenalizedMSELoss whose bin_mean_ema and bin_update_count
# must live on the same device as the predictions/targets).  Harmless
# for stateless losses like nn.MSELoss.
criterion = criterion.to(device)

print0(f"{'='*60}\n")


# ===== SCALE LEARNING RATE FOR DDP =====
if is_distributed:
    scaled_lr = initial_lr * (world_size ** 0.25)
    print0(f"Scaling learning rate: {initial_lr} → {scaled_lr:.4f}")
else:
    scaled_lr = initial_lr


# ===== METRIC REDUCTION HELPER =====
def reduce_metrics(metrics_dict: dict) -> dict:
    """All-reduce metrics across all DDP ranks."""
    if not is_distributed:
        return metrics_dict
    metrics_tensor = torch.tensor(
        list(metrics_dict.values()), dtype=torch.float32, device=device
    )
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    return {key: metrics_tensor[i].item() for i, key in enumerate(metrics_dict.keys())}


# ===== OPTIMIZER AND SCHEDULER =====
optimizer = torch.optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)


# ===== CHECKPOINT RESUME =====
start_epoch = 0
if resume_checkpoint and os.path.isfile(resume_checkpoint):
    print0(f"Resuming from checkpoint: {resume_checkpoint}")
    ckpt_resume = torch.load(resume_checkpoint, map_location=device)

    # Model state
    base_model = model.module if is_distributed else model
    base_model.load_state_dict(ckpt_resume['model_state_dict'])
    print0("  ✓ Model state restored")

    # Optimizer state
    if 'optimizer_state_dict' in ckpt_resume:
        optimizer.load_state_dict(ckpt_resume['optimizer_state_dict'])
        print0("  ✓ Optimizer state restored")

    # Scheduler state
    if 'scheduler_state_dict' in ckpt_resume:
        scheduler.load_state_dict(ckpt_resume['scheduler_state_dict'])
        print0("  ✓ Scheduler state restored")

    # Criterion state (EMA buffers, update counts, etc.)
    if 'criterion_state_dict' in ckpt_resume:
        criterion.load_state_dict(ckpt_resume['criterion_state_dict'])
        print0("  ✓ Criterion state restored")

    # Resume epoch
    start_epoch = ckpt_resume.get('epoch', 0)
    print0(f"  Resuming from epoch {start_epoch}")

    # Restore metric histories
    if rank == 0:
        train_losses = ckpt_resume.get('train_losses', [])
        train_accuracies = ckpt_resume.get('train_accuracies', [])
        train_relative_accuracies = ckpt_resume.get('train_relative_accuracies', [])
        test_losses = ckpt_resume.get('test_losses', [])
        test_accuracies = ckpt_resume.get('test_accuracies', [])
        test_relative_accuracies = ckpt_resume.get('test_relative_accuracies', [])
        learning_rates = ckpt_resume.get('learning_rates', [])

    del ckpt_resume  # free memory

elif resume_checkpoint:
    print0(f"WARNING: --resume file not found: {resume_checkpoint}.  Training from scratch.")


start_time = timer()
best_val_loss, best_epoch = float('inf'), 0      # TEASER: best-validation tracking


# ===== DIAGNOSTICS AND HEALTH MONITOR =====
diagnostics = TrainingDiagnostics(rank=rank, enable_detailed=diagnostic_mode)

if enable_health_monitor:
    health_monitor = TrainingHealthMonitor(
        patience=5,
        min_pred_std=0.01,
        min_grad_active_ratio=0.3,
        rank=rank
    )
    print0("Health monitoring ENABLED (patience=5, auto-recovery on collapse)")
else:
    health_monitor = None
    print0("Health monitoring DISABLED")

# Health_check_frequency defined here, before the training loop.
# Running every 50 batches balances responsiveness vs. overhead.
health_check_frequency = 50


# ===== TRAINING LOOP =====
print0("\nStarting training...")
print0("RelAcc = Relative Accuracy (predictions within ±{}% of true value)".format(
    int(relative_threshold * 100)))
print0("-" * 70)

try:
    for epoch in range(start_epoch, num_epochs):
        batch_accuracies = []

        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ------------------------------------------------------------------ #
        #  TRAINING PHASE
        # ------------------------------------------------------------------ #
        model.train()
        running_loss = 0.0
        correct = 0
        relative_correct = 0
        total = 0

        # Fix: enumerate() provides batch_idx used throughout the loop.
        for batch_idx, (images, signal_amplitudes) in enumerate(train_loader):
            images = images.to(device)
            signal_amplitudes = signal_amplitudes.to(device).float()

            # Log input ranges on the very first batch of training
            if epoch == 0 and total == 0:
                print0(f"First batch input range:  "
                       f"[{images.min().item():.4f}, {images.max().item():.4f}]")
                print0(f"First batch target range: "
                       f"[{signal_amplitudes.min().item():.4f}, "
                       f"{signal_amplitudes.max().item():.4f}]")

            # ---- Check for NaN in inputs ----
            has_nan = torch.isnan(images).any() or torch.isnan(signal_amplitudes).any()
            if is_distributed:
                has_nan_tensor = torch.tensor([1.0 if has_nan else 0.0], device=device)
                dist.all_reduce(has_nan_tensor, op=dist.ReduceOp.MAX)
                has_nan = has_nan_tensor.item() > 0
            if has_nan:
                print0(f"WARNING: NaN in batch inputs at epoch {epoch+1}, batch {batch_idx} — skipping")
                continue

            # ---- Forward pass ----
            if is_heteroscedastic:
                mean_pred, log_var_pred = model(images)
                loss = criterion(mean_pred, log_var_pred, signal_amplitudes)
                outputs = mean_pred

                # Uncertainty monitoring runs only for the diagnostic_mode
                # for heteroscedastic losses, every 50 batches.
                if diagnostic_mode and batch_idx % 50 == 0:
                    with torch.no_grad():
                        mean_uncertainty = torch.sqrt(torch.exp(log_var_pred)).mean().item()
                        max_log_var = log_var_pred.max().item()
                        min_log_var = log_var_pred.min().item()
                        print0(f"  Mean σ: {mean_uncertainty:.4f},  "
                               f"log_var range: [{min_log_var:.2f}, {max_log_var:.2f}]")
                        if max_log_var > log_var_max * 0.9:
                            print0(f"  ⚠️  log_var approaching upper bound "
                                   f"({max_log_var:.2f} vs limit {log_var_max})")
            else:
                outputs = model(images)
                loss = criterion(outputs, signal_amplitudes)

            # ---- Diagnostic quick-check ----
            if diagnostic_mode:
                has_issues = diagnostics.quick_check(loss, outputs, signal_amplitudes)
                if has_issues:
                    print0(f"\n⚠️  Issues detected at epoch {epoch+1}, "
                           f"batch {batch_idx} — running full diagnostics...")
                    diagnostics.diagnose_batch(
                        model=model.module if is_distributed else model,
                        batch_input=images,
                        batch_target=signal_amplitudes,
                        loss=loss,
                        outputs=outputs,
                        epoch=epoch + 1,
                        batch_idx=batch_idx,
                        optimizer=None,
                        criterion=criterion
                    )

            # Occasional diagnostic for extreme weighting (not behind --diagnostic-mode)
            if epoch % 50 == 0 and total == 0:
                with torch.no_grad():
                    extreme_low = signal_amplitudes < 0.5
                    extreme_high = signal_amplitudes > 3.5
                    if extreme_low.any() or extreme_high.any():
                        print0(f"  Batch has {extreme_low.sum().item()} low / "
                               f"{extreme_high.sum().item()} high extremes,  "
                               f"loss = {loss.item():.4f}")

            # ---- Check for NaN in loss ----
            loss_is_nan = torch.isnan(loss).item()
            if is_distributed:
                nan_loss_tensor = torch.tensor([1.0 if loss_is_nan else 0.0], device=device)
                dist.all_reduce(nan_loss_tensor, op=dist.ReduceOp.MAX)
                loss_is_nan = nan_loss_tensor.item() > 0

            if loss_is_nan:
                print0(f"WARNING: NaN loss at epoch {epoch+1}, batch {batch_idx} — "
                       "falling back to MSE for this batch")
                # Explicit branch for heteroscedastic case
                if is_heteroscedastic:
                    loss = nn.functional.mse_loss(mean_pred, signal_amplitudes)
                else:
                    loss = nn.functional.mse_loss(outputs, signal_amplitudes)

                # ---- Synchronize the fallback-NaN decision across all ranks ----
                fallback_nan = torch.isnan(loss).item()
                if is_distributed:
                    fb_nan_t = torch.tensor([1.0 if fallback_nan else 0.0], device=device)
                    dist.all_reduce(fb_nan_t, op=dist.ReduceOp.MAX)
                    fallback_nan = fb_nan_t.item() > 0
                if fallback_nan:
                    print0("  Still NaN after MSE fallback — skipping batch")
                    continue

            optimizer.zero_grad()
            loss.backward()

            # ---- Full gradient diagnostic ----
            if diagnostic_mode and (batch_idx % diagnostic_frequency == 0):
                diagnostics.diagnose_batch(
                    model=model.module if is_distributed else model,
                    batch_input=images,
                    batch_target=signal_amplitudes,
                    loss=loss,
                    outputs=outputs,
                    epoch=epoch + 1,
                    batch_idx=batch_idx,
                    optimizer=optimizer,
                    criterion=criterion
                )

            # ---- Health monitoring ----
            if enable_health_monitor and (batch_idx % health_check_frequency == 0):
                base_model = model.module if is_distributed else model
                is_healthy, health_diag = health_monitor.check_health(outputs, base_model)

                # Synchronize the intervention decision: health_monitor uses per-rank outputs 
                # Use MAX reduction: if ANY rank detects collapse, ALL ranks act.
                needs_intervention = health_diag.get('needs_intervention', False)
                if is_distributed:
                    ni_t = torch.tensor([1.0 if needs_intervention else 0.0], device=device)
                    dist.all_reduce(ni_t, op=dist.ReduceOp.MAX)
                    needs_intervention = ni_t.item() > 0

                if needs_intervention:
                    print0(f"\n{'!'*70}")
                    print0(f"!!! COLLAPSE DETECTED at epoch {epoch+1}, batch {batch_idx} !!!")
                    print0(f"!!! Prediction std: {health_diag['pred_std']:.6f}")
                    print0(f"!!! Active gradients: {health_diag['grad_active_ratio']:.1%}")
                    print0(f"!!! Triggering emergency reinitialization...")
                    print0(f"{'!'*70}\n")

                    if is_distributed:
                        emergency_reinitialize(model.module, target_mean=target_mean,
                                               leaky_slope=leaky_relu_slope)
                    else:
                        emergency_reinitialize(model, target_mean=target_mean,
                                               leaky_slope=leaky_relu_slope)

                    # Recalibrate BatchNorm after reinitialization
                    print0("    Calibrating BatchNorm statistics after reinitialization...")
                    calibrate_batchnorm(
                        model.module if is_distributed else model,
                        train_loader,
                        device,
                        num_batches=100,
                        rank=rank
                    )

                    # TEASER FIX: make all replicas identical again.  emergency_reinitialize
                    # draws new random weights independently on every rank and
                    # calibrate_batchnorm uses each rank's own data shard; DDP synchronises
                    # gradients only, never parameters, so without this broadcast the ranks
                    # train four different models whose gradients are averaged (seen in the
                    # first cluster pass: T0_WHITE/mse had one intervention, after which the
                    # reduced validation loss was ~1e5-1e8 for the remaining 698 epochs while
                    # rank 0's own model evaluated sanely).
                    if is_distributed:
                        for tensor in list(model.module.parameters()) + list(model.module.buffers()):
                            dist.broadcast(tensor.data, src=0)
                        dist.barrier()
                        print0("    Broadcast rank-0 parameters and BatchNorm buffers to all ranks")

                    # Reset optimizer with reduced learning rate
                    recovery_lr = scaled_lr * 0.5
                    optimizer = torch.optim.SGD(
                        model.parameters(), lr=recovery_lr,
                        momentum=0.9, weight_decay=1e-4
                    )
                    remaining_epochs = num_epochs - epoch
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=remaining_epochs, eta_min=1e-6
                    )
                    print0(f"    Optimizer reset: LR={recovery_lr:.6f}, "
                           f"{remaining_epochs} epochs remaining")

                    health_monitor.reset()
                    continue   # Skip gradient update for this batch

            # ---- Gradient clipping ----
            if is_distributed:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.module.parameters(), max_norm=200
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=200
                )

            if grad_norm > 200 and batch_idx % 50 == 0:
                print0(f"  Gradient clipped: norm={grad_norm:.2f} → 200")

            optimizer.step()

            # ---- Accumulate metrics ----
            if not loss_is_nan:
                running_loss += loss.item() * images.size(0)
                total += signal_amplitudes.size(0)
                correct += (
                    (outputs - signal_amplitudes).abs() < absolute_threshold
                ).sum().item()
                relative_errors = (
                    (outputs - signal_amplitudes).abs() / (signal_amplitudes + 1e-8)
                )
                relative_correct += (relative_errors < relative_threshold).sum().item()

            # [FIX C6] Append to loss history every batch so stuck-loss detection works.
            if not loss_is_nan:
                diagnostics.loss_history.append(loss.item())

        # ---- End-of-epoch defensive check ----
        if total == 0:
            print0(f"WARNING: No valid batches in epoch {epoch+1}")
            running_loss = 0.0
            correct = 0
            relative_correct = 0
            total = 1

        # ---- Aggregate training metrics across ranks ----
        metrics = {
            'loss': running_loss, 'correct': correct,
            'rel_correct': relative_correct, 'total': total
        }
        metrics = reduce_metrics(metrics)

        if metrics['total'] > 0:
            avg_loss = metrics['loss'] / metrics['total']
            accuracy = 100.0 * metrics['correct'] / metrics['total']
            relative_accuracy = 100.0 * metrics['rel_correct'] / metrics['total']
        else:
            print0(f"ERROR: No samples processed in epoch {epoch+1}")
            avg_loss = float('inf')
            accuracy = 0.0
            relative_accuracy = 0.0

        if rank == 0:
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            train_relative_accuracies.append(relative_accuracy)

        # ------------------------------------------------------------------ #
        #  VALIDATION PHASE
        # ------------------------------------------------------------------ #
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_relative_correct = 0
        val_total = 0
        val_batch_losses = []
        extreme_val_batch_detected = False

        with torch.no_grad():
            for val_batch_idx, (images, signal_amplitudes) in enumerate(test_loader):
                images = images.to(device)
                signal_amplitudes = signal_amplitudes.to(device).float()

                if torch.isnan(images).any() or torch.isnan(signal_amplitudes).any():
                    continue

                if is_heteroscedastic:
                    mean_pred, log_var_pred = model(images)
                    loss = criterion(mean_pred, log_var_pred, signal_amplitudes)
                    outputs = mean_pred
                else:
                    outputs = model(images)
                    loss = criterion(outputs, signal_amplitudes)

                if torch.isnan(loss):
                    continue

                batch_loss_value = loss.item()
                val_batch_losses.append(batch_loss_value)

                # Diagnostic: detect extreme validation loss spikes
                if diagnostic_mode and batch_loss_value > 10.0:
                    if not extreme_val_batch_detected:
                        extreme_val_batch_detected = True
                        print0(f"\n{'!'*70}")
                        print0("!!! EXTREME VALIDATION LOSS DETECTED !!!")
                        print0(f"!!! Epoch {epoch+1}, Validation Batch {val_batch_idx}")
                        print0(f"!!! Batch Loss: {batch_loss_value:.4f}")
                        print0(f"{'!'*70}")
                        errors = (outputs - signal_amplitudes).abs()
                        print0(f"  Target range: [{signal_amplitudes.min():.4f}, "
                               f"{signal_amplitudes.max():.4f}]")
                        print0(f"  Output range: [{outputs.min():.4f}, "
                               f"{outputs.max():.4f}]")
                        worst_idx = errors.argmax()
                        print0(f"  Worst prediction: target={signal_amplitudes[worst_idx].item():.4f}, "
                               f"pred={outputs[worst_idx].item():.4f}, "
                               f"error={errors[worst_idx].item():.4f}")
                        print0(f"{'!'*70}\n")

                val_loss += batch_loss_value * images.size(0)
                val_total += signal_amplitudes.size(0)
                val_correct += (
                    (outputs - signal_amplitudes).abs() < absolute_threshold
                ).sum().item()
                relative_errors = (
                    (outputs - signal_amplitudes).abs() / (signal_amplitudes.abs() + 1e-8)
                )
                val_relative_correct += (relative_errors < relative_threshold).sum().item()

                batch_acc = (
                    (outputs - signal_amplitudes).abs() < absolute_threshold
                ).float().mean()
                batch_accuracies.append(batch_acc.item())

        # ---- Aggregate validation metrics across ranks ----
        val_metrics = {
            'loss': val_loss, 'correct': val_correct,
            'rel_correct': val_relative_correct, 'total': val_total
        }
        val_metrics = reduce_metrics(val_metrics)

        if val_metrics['total'] > 0:
            avg_val_loss = val_metrics['loss'] / val_metrics['total']
            val_accuracy = 100.0 * val_metrics['correct'] / val_metrics['total']
            val_relative_accuracy = 100.0 * val_metrics['rel_correct'] / val_metrics['total']
        else:
            avg_val_loss = float('inf')
            val_accuracy = 0.0
            val_relative_accuracy = 0.0

        if rank == 0:
            test_losses.append(avg_val_loss)
            test_accuracies.append(val_accuracy)
            test_relative_accuracies.append(val_relative_accuracy)

            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            batch_std = np.std(batch_accuracies) if batch_accuracies else 0.0

            # Diagnostic: check for validation loss spikes
            if diagnostic_mode and val_batch_losses:
                max_val_loss = max(val_batch_losses)
                mean_val_loss = np.mean(val_batch_losses)
                if max_val_loss > 10.0 or max_val_loss > mean_val_loss * 10:
                    print(f"\n⚠️  VALIDATION SPIKE WARNING:")
                    print(f"    Max batch loss:  {max_val_loss:.4f}")
                    print(f"    Mean batch loss: {mean_val_loss:.4f}")
                    print(f"    Ratio:           {max_val_loss / mean_val_loss:.2f}x")
                    if loss_function == "weighted_mse":
                        print("    ℹ️  Weighted loss is enabled — this may amplify errors")

            print(f"Epoch [{epoch+1}/{num_epochs}]  "
                  f"LR: {current_lr:.5f}  "
                  f"Train Loss: {avg_loss:.4f}  Train RelAcc: {relative_accuracy:.1f}%  "
                  f"Test Loss: {avg_val_loss:.4f}  Test RelAcc: {val_relative_accuracy:.1f}%  "
                  f"Batch std: {batch_std:.3f}")

        scheduler.step()

        # TEASER: keep the best-validation state as <checkpoint>.best.pth (rank 0).
        # The final-epoch checkpoint is still written at the end as before; the
        # evaluation can be pointed at either.
        if rank == 0 and np.isfinite(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(dict(model_state_dict=(model.module if is_distributed else model).state_dict(),
                            epoch=best_epoch, val_loss=best_val_loss,
                            config=dict(loss_function=loss_function, h5_file=h5_file_path,
                                        is_heteroscedastic=is_heteroscedastic,
                                        leaky_relu_slope=leaky_relu_slope, num_epochs=num_epochs,
                                        initial_lr=initial_lr, scaled_lr=scaled_lr,
                                        batch_size=batch_size, num_gpus=world_size,
                                        val_split=val_split),
                            teaser=dict(norm_min=min_val, norm_max=max_val, L=dataset_L,
                                        model=dataset_model, h5_attrs=h5_attrs,
                                        augmentation=('none' if args.no_augment else 'D4'),
                                        seed=seed_value, best_of='validation loss',
                                        script='train_teaser_ddp.py')),
                       str(saved_checkpoint) + '.best.pth')
            if (epoch + 1) % 50 == 0 or epoch < 5:
                print(f"  [best-val checkpoint updated: epoch {best_epoch}, val loss {best_val_loss:.4f}]")

        # Periodic EMA bias summary (rank 0 only)
        if (loss_function == "ema_bias_corrected_mse"
                and rank == 0
                and (epoch + 1) % bias_summary_freq == 0):
            summary = criterion.conditional_bias_summary()
            print(f"\n--- EMA Bias Summary (Epoch {epoch+1}) ---")
            for k, edge in enumerate(summary['bin_edges']):
                print(f"  Bin {k} [{edge[0]:.3f}, {edge[1]:.3f}]: "
                      f"mean_bias={summary['bin_means'][k]:+.6f}, "
                      f"n_updates={summary['n_updates'][k]}")
            print(f"  Max |bias|: {summary['max_abs_bias']:.6f}")
            print(f"  RMS bias:   {summary['rms_bias']:.6f}")
            print(f"---{'─'*40}---\n")

        # Periodic diagnostic epoch summary
        if diagnostic_mode and ((epoch + 1) % 5 == 0):
            create_diagnostics_summary(diagnostics, epoch + 1)


    # ================================================================== #
    #  POST-TRAINING
    # ================================================================== #
    end_time = timer()
    print0(f"\nTotal training time: {end_time - start_time:.3f} seconds")

    if rank == 0:
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)

        save_model = model.module if is_distributed else model

        final_test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=4 if not load_to_memory else 2,
            pin_memory=(device.type == 'cuda'),
            worker_init_fn=dataset.worker_init_fn if not load_to_memory else None
        )

        eval_results = stratified_evaluation(
            save_model, final_test_loader, device,
            absolute_threshold, relative_threshold
        )

        if is_heteroscedastic:
            predictions, uncertainties, actuals = eval_results
        else:
            predictions, actuals = eval_results
            uncertainties = None

        # ---- Build checkpoint ----
        ckpt = dict(
            model_state_dict=save_model.state_dict(),
            criterion_state_dict=criterion.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            train_losses=train_losses,
            train_accuracies=train_accuracies,
            train_relative_accuracies=train_relative_accuracies,
            test_losses=test_losses,
            test_accuracies=test_accuracies,
            test_relative_accuracies=test_relative_accuracies,
            learning_rates=learning_rates,
            epoch=num_epochs,
            absolute_threshold=absolute_threshold,
            relative_threshold=relative_threshold,
            final_predictions=predictions,
            final_actuals=actuals,
            config=dict(
                h5_file=h5_file_path,
                batch_size=batch_size,
                total_batch_size=batch_size * world_size if is_distributed else batch_size,
                num_gpus=world_size,
                num_epochs=num_epochs,
                initial_lr=initial_lr,
                scaled_lr=scaled_lr,
                val_split=val_split,
                load_to_memory=load_to_memory,
                loss_function=loss_function,
                is_heteroscedastic=is_heteroscedastic,
                leaky_relu_slope=leaky_relu_slope,
                health_monitor_enabled=enable_health_monitor,
            )
        )

        if is_heteroscedastic:
            ckpt['final_uncertainties'] = uncertainties
            ckpt['config']['log_var_min'] = log_var_min
            ckpt['config']['log_var_max'] = log_var_max
            ckpt['config']['use_softplus_variance'] = use_softplus_variance
            ckpt['config']['beta'] = (
                beta_param
                if loss_function in ("beta_nll", "original_beta_nll", "slope_penalized_nll")
                else 1.0
            )

        # TEASER: everything the evaluation needs to rebuild the input map
        ckpt['teaser'] = dict(
            norm_min=min_val, norm_max=max_val, L=dataset_L, model=dataset_model,
            h5_attrs=h5_attrs, augmentation=('none' if args.no_augment else 'D4'),
            input_path='float32 (x-min)/(max-min), no quantisation',
            bias_bins=(dict(n_bins=n_bins, y_min=y_min, y_max=y_max, lambda_cond=lambda_cond)
                       if loss_function in ('ema_bias_corrected_mse', 'bias_corrected_mse')
                       else None),
            seed=seed_value, script='train_teaser_ddp.py',
            best_val=dict(epoch=best_epoch, val_loss=best_val_loss,
                          path=str(saved_checkpoint) + '.best.pth'),
        )

        # Health monitor summary saved to checkpoint for post-hoc analysis
        if enable_health_monitor:
            ckpt['health_monitor_stats'] = health_monitor.get_summary()

        torch.save(ckpt, saved_checkpoint)

        print(f"\n{'='*60}")
        print("METRIC STABILITY:")
        print(f"{'='*60}")
        print(f"Test Absolute Accuracy std dev: {np.std(test_accuracies):.2f}%")
        print(f"Test Relative Accuracy std dev: {np.std(test_relative_accuracies):.2f}%")

        if enable_health_monitor:
            print(f"\n{'='*60}")
            print("HEALTH MONITOR SUMMARY:")
            print(f"{'='*60}")
            summary = health_monitor.get_summary()
            print(f"Total health checks:           {summary['total_checks']}")
            print(f"Emergency reinitializations:   {summary['intervention_count']}")
            if summary['intervention_count'] > 0:
                print(f"  ⚠️  Model required {summary['intervention_count']} reinitialization(s)")
            else:
                print("  ✓ No collapse detected during training")

        if diagnostic_mode:
            print("\n" + "#"*70)
            print("FINAL DIAGNOSTIC SUMMARY")
            print("#"*70)
            print(f"Total issues detected:  {diagnostics.issue_count}")
            print(f"Total batches tracked:  {len(diagnostics.loss_history)}")
            if diagnostics.issue_count > 0:
                print(f"\n⚠️  Training completed with {diagnostics.issue_count} issues detected.")
                print("Please review the diagnostic output above for details.")
            else:
                print("\n✓ Training completed without detected issues.")
            print("#"*70 + "\n")

        print("%%%%%%%%%%%%%%%%%%%%%%")
        print(f"Saved checkpoint → {saved_checkpoint}")

        if is_heteroscedastic:
            print(f"\nHeteroscedastic model — uncertainties saved in checkpoint")
            print(f"  'final_predictions'  → mean values (compatible with plotting)")
            print(f"  'final_uncertainties' → predicted standard deviations σ")


except Exception as e:
    print0(f"Training failed with error: {e}")
    import traceback
    traceback.print_exc()

finally:
    dataset.cleanup()
    if is_distributed:
        dist.destroy_process_group()
    print0("Cleanup completed")
