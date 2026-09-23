"""Utilities for ResNet regression with HDF5 data - DDP compatible -- TEASER version.

Derived 2026-09-05 from utils_emaloss_ddp.py for the Paper-I teaser runs
(Paper I Sec. 6.1).  The ONLY changes are in the data path:
HDF5ImageDataset now returns float32 tensors (no uint8/PIL quantisation) and
augmentation is the exact dihedral group D4 (D4Augment) instead of
torchvision RandomRotation + flips.  Models, losses, health monitor and
diagnostics are byte-identical to the emaloss version.

Original header follows.

Optimized utilities for ResNet regression with HDF5 data - DDP compatible.

Key features:
- Proper per-worker HDF5 file handle management
- Optional memory loading for maximum speed
- Thread-safe file operations
- Optimized for multi-GPU DDP training

STABILITY ENHANCEMENTS (v2.2 fixed):
- LeakyReLU activation throughout (default slope=0.1, never falls back to ReLU)
- Proper final layer initialization for regression tasks
- Training health monitoring with automatic recovery from collapse
- BatchNorm calibration after emergency reinitialization
- LOG-VARIANCE CLAMPING: Model-level bounds to prevent exp() overflow
  (defaults: [-8, 8], configurable via --log-var-min / --log-var-max)
- Optional Softplus variance parameterization as alternative
"""

import os
import warnings
from typing import Tuple, List
import argparse
import math
import pathlib
import threading

import numpy as np
import h5py
from scipy import stats as scipy_stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------------------------------------------------- #
#  Device helpers
# --------------------------------------------------------------------- #

def get_default_device() -> torch.device:
    """Return *cuda*, *mps* or *cpu* depending on availability."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data configuration
    p.add_argument("--h5", required=True, metavar="FILE",
                   help="HDF5 file containing image and target datasets")
    p.add_argument("--checkpoint", default="resnet_regression.pth",
                   help="Where to save model & metrics")

    # Training configuration
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size PER GPU (total = batch-size * num_gpus in DDP)")
    p.add_argument("--lr", type=float, default=0.01,
                   help="Base learning rate (auto-scaled in DDP)")

    # Evaluation thresholds
    p.add_argument("--absolute-threshold", type=float, default=0.5,
                   help="Absolute threshold for accuracy metrics")
    p.add_argument("--relative-threshold", type=float, default=0.2,
                   help="Relative threshold as fraction (0.2 = 20%%)")

    # Data split
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data used for validation")

    # Random seed
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")

    # HPC/logging
    p.add_argument("--hpc", action="store_true",
                   help="Redirect stdout/stderr to timestamped log files")
    p.add_argument("--log-dir", default=None,
                   help="Directory for log files (when --hpc is set)")

    # Checkpoint resume
    p.add_argument("--resume", default=None, metavar="FILE",
                   help="Path to checkpoint to resume training from "
                        "(restores model, optimizer, scheduler, and criterion state)")

    # Performance optimization
    p.add_argument("--load-to-memory", action="store_true",
                   help="Load entire HDF5 dataset to memory for faster access")

    # Loss function selection
    p.add_argument("--loss-function", type=str, default="mse",
                   choices=["mse", "weighted_mse", "slope_penalized_mse",
                            "bias_corrected_mse", "ema_bias_corrected_mse", 
                            "beta_nll", "gaussian_nll", "original_beta_nll",
                            "slope_penalized_nll"],
                   help="Loss function: mse (standard MSE), weighted_mse (ExtremesWeightedMSELoss), "
                        "slope_penalized_mse (MSE + global slope penalty), "
                        "bias_corrected_mse (MSE + per-bin conditional-bias penalty), "
                        "ema_bias_corrected_mse (EMA-based calculation of bias penalty), "
                        "beta_nll (β-NLL for heteroscedastic regression), "
                        "gaussian_nll (standard Gaussian NLL, β=1.0), "
                        "original_beta_nll (Seitzer et al. β-NLL with stop-gradient), "
                        "slope_penalized_nll (β-NLL + slope/bias penalty)")
    p.add_argument("--beta", type=float, default=0.5,
                   help="Beta parameter for beta_nll losses (default: 0.5, range: 0.1-2.0)")

    # Legacy argument for backward compatibility
    p.add_argument("--use-weighted-loss", action="store_true",
                   help="[DEPRECATED] Use --loss-function weighted_mse instead")

    # EMA bias summary logging
    p.add_argument("--bias-summary-freq", type=int, default=10,
                   help="Log EMA conditional-bias summary every N epochs "
                        "(only active for ema_bias_corrected_mse loss; default: 10)")

    # Diagnostic options
    p.add_argument("--diagnostic-mode", action="store_true",
                   help="Enable comprehensive diagnostic output")
    p.add_argument("--diagnostic-frequency", type=int, default=10,
                   help="How often to run full diagnostics (in batches)")

    # Stability options
    p.add_argument("--disable-health-monitor", action="store_true",
                   help="Disable automatic health monitoring and recovery (not recommended)")
    p.add_argument("--leaky-relu-slope", type=float, default=0.1,
                   help="Negative slope for LeakyReLU activations (default: 0.1)")

    # Variance parameterization options
    p.add_argument("--use-softplus-variance", action="store_true",
                   help="Use softplus parameterization for variance instead of direct log_var output. "
                        "Softplus guarantees positive variance without relying on clamping.")
    p.add_argument("--log-var-min", type=float, default=-8.0,
                   help="Minimum log-variance bound applied in model forward pass "
                        "(default: -8.0, corresponds to σ_min ≈ 0.018)")
    p.add_argument("--log-var-max", type=float, default=8.0,
                   help="Maximum log-variance bound applied in model forward pass "
                        "(default: 8.0, corresponds to σ_max ≈ 54)")

    # ---- TEASER additions ----
    p.add_argument("--ema-ymax", type=float, default=None,
                   help="Upper edge of the EMA bias-correction bins; default: the "
                        "dataset's L (HDF5 attr), falling back to 5.0")
    p.add_argument("--ema-nbins", type=int, default=None,
                   help="Number of bias-correction bins over [0, ema-ymax]; default: the "
                        "legacy value per loss (bias_corrected_mse: 8, ema_bias_corrected_mse: 10)")
    p.add_argument("--lambda-cond", type=float, default=None,
                   help="Penalty strength of the bias-corrected losses; default keeps the "
                        "values of train_biascorr_ddp.py / train_emaloss_ddp.py "
                        "(bias_corrected_mse: 15.0, ema_bias_corrected_mse: 10.0)")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable the D4 (flip/rot90) training augmentation")

    return p.parse_args()


# --------------------------------------------------------------------- #
#  Stability utilities
# --------------------------------------------------------------------- #

def replace_relu_with_leaky(model: nn.Module, negative_slope: float = 0.1) -> nn.Module:
    """Recursively replace all ReLU activations in a model with LeakyReLU."""
    for name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, name, nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        else:
            replace_relu_with_leaky(child, negative_slope)
    return model


def initialize_regression_head(model: nn.Module, target_mean: float = 1.0,
                                weight_std: float = 0.01) -> nn.Module:
    """
    Initialize the final FC layer(s) with small weights and appropriate bias.

    For heteroscedastic models:
      - fc_mean bias is set to target_mean (sensible starting prediction)
      - fc_logvar bias is set to 0.0 (corresponds to σ ≈ 1, a neutral starting uncertainty)
    """
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        nn.init.normal_(model.fc.weight, mean=0.0, std=weight_std)
        nn.init.constant_(model.fc.bias, target_mean)
    if hasattr(model, 'fc_mean') and isinstance(model.fc_mean, nn.Linear):
        nn.init.normal_(model.fc_mean.weight, mean=0.0, std=weight_std)
        nn.init.constant_(model.fc_mean.bias, target_mean)
    if hasattr(model, 'fc_logvar') and isinstance(model.fc_logvar, nn.Linear):
        nn.init.normal_(model.fc_logvar.weight, mean=0.0, std=weight_std)
        nn.init.constant_(model.fc_logvar.bias, 0.0)
    return model


def emergency_reinitialize(model: nn.Module, target_mean: float = 1.0,
                            leaky_slope: float = 0.1) -> None:
    """
    Emergency full reinitialization of model weights after training collapse.

    Resets all Conv2d, BatchNorm2d, and Linear layers to a fresh state while
    preserving the architecture.  After calling this, BatchNorm running statistics
    are stale and must be recalibrated with calibrate_batchnorm().

    Linear layer weight strategy
    ----------------------------
    Default: Normal(0, 0.01) — very small weights, safe starting point.  The final
    prediction/logvar heads are then corrected via initialize_regression_head().

    Alternative (commented out below): Kaiming-normal initialization, which gives
    variance-preserving scale for LeakyReLU networks but results in larger initial
    weights that can cause instability after a collapse event.

    Parameters
    ----------
    model : nn.Module
        The (unwrapped) model to reinitialize.
    target_mean : float
        Initial bias for the mean prediction head(s).
    leaky_slope : float
        Negative slope used in Kaiming initialization for Conv2d layers.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='leaky_relu', a=leaky_slope)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            m.reset_running_stats()
        elif isinstance(m, nn.Linear):
            # Default: small Normal weights — conservative and stable after collapse.
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            # Alternative (Kaiming): gives proper variance-preserving scale for
            # LeakyReLU networks, but may be too large immediately post-collapse:
            #   nn.init.kaiming_normal_(m.weight, mode='fan_out',
            #                           nonlinearity='leaky_relu', a=leaky_slope)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Override the prediction and log-variance heads to their proper starting values.
    # This corrects the bias=0 set above for fc/fc_mean, and ensures fc_logvar
    # starts at log_var=0 (σ≈1) rather than target_mean.
    initialize_regression_head(model, target_mean=target_mean, weight_std=0.01)


# --------------------------------------------------------------------- #
#  Model definition
# --------------------------------------------------------------------- #

def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3×3 convolution with padding and no bias."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
    # """3×3 convolution with **CIRCULAR** padding and no bias."""
    # return nn.Conv2d(in_channels, out_channels, kernel_size=3,
    #                  stride=stride, padding=1, bias=False, padding_mode='circular')

class ResidualBlock(nn.Module):
    """
    Post-activation residual block with LeakyReLU activation.

    Architecture (unchanged from original):
        Conv → BN → LeakyReLU → Conv → BN → (+identity) → LeakyReLU

    The negative_slope parameter controls the LeakyReLU slope.  Default is 0.1,
    which prevents dead neurons without significantly altering the gradient flow.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: nn.Module | None = None, negative_slope: float = 0.1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        # Always LeakyReLU — slope 0.1 by default; set negative_slope=0.0 to
        # obtain gradient-equivalent behaviour to standard ReLU if ever needed.
        self.relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    Flexible ResNet for single-channel regression with LeakyReLU and log-var clamping.

    Parameters
    ----------
    block : type[ResidualBlock]
    layers : List[int]
        Number of residual blocks in each stage.  Between 1 and 6 stages are
        supported; the count is inferred automatically from ``len(layers)``.
        Examples:
            [3, 4, 6, 3]       → 4-stage ResNet-34-style model (original default)
            [5, 5, 5]          → 3-stage model
            [2, 3, 4, 5, 4, 2] → 6-stage model
        Channel widths follow the standard doubling pattern:
            stage 1→16, 2→32, 3→64, 4→128, 5→256, 6→512
        (all scaled by width_multiplier).
    width_multiplier : float
    heteroscedastic : bool
        If True, model outputs (mean, log_var); if False, outputs single prediction
    negative_slope : float
        Negative slope for LeakyReLU activations (default: 0.1; always LeakyReLU)
    log_var_bounds : tuple(float, float)
        Hard clamp applied to log_var in forward().  Default (-8, 8) gives
        σ ∈ [~0.018, ~54].  Override via --log-var-min / --log-var-max.
    use_softplus_var : bool
        If True, interpret fc_logvar output as raw softplus input, then take log.
        Provides an alternative soft-positivity constraint; the clamp is still
        applied afterwards.
    """

    def __init__(self, block, layers, width_multiplier=1.0, heteroscedastic=False,
                 negative_slope=0.1, log_var_bounds=(-8.0, 8.0), use_softplus_var=False):
        super().__init__()

        num_stages = len(layers)
        if not (1 <= num_stages <= 6):
            raise ValueError(
                f"ResNet supports between 1 and 6 stages; got {num_stages}."
            )

        self.heteroscedastic = heteroscedastic
        self.negative_slope = negative_slope
        self.log_var_bounds = log_var_bounds
        self.use_softplus_var = use_softplus_var
        self.num_stages = num_stages

        # Channel widths for up to 6 stages, following the standard doubling pattern.
        # Stages 5 and 6 extend the original [16, 32, 64, 128] to [256, 512].
        base_widths = [16, 32, 64, 128, 256, 512]
        widths = [int(w * width_multiplier) for w in base_widths[:num_stages]]
        self.in_channels = widths[0]

        self.conv1 = conv3x3(1, widths[0])
        self.bn1 = nn.BatchNorm2d(widths[0])
        # Always LeakyReLU (same slope as blocks for consistency)
        self.relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        # Build stages dynamically based on len(layers).
        # Stage 0 keeps spatial resolution (stride=1); each subsequent stage
        # halves it (stride=2), following standard ResNet convention.
        self.stages = nn.ModuleList()
        for i, (w, n) in enumerate(zip(widths, layers)):
            stride = 1 if i == 0 else 2
            self.stages.append(self._make_layer(block, w, n, stride=stride))

        final_channels = widths[-1]

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if self.heteroscedastic:
            self.fc_mean = nn.Linear(final_channels, 1)
            self.fc_logvar = nn.Linear(final_channels, 1)
        else:
            self.fc = nn.Linear(final_channels, 1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels, momentum=0.1),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample,
                        negative_slope=self.negative_slope)]
        self.in_channels = out_channels
        layers.extend(block(out_channels, out_channels, negative_slope=self.negative_slope)
                      for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Returns
        -------
        heteroscedastic=False:
            prediction : Tensor, shape (batch_size,)
        heteroscedastic=True:
            mean    : Tensor, shape (batch_size,)
            log_var : Tensor, shape (batch_size,)
                Guaranteed to lie in [log_var_bounds[0], log_var_bounds[1]].
        """
        out = self.relu(self.bn1(self.conv1(x)))
        for stage in self.stages:
            out = stage(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)

        if self.heteroscedastic:
            mean = self.fc_mean(out).squeeze(-1)
            log_var_raw = self.fc_logvar(out).squeeze(-1)

            if self.use_softplus_var:
                # Softplus guarantees positive variance before log; clamp still applied.
                variance = F.softplus(log_var_raw) + 1e-6
                log_var = torch.log(variance)
            else:
                log_var = log_var_raw

            # Hard clamp: single source of truth for variance bounds.
            # All NLL loss functions rely on this and do NOT clamp internally.
            log_var = torch.clamp(log_var,
                                  min=self.log_var_bounds[0],
                                  max=self.log_var_bounds[1])
            return mean, log_var
        else:
            return self.fc(out).squeeze(-1)


# --------------------------------------------------------------------- #
#  Optimized HDF5 Dataset
# --------------------------------------------------------------------- #

class HDF5ImageDataset(Dataset):
    """
    HDF5 dataset for multi-GPU training -- TEASER (float32) VERSION.

    Differences from the legacy ``utils_emaloss_ddp.HDF5ImageDataset`` (the only
    ones; file-handle management, memory loading and the DDP contract are
    unchanged):

    * NO 8-bit quantisation.  The legacy loader mapped every map through the
      global train-set min/max onto uint8 (0..255) and handed it to PIL.  That
      is a nonlinear, information-destroying step the matched filter of the
      eta computation never suffers, and its step size scales with the
      dataset's dynamic range -- for the spectral-tilt mixture (T1_PSRAND_WN)
      the rare steep-slope maps set a range of ~+-200, i.e. a step of ~1.6 on
      top of a white floor of sigma_w = 0.27.  Here the SAME affine scaling
      x -> (x - min)/(max - min) is applied in float32 (so the network sees
      inputs in [0, 1] as before and the validated hyper-parameters carry
      over), but nothing is rounded.  Global constants only: no per-map
      normalisation, no mean subtraction -- the DC channel is untouched.

    * Augmentation is passed as a callable on a (1, H, W) float tensor (see
      ``D4Augment``), not a torchvision PIL pipeline.

    Returns (image tensor (1, H, W) float32, target float32) per item.
    """

    def __init__(self, h5_path, *, image_key="noisy", target_key="I0",
                 transform=None, load_to_memory=False):
        super().__init__()
        self.h5_path = pathlib.Path(h5_path)
        self.image_key = image_key
        self.target_key = target_key
        self.transform = transform
        self.load_to_memory = load_to_memory
        self._local = threading.local()

        with h5py.File(self.h5_path, "r") as f:
            if image_key not in f or target_key not in f:
                raise KeyError(
                    f"Expected datasets '{image_key}' and '{target_key}' in {h5_path}"
                )
            self.length = len(f[image_key])
            self.image_shape = f[image_key].shape[1:]
            self.target_shape = f[target_key].shape[1:] if f[target_key].ndim > 1 else ()
            if len(self.image_shape) == 3:
                self.img_size = self.image_shape[-1]
            elif len(self.image_shape) == 2:
                self.img_size = self.image_shape[-1]
            else:
                self.img_size = int(math.isqrt(self.image_shape[0]))
                if self.img_size ** 2 != self.image_shape[0]:
                    raise ValueError(f"Images not square: {self.image_shape[0]} pixels")

        self.min_val = 0.0
        self.max_val = 1.0

        if self.load_to_memory:
            print(f"Loading {self.length} images to memory...")
            with h5py.File(self.h5_path, "r") as f:
                self.images_mem = np.array(f[image_key])
                self.targets_mem = np.array(f[target_key])
            print(f"Loaded {self.images_mem.nbytes / 1e9:.2f} GB to memory")

    def set_normalization(self, min_val: float, max_val: float) -> None:
        """Global affine scaling constants (train split min/max)."""
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def normalize(self, img_arr: np.ndarray) -> np.ndarray:
        """x -> (x - min)/(max - min) in float32; identity if max <= min.
        Exposed so that the evaluation script applies the identical map."""
        img_arr = np.asarray(img_arr, dtype=np.float32)
        if self.max_val > self.min_val:
            return (img_arr - np.float32(self.min_val)) / np.float32(self.max_val - self.min_val)
        return img_arr

    def _get_file_handle(self):
        if self.load_to_memory:
            return None
        if not hasattr(self._local, 'h5file') or self._local.h5file is None:
            self._local.h5file = h5py.File(self.h5_path, "r", libver='latest', swmr=True)
            self._local.images_ds = self._local.h5file[self.image_key]
            self._local.targets_ds = self._local.h5file[self.target_key]
        return self._local.h5file

    def worker_init_fn(self, worker_id: int) -> None:
        if not self.load_to_memory:
            self._get_file_handle()

    def cleanup(self) -> None:
        if hasattr(self._local, 'h5file') and self._local.h5file is not None:
            self._local.h5file.close()
            self._local.h5file = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.load_to_memory:
            img_arr = self.images_mem[idx]
            target = self.targets_mem[idx]
        else:
            self._get_file_handle()
            img_arr = self._local.images_ds[idx]
            target = self._local.targets_ds[idx]

        img_arr = np.asarray(img_arr)
        if img_arr.ndim == 1:
            img_arr = img_arr.reshape(self.img_size, self.img_size)
        elif img_arr.ndim == 3:
            img_arr = img_arr.squeeze(0)

        img = torch.from_numpy(self.normalize(img_arr)).unsqueeze(0)   # (1, H, W) float32
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.as_tensor(target, dtype=torch.float32)

    def get_images(self, indices: np.ndarray) -> np.ndarray:
        """Batch-fetch raw images (used for normalization / noise estimation)."""
        if self.load_to_memory:
            return self.images_mem[indices]
        with h5py.File(self.h5_path, "r") as f:
            if np.all(np.diff(indices) == 1):
                return np.array(f[self.image_key][indices[0]:indices[-1] + 1])
            else:
                return np.array(f[self.image_key][sorted(indices)])

    def get_targets(self, indices: np.ndarray) -> np.ndarray:
        """Batch-fetch targets."""
        if self.load_to_memory:
            return self.targets_mem[indices]
        with h5py.File(self.h5_path, "r") as f:
            if np.all(np.diff(indices) == 1):
                return np.array(f[self.target_key][indices[0]:indices[-1] + 1])
            else:
                return np.array(f[self.target_key][sorted(indices)])


class D4Augment:
    """Random element of the dihedral group D4 applied to a (C, H, W) tensor.

    The eight symmetries of the square (4 rotations x optional transpose) are
    EXACT symmetries of an isotropic, centred-source model: the map is
    re-indexed, never resampled, so the noise statistics, the source and the
    DC mode are untouched.  This replaces the legacy
    ``RandomRotation(180) + flips`` pipeline, whose arbitrary-angle rotation
    resampled the noise (nearest-neighbour aliasing) and filled the corners
    with the global-minimum value -- a train/test mismatch that can only make
    the network worse and, for a Tier-1 cell, would bias its gain low.

    Do NOT use for anisotropic models (scan direction, row-median residuals):
    there only the identity and the symmetries that preserve the anisotropy
    axis are admissible.

    Each worker draws from torch's per-worker RNG, so the augmentation is
    reproducible under a fixed --seed.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return img
        k = int(torch.randint(0, 4, (1,)))
        if k:
            img = torch.rot90(img, k, dims=(-2, -1))
        if bool(torch.randint(0, 2, (1,))):
            img = img.transpose(-2, -1)
        return img.contiguous()


class NoAugment:
    """Identity transform (validation / evaluation)."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img


# --------------------------------------------------------------------- #
#  Loss functions
# --------------------------------------------------------------------- #

class ExtremesWeightedMSELoss(nn.Module):
    """
    Weighted MSE that emphasizes errors at low/high target ranges, with optional
    directional penalties to counter systematic bias.

    Args:
        low_thresh, high_thresh: Range thresholds for extreme weighting.
        w_low, w_high: Weight multipliers for extreme ranges (≥1).
        smooth: Use sigmoid gates (True) or hard step (False).
        tau: Smoothness parameter for sigmoid transitions.
        directional: Enable asymmetric penalties for error direction.
        over_weight_low: Extra penalty when pred > target in low range.
        under_weight_high: Extra penalty when pred < target in high range.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, low_thresh=None, high_thresh=None, w_low=1.0, w_high=1.0,
                 smooth=True, tau=0.1, directional=True, over_weight_low=1.0,
                 under_weight_high=1.0, reduction="mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.w_low = float(w_low)
        self.w_high = float(w_high)
        self.smooth = bool(smooth)
        self.tau = float(tau) if tau is not None else 0.1
        self.directional = bool(directional)
        self.over_weight_low = float(over_weight_low)
        self.under_weight_high = float(under_weight_high)
        self.reduction = reduction

    def _gate_sigmoid(self, x, center, sign):
        z = (x - center) / self.tau if sign > 0 else (center - x) / self.tau
        return torch.sigmoid(z)

    def _low_gate(self, target):
        if self.low_thresh is None:
            return torch.zeros_like(target)
        lt = torch.as_tensor(self.low_thresh, dtype=target.dtype, device=target.device)
        return self._gate_sigmoid(target, lt, sign=-1) if self.smooth else (target <= lt).to(target.dtype)

    def _high_gate(self, target):
        if self.high_thresh is None:
            return torch.zeros_like(target)
        ht = torch.as_tensor(self.high_thresh, dtype=target.dtype, device=target.device)
        return self._gate_sigmoid(target, ht, sign=+1) if self.smooth else (target >= ht).to(target.dtype)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        base_w = torch.ones_like(target)
        low_gate = self._low_gate(target)
        high_gate = self._high_gate(target)
        if self.w_low != 1.0:
            base_w = base_w * (1.0 + (self.w_low - 1.0) * low_gate)
        if self.w_high != 1.0:
            base_w = base_w * (1.0 + (self.w_high - 1.0) * high_gate)
        if self.directional:
            if self.over_weight_low != 1.0:
                over_low = (err > 0).to(target.dtype)
                base_w = base_w * (1.0 + (self.over_weight_low - 1.0) * low_gate * over_low)
            if self.under_weight_high != 1.0:
                under_high = (err < 0).to(target.dtype)
                base_w = base_w * (1.0 + (self.under_weight_high - 1.0) * high_gate * under_high)
        loss = base_w * err.pow(2)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SlopePenalizedMSELoss(nn.Module):
    """
    Standard MSE plus a penalty that discourages regression-to-mean behaviour.

    The penalty measures the deviation of the batch-level OLS slope from 1.0:
        L = MSE + λ_slope * (1 - slope)²

    Parameters
    ----------
    lambda_slope : float
        Penalty strength.  Typical range: [5.0, 50.0].
    """

    def __init__(self, lambda_slope: float = 10.0):
        super().__init__()
        self.lambda_slope = lambda_slope

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((predictions - targets) ** 2)
        pred_centered = predictions - predictions.mean()
        target_centered = targets - targets.mean()
        covariance = torch.mean(pred_centered * target_centered)
        target_variance = torch.mean(target_centered ** 2)
        slope = covariance / (target_variance + 1e-8)
        slope_penalty = (1.0 - slope) ** 2
        return mse + self.lambda_slope * slope_penalty


class ConditionalBiasPenalizedMSELoss(nn.Module):
    """
    MSE loss with a conditional-bias penalty.

    The slope penalty in ``SlopePenalizedMSELoss`` corrects global regression-
    to-the-mean, but it is insensitive to *pointwise* (nonlinear) systematic
    errors.  This loss directly penalises the conditional bias: the mean
    residual within each bin of the target range.

    The full loss is

        L = MSE + λ_cond * (1/K') Σ_k  [mean_{i ∈ bin_k}(ŷ_i − y_i)]²

    where K' is the number of non-empty bins, and the bin assignment is
    determined by the target values y alone (so no gradient discontinuity
    is introduced — gradients flow through the residuals as usual).

    Parameters
    ----------
    lambda_cond : float
        Conditional-bias penalty strength.  Typical range: [5.0, 50.0].
    n_bins : int
        Number of equal-width bins covering [y_min, y_max].  Default: 10.
    y_min : float
        Lower edge of the binning range.  Default: 0.0.
    y_max : float
        Upper edge of the binning range.  Default: 2.0.

    Notes
    -----
    Targets outside [y_min, y_max] are clamped into the nearest boundary bin
    rather than discarded, so no samples are silently ignored.  For a batch
    where every bin is empty (degenerate edge case) the penalty is zero and
    a warning is raised.
    """

    def __init__(
        self,
        lambda_cond: float = 10.0,
        n_bins: int = 10,
        y_min: float = 0.0,
        y_max: float = 2.0,
    ):
        super().__init__()
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        if y_max <= y_min:
            raise ValueError(f"y_max ({y_max}) must be greater than y_min ({y_min})")
        self.lambda_cond = lambda_cond
        self.n_bins = n_bins
        self.y_min = y_min
        self.y_max = y_max

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # ── Standard MSE ──────────────────────────────────────────────────────
        mse = torch.mean((predictions - targets) ** 2)

        # ── Bin assignment (based on targets only — no gradient wrt params) ──
        bin_width = (self.y_max - self.y_min) / self.n_bins
        # floor division gives 0-based index; clamp handles out-of-range targets
        bin_idx = torch.floor(
            (targets - self.y_min) / bin_width
        ).long().clamp(0, self.n_bins - 1)

        residuals = predictions - targets  # shape (N,)

        # ── Vectorised accumulation of per-bin residual sums and counts ───────
        # scatter_add_ operates on a freshly allocated tensor, so autograd
        # correctly propagates gradients through `residuals`.
        bin_sum = torch.zeros(
            self.n_bins, device=predictions.device, dtype=predictions.dtype
        )
        bin_count = torch.zeros(
            self.n_bins, device=predictions.device, dtype=predictions.dtype
        )
        bin_sum.scatter_add_(0, bin_idx, residuals)
        bin_count.scatter_add_(0, bin_idx, torch.ones_like(residuals))

        # ── Conditional-bias penalty (average over non-empty bins) ───────────
        nonempty = bin_count > 0
        if not nonempty.any():
            warnings.warn(
                "ConditionalBiasPenalizedMSELoss: all bins are empty — "
                "conditional-bias penalty is zero.  Check y_min/y_max."
            )
            return mse

        bin_mean_residual = bin_sum[nonempty] / bin_count[nonempty]
        cond_bias_penalty = (bin_mean_residual ** 2).mean()

        return mse + self.lambda_cond * cond_bias_penalty


"""
EMABasedBiasPenalizedMSELoss
============================
Drop-in replacement for ``ConditionalBiasPenalizedMSELoss`` that replaces
noisy mini-batch bin-mean estimates with an Exponential Moving Average (EMA)
accumulated across batches.

The gradient pathology in the original loss comes from computing bin-mean
residuals over O(batch_size / n_bins) samples, which are too few to give a
stable gradient direction.  The EMA approach decouples *what the gradient
direction is* (determined by the slowly-updated EMA, low variance) from *where
the gradient acts* (the current batch predictions, so autograd works normally).

Requires: Python ≥ 3.9, PyTorch ≥ 1.12
Optional: torch.distributed (for DDP sync via dist_sync=True)
"""

class EMABasedBiasPenalizedMSELoss(nn.Module):
    """
    MSE loss with a conditional-bias penalty whose gradient direction is
    stabilised by an Exponential Moving Average of per-bin residual means.

    Background
    ----------
    ``ConditionalBiasPenalizedMSELoss`` computes the penalty

        (λ / K') · Σ_k  [r̄_k^batch]²,    r̄_k = mean residual in bin k

    from the current mini-batch.  With n_bins=10 and batch size B, each bin
    contains on average B/10 samples.  The gradient for sample i in bin k is

        ∂L/∂ŷ_i  =  (2λ / K') · r̄_k^batch / n_k

    so the gradient is directly proportional to the noisy batch bin-mean.
    This creates two coupled problems:

      1. **Gradient variance**: SE(r̄_k) ≈ σ_r / √(B/K).  For B=128, K=10,
         σ_r≈0.49, this is ≈ 0.19 — roughly 40 % of the per-sample RMS.
         With λ=15 amplifying this noise 15×, the penalty gradient is
         frequently larger than the MSE gradient and points in a random
         direction from batch to batch.

      2. **Train/test loss gap**: the network implicitly memorises the
         bin-composition of training batches (aided by BatchNorm's training-
         mode statistics), so batch bin-means are near zero on training data
         but large on unseen test batches.

    EMA solution
    ------------
    Maintain a running estimate μ_k^EMA of the true population bin-mean
    residual, updated after each training forward pass:

        μ_k^EMA  ←  β · μ_k^EMA  +  (1−β) · r̄_k^batch .detach()

    Replace the quadratic penalty with its first-order Taylor linearisation
    around μ_k^EMA:

        penalty_k  =  2 · μ_k^EMA · r̄_k^batch  −  (μ_k^EMA)²          (*)

    The second term is constant w.r.t. the network parameters (EMA buffer,
    no gradient), so the gradient is

        ∂L/∂ŷ_i  =  (2λ / K') · μ_k^EMA / n_k

    Properties:
    - **Stable**: μ_k^EMA has variance ≈ (1−β)² · σ_r² / n_k per EMA step,
      which decays rapidly; with β=0.99 the effective window is 100 steps.
    - **Correct direction**: the gradient pushes predictions in the direction
      that reduces the EMA-estimated conditional bias.
    - **No train/test gap from loss pathology**: the EMA is accumulated from
      training data, but the penalty value (*) is bounded (≤ the true penalty)
      and the test penalty is computed with the same stable EMA coefficient,
      so it no longer reflects mini-batch composition noise.
    - **At convergence**: when r̄_k^batch ≈ μ_k^EMA ≈ 0, (*) → 0, and the
      loss reduces to pure MSE.  ✓

    Parameters
    ----------
    lambda_cond : float
        Conditional-bias penalty strength.  The EMA gradient has the same
        *expected magnitude* as the original batch-based gradient (both are
        proportional to the population bin-mean bias), but much lower
        variance.  You can therefore use the same λ value as before and
        expect smoother convergence, or reduce it if the gradient was
        already too strong.  Suggested starting range: [3, 15].  Default: 5.0.
    n_bins : int
        Number of equal-width bins covering [y_min, y_max].  Default: 10.
    y_min : float
        Lower edge of the binning range.  Targets below y_min are clamped
        into bin 0.  Default: 0.0.
    y_max : float
        Upper edge of the binning range.  Targets above y_max are clamped
        into bin n_bins-1.  Default: 2.0.
    ema_decay : float
        EMA decay factor β ∈ (0, 1).  Effective history window ≈ 1/(1−β)
        steps.  Higher values → slower adaptation (lower gradient variance,
        slower response to shifts in model predictions).
        β=0.99 (window ≈ 100 steps) is a safe default; β=0.999 (≈1000 steps)
        is appropriate when you want the EMA to span a full epoch.
        Default: 0.99.
    dist_sync : bool
        If True, all-reduces bin_sum and bin_count across DDP processes
        *before* the EMA update so that all GPUs accumulate the same global
        bin-mean estimate rather than each GPU's local shard.  Set to True
        whenever using DistributedDataParallel.  Default: False.

    Buffers (saved/loaded with state_dict, moved with .to(device))
    --------------------------------------------------------------
    bin_mean_ema : Tensor, shape (n_bins,)
        Raw (non-bias-corrected) EMA of per-bin mean residuals.
    bin_update_count : Tensor, shape (n_bins,), dtype=long
        Number of EMA updates per bin.  Used for Adam-style bias correction
        that compensates for the zero initialisation of bin_mean_ema.

    Notes
    -----
    * The EMA buffers are NOT automatically synchronised across DDP processes
      (register_buffer does not trigger all_reduce).  Use dist_sync=True.
    * Evaluation mode (model.eval()): EMA is NOT updated, and the penalty
      value uses the EMA accumulated during training, making the eval loss
      stable and comparable across epochs.
    * The penalty can be negative early in training (when r̄_k^batch and
      μ_k^EMA have opposite signs due to zero initialisation of the EMA).
      This is harmless; the gradient direction is still correct, and the
      penalty value converges to the true squared-bias once the EMA warms up.
    """

    def __init__(
        self,
        lambda_cond: float = 5.0,
        n_bins: int = 10,
        y_min: float = 0.0,
        y_max: float = 2.0,
        ema_decay: float = 0.99,
        dist_sync: bool = False,
    ) -> None:
        super().__init__()

        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        if y_max <= y_min:
            raise ValueError(f"y_max ({y_max}) must be greater than y_min ({y_min})")
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")

        self.lambda_cond  = lambda_cond
        self.n_bins       = n_bins
        self.y_min        = y_min
        self.y_max        = y_max
        self.ema_decay    = ema_decay
        self.dist_sync    = dist_sync

        # EMA state — zero-initialised; bias correction compensates for this.
        # dtype=float32 matches typical model dtype; promote to float64 here
        # if targets are double.
        self.register_buffer("bin_mean_ema",    torch.zeros(n_bins))
        self.register_buffer("bin_update_count", torch.zeros(n_bins, dtype=torch.long))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def bias_corrected_ema(self) -> torch.Tensor:
        """
        Return the Adam-style bias-corrected EMA estimate for each bin.

        Raw EMA is zero-initialised, so early estimates are systematically
        pulled toward zero.  The correction factor 1/(1 − β^n) removes this
        bias, analogously to Adam's first-moment correction.

        Returns
        -------
        mu_hat : Tensor, shape (n_bins,)
            Bias-corrected per-bin mean residual estimate.  Bins that have
            never been updated return 0.
        """
        n = self.bin_update_count.float()              # (n_bins,)
        # Clamp n to 1 before exponentiation to avoid 0^0 = 1 for empty bins,
        # which would give correction = 0 and a divide-by-zero.  The torch.where
        # below handles the n=0 case by returning 0 regardless.
        correction = 1.0 - self.ema_decay ** n.clamp(min=1)   # (n_bins,)
        mu_hat = self.bin_mean_ema / correction
        # Bins with no update history: return 0 (not nan/inf from 0/tiny)
        mu_hat = torch.where(self.bin_update_count > 0, mu_hat, torch.zeros_like(mu_hat))
        return mu_hat

    def conditional_bias_summary(self) -> dict:
        """
        Diagnostic snapshot of the current EMA-estimated conditional bias.

        Useful for logging during training, e.g.::

            loss_fn = EMABasedBiasPenalizedMSELoss(...)
            ...
            if step % log_every == 0:
                logger.info(loss_fn.conditional_bias_summary())

        Returns
        -------
        dict with keys:
            bin_edges : list of (float, float) — lower/upper edge of each bin.
            bin_means : list of float — bias-corrected EMA per bin.
            n_updates : list of int — number of EMA updates per bin.
            max_abs_bias : float — max |μ̂_k| over all bins.
            rms_bias : float — √(mean_k μ̂_k²), the penalty before λ scaling.
        """
        mu   = self.bias_corrected_ema().detach().cpu()
        bw   = (self.y_max - self.y_min) / self.n_bins
        edges = [
            (round(self.y_min + k * bw, 6), round(self.y_min + (k + 1) * bw, 6))
            for k in range(self.n_bins)
        ]
        return {
            "bin_edges"    : edges,
            "bin_means"    : mu.tolist(),
            "n_updates"    : self.bin_update_count.tolist(),
            "max_abs_bias" : mu.abs().max().item(),
            "rms_bias"     : mu.pow(2).mean().sqrt().item(),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.view(-1)
        targets     = targets.view(-1)

        # ── Standard MSE ──────────────────────────────────────────────────────
        mse = torch.mean((predictions - targets) ** 2)

        # ── Bin assignment (target-based only; no gradient through bin_idx) ───
        bin_width = (self.y_max - self.y_min) / self.n_bins
        bin_idx = (
            torch.floor((targets - self.y_min) / bin_width)
            .long()
            .clamp(0, self.n_bins - 1)
        )

        residuals = predictions - targets          # gradient flows through here

        # ── Accumulate per-bin residual sums and sample counts ────────────────
        # scatter_add_ on a freshly allocated tensor preserves the autograd
        # graph through `residuals` (same approach as the original class).
        bin_sum = torch.zeros(
            self.n_bins, device=predictions.device, dtype=predictions.dtype
        )
        bin_count = torch.zeros(
            self.n_bins, device=predictions.device, dtype=predictions.dtype
        )
        bin_sum.scatter_add_(0, bin_idx, residuals)
        bin_count.scatter_add_(0, bin_idx, torch.ones_like(residuals))

        # ── DDP sync via straight-through estimator ─────────────────────────
        #
        # Problem: dist.all_reduce has no autograd kernel. Calling it on
        # tensors inside the computation graph triggers a deprecation warning
        # today and will become a hard error in a future PyTorch release.
        # But using LOCAL bin_sum / bin_count for the penalty changes the
        # gradient magnitude by a factor of world_size (local n_k ≈
        # global n_k / W → per-sample gradient ×W → effective λ ×W).
        #
        # Solution: all-reduce only *detached clones*, then reconstruct a
        # tensor whose forward VALUE equals the global statistic but whose
        # GRADIENT flows through the local contribution:
        #
        #     bin_sum_synced = bin_sum + (bin_sum_global - bin_sum.detach())
        #
        # Forward: = bin_sum + bin_sum_global - bin_sum = bin_sum_global  ✓
        # Backward: ∂/∂(bin_sum) = 1  (only the first term carries grad)  ✓
        #
        # This preserves the original gradient (∂L/∂ŷ_i = 2λ μ_k / K'n_k
        # with GLOBAL n_k) while avoiding the autograd-through-all_reduce
        # issue.  For dist_sync=False the passthrough is a no-op.
        if self.dist_sync:
            bin_sum_global = bin_sum.detach().clone()
            bin_count_global = bin_count.detach().clone()
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.all_reduce(bin_sum_global,   op=dist.ReduceOp.SUM)
                    dist.all_reduce(bin_count_global, op=dist.ReduceOp.SUM)
            except ImportError:
                warnings.warn(
                    "EMABasedBiasPenalizedMSELoss: dist_sync=True but "
                    "torch.distributed is unavailable.  Skipping sync."
                )
                bin_sum_global = bin_sum.detach()
                bin_count_global = bin_count.detach()

            # Straight-through: value = global, gradient = local
            bin_sum_synced   = bin_sum + (bin_sum_global - bin_sum.detach())
            bin_count_synced = bin_count_global          # counts carry no gradient
        else:
            bin_sum_synced   = bin_sum
            bin_count_synced = bin_count

        # ── Identify non-empty bins (GLOBAL counts) ──────────────────────────
        nonempty = bin_count_synced > 0

        if not nonempty.any():
            warnings.warn(
                "EMABasedBiasPenalizedMSELoss: all bins are empty — "
                "check y_min / y_max.  Returning MSE only."
            )
            return mse

        # ── Current batch bin means (WITH gradient for non-empty bins) ────────
        # Empty bins get zero; they do not contribute to the penalty gradient.
        batch_bin_means = torch.where(
            nonempty,
            bin_sum_synced / bin_count_synced.clamp(min=1.0),
            torch.zeros_like(bin_sum_synced),
        )

        # ── Bias-corrected EMA from previous steps (NO gradient) ──────────────
        # This is the stable estimate of the true conditional bias per bin.
        # Detached from the computation graph — all gradient signal comes
        # through batch_bin_means below.
        mu_ema = self.bias_corrected_ema()         # (n_bins,), no grad

        # ── Linearised EMA penalty ────────────────────────────────────────────
        #
        # penalty_k = 2 · μ_k^EMA · r̄_k^batch  −  (μ_k^EMA)²
        #
        # First-order Taylor expansion of (r̄_k^batch)² around μ_k^EMA:
        #   (r̄_k)²  ≈  (μ_EMA)² + 2·μ_EMA·(r̄_k − μ_EMA)
        #            =  2·μ_EMA·r̄_k − (μ_EMA)²
        #
        # Gradient w.r.t. predictions[i] for i ∈ bin k:
        #   ∂L/∂ŷ_i  =  (2λ / K') · μ_k^EMA / n_k        ← stable, EMA-scaled
        #
        # Compare to original (noisy):
        #   ∂L/∂ŷ_i  =  (2λ / K') · r̄_k^batch / n_k     ← high variance
        #
        # Value properties:
        #   · At convergence (r̄ ≈ μ_EMA ≈ 0): penalty → 0                  ✓
        #   · When r̄ = μ_EMA = c ≠ 0: penalty = c²  (matches true penalty)  ✓
        #   · Sign: can be negative early in training (zero-initialised EMA
        #     vs. positive batch mean), but the gradient direction is correct.

        active  = nonempty                          # mask for active bins
        K_prime = active.sum().float()              # number of active bins

        # Only sum over active bins to avoid polluting the mean with zeros
        # from empty bins (matches the original implementation's `nonempty` mask).
        linearised_terms = (
            2.0 * mu_ema[active] * batch_bin_means[active]
            - mu_ema[active] ** 2
        )
        cond_bias_penalty = linearised_terms.sum() / K_prime

        # ── EMA update ────────────────────────────────────────────────────────
        # Applied AFTER the penalty so that the gradient at this step uses the
        # EMA from the *previous* step — a clean separation analogous to a
        # target network in RL.  Only runs in training mode so that validation
        # forward passes do not corrupt the accumulated EMA.
        # batch_bin_means already contains global statistics (via the
        # straight-through sync), so its .detach() is the global bin mean.
        if self.training:
            with torch.no_grad():
                upd = nonempty
                if upd.any():
                    self.bin_mean_ema[upd] = (
                        self.ema_decay * self.bin_mean_ema[upd]
                        + (1.0 - self.ema_decay) * batch_bin_means.detach()[upd]
                    )
                    self.bin_update_count[upd] += 1

        return mse + self.lambda_cond * cond_bias_penalty




class BetaNLLLoss(nn.Module):
    """
    β-NLL Loss with Stop-Gradient (Seitzer et al., 2022).

    Behaves identically to Gaussian NLL in the forward pass, but scales the
    gradient that flows through the variance term by (1 − β), which stabilises
    training by preventing variance collapse / explosion.

    Clamping note: this loss does NOT clamp log_var.  The ResNet model's forward
    pass guarantees that log_var is already bounded.

    Parameters
    ----------
    beta : float
        β = 0.0 → only the mean is trained (variance frozen).
        β = 0.5 → recommended balanced setting.
        β = 1.0 → standard Gaussian NLL (full variance gradients).
    reduction : str
        'mean', 'sum', or 'none'.
    """

    def __init__(self, beta: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        assert beta >= 0, f"Beta must be non-negative, got {beta}"
        if beta > 0.8:
            warnings.warn(
                f"BetaNLLLoss: beta={beta} > 0.8 may lead to variance collapse. "
                "Recommended range: [0.3, 0.7]."
            )
        self.beta = beta
        self.reduction = reduction

    def forward(self, mean: torch.Tensor, log_var: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if mean.dim() > 1:
            mean = mean.view(-1)
        if log_var.dim() > 1:
            log_var = log_var.view(-1)
        if target.dim() > 1:
            target = target.view(-1)

        var = torch.exp(log_var)
        squared_error = (target - mean) ** 2

        if self.beta > 0:
            var_beta_sg = var.pow(self.beta).detach()
            var_one_minus_beta = var.pow(1 - self.beta)
            effective_var = var_beta_sg * var_one_minus_beta
        else:
            effective_var = var

        loss = 0.5 * (squared_error / effective_var + log_var)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class GaussianNLLLoss(nn.Module):
    """
    Standard Gaussian Negative Log-Likelihood (equivalent to BetaNLLLoss with β=1.0).

    Clamping removed: the ResNet model guarantees bounded log_var.
    """

    def __init__(self, reduction: str = 'mean', epsilon: float = 1e-6):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), f"Invalid reduction: {reduction}"
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, mean: torch.Tensor, log_var: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if mean.dim() > 1:
            mean = mean.squeeze()
        if log_var.dim() > 1:
            log_var = log_var.squeeze()
        if target.dim() > 1:
            target = target.squeeze()

        precision = torch.exp(-log_var)
        squared_error = (target - mean) ** 2
        loss = 0.5 * (precision * squared_error + log_var)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class OriginalbetaNLLLoss(nn.Module):
    """
    Seitzer et al. (2022) β-NLL with stop-gradient.

    Functionally identical to BetaNLLLoss; kept as a separate class for
    checkpoint compatibility.  Clamping removed (model handles it).
    """

    def __init__(self, beta: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, mean: torch.Tensor, log_var: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        mean = mean.view(-1)
        log_var = log_var.view(-1)
        target = target.view(-1)

        var = torch.exp(log_var)
        squared_error = (target - mean) ** 2

        var_beta_sg = var.pow(self.beta).detach()
        var_one_minus_beta = var.pow(1 - self.beta)
        effective_var = var_beta_sg * var_one_minus_beta

        loss = 0.5 * (squared_error / effective_var + log_var)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SlopePenalizedBetaNLLLoss(nn.Module):
    """
    Heteroscedastic β-NLL loss with slope and bias penalties.

    Combines the Seitzer et al. stop-gradient β-NLL with explicit slope/bias
    penalties to reduce regression-to-mean bias while maintaining calibrated
    uncertainties.

        Loss = β-NLL + λ_slope * (1 - slope)² + λ_bias * bias²

    Clamping removed: the ResNet model guarantees bounded log_var.

    Parameters
    ----------
    beta : float
        β parameter (0.5 is typical; 1.0 = standard NLL).
    lambda_slope : float
        Penalty strength for slope deviation from 1.0.
    lambda_bias : float
        Penalty strength for mean bias (often 0).
    min_var : float
        Small additive floor on variance for numerical stability.
    min_target_var : float
        Minimum target variance in a batch before slope penalty is skipped.
    verbose : bool
        Print detailed diagnostics every 50 forward calls.
    """

    def __init__(self, beta=0.5, lambda_slope=10.0, lambda_bias=0.0, min_var=1e-6,
                 min_target_var=1e-3, verbose=False):
        super().__init__()
        assert beta >= 0, f"Beta must be non-negative, got {beta}"
        self.beta = beta
        self.lambda_slope = lambda_slope
        self.lambda_bias = lambda_bias
        self.min_var = min_var
        self.min_target_var = min_target_var
        self.verbose = verbose
        self._call_count = 0

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        if mu.dim() > 1:
            mu = mu.view(-1)
        if log_var.dim() > 1:
            log_var = log_var.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)

        var = torch.exp(log_var) + self.min_var
        sigma = torch.sqrt(var)
        residuals = mu - targets
        sq_residuals = residuals ** 2

        # β-NLL with Seitzer stop-gradient
        var_beta_sg = var.pow(self.beta).detach()
        var_one_minus_beta = var.pow(1 - self.beta)
        effective_var = var_beta_sg * var_one_minus_beta
        nll = sq_residuals / (2 * effective_var) + 0.5 * log_var
        nll_mean = torch.mean(nll)

        # Slope penalty (batch OLS estimator)
        mu_centered = mu - mu.mean()
        target_centered = targets - targets.mean()
        covariance = torch.mean(mu_centered * target_centered)
        target_variance = torch.mean(target_centered ** 2)

        if target_variance > self.min_target_var:
            slope = covariance / target_variance
            slope_penalty = (1.0 - slope) ** 2
        else:
            slope = torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
            slope_penalty = torch.tensor(0.0, device=mu.device, dtype=mu.dtype)

        bias = torch.mean(residuals)
        bias_penalty = bias ** 2
        loss = nll_mean + self.lambda_slope * slope_penalty + self.lambda_bias * bias_penalty

        if self.verbose:
            self._call_count += 1
            should_print = (self._call_count % 50 == 0) or torch.isinf(loss) or torch.isnan(loss)
            if should_print:
                with torch.no_grad():
                    rmse = torch.sqrt(torch.mean(sq_residuals))
                    mean_sigma = torch.mean(sigma)
                    sigma_std = torch.std(sigma)
                    calibration_ratio = mean_sigma / (rmse + 1e-8)
                    has_inf = torch.isinf(loss) or torch.isinf(nll_mean)
                    has_nan = torch.isnan(loss) or torch.isnan(nll_mean)
                    log_var_mean = log_var.mean().item()
                    log_var_min = log_var.min().item()
                    log_var_max = log_var.max().item()
                    print(f"\n  {'='*70}")
                    print(f"  [SlopePenalizedBetaNLL Diagnostics - Call #{self._call_count}]")
                    print(f"  {'='*70}")
                    print(f"  Loss Components:")
                    print(f"    Total Loss      = {loss.item():.6f} {'[!!! INF !!!]' if has_inf else ''}")
                    print(f"    NLL term        = {nll_mean.item():.6f}")
                    print(f"    Slope penalty   = {(self.lambda_slope * slope_penalty).item():.6f}")
                    print(f"    Bias penalty    = {(self.lambda_bias * bias_penalty).item():.6f}")
                    print(f"  Regression Metrics:")
                    print(f"    Slope           = {slope.item():.4f}  (target: 1.0)")
                    print(f"    Bias            = {bias.item():.6f}")
                    print(f"    RMSE            = {rmse.item():.6f}")
                    print(f"    Target variance = {target_variance.item():.6f}"
                          f"  (threshold: {self.min_target_var})")
                    print(f"  Uncertainty Metrics:")
                    print(f"    Mean σ          = {mean_sigma.item():.6f}")
                    print(f"    Std(σ)          = {sigma_std.item():.6f}")
                    print(f"    log_var range   = [{log_var_min:.2f}, {log_var_max:.2f}]")
                    print(f"    Calibration     = {calibration_ratio.item():.3f}  (ideal: 1.0)")
                    print(f"  Beta parameter  = {self.beta}")
                    print(f"  Expected σ²/res² ≈ {1-self.beta:.2f}  (β-NLL equilibrium)")
                    if has_nan:
                        print(f"  ⚠️  WARNING: NaN detected in loss!")
                    print(f"  {'='*70}\n")

        return loss


# --------------------------------------------------------------------- #
#  Training health monitoring
# --------------------------------------------------------------------- #

class TrainingHealthMonitor:
    """
    Monitors training health and detects prediction/gradient collapse.

    After check_health() accumulates `patience` consecutive unhealthy checks,
    should_intervene() returns True and the diagnostics dict contains
    'needs_intervention': True, so the training loop can trigger recovery.
    """

    def __init__(self, patience: int = 5, min_pred_std: float = 0.01,
                 min_grad_active_ratio: float = 0.3, rank: int = 0):
        self.patience = patience
        self.min_pred_std = min_pred_std
        self.min_grad_active_ratio = min_grad_active_ratio
        self.rank = rank
        self.collapse_counter = 0
        self.total_checks = 0
        self.intervention_count = 0

    def check_health(self, predictions: torch.Tensor,
                     model: nn.Module) -> Tuple[bool, dict]:
        """Returns (is_healthy, diagnostics_dict)."""
        self.total_checks += 1
        diagnostics = {}
        is_healthy = True

        with torch.no_grad():
            pred_std = predictions.std().item()
            pred_range = (predictions.max() - predictions.min()).item()

        diagnostics['pred_std'] = pred_std
        diagnostics['pred_range'] = pred_range
        diagnostics['prediction_collapse'] = pred_std < self.min_pred_std
        if diagnostics['prediction_collapse']:
            is_healthy = False

        total_params_with_grad = 0
        active_gradients = 0
        for param in model.parameters():
            if param.grad is not None:
                total_params_with_grad += 1
                if param.grad.norm().item() > 1e-7:
                    active_gradients += 1
        grad_active_ratio = (active_gradients / total_params_with_grad
                             if total_params_with_grad > 0 else 0.0)
        diagnostics['grad_active_ratio'] = grad_active_ratio
        diagnostics['gradient_collapse'] = grad_active_ratio < self.min_grad_active_ratio
        if diagnostics['gradient_collapse']:
            is_healthy = False

        if not is_healthy:
            self.collapse_counter += 1
        else:
            self.collapse_counter = 0

        diagnostics['collapse_counter'] = self.collapse_counter
        diagnostics['needs_intervention'] = self.collapse_counter >= self.patience
        return is_healthy, diagnostics

    def should_intervene(self) -> bool:
        return self.collapse_counter >= self.patience

    def reset(self) -> None:
        self.collapse_counter = 0
        self.intervention_count += 1

    def get_summary(self) -> dict:
        return {
            'total_checks': self.total_checks,
            'intervention_count': self.intervention_count,
            'current_collapse_counter': self.collapse_counter,
        }


def calibrate_batchnorm(model: nn.Module, train_loader, device,
                         num_batches: int = 100, rank: int = 0) -> None:
    """
    Recalibrate BatchNorm running statistics after emergency reinitialization.

    Runs the model in train() mode for `num_batches` batches without updating
    weights, so that the running_mean / running_var buffers are re-estimated on
    real data.  This must be called before the next validation pass.
    """
    model.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if i >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
    if rank == 0:
        print(f"    BatchNorm calibrated with {min(i + 1, num_batches)} batches")


# --------------------------------------------------------------------- #
#  Diagnostics
# --------------------------------------------------------------------- #

class TrainingDiagnostics:
    """Comprehensive diagnostic monitor for training stability."""

    def __init__(self, rank: int = 0, enable_detailed: bool = False):
        self.rank = rank
        self.enable_detailed = enable_detailed
        self.loss_history: list = []
        self.grad_norm_history: list = []
        self.issue_count: int = 0

    def quick_check(self, loss: torch.Tensor, outputs: torch.Tensor,
                    batch_target: torch.Tensor) -> bool:
        """Fast per-batch sanity check for NaN / Inf and stuck losses.
        Returns True if any issue was detected.  Issues are counted on rank 0 only."""
        issues = []
        if torch.isnan(loss).any():
            issues.append("Loss is NaN")
        if torch.isnan(outputs).any():
            issues.append("Outputs contain NaN")
        if torch.isnan(batch_target).any():
            issues.append("Targets contain NaN")
        if torch.isinf(outputs).any():
            issues.append("Outputs contain Inf")
        if torch.isinf(batch_target).any():
            issues.append("Targets contain Inf")
        if len(self.loss_history) > 5:
            if np.std(self.loss_history[-5:]) < 1e-10:
                issues.append(f"Loss stuck at {loss.item():.6f}")

        # Count issues only on rank 0 to avoid double-counting across GPUs
        if issues and self.rank == 0:
            print(f"\n⚠️  Quick check found issues: {', '.join(issues)}")
            self.issue_count += len(issues)

        return len(issues) > 0

    def diagnose_batch(self, model: nn.Module, batch_input: torch.Tensor,
                       batch_target: torch.Tensor, loss: torch.Tensor,
                       outputs: torch.Tensor | None = None,
                       epoch: int | None = None, batch_idx: int | None = None,
                       optimizer=None, criterion=None) -> None:
        """Full diagnostic pass: loss, output stats, gradients, parameters, activations."""
        if self.rank != 0:
            return

        header = "FULL DIAGNOSTIC CHECK"
        if epoch is not None and batch_idx is not None:
            header = f"FULL DIAGNOSTIC  epoch={epoch}  batch={batch_idx}"

        print(f"\n{'='*70}")
        print(header)
        print(f"{'='*70}")

        self._check_loss(loss)

        if outputs is not None:
            self._check_outputs_targets(outputs, batch_target)
        else:
            self._check_outputs_targets(batch_input, batch_target)

        self._check_gradients(model)
        self._check_parameters(model)
        self._check_activations(model, batch_input)

        print(f"{'='*70}\n")

    def _check_loss(self, loss: torch.Tensor) -> None:
        print(f"Loss value: {loss.item():.6f}")
        if torch.isnan(loss):
            print("  ⚠️  CRITICAL: Loss is NaN!")
            self.issue_count += 1
        if torch.isinf(loss):
            print("  ⚠️  CRITICAL: Loss is Inf!")
            self.issue_count += 1

    def _check_outputs_targets(self, outputs: torch.Tensor,
                                targets: torch.Tensor) -> None:
        print("\nPrediction statistics:")
        print(f"  Mean: {outputs.mean().item():.6f}")
        print(f"  Std:  {outputs.std().item():.6f}")
        print(f"  Min:  {outputs.min().item():.6f}")
        print(f"  Max:  {outputs.max().item():.6f}")
        if torch.isnan(outputs).any():
            print("  ⚠️  WARNING: Outputs contain NaN!")
            self.issue_count += 1
        if torch.isinf(outputs).any():
            print("  ⚠️  WARNING: Outputs contain Inf!")
            self.issue_count += 1

        print("\nTarget statistics:")
        print(f"  Mean: {targets.mean().item():.6f}")
        print(f"  Std:  {targets.std().item():.6f}")
        print(f"  Min:  {targets.min().item():.6f}")
        print(f"  Max:  {targets.max().item():.6f}")

    def _check_gradients(self, model: nn.Module) -> None:
        total_norm = 0.0
        num_params = 0
        num_zero_grads = 0
        num_nan_grads = 0
        max_grad = float('-inf')
        min_grad = float('inf')

        for name, param in model.named_parameters():
            if param.grad is not None:
                num_params += 1
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                g_min = param.grad.min().item()
                g_max = param.grad.max().item()
                max_grad = max(max_grad, g_max)
                min_grad = min(min_grad, g_min)
                if torch.isnan(param.grad).any():
                    num_nan_grads += 1
                if (param.grad.abs() < 1e-15).all():
                    num_zero_grads += 1

        total_norm = total_norm ** 0.5
        self.grad_norm_history.append(total_norm)

        print(f"\nGradient statistics:")
        print(f"  Total gradient norm: {total_norm:.6e}")
        print(f"  Parameters with gradients: {num_params}")
        print(f"  Zero-gradient params: {num_zero_grads}")
        print(f"  NaN-gradient params:  {num_nan_grads}")
        if num_params > 0:
            print(f"  Gradient range: [{min_grad:.6e}, {max_grad:.6e}]")

        if num_nan_grads > 0:
            print(f"  ⚠️  CRITICAL: {num_nan_grads} params have NaN gradients!")
            self.issue_count += 1
        if num_params > 0 and num_zero_grads == num_params:
            print("  ⚠️  CRITICAL: ALL gradients are zero!")
            self.issue_count += 1
        elif num_params > 0 and num_zero_grads > num_params * 0.5:
            print(f"  ⚠️  WARNING: {num_zero_grads}/{num_params} gradients are zero")
            self.issue_count += 1
        if total_norm < 1e-10:
            print(f"  ⚠️  WARNING: Gradient norm extremely small ({total_norm:.6e})")
            self.issue_count += 1

    def _check_parameters(self, model: nn.Module) -> None:
        total_params = 0
        total_norm = 0.0
        num_nan_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()
            total_norm += param.data.norm(2).item() ** 2
            if torch.isnan(param).any():
                num_nan_params += 1
                print(f"  ⚠️  WARNING: Parameter '{name}' contains NaN!")
                self.issue_count += 1

        total_norm = total_norm ** 0.5
        print(f"\nModel parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Parameter norm: {total_norm:.6e}")
        print(f"  Params with NaN: {num_nan_params}")

    def _check_activations(self, model: nn.Module, batch_input: torch.Tensor) -> None:
        activations = {}

        def hook_fn(name):
            def hook(module, inp, output):
                activations[name] = output.detach()
            return hook

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.LeakyReLU, nn.ReLU, nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        with torch.no_grad():
            _ = model(batch_input)

        print("\nSample intermediate activations (first 10 layers):")
        for name, act in list(activations.items())[:10]:
            mean = act.mean().item()
            std = act.std().item()
            zero_frac = (act == 0).float().mean().item()
            has_nan = torch.isnan(act).any().item()
            print(f"  {name[:40]:40s} | "
                  f"mean={mean:8.3e}, std={std:8.3e}, "
                  f"zeros={zero_frac*100:5.1f}%, nan={has_nan}")
            if zero_frac > 0.99:
                print("    ⚠️  WARNING: >99% of activations are zero")
                self.issue_count += 1

        for hook in hooks:
            hook.remove()


def create_diagnostics_summary(diagnostics: TrainingDiagnostics, epoch: int) -> None:
    """Print a summary of diagnostic statistics at the end of an epoch."""
    if diagnostics.rank != 0:
        return

    print(f"\n{'#'*70}")
    print(f"DIAGNOSTIC SUMMARY - End of Epoch {epoch}")
    print(f"{'#'*70}")
    print(f"Total issues detected: {diagnostics.issue_count}")

    if diagnostics.loss_history:
        recent = diagnostics.loss_history[-20:]
        print(f"\nLoss history (last {len(recent)} values):")
        print(f"  Mean: {np.mean(recent):.6f}")
        print(f"  Std:  {np.std(recent):.6e}")
        print(f"  Min:  {np.min(recent):.6f}")
        print(f"  Max:  {np.max(recent):.6f}")

    if diagnostics.grad_norm_history:
        recent_g = diagnostics.grad_norm_history[-20:]
        print(f"\nGradient norm history (last {len(recent_g)} values):")
        print(f"  Mean: {np.mean(recent_g):.6e}")
        print(f"  Std:  {np.std(recent_g):.6e}")
        print(f"  Min:  {np.min(recent_g):.6e}")
        print(f"  Max:  {np.max(recent_g):.6e}")

    print(f"{'#'*70}\n")


def print_data_diagnostics(train_targets: np.ndarray, rank: int = 0) -> None:
    """
    Print initial data diagnostics.

    Always runs regardless of --diagnostic-mode, since data-level issues
    (e.g. NaN, extreme ranges) must be caught before training starts.
    """
    if rank != 0:
        return

    print("\n" + "="*70)
    print("INITIAL DATA DIAGNOSTICS")
    print("="*70)
    print("\nTraining set target statistics:")
    print(f"  Count: {len(train_targets)}")
    print(f"  Mean:  {train_targets.mean():.6f}")
    print(f"  Std:   {train_targets.std():.6f}")
    print(f"  Min:   {train_targets.min():.6f}")
    print(f"  Max:   {train_targets.max():.6f}")
    print(f"  Percentiles:")
    for pct in [1, 25, 50, 75, 99]:
        print(f"    {pct:3d}%:  {np.percentile(train_targets, pct):.6f}")

    if train_targets.std() < 1e-6:
        print("\n  ⚠️  WARNING: Target std is extremely small — training will be unstable!")
    if np.isnan(train_targets).any():
        print(f"\n  ⚠️  CRITICAL: Training targets contain {np.isnan(train_targets).sum()} NaN values!")
    if np.isinf(train_targets).any():
        print(f"\n  ⚠️  CRITICAL: Training targets contain {np.isinf(train_targets).sum()} Inf values!")

    print("="*70 + "\n")


# --------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------- #

def stratified_evaluation(model, test_loader, device,
                           absolute_threshold=0.5, relative_threshold=0.2):
    """
    Comprehensive stratified evaluation of a trained model.

    Automatically detects heteroscedastic vs. standard models from the forward
    output shape.  For heteroscedastic models the log_var is guaranteed bounded
    by the model's forward pass, so std = sqrt(exp(log_var)) is always finite.

    Returns
    -------
    Standard model:
        predictions, actuals : np.ndarray
    Heteroscedastic model:
        predictions, uncertainties, actuals : np.ndarray
        (uncertainties are predicted standard deviations σ)
    """
    model.eval()
    all_outputs, all_uncertainties, all_targets = [], [], []

    with torch.no_grad():
        sample_images = next(iter(test_loader))[0][:1].to(device)
        sample_output = model(sample_images)
        is_heteroscedastic = isinstance(sample_output, tuple) and len(sample_output) == 2

    with torch.no_grad():
        for images, signal_amplitudes in test_loader:
            images = images.to(device)
            signal_amplitudes = signal_amplitudes.to(device).float()

            if is_heteroscedastic:
                mean_pred, log_var_pred = model(images)
                std_pred = torch.sqrt(torch.exp(log_var_pred))
                all_outputs.extend(mean_pred.cpu().numpy())
                all_uncertainties.extend(std_pred.cpu().numpy())
            else:
                outputs = model(images)
                all_outputs.extend(outputs.cpu().numpy())

            all_targets.extend(signal_amplitudes.cpu().numpy())

    outputs = np.array(all_outputs)
    targets = np.array(all_targets)
    abs_errors = np.abs(outputs - targets)
    relative_errors = abs_errors / (targets + 1e-8)

    print(f'\n{"="*60}')
    print("FINAL TEST RESULTS - COMPREHENSIVE EVALUATION")
    print(f'{"="*60}')

    print("\nOverall Performance:")
    print(f"  Mean Absolute Error (MAE): {abs_errors.mean():.4f}")
    print(f"  Root Mean Squared Error (RMSE): {np.sqrt((abs_errors**2).mean()):.4f}")
    print(f"  Mean Relative Error: {relative_errors.mean()*100:.2f}%")
    print(f"  Median Relative Error: {np.median(relative_errors)*100:.2f}%")

    abs_accuracy = (abs_errors < absolute_threshold).mean() * 100
    rel_accuracy = (relative_errors < relative_threshold).mean() * 100
    print(f"\n  Absolute Accuracy (error < {absolute_threshold}): {abs_accuracy:.1f}%")
    print(f"  Relative Accuracy (error < {relative_threshold*100:.0f}%): {rel_accuracy:.1f}%")

    print(f"\n{'='*60}")
    print("BIAS ANALYSIS")
    print(f"{'='*60}")
    residuals = outputs - targets
    print(f"  Mean Residual (bias): {residuals.mean():.4f}")
    print(f"  Residual Std:         {residuals.std():.4f}")

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(targets, outputs)
    print(f"\n  Regression (pred vs actual):")
    print(f"    Slope:     {slope:.4f}  (ideal=1.0)")
    print(f"    Intercept: {intercept:.4f}  (ideal=0.0)")
    print(f"    R²:        {r_value**2:.4f}")
    if slope < 0.9:
        print("    ⚠️  Slope < 0.9 suggests regression-to-mean bias")

    print(f"\n{'='*60}")
    print("STRATIFIED ANALYSIS BY TARGET RANGE")
    print(f"{'='*60}")

    strata = [
        ("Very Low (0–0.3)",  0.0, 0.3),
        ("Low (0.3–0.7)",     0.3, 0.7),
        ("Medium (0.7–1.3)",  0.7, 1.3),
        ("High (1.3–1.7)",    1.3, 1.7),
        ("Very High (1.7+)",  1.7, float('inf')),
    ]
    for name, low, high in strata:
        mask = (targets >= low) & (targets < high)
        n = mask.sum()
        if n > 0:
            s_mae = abs_errors[mask].mean()
            s_rel = relative_errors[mask].mean() * 100
            s_rel_acc = (relative_errors[mask] < relative_threshold).mean() * 100
            print(f"\n  {name}: n={n}")
            print(f"    MAE: {s_mae:.4f},  Mean Rel Error: {s_rel:.1f}%")
            print(f"    Relative Accuracy: {s_rel_acc:.1f}%")

    if is_heteroscedastic:
        uncertainties = np.array(all_uncertainties)
        n_inf = np.isinf(uncertainties).sum()
        n_nan = np.isnan(uncertainties).sum()

        print(f"\n{'='*60}")
        print("UNCERTAINTY ANALYSIS (Heteroscedastic Model)")
        print(f"{'='*60}")
        print(f"  Mean predicted σ:  {uncertainties.mean():.4f}")
        print(f"  Std of σ:          {uncertainties.std():.4f}")
        print(f"  σ range:           [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")

        if n_inf > 0:
            print(f"  ⚠️  WARNING: {n_inf} infinite uncertainties detected!")
        if n_nan > 0:
            print(f"  ⚠️  WARNING: {n_nan} NaN uncertainties detected!")

        within_1sigma = (abs_errors < uncertainties).mean() * 100
        within_2sigma = (abs_errors < 2 * uncertainties).mean() * 100
        print(f"\n  Calibration:")
        print(f"    Errors within 1σ: {within_1sigma:.1f}%  (ideal: 68%)")
        print(f"    Errors within 2σ: {within_2sigma:.1f}%  (ideal: 95%)")

        corr = np.corrcoef(abs_errors, uncertainties)[0, 1]
        print(f"    Error–uncertainty correlation: {corr:.3f}  (higher = better calibrated)")

        return outputs, uncertainties, targets

    return outputs, targets
