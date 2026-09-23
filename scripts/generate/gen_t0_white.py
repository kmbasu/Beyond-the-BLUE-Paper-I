#!/usr/bin/env python
"""
gen_t0_white.py — Tier-0 driver: pure white Gaussian noise
==========================================================

Generates the benchmark dataset of paper Sec. 4.1: a central beta-model
source (beam-convolved) in pure white Gaussian noise.  This is the
CNN-vs-MF calibration case — Gaussian noise with known (flat) covariance,
so eta = 1 and the CNN can never beat the matched filter; the dataset's
role is to anchor the bias-variance phenomenology and validate the
pipeline (the noise-only version of this ensemble doubles as a Gaussian
null test for the eta machinery).

Pipeline per image (legacy map_realnoise convention):
  1. beta-model signal (or zeros)          [pre-beam]
  2. beam smoothing of the signal          (white noise is NEVER smoothed —
                                            it originates in detector
                                            timestreams)
  3. white noise added after smoothing

Output: HDF5 with datasets noisy/clean/I0/has_source and full parameter
attributes (see noise_lib.io_h5).  Set ``noise_only = True`` to produce
the source-free ensemble for the eta-ceiling calculation.

Run in Spyder cell-by-cell (#%%) or as a script:  python gen_t0_white.py
"""

#%% ================================================================================
# === IMPORTS AND PATH SETUP ===
# ==================================================================================

import sys
from pathlib import Path

import numpy as np

# Locate noise_lib by walking up from this file (works from Spyder and CLI)
_here = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
for _p in (_here, *_here.parents):
    if (_p / 'noise_lib').is_dir():
        sys.path.insert(0, str(_p))
        break

import noise_lib as nl

#%% ================================================================================
# === CONFIG ===
# ==================================================================================

model_name = 'T0_WHITE'
output_h5  = 'T0_white_test.h5'

num_images  = 400
image_size  = 128
master_seed = 12345

# --- source ---
source_type = 'beta'      # 'beta' | 'point' | 'none'
flux_mode   = 'lin'       # 'lin' | 'log' | 'gauss'
rc_mean, rc_sigma = 7.0, 0.0
I0_range   = (0.0, 5.0)
I0_fixed   = None
I0_mean, I0_sigma = 1.0, 0.3

noise_only         = False   # True -> source off everywhere (eta ensembles)
zero_fraction      = 0.0     # fraction of source-free images
zero_deterministic = True    # exact count at random positions (legacy semantics)

# --- noise ---
white_noise_amplitude   = 1.0
gaussian_smoothing_fwhm = 5.0    # beam FWHM (pixels); applied to signal only here

params = {k: v for k, v in dict(
    model_name=model_name, num_images=num_images, image_size=image_size,
    source_type=source_type, flux_mode=flux_mode, rc_mean=rc_mean,
    rc_sigma=rc_sigma, I0_range=I0_range, I0_fixed=I0_fixed, I0_mean=I0_mean,
    I0_sigma=I0_sigma, noise_only=noise_only, zero_fraction=zero_fraction,
    zero_deterministic=zero_deterministic,
    white_noise_amplitude=white_noise_amplitude,
    gaussian_smoothing_fwhm=gaussian_smoothing_fwhm).items()}

#%% ================================================================================
# === MAIN LOOP ===
# ==================================================================================

ss = np.random.SeedSequence(master_seed)
child_seeds = ss.spawn(num_images + 1)          # [-1] reserved for bookkeeping
zrng = np.random.default_rng(child_seeds[-1])

if noise_only:
    zero_indices = set(range(num_images))
elif zero_deterministic and zero_fraction > 0:
    n_zero = int(round(zero_fraction * num_images))
    zero_indices = set(zrng.choice(num_images, size=n_zero, replace=False))
else:
    zero_indices = None

images_noisy, images_clean, labels_I0, has_source = [], [], [], []

for i in range(num_images):
    rng = np.random.default_rng(child_seeds[i])

    if zero_indices is not None:
        is_empty = i in zero_indices
    else:
        is_empty = rng.random() < zero_fraction

    img, I0, _rc = nl.draw_source(
        rng, image_size, source_type=source_type, flux_mode=flux_mode,
        rc_mean=rc_mean, rc_sigma=rc_sigma, I0_range=I0_range,
        I0_fixed=I0_fixed, I0_mean=I0_mean, I0_sigma=I0_sigma,
        is_empty=is_empty)

    w = nl.white_noise((image_size, image_size), white_noise_amplitude, rng)
    final = nl.beam_smooth(img, gaussian_smoothing_fwhm) + w

    images_clean.append(img.astype(np.float32))
    images_noisy.append(final.astype(np.float32))
    labels_I0.append(float(I0))
    has_source.append(0 if is_empty else 1)

print(f"Generated {num_images} images  [white sigma={white_noise_amplitude}  "
      f"beam FWHM={gaussian_smoothing_fwhm}px  "
      f"noise_only={'ON' if noise_only else 'off'}]")

#%% ================================================================================
# === SAVE DATASET ===
# ==================================================================================

nl.save_dataset(output_h5, images_noisy, images_clean, labels_I0, has_source,
                params=params, latents=None, master_seed=master_seed,
                model_name=model_name)

#%% ================================================================================
# === DIAGNOSTICS ===
# ==================================================================================

noisy_arr = np.stack(images_noisy)
m = nl.diagnostics.moment_summary(noisy_arr)
print(f"pixel moments: std={m['std']:.4f} (expect ~{white_noise_amplitude})  "
      f"skew={m['skewness']:.4f}  ex.kurt={m['excess_kurtosis']:.4f}  (expect ~0)")
nl.diagnostics.quicklook(noisy_arr, np.array(labels_I0),
                         title=f'{model_name}: white noise benchmark')
