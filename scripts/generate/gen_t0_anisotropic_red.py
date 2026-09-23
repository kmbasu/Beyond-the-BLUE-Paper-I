#!/usr/bin/env python
"""
gen_t0_anisotropic_red.py — Tier-0 driver: FIXED-direction anisotropic noise
============================================================================

Fixed-orientation two-component anisotropic GAUSSIAN noise,

    P(kx, ky) = A_scan^2 / |kx|^alpha_scan + A_iso^2 / |k|^alpha_iso,

with NO nonlinear processing.  This is the Tier-0 control completing the
anisotropy triptych (added 2026-08-27 — the case
was implicit in gen_t2_aniso_median with use_mod2_row_median=False but
deserves its own Tier-0 driver for taxonomic clarity):

* HERE (Tier-0)  : anisotropic, fixed direction, Gaussian.  The covariance
  is fixed and known; per M4 Sec. 5.1 "fixed-direction anisotropy is
  Gaussian" — an MF built with the full 2-D PSD is the MVUE, eta = 1, and
  no CNN advantage exists.  The instructive twist: an MF built with a
  RADIALLY AVERAGED 1-D PS is a mismatched linear filter here (it
  isotropizes the stripe power), so this dataset cleanly demonstrates
  that "use the 2-D PSD" is a requirement of correct LINEAR estimation,
  not a CNN capability.
* gen_t1_random_orientation (Tier-1) : same spectrum, direction random
  per image — no fixed filter is per-image optimal; conditionally
  Gaussian covariance mixture.
* gen_t2_aniso_median (Tier-2) : fixed direction + row-median removal —
  non-Gaussianity from nonlinear processing.

The noise-only ensemble doubles as a Gaussian null test of the eta
pipeline with ANISOTROPIC 2-D whitening (eta memo Sec. 15.1) — a stricter
null than the isotropic Tier-0 cases.

Pipeline per image:
  1. beta-model signal (or zeros)
  2. anisotropic 1/f noise (fixed direction, stripes along x)
  3. beam smoothing of (signal + noise);  white noise added after, unsmoothed

Run in Spyder cell-by-cell (#%%) or as a script.
"""

#%% ================================================================================
# === IMPORTS AND PATH SETUP ===
# ==================================================================================

import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
for _p in (_here, *_here.parents):
    if (_p / 'noise_lib').is_dir():
        sys.path.insert(0, str(_p))
        break

import noise_lib as nl

#%% ================================================================================
# === CONFIG ===
# ==================================================================================

model_name = 'T0_ANISO_RED'
output_h5  = 'T0_aniso_red_test.h5'

num_images  = 400
image_size  = 128
master_seed = 12345

# --- source ---
source_type = 'beta'
flux_mode   = 'lin'
rc_mean, rc_sigma = 7.0, 0.0
I0_range   = (0.0, 5.0)
I0_fixed   = None
I0_mean, I0_sigma = 1.0, 0.3

noise_only         = False
zero_fraction      = 0.0
zero_deterministic = True

# --- anisotropic noise (same defaults as gen_t2_aniso_median) ---
one_over_f_slope_iso      = 3.0
one_over_f_amplitude_iso  = 5.0
one_over_f_slope_scan     = 3.0
one_over_f_amplitude_scan = 0.5

white_noise_amplitude   = 0.0
gaussian_smoothing_fwhm = 5.0

params = dict(
    model_name=model_name, num_images=num_images, image_size=image_size,
    source_type=source_type, flux_mode=flux_mode, rc_mean=rc_mean,
    rc_sigma=rc_sigma, I0_range=I0_range, I0_fixed=I0_fixed, I0_mean=I0_mean,
    I0_sigma=I0_sigma, noise_only=noise_only, zero_fraction=zero_fraction,
    zero_deterministic=zero_deterministic,
    one_over_f_slope_iso=one_over_f_slope_iso,
    one_over_f_amplitude_iso=one_over_f_amplitude_iso,
    one_over_f_slope_scan=one_over_f_slope_scan,
    one_over_f_amplitude_scan=one_over_f_amplitude_scan,
    white_noise_amplitude=white_noise_amplitude,
    gaussian_smoothing_fwhm=gaussian_smoothing_fwhm)

#%% ================================================================================
# === MAIN LOOP ===
# ==================================================================================

shape = (image_size, image_size)
P2d = nl.psd_two_component(shape,
                           one_over_f_slope_scan, one_over_f_amplitude_scan,
                           one_over_f_slope_iso, one_over_f_amplitude_iso)

ss = np.random.SeedSequence(master_seed)
child_seeds = ss.spawn(num_images + 1)
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

    w = nl.white_noise(shape, white_noise_amplitude, rng)
    n1f = nl.grf_from_psd(P2d, rng)

    final = nl.beam_smooth(img + n1f, gaussian_smoothing_fwhm) + w

    images_clean.append(img.astype(np.float32))
    images_noisy.append(final.astype(np.float32))
    labels_I0.append(float(I0))
    has_source.append(0 if is_empty else 1)

print(f"Generated {num_images} images  [fixed-direction aniso: "
      f"A_scan={one_over_f_amplitude_scan}  A_iso={one_over_f_amplitude_iso}  "
      f"Gaussian, no processing  "
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
print(f"pixel moments: skew={m['skewness']:.4f}  ex.kurt={m['excess_kurtosis']:.4f}  "
      f"(Gaussian: both ~0 — contrast with gen_t2_aniso_median)")

# Anisotropy check via ensemble 2-D PSD: power along the kx=0 column
# (y-only structure, fed by the scan term) vs the ky=0 row
P_hat = nl.diagnostics.estimate_psd2d(noisy_arr[:100])
aniso = P_hat[1:, 0].mean() / P_hat[0, 1:].mean()
print(f"ensemble 2-D PSD anisotropy (kx=0 col / ky=0 row): {aniso:.2f}  "
      f"(> 1: stripe power along x; an isotropized 1-D PS misses this)")

nl.diagnostics.quicklook(noisy_arr, np.array(labels_I0),
                         title=f'{model_name}: fixed-direction anisotropic (Gaussian)')
