#!/usr/bin/env python
"""
gen_t2_glitches.py — Tier-2 driver: sub-threshold cosmic-ray glitch residuals
=============================================================================

The faint-numerous-glitch model of paper Sec. 6.1: positive-definite,
beam-convolved glitch stamps (causal exponential along scan x Gaussian
across scan) with Pareto amplitudes, injected on an anisotropic Gaussian
background.  One-signed marks => skewed maps, bispectrum-led Tier-2
non-Gaussianity; a Campbell (marked-Poisson) model in position space, so
the analytic eta forecast of eta memo Sec. 18 applies directly with
(lambda, <a^n>, psi) read off the parameters below — the reason the
per-glitch latents (x0, y0, amp) are STORED in the HDF5.

Expected regime (eta memo Sec. 21.1): perturbative by construction,
eta - 1 ∝ f_s^2 xi^2 — the quantitative post-mortem of the measured
10-20% plateau; severity sweeps move (amp_scale, n_glitches).

Pipeline per image (legacy anisotropic_glitches.py):
  1. beta-model signal (or zeros)
  2. anisotropic 1/f noise
  3. beam smoothing of (signal + noise);  white noise added after
  4. glitch residuals injected into the combined map
  5. [Mod 2] optional row-median removal

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

model_name = 'T2_GLITCHES'
output_h5  = 'T2_glitches_test.h5'

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

# --- background noise (legacy defaults) ---
one_over_f_slope_iso      = 3.0
one_over_f_amplitude_iso  = 5.0
one_over_f_slope_scan     = 3.0
one_over_f_amplitude_scan = 0.5

white_noise_amplitude   = 0.0
gaussian_smoothing_fwhm = 5.0

# --- glitches (legacy defaults) ---
n_glitches         = 25     # events per image (fixed count, legacy convention)
tau_pix            = 5.0    # exponential decay length (pixels)
sigma_beam_glitch  = 2.0    # beam sigma for the scan-direction convolution
sigma_cross        = 2.0    # cross-scan Gaussian sigma (pixels)
amp_scale_glitch   = 0.3    # Pareto scale; mean amplitude ~ 1.67 * amp_scale

# --- Mod 2 ---
use_mod2_row_median = False
mod2_axis           = 1

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
    gaussian_smoothing_fwhm=gaussian_smoothing_fwhm,
    n_glitches=n_glitches, tau_pix=tau_pix,
    sigma_beam_glitch=sigma_beam_glitch, sigma_cross=sigma_cross,
    amp_scale_glitch=amp_scale_glitch,
    use_mod2_row_median=use_mod2_row_median, mod2_axis=mod2_axis)

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
glitch_infos = []   # per-image event lists -> HDF5 latents

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

    final, ginfo = nl.add_glitch_residuals(
        final, n_glitches, tau_pix, amp_scale_glitch,
        sigma_beam=sigma_beam_glitch, sigma_cross=sigma_cross, rng=rng)
    glitch_infos.append(ginfo)

    if use_mod2_row_median:
        final, _baseline = nl.row_median_removal(final, axis=mod2_axis)

    images_clean.append(img.astype(np.float32))
    images_noisy.append(final.astype(np.float32))
    labels_I0.append(float(I0))
    has_source.append(0 if is_empty else 1)

latents = nl.io_h5.pack_event_latents(glitch_infos, ('x0', 'y0', 'amp'), 'glitch')

print(f"Generated {num_images} images  [glitches: n={n_glitches}  tau={tau_pix}px  "
      f"amp_scale={amp_scale_glitch}  |  A_iso={one_over_f_amplitude_iso}  "
      f"Mod2={'ON' if use_mod2_row_median else 'off'}  "
      f"noise_only={'ON' if noise_only else 'off'}]")

#%% ================================================================================
# === SAVE DATASET ===
# ==================================================================================

nl.save_dataset(output_h5, images_noisy, images_clean, labels_I0, has_source,
                params=params, latents=latents, master_seed=master_seed,
                model_name=model_name)

#%% ================================================================================
# === DIAGNOSTICS ===
# ==================================================================================

noisy_arr = np.stack(images_noisy)
m = nl.diagnostics.moment_summary(noisy_arr)
print(f"pixel moments: skew={m['skewness']:.4f} (>0: one-signed glitches)  "
      f"ex.kurt={m['excess_kurtosis']:.4f}")

amps = latents['glitch_amp']
print(f"glitch amplitudes: n={len(amps)}  mean={amps.mean():.3f}  "
      f"max={amps.max():.2f}  (Pareto tail)")

nl.diagnostics.quicklook(noisy_arr, np.array(labels_I0),
                         title=f'{model_name}: sub-threshold glitches')
