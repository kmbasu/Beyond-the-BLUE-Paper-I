#!/usr/bin/env python
"""
gen_t2_aniso_median.py — Tier-2 driver: anisotropic noise + row-median removal
==============================================================================

The processing-induced non-Gaussianity model of paper Sec. 6.1 (mild case):
fixed-direction two-component anisotropic Gaussian noise,

    P(kx, ky) = A_scan^2 / |kx|^alpha_scan + A_iso^2 / |k|^alpha_iso,

followed by per-row MEDIAN baseline removal — a nonlinear, data-dependent
operator (M4 Sec. 5.3 dichotomy) that converts Gaussian input into mildly
non-Gaussian output.  With ``use_mod2_row_median = False`` the model is
fixed-direction anisotropic GAUSSIAN noise: a Tier-0 control (the MF with
the full 2-D PSD is optimal, eta = 1) — the contrast between the two flags
IS the Sec. 6.1 story ("SET THE DIFFERENCE FROM TIER-1 ANISOTROPIC
SCANNING": here the direction is fixed and the non-Gaussianity comes from
processing; in gen_t1_random_orientation the direction varies and the
noise stays conditionally Gaussian).

Pipeline per image (legacy anisotropic_median.py):
  1. beta-model signal (or zeros)
  2. anisotropic 1/f noise (fixed direction, theta = 0)
  3. beam smoothing of (signal + noise);  white noise added after
  4. [Mod 2] row-median removal on the combined map

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

model_name = 'T2_ANISO_MEDIAN'
output_h5  = 'T2_aniso_median_test.h5'

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

# --- anisotropic noise (legacy defaults) ---
one_over_f_slope_iso      = 3.0
one_over_f_amplitude_iso  = 5.0
one_over_f_slope_scan     = 3.0
one_over_f_amplitude_scan = 0.5

white_noise_amplitude   = 0.0
gaussian_smoothing_fwhm = 5.0

# --- Mod 2: the Tier-2 mechanism of this driver ---
use_mod2_row_median = True     # False -> Gaussian anisotropic control (Tier-0)
mod2_axis           = 1        # 1 = per-row median (scan direction)

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

    if use_mod2_row_median:
        final, _baseline = nl.row_median_removal(final, axis=mod2_axis)

    images_clean.append(img.astype(np.float32))
    images_noisy.append(final.astype(np.float32))
    labels_I0.append(float(I0))
    has_source.append(0 if is_empty else 1)

print(f"Generated {num_images} images  [aniso: A_scan={one_over_f_amplitude_scan} "
      f"A_iso={one_over_f_amplitude_iso}  "
      f"Mod2={'ON' if use_mod2_row_median else 'off'}  "
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
      f"(median removal -> mild non-Gaussianity; flag OFF -> both ~0)")

# Anisotropy check: kx cut (red, scan term) vs ky cut
kxk, kxp = nl.diagnostics.psd_1d_cut(noisy_arr[0], 'kx')
kyk, kyp = nl.diagnostics.psd_1d_cut(noisy_arr[0], 'ky')
print(f"low-k anisotropy (single image): kx-cut={kxp[:5].mean():.1f}  "
      f"ky-cut={kyp[:5].mean():.1f}")

nl.diagnostics.quicklook(noisy_arr, np.array(labels_I0),
                         title=f'{model_name}: anisotropic + row median')
