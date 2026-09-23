#!/usr/bin/env python
"""
gen_t1_random_orientation.py — Tier-1 driver: randomly-oriented anisotropic noise
=================================================================================

The second Tier-1 model of paper Sec. 5.2, NEWLY IMPLEMENTED in this
refactor (the legacy code only had the fixed-direction anisotropy): the
noise spectrum is the two-component anisotropic model

    P_i(kx, ky) = A_scan^2 / |k_par,i|^alpha_scan
                + A_iso^2  / (kx^2 + ky^2)^(alpha_iso / 2),

    k_par,i = kx cos(theta_i) + ky sin(theta_i),

with the anisotropy ORIENTATION theta_i ~ Uniform(0, pi) drawn per image
(the spectrum is invariant under theta -> theta + pi).  Each image is a
Gaussian random field given theta_i — a conditionally-Gaussian structure
mixture (M4 Sec. 5.2, "random-direction-per-image anisotropy").  The
ensemble-averaged PSD isotropizes the scan term, so the empirical MF is
mismatched on every single image; the recoverable advantage is Tier-1
noise-modeling, bounded by the oracle (per-image true P_i), and the exact
eta_marg follows from 1-D latent quadrature over theta (eta memo Sec. 19).

Per-image latent theta_i is STORED in the HDF5 (required by the oracle MF
and the quadrature).  No median subtraction here — this case must remain
purely conditionally Gaussian.

Pipeline per image:
  1. beta-model signal (or zeros)
  2. rotated two-component anisotropic 1/f noise (angle theta_i)
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

model_name = 'T1_RANDOM_ORIENTATION'
output_h5  = 'T1_random_orientation_test.h5'

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

# --- anisotropic noise (legacy two-component parameters + random angle) ---
one_over_f_slope_iso      = 3.0
one_over_f_amplitude_iso  = 5.0
one_over_f_slope_scan     = 3.0
one_over_f_amplitude_scan = 0.5    # scan/iso ratio sets Tier-1 severity;
                                   # sweep in Phase B if a stronger case is needed

angle_min, angle_max = 0.0, np.pi  # uniform orientation prior

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
    angle_min=angle_min, angle_max=angle_max,
    white_noise_amplitude=white_noise_amplitude,
    gaussian_smoothing_fwhm=gaussian_smoothing_fwhm)

#%% ================================================================================
# === MAIN LOOP ===
# ==================================================================================

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
lat_angles = []

shape = (image_size, image_size)

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

    theta = float(rng.uniform(angle_min, angle_max))
    P_i = nl.psd_two_component(shape,
                               one_over_f_slope_scan, one_over_f_amplitude_scan,
                               one_over_f_slope_iso, one_over_f_amplitude_iso,
                               scan_angle=theta)
    n1f = nl.grf_from_psd(P_i, rng)
    lat_angles.append(theta)

    final = nl.beam_smooth(img + n1f, gaussian_smoothing_fwhm) + w

    images_clean.append(img.astype(np.float32))
    images_noisy.append(final.astype(np.float32))
    labels_I0.append(float(I0))
    has_source.append(0 if is_empty else 1)

latents = {'angle': np.array(lat_angles)}

print(f"Generated {num_images} images  [random-orientation anisotropy: "
      f"A_scan={one_over_f_amplitude_scan}  A_iso={one_over_f_amplitude_iso}  "
      f"theta ~ U({angle_min:.2f}, {angle_max:.2f})  "
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
print(f"pooled pixel moments: skew={m['skewness']:.3f}  "
      f"ex.kurt={m['excess_kurtosis']:.3f}")

# Orientation check on a few images: the periodogram should be elongated
# perpendicular to the stripes, at the recorded angle.
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, idx in zip(axes.ravel(), range(0, 8)):
    ax.imshow(noisy_arr[idx], cmap='RdBu_r')
    ax.set_title(f"θ={np.rad2deg(lat_angles[idx]):.0f}°", fontsize=9)
    ax.axis('off')
fig.suptitle(f'{model_name}: per-image anisotropy orientation (stripes ⊥ θ)')
fig.tight_layout()
plt.show()

nl.diagnostics.quicklook(noisy_arr, np.array(labels_I0),
                         title=f'{model_name}: random-orientation anisotropy')
