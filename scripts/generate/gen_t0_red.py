#!/usr/bin/env python
"""
gen_t0_red.py — Tier-0 driver: isotropic 1/f^alpha ("atmospheric") noise
========================================================================

Generates the Gaussian red-noise datasets of paper Secs. 4.2-4.3 in their
three configurations, all controlled from the CONFIG cell:

* pure 1/f^3         : white_noise_amplitude = 0            (Sec. 4.2)
* "realistic"        : white_noise_amplitude > 0 — beam-smoothed red noise
                       plus UNSMOOTHED white noise           (Sec. 4.2)
* high-pass filtered : use_highpass = True — linear k-space filter with
                       POWER transfer T(k) = 1 - exp(-k^2/2 k_filt^2),
                       applied to the FULLY ASSEMBLED map    (Sec. 4.3)

High-pass safety (DELIBERATE CHANGE vs legacy Mod 1).  The legacy
modified_noisemap.ipynb applied the filter to the 1/f noise map ALONE,
before the signal was added.  With a source present that is exactly the
"filter the noise but not the signal" variant that M4 Sec. 5.1 forbids:
since T(0) = 0, the noise DC mode becomes exactly noiseless while the
signal keeps its DC content — a noiseless signal channel, the same
manufactured-advantage trap as the DC-zeroing cautionary tale (paper
Sec. 4.4).  Here the filter is instead applied to the assembled map
(signal + red + white TOGETHER), the physical "filter applied to the
data" operation: per-mode SNR is invariant, the noise stays Gaussian, and
the MF built with the FILTERED template tau' (amplitude x sqrt(T)) and the
filtered PSD remains the MVUE — eta = 1.  Any MF / eta run on a
use_highpass dataset MUST use that filtered template.

All three configurations are Gaussian with fixed, known covariance:
eta = 1, no CNN advantage exists (M4 Sec. 5.1), and the noise-only
ensembles double as the Gaussian null tests of the eta pipeline (eta memo
Sec. 15.1).

Pipeline per image:
  1. beta-model signal (or zeros)
  2. isotropic 1/f noise
  3. beam smoothing of (signal + 1/f);  white noise added after, unsmoothed
  4. [optional] high-pass filter on the assembled map (signal + all noise)

Amplitude note: the GRF generator delivers the full target spectrum
P = A^2/|k|^alpha; legacy strict_1overf_noise delivered P/2 (see
noise_lib.grf).  To match a legacy dataset's noise level, use
A_legacy/sqrt(2).

Run in Spyder cell-by-cell (#%%) or as a script:  python gen_t0_red.py
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

model_name = 'T0_RED'
output_h5  = 'T0_red_filtered.h5'

num_images  = 400
image_size  = 128
master_seed = 42

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

# --- noise ---
one_over_f_slope     = 3.0    # alpha : P ~ 1/|k|^alpha
one_over_f_amplitude = 5.0    # A     (full-spectrum convention; see docstring)

white_noise_amplitude   = 0.0    # > 0 -> "realistic" configuration
gaussian_smoothing_fwhm = 5.0    # beam FWHM (pixels)

# --- high-pass filter (applied to the assembled map; see docstring) ---
# T(k) = 1 - exp(-k^2 / 2 kfilt_scale^2) is the POWER transfer (Gaussian-
# complement taper): T(0)=0 (DC removed), T(kfilt_scale) = 1 - e^{-1/2}
# ~ 0.39 (the ~half-power point), T -> 1 for k >> kfilt_scale.  Guideline:
# keep kfilt_scale well below the source's characteristic frequency
# k_src ~ 1/(2 pi rc) ~ 0.023 cyc/px for rc = 7 px, so the template loses
# little power (kfilt_scale = 0.01 -> ~7% power attenuation at k_src).
use_highpass = False
kfilt_scale  = 0.01              # cycles/pixel (np.fft.fftfreq units, 0-0.5)

params = dict(
    model_name=model_name, num_images=num_images, image_size=image_size,
    source_type=source_type, flux_mode=flux_mode, rc_mean=rc_mean,
    rc_sigma=rc_sigma, I0_range=I0_range, I0_fixed=I0_fixed, I0_mean=I0_mean,
    I0_sigma=I0_sigma, noise_only=noise_only, zero_fraction=zero_fraction,
    zero_deterministic=zero_deterministic,
    one_over_f_slope=one_over_f_slope,
    one_over_f_amplitude=one_over_f_amplitude,
    white_noise_amplitude=white_noise_amplitude,
    gaussian_smoothing_fwhm=gaussian_smoothing_fwhm,
    use_highpass=use_highpass, kfilt_scale=kfilt_scale)

#%% ================================================================================
# === MAIN LOOP ===
# ==================================================================================

# Fixed spectrum: precompute once
P2d = nl.psd_isotropic_powerlaw((image_size, image_size),
                                one_over_f_slope, one_over_f_amplitude)

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

    w = nl.white_noise((image_size, image_size), white_noise_amplitude, rng)
    n1f = nl.grf_from_psd(P2d, rng)

    final = nl.beam_smooth(img + n1f, gaussian_smoothing_fwhm) + w

    if use_highpass:
        # Filter the ASSEMBLED map — signal and noise together (M4 Sec. 5.1
        # safe variant; see module docstring).  MF/eta consumers must use
        # the filtered template tau' = ifft2(sqrt(T) * fft2(tau)).
        final, _T = nl.kspace_highpass(final, kfilt_scale)

    images_clean.append(img.astype(np.float32))
    images_noisy.append(final.astype(np.float32))
    labels_I0.append(float(I0))
    has_source.append(0 if is_empty else 1)

print(f"Generated {num_images} images  [1/f^{one_over_f_slope} A={one_over_f_amplitude}  "
      f"white={white_noise_amplitude}  highpass={'ON' if use_highpass else 'off'}  "
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
print(f"pixel moments: std={m['std']:.3f}  skew={m['skewness']:.4f}  "
      f"ex.kurt={m['excess_kurtosis']:.4f}  (Gaussian: both ~0)")

# radial PSD slope check against the generation parameter
r, psd = nl.diagnostics.radial_psd(noisy_arr[0])
psd_mean = np.mean([nl.diagnostics.radial_psd(im)[1]
                    for im in noisy_arr[:50]], axis=0)
slope_fit, _ = nl.diagnostics.fit_psd_slope(r, psd_mean, r_range=(2, 20))
print(f"fitted radial PSD slope: {slope_fit:.2f}  "
      f"(generation: -{one_over_f_slope}; beam steepens the high-k end)")

nl.diagnostics.quicklook(noisy_arr, np.array(labels_I0),
                         title=f'{model_name}: isotropic red noise')
