#!/usr/bin/env python
"""
gen_t1_psrand.py — Tier-1 driver: per-image randomized 1/f PSD (PSRAND)
=======================================================================

The flagship Tier-1 model of paper Sec. 5.1: each image is a Gaussian
random field, but its spectral parameters vary image to image,

    amplitude_i ~ max(1e-6, N(amplitude_mean, amplitude_sigma))
    slope_i     ~ max(1,    N(slope_mean,     slope_sigma))
    P_i(k)      = amplitude_i^2 / |k|^slope_i ,

i.e. a conditionally-Gaussian covariance mixture with a 1-2 dimensional
latent.  The per-image (slope, amplitude) latents are STORED in the HDF5 —
they are required by the exact Tier-1 latent quadrature (eta memo Sec. 19),
the oracle matched filter, and the M4 Sec. 5.2 dispersion cross-check.
Expected phenomenology: advantage ~ Var(alpha) x Var_f[log k]; exactly
zero for amplitude-only randomization (set slope_sigma = 0 to run that
theorem-level null).

``n_patches`` connects to the historical datasets: n_patches = 1 (default)
is the per-image model above; n_patches = 4 reproduces the legacy Mod-3
patchwise non-stationary variant used for PSRAND1/2 (32-dim latent stored
as per-image parameter grids; eta then only via the variational method).

Pipeline per image:
  1. beta-model signal (or zeros)
  2. randomized-PSD 1/f noise (per-image or patchwise)
  3. beam smoothing of (signal + 1/f);  white noise added after, unsmoothed

Run in Spyder cell-by-cell (#%%) or as a script:  python gen_t1_psrand.py
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

model_name = 'T1_PSRAND'
output_h5  = 'T1_psrand_test.h5'

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

# --- randomized-PSD noise (legacy Mod-3 parameter priors) ---
slope_mean      = 3.0
slope_sigma     = 0.5     # 0 -> fixed slope (amplitude-only randomization null)
amplitude_mean  = 5.0
amplitude_sigma = 0.8     # 0 -> fixed amplitude (slope-only randomization)

n_patches = 1             # 1 = per-image PSRAND (flagship); 4 = legacy patchwise

white_noise_amplitude   = 0.0
gaussian_smoothing_fwhm = 5.0

params = dict(
    model_name=model_name, num_images=num_images, image_size=image_size,
    source_type=source_type, flux_mode=flux_mode, rc_mean=rc_mean,
    rc_sigma=rc_sigma, I0_range=I0_range, I0_fixed=I0_fixed, I0_mean=I0_mean,
    I0_sigma=I0_sigma, noise_only=noise_only, zero_fraction=zero_fraction,
    zero_deterministic=zero_deterministic,
    slope_mean=slope_mean, slope_sigma=slope_sigma,
    amplitude_mean=amplitude_mean, amplitude_sigma=amplitude_sigma,
    n_patches=n_patches, white_noise_amplitude=white_noise_amplitude,
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
lat_slopes, lat_amps = [], []          # per-image latents (n_patches=1)
lat_slope_grids, lat_amp_grids = [], []  # patch grids   (n_patches>1)

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

    if n_patches == 1:
        n1f, lat = nl.psrand_1overf_noise(
            (image_size, image_size), slope_mean, slope_sigma,
            amplitude_mean, amplitude_sigma, rng=rng)
        lat_slopes.append(lat['slope'])
        lat_amps.append(lat['amplitude'])
    else:
        n1f, grid = nl.patchwise_1overf_noise(
            (image_size, image_size), slope_mean, slope_sigma,
            amplitude_mean, amplitude_sigma, n_patches=n_patches, rng=rng)
        lat_slope_grids.append(grid['slopes'])
        lat_amp_grids.append(grid['amplitudes'])

    final = nl.beam_smooth(img + n1f, gaussian_smoothing_fwhm) + w

    images_clean.append(img.astype(np.float32))
    images_noisy.append(final.astype(np.float32))
    labels_I0.append(float(I0))
    has_source.append(0 if is_empty else 1)

if n_patches == 1:
    latents = {'slope': np.array(lat_slopes), 'amplitude': np.array(lat_amps)}
else:
    latents = {'slope_grid': np.stack(lat_slope_grids),
               'amplitude_grid': np.stack(lat_amp_grids)}

print(f"Generated {num_images} images  [PSRAND n_patches={n_patches}  "
      f"slope={slope_mean}±{slope_sigma}  amp={amplitude_mean}±{amplitude_sigma}  "
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
      f"ex.kurt={m['excess_kurtosis']:.3f}  "
      f"(covariance mixture -> leptokurtic pooled distribution, per-image Gaussian)")

if n_patches == 1:
    print(f"latent check: slope mean={np.mean(lat_slopes):.3f}±{np.std(lat_slopes):.3f}  "
          f"amp mean={np.mean(lat_amps):.3f}±{np.std(lat_amps):.3f}")
    # per-image PSD variation: fitted slopes should track the latent slopes
    fits = []
    for im, s_true in list(zip(noisy_arr, lat_slopes))[:30]:
        r, psd = nl.diagnostics.radial_psd(im)
        s_fit, _ = nl.diagnostics.fit_psd_slope(r, psd, r_range=(2, 12))
        fits.append((-s_fit, s_true))
    fits = np.array(fits)
    corr = np.corrcoef(fits[:, 0], fits[:, 1])[0, 1]
    print(f"per-image fitted-vs-true slope correlation (30 imgs): {corr:.2f} "
          f"(should be strongly positive — the slope is an inferable latent)")

nl.diagnostics.quicklook(noisy_arr, np.array(labels_I0),
                         title=f'{model_name}: per-image randomized PSD')
