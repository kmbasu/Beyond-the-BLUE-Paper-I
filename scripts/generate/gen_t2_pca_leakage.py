#!/usr/bin/env python
"""
gen_t2_pca_leakage.py — Tier-2 driver: PCA leaked-mode residuals
================================================================

The PCA mode-truncation model of paper Sec. 6.2: a Poisson number of
smooth, anisotropically-correlated eigenmode patterns leak through
imperfect PCA atmospheric subtraction, each with a both-signed heavy-tailed
(Student-t) amplitude tied to the parent noise level — a Campbell model in
MODE space (eta memo Sec. 21.3).  Sign-symmetric marks => excess kurtosis
without skewness (quadratic-rung null expected in Phase B); heavy tails
admit a Gaussian-scale-mixture reading, so this is the Tier-boundary case
whose eta_T1/eta_T2 factorization is reported rather than a taxonomy
ruling made.  Per-mode amplitudes are stored as latents; the mode FIELDS
are reproducible from (master_seed, image index) via the per-image child
generator, since each image's full draw sequence is deterministic.

Optionally the deterministic part of PCA subtraction is modeled by the
linear transfer function T = 1 - exp(-kx^2/2kx_cut^2 - ky^2/2ky_cut^2)
applied to the 1/f noise (use_pca_transfer; linear => no Tier-2 content by
itself, M4 Sec. 5.3).

Pipeline per image (legacy PCA_leakage.py):
  1. beta-model signal (or zeros)
  2. isotropic 1/f noise  [optional PCA transfer function]
  3. beam smoothing of (signal + noise);  white noise added after
  4. leaked-mode artifacts injected
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

model_name = 'T2_PCA_LEAKAGE'
output_h5  = 'T2_pca_leakage_test.h5'

num_images  = 400
image_size  = 128
master_seed = None

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

# --- background noise (legacy defaults: isotropic only) ---
one_over_f_slope_iso      = 3.0
one_over_f_amplitude_iso  = 5.0

white_noise_amplitude   = 0.0
gaussian_smoothing_fwhm = 5.0

# --- PCA leaked modes (legacy defaults) ---
n_leaked_mean   = 3       # Poisson mean leaked modes per map
leak_efficiency = 0.15    # amplitude fraction of A_iso
leak_df         = 2.5     # Student-t dof
mode_corr_scan  = 20.0    # along-scan correlation length (pixels)
mode_corr_cross = 40.0    # across-scan correlation length (pixels)

# --- PCA transfer function (optional, linear) ---
use_pca_transfer = False
pca_kx_cut       = 0.05    # cycles/pixel  (~ 1/mode_corr_scan)
pca_ky_cut       = 0.025   # cycles/pixel  (~ 1/mode_corr_cross)

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
    white_noise_amplitude=white_noise_amplitude,
    gaussian_smoothing_fwhm=gaussian_smoothing_fwhm,
    n_leaked_mean=n_leaked_mean, leak_efficiency=leak_efficiency,
    leak_df=leak_df, mode_corr_scan=mode_corr_scan,
    mode_corr_cross=mode_corr_cross, use_pca_transfer=use_pca_transfer,
    pca_kx_cut=pca_kx_cut, pca_ky_cut=pca_ky_cut,
    use_mod2_row_median=use_mod2_row_median, mod2_axis=mod2_axis)

#%% ================================================================================
# === MAIN LOOP ===
# ==================================================================================

shape = (image_size, image_size)
P2d = nl.psd_isotropic_powerlaw(shape, one_over_f_slope_iso,
                                one_over_f_amplitude_iso)

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
leak_infos = []

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
    if use_pca_transfer:
        n1f = nl.pca_transfer(n1f, pca_kx_cut, pca_ky_cut)

    final = nl.beam_smooth(img + n1f, gaussian_smoothing_fwhm) + w

    art, n_leak, linfo = nl.pca_leaked_modes(
        shape, n_leaked_mean=n_leaked_mean, leak_efficiency=leak_efficiency,
        A_iso=one_over_f_amplitude_iso, leak_df=leak_df,
        mode_corr_scan=mode_corr_scan, mode_corr_cross=mode_corr_cross,
        rng=rng)
    final += art
    leak_infos.append(linfo)

    if use_mod2_row_median:
        final, _baseline = nl.row_median_removal(final, axis=mod2_axis)

    images_clean.append(img.astype(np.float32))
    images_noisy.append(final.astype(np.float32))
    labels_I0.append(float(I0))
    has_source.append(0 if is_empty else 1)

latents = nl.io_h5.pack_event_latents(leak_infos, ('amp',), 'leak')

n_arr = latents['leak_count']
print(f"Generated {num_images} images  [PCA leakage: mean={n_arr.mean():.1f} "
      f"(Poisson λ={n_leaked_mean})  η={leak_efficiency}  df={leak_df}  "
      f"L=({mode_corr_scan},{mode_corr_cross})px  "
      f"transfer={'ON' if use_pca_transfer else 'off'}  "
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
print(f"pixel moments: skew={m['skewness']:.4f} (expect ~0: sign-symmetric)  "
      f"ex.kurt={m['excess_kurtosis']:.4f} (expect >0: heavy-tailed modes)")

amps = latents['leak_amp']
if len(amps):
    print(f"leaked-mode amplitudes: n={len(amps)}  mean={amps.mean():+.3f}  "
          f"std={amps.std():.3f}")

nl.diagnostics.quicklook(noisy_arr, np.array(labels_I0),
                         title=f'{model_name}: PCA leaked modes')
