#!/usr/bin/env python
"""
gen_t2_scan_crossings.py — Tier-2 driver: sparse scan-crossing residuals
========================================================================

The scan-coherent residual model (formerly scanning_residuals.py, RELOCATED
from the Tier-1 folder per the taxonomy decision of 2026-08-26): a Poisson
number of imperfectly-filtered scan crossings per map, each a band of
~beam_sigma_rows rows carrying an along-scan-correlated residual with a
heavy-tailed Student-t amplitude, on an isotropic 1/f background.  As a
sparse location mixture with heavy-tailed marks this is genuinely Tier-2
(M4 taxonomy); its eta_T1/eta_T2 factorization will nevertheless be
computed and REPORTED in Phase B, per the "adjudicate empirically" decision.

Sign convention (the key statistical switch; see noise_lib.artifacts):
  'symmetric' (default) — signed amplitudes: odd cumulants vanish,
      trispectrum-led; the eta-memo Sec. 21.2 quadratic-rung ~ 1 null
      test applies.
  'positive'            — |t| amplitudes + positive-definite profiles
      (legacy scanning_residuals.py): skewed, bispectrum-led.
Phase B runs eta for BOTH; the per-crossing latents (row0, amp) are stored.

Pipeline per image (legacy scanning_residuals.py):
  1. beta-model signal (or zeros)
  2. isotropic 1/f noise (continuous scan term off by default — scan noise
     is modeled by the sparse crossings instead)
  3. beam smoothing of (signal + noise);  white noise added after
  4. crossing artifacts injected (already beam-coherent across rows)
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

model_name = 'T2_SCAN_CROSSINGS'
output_h5  = 'T2_scan_crossings_test.h5'

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

# --- background noise (legacy defaults: isotropic only) ---
one_over_f_slope_iso      = 3.0
one_over_f_amplitude_iso  = 5.0
one_over_f_slope_scan     = 3.0
one_over_f_amplitude_scan = 0.0   # scan noise modeled by the crossings

white_noise_amplitude   = 0.0
gaussian_smoothing_fwhm = 5.0

# --- scan crossings (legacy defaults) ---
n_crossings_mean = 8       # Poisson mean crossings per map
along_scan_corr  = 30.0    # along-scan correlation length (pixels)
beam_sigma_rows  = 2.0     # cross-scan beam sigma (pixels)
amp_scale_cross  = 0.5     # amplitude scale
amp_df           = 2.5     # Student-t dof (heavy tails)
sign_convention  = 'symmetric'   # 'symmetric' | 'positive'  (see docstring)

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
    n_crossings_mean=n_crossings_mean, along_scan_corr=along_scan_corr,
    beam_sigma_rows=beam_sigma_rows, amp_scale_cross=amp_scale_cross,
    amp_df=amp_df, sign_convention=sign_convention,
    use_mod2_row_median=use_mod2_row_median, mod2_axis=mod2_axis)

#%% ================================================================================
# === MAIN LOOP ===
# ==================================================================================

shape = (image_size, image_size)
P2d = nl.psd_two_component(shape,
                           one_over_f_slope_scan, one_over_f_amplitude_scan,
                           one_over_f_slope_iso, one_over_f_amplitude_iso) \
      if one_over_f_amplitude_scan != 0 else \
      nl.psd_isotropic_powerlaw(shape, one_over_f_slope_iso,
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
crossing_infos = []

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

    art, n_cross, cinfo = nl.scan_crossing_artifacts(
        shape, n_crossings_mean=n_crossings_mean,
        along_scan_corr=along_scan_corr, beam_sigma_rows=beam_sigma_rows,
        amp_scale=amp_scale_cross, amp_df=amp_df,
        sign_convention=sign_convention, rng=rng)
    final += art
    crossing_infos.append(cinfo)

    if use_mod2_row_median:
        final, _baseline = nl.row_median_removal(final, axis=mod2_axis)

    images_clean.append(img.astype(np.float32))
    images_noisy.append(final.astype(np.float32))
    labels_I0.append(float(I0))
    has_source.append(0 if is_empty else 1)

latents = nl.io_h5.pack_event_latents(crossing_infos, ('row0', 'amp'), 'crossing')

n_arr = latents['crossing_count']
print(f"Generated {num_images} images  [crossings: mean={n_arr.mean():.1f} "
      f"(Poisson λ={n_crossings_mean})  sign={sign_convention}  "
      f"amp_scale={amp_scale_cross}  df={amp_df}  "
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
expect = ("skew ~ 0, ex.kurt > 0 (trispectrum-led)" if sign_convention == 'symmetric'
          else "skew > 0 (bispectrum-led)")
print(f"pixel moments: skew={m['skewness']:.4f}  ex.kurt={m['excess_kurtosis']:.4f}  "
      f"[{sign_convention}: expect {expect}]")

amps = latents['crossing_amp']
print(f"crossing amplitudes: n={len(amps)}  mean={amps.mean():+.3f}  "
      f"std={amps.std():.3f}  min={amps.min():+.2f}  max={amps.max():+.2f}")

nl.diagnostics.quicklook(noisy_arr, np.array(labels_I0),
                         title=f'{model_name}: scan crossings ({sign_convention})')
