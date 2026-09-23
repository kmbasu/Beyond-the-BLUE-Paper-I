"""
psrand.py — randomized-PSD 1/f noise (the Tier-1 family)
========================================================

Two generators for conditionally-Gaussian noise whose covariance varies
between (or within) images:

``psrand_1overf_noise``  — PER-IMAGE randomization: one (slope, amplitude)
    pair drawn per image, one stationary GRF generated with that spectrum.
    This is the "PSRAND" model of the theory memos (M4 Sec. 5.2; eta memo
    Secs. 10, 19, 21.5): a Gaussian scale/shape mixture with a 1-2
    dimensional latent, for which the marginal-score eta is computable
    EXACTLY by latent quadrature.  This is the paper's flagship Tier-1
    Sec. 5.1 case.

``patchwise_1overf_noise`` — WITHIN-IMAGE randomization (legacy Mod 3 of
    modified_noisemap.ipynb, used for the historical PSRAND1/2 datasets):
    an n_patches x n_patches grid of independent (amplitude, slope) draws,
    blended by Gaussian weight masks

        noise(r) = sum_ij w_ij(r) f_ij(r) / sqrt(sum_ij w_ij(r)^2).

    The result is non-stationary within each image and carries a
    32-dimensional latent (for n_patches=4) — exact quadrature is
    intractable and eta must come from the variational method.  Retained
    for regression against the legacy datasets and as a secondary
    robustness case.

    NOTE: n_patches=1 makes the blend collapse identically to a single
    stationary GRF (w/sqrt(w^2) = 1 pointwise), i.e. the per-image model —
    but use ``psrand_1overf_noise`` for that case: it skips the blending
    machinery and exposes the latents cleanly.

Both use the truncated-normal parameter priors of the legacy Mod 3
(clipping, not redraw):  amplitude >= 1e-6,  slope >= 1.

Amplitude convention: the underlying GRFs come from grf.grf_from_psd (full
target power); legacy Mod 3 fields carried half that power (see grf.py).
"""

import numpy as np

from .grf import grf_from_psd
from .spectra import psd_isotropic_powerlaw


def _draw_params(rng, slope_mean, slope_sigma, amplitude_mean, amplitude_sigma):
    """One truncated-normal (amplitude, slope) draw, legacy Mod-3 clipping."""
    amp = max(1e-6, float(rng.normal(amplitude_mean, amplitude_sigma)))
    slope = max(1.0, float(rng.normal(slope_mean, slope_sigma)))
    return amp, slope


def psrand_1overf_noise(shape, slope_mean, slope_sigma,
                        amplitude_mean, amplitude_sigma, rng=None):
    """Per-image randomized isotropic 1/f noise (flagship Tier-1 model).

    Draws ONE (amplitude, slope) pair for the whole image,

        amplitude ~ max(1e-6, N(amplitude_mean, amplitude_sigma))
        slope     ~ max(1,    N(slope_mean,     slope_sigma))

    and returns a stationary GRF with spectrum amplitude^2 / |k|^slope.
    Setting either sigma to 0 pins that parameter — e.g. slope_sigma=0
    gives the amplitude-only randomization whose exact Tier-1 ceiling is
    eta_marg = 1 (the scale-free null test of eta memo Sec. 21.5).

    Returns
    -------
    noise   : (ny, nx) ndarray
    latents : {'slope': float, 'amplitude': float} — REQUIRED downstream by
              the Sec. 19 latent quadrature and the oracle MF; always store.
    """
    if rng is None:
        rng = np.random.default_rng()
    amp, slope = _draw_params(rng, slope_mean, slope_sigma,
                              amplitude_mean, amplitude_sigma)
    P = psd_isotropic_powerlaw(shape, slope, amp)
    noise = grf_from_psd(P, rng)
    return noise, {'slope': slope, 'amplitude': amp}


def patchwise_1overf_noise(shape, slope_mean, slope_sigma,
                           amplitude_mean, amplitude_sigma,
                           n_patches=4, rng=None):
    """Within-image patchwise nonstationary 1/f noise (legacy Mod 3).

    See module docstring.  Port of generate_nonstationary_1overf_noise with
    the GRF construction replaced by grf.grf_from_psd; blending geometry
    (patch centers on linspace(0, n-1, n_patches), Gaussian weight sigma =
    0.8 * ny/n_patches, sqrt-sum-of-squares normalization) is unchanged.

    Returns
    -------
    noise      : (ny, nx) ndarray (non-stationary)
    param_grid : {'amplitudes': (n_patches, n_patches) ndarray,
                  'slopes'    : (n_patches, n_patches) ndarray}
    """
    if rng is None:
        rng = np.random.default_rng()

    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)

    y_centers = np.linspace(0, ny - 1, n_patches)
    x_centers = np.linspace(0, nx - 1, n_patches)
    sigma_patch = (ny / n_patches) * 0.8

    noise_acc = np.zeros(shape)
    weight_sq = np.zeros(shape)
    amplitudes = np.zeros((n_patches, n_patches))
    slopes = np.zeros((n_patches, n_patches))

    for i, yc in enumerate(y_centers):
        for j, xc in enumerate(x_centers):
            amp, slope = _draw_params(rng, slope_mean, slope_sigma,
                                      amplitude_mean, amplitude_sigma)
            amplitudes[i, j] = amp
            slopes[i, j] = slope

            P = psd_isotropic_powerlaw(shape, slope, amp)
            patch_noise = grf_from_psd(P, rng)

            dist2 = (yy - yc)**2 + (xx - xc)**2
            weight = np.exp(-dist2 / (2.0 * sigma_patch**2))

            noise_acc += weight * patch_noise
            weight_sq += weight**2

    noise_out = noise_acc / np.sqrt(weight_sq + 1e-12)
    return noise_out, {'amplitudes': amplitudes, 'slopes': slopes}
