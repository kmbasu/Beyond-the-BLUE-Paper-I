"""
whitening.py — splits, 2-D PSD estimation, whitening, templates, σ_MF
=====================================================================

Memo §14 infrastructure.  All spectra use the unnormalized-FFT convention
E[|fft2(n)|²] = P(k) (matching noise_lib), so quadratic forms obey

    τ† N⁻¹ d = Σ_k τ̃*_k d̃_k / P_k ,      σ_MF² = 1 / Σ_k |τ̃|²/P_k .

Whitening filter √(npix/P̂) makes the whitened field unit-white
(E[|fft2(x)|²] = npix ⇔ E[x_p x_q] ≈ δ_pq), and the whitened template
t = ifft2(fft2(τ)·√(npix/P̂)) then satisfies ‖t‖² = 1/σ_MF² (Parseval),
so t̂ = σ_MF·t is the unit whitened template of memo §3.

STATIONARITY caveat: a single 2-D PSD whitens exactly only stationary
noise.  For non-stationary models (patchwise PSRAND, median residuals to
some degree) the whitening removes the stationary part and the linear rung
of the variational ladder returns η_lin ≥ 1 — the better-linear-filter
forecast — instead of exactly 1 (memo §14, route b).
"""

import numpy as np

#  Block size for the stacked transforms below.  These three routines used to
#  transform a whole ensemble at once, which for the confusion rung's 60k-map
#  ensembles allocated ~21 GB of complex128 in a single expression and made a
#  MacBook Air swap rather than compute.  Worse, ``whiten_maps`` returned
#  ``ifft2(...).real`` -- a VIEW into the complex temporary, so every whitened
#  array retained 16 bytes per element instead of 8 for the whole run.
#  Chunking is exact (each image transforms independently) and the returned
#  array now owns contiguous float64 storage.  Verified bit-identical to the
#  pre-2026-09-03 implementation on white, red and confusion ensembles.
_CHUNK = 512
_FULL_MAX = 2048        # below this, estimate_psd2d keeps the exact legacy path


# ----------------------------------------------------------------------------------
# Splits (memo §14, + separate VAL split; see package docstring)
# ----------------------------------------------------------------------------------

def make_splits(n_images, f_psd=0.25, f_fit=0.4, f_val=0.1, seed=12345):
    """Disjoint index sets {'psd', 'fit', 'val', 'eval'} covering range(n).

    Fractions are (f_psd, f_fit, f_val); the remainder is EVAL.  A fixed
    seed makes the split reproducible per ensemble; images are assumed
    exchangeable (i.i.d. by construction in the drivers).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_images)
    n_psd = int(f_psd * n_images)
    n_fit = int(f_fit * n_images)
    n_val = int(f_val * n_images)
    return {'psd':  np.sort(idx[:n_psd]),
            'fit':  np.sort(idx[n_psd:n_psd + n_fit]),
            'val':  np.sort(idx[n_psd + n_fit:n_psd + n_fit + n_val]),
            'eval': np.sort(idx[n_psd + n_fit + n_val:])}


# ----------------------------------------------------------------------------------
# PSD estimation and whitening
# ----------------------------------------------------------------------------------

def estimate_psd2d(images, method='mean'):
    """Ensemble 2-D PSD (mean or median periodogram), unnormalized-FFT
    convention.  Pass ONLY the SPLIT_PSD images — reusing fitting or
    evaluation images here is the leakage failure mode of memo §14.

    The median option is robust to heavy-tailed ensembles but is biased low
    for chi-squared periodograms (median/mean = ln 2 per 2-dof mode); when
    used, the bias is irrelevant for η only if applied consistently, and
    the Gaussian null test will say so — default is 'mean'.
    """
    images = np.asarray(images)
    if method == 'median':
        #  the median needs every value per mode, so this path keeps the
        #  full-stack transform (n x 16384 x 16 bytes of complex128)
        pgrams = np.abs(np.fft.fft2(images, axes=(-2, -1)))**2
        return np.maximum(np.median(pgrams, axis=0), 1e-20)
    if len(images) <= _FULL_MAX:
        #  BIT-IDENTICAL to the pre-2026-09-03 implementation.  Chunked
        #  accumulation changes the floating-point summation order and moves
        #  P_hat by ~4e-15 relative -- physically nothing, but every campaign
        #  result so far used a SPLIT_PSD of 1500 maps, and those must stay
        #  reproducible to the last bit when re-run or merged.  Chunking is
        #  therefore reserved for ensembles that would not have fitted anyway.
        pgrams = np.abs(np.fft.fft2(images, axes=(-2, -1)))**2
        return np.maximum(pgrams.mean(axis=0), 1e-20)
    acc = np.zeros(images.shape[-2:], dtype=np.float64)
    for i in range(0, len(images), _CHUNK):
        acc += (np.abs(np.fft.fft2(images[i:i + _CHUNK], axes=(-2, -1)))**2
                ).sum(axis=0)
    return np.maximum(acc / len(images), 1e-20)


def whiten_maps(images, P_hat):
    """Whiten maps with the SPLIT_PSD spectrum: x = ifft2(fft2(n)·√(npix/P̂)).

    Returns float64 maps with E[x_p²] ≈ 1 for noise drawn from P̂.
    """
    images = np.asarray(images)
    ny, nx = images.shape[-2:]
    H = np.sqrt(ny * nx / P_hat)
    if images.ndim == 2:
        return np.ascontiguousarray(
            np.fft.ifft2(np.fft.fft2(images) * H).real)
    out = np.empty(images.shape, dtype=np.float64)
    for i in range(0, len(images), _CHUNK):
        blk = images[i:i + _CHUNK]
        out[i:i + _CHUNK] = np.fft.ifft2(
            np.fft.fft2(blk, axes=(-2, -1)) * H, axes=(-2, -1)).real
    return out


def whitened_template(tau, P_hat):
    """Unit whitened template t̂ and σ_MF for template τ under spectrum P̂.

    Returns (t_hat, sigma_mf):  t̂ = σ_MF · ifft2(fft2(τ)·√(npix/P̂)),
    normalized so ‖t̂‖ = 1; σ_MF = (Σ|τ̃|²/P̂)^(-1/2).
    """
    ny, nx = tau.shape
    tau_F = np.fft.fft2(tau)
    norm = np.sum(np.abs(tau_F)**2 / P_hat)
    sigma_mf = 1.0 / np.sqrt(norm)
    t = np.fft.ifft2(tau_F * np.sqrt(ny * nx / P_hat)).real
    t_hat = t / np.sqrt((t**2).sum())
    return t_hat, sigma_mf


def sigma_mf_from_psd(tau, P_hat):
    """σ_MF = (Σ|τ̃|²/P̂)^(-1/2) — matched-filter error for template τ."""
    tau_F = np.fft.fft2(tau)
    return 1.0 / np.sqrt(np.sum(np.abs(tau_F)**2 / P_hat))


def mf_amplitudes(images, tau, P_hat):
    """Apply the PSD-based MF to maps; returns per-image amplitude estimates.

    Â_i = Σ_k τ̃* d̃_i / P̂ / Σ_k |τ̃|²/P̂ — used for empirical σ_MF
    cross-checks against the predicted value (they agree to a few % when
    conventions are right; part of the validation battery).
    """
    tau_F = np.fft.fft2(tau)
    norm = np.sum(np.abs(tau_F)**2 / P_hat)
    images = np.asarray(images)
    w = np.conj(tau_F) / P_hat
    out = np.empty(len(images), dtype=np.float64)
    for i in range(0, len(images), _CHUNK):
        F = np.fft.fft2(images[i:i + _CHUNK], axes=(-2, -1))
        out[i:i + _CHUNK] = (w * F).sum(axis=(-2, -1)).real
    return out / norm


# ----------------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------------

def bootstrap_mean(values, n_boot=2000, seed=0):
    """Bootstrap mean and std of the mean, resampling IMAGES (memo §14).

    `values` are per-image contributions (e.g. c_i = 2 t̂·∇f_i − f_i² for
    the variational bound); returns (mean, boot_std).
    """
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(means.std())
