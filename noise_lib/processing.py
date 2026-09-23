"""
processing.py — deterministic map-processing operators
======================================================

The linear and nonlinear processing steps applied to simulated maps.  All
four are bit-exact ports of their legacy counterparts (same FFT conventions,
same regularization); only the module organization is new.

Taxonomy reminder (M4 Sec. 5.3, the linear/nonlinear dichotomy):

* ``kspace_highpass``, ``pca_transfer`` and ``beam_smooth`` are LINEAR,
  data-independent operators — they map Gaussian noise to Gaussian noise
  and merely reshape the spectrum (no Tier-2 channel is created).
* ``row_median_removal`` is NONLINEAR (the sample median is data-dependent)
  — it is the paper's canonical processing-induced Tier-2 mechanism.
"""

import numpy as np


def row_median_removal(obs_map, axis=1):
    """Per-row (axis=1) or per-column (axis=0) median baseline removal.

    Nonlinear operator (Mod 2 of the legacy framework): subtracts the median
    pixel value of each row/column from that row/column, mimicking
    scan-synchronous baseline removal in survey pipelines.

    Returns
    -------
    cleaned  : map with baselines removed
    baseline : 1-D array of the removed medians
    """
    baseline = np.median(obs_map, axis=axis, keepdims=True)
    return obs_map - baseline, baseline.squeeze()


def kspace_highpass(noise_map, k_filt):
    """Linear k-space high-pass filter (Mod 1 of the legacy framework).

    T(k) = 1 - exp(-k^2 / (2 k_filt^2)), k = sqrt(kx^2 + ky^2) in
    cycles/pixel, is the POWER transfer function: the amplitude spectrum is
    multiplied by sqrt(T), so the filtered field has P'(k) = T(k) P(k)
    (legacy convention, apply_kspace_highpass_filter).  T(0) = 0 (DC
    removed), T -> 1 at high k, T(k_filt) ~ 0.393.

    Returns
    -------
    filtered : filtered map
    T        : the 2-D POWER transfer function (for diagnostics)
    """
    ny, nx = noise_map.shape
    fy = np.fft.fftfreq(ny).reshape(-1, 1)
    fx = np.fft.fftfreq(nx).reshape(1, -1)
    k2 = fx**2 + fy**2
    T = 1.0 - np.exp(-k2 / (2.0 * k_filt**2))
    filtered = np.fft.ifft2(np.fft.fft2(noise_map) * np.sqrt(T)).real
    return filtered, T


def pca_transfer(noise_map, kx_cut, ky_cut):
    """Anisotropic PCA-like transfer function (legacy apply_pca_transfer).

    T(kx, ky) = 1 - exp(-kx^2/(2 kx_cut^2) - ky^2/(2 ky_cut^2)):
    removes the low-|k| modes a successful PCA subtraction would take out,
    with independent along-scan / cross-scan cutoffs.  Linear operator.
    """
    ny, nx = noise_map.shape
    kx = np.fft.fftfreq(nx).reshape(1, -1)
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    T = 1.0 - np.exp(-kx**2 / (2.0 * kx_cut**2)
                     - ky**2 / (2.0 * ky_cut**2))
    return np.fft.ifft2(np.fft.fft2(noise_map) * T).real


def beam_smooth(image, fwhm_pixels):
    """Gaussian beam smoothing via circular convolution in Fourier space.

    Beam transfer  B(k) = exp(-2 pi^2 sigma^2 k^2), sigma = FWHM/2.355;
    B(0) = 1, so the map sum (DC mode) is preserved.  Bit-exact port of the
    legacy smooth_image_fourier / apply_gaussian_smoothing.
    ``fwhm_pixels <= 0`` returns an unmodified copy.
    """
    if fwhm_pixels <= 0:
        return image.copy()
    sigma = fwhm_pixels / 2.355
    ny, nx = image.shape
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    kx = np.fft.fftfreq(nx).reshape(1, -1)
    beam_amp = np.exp(-2.0 * np.pi**2 * sigma**2 * (kx**2 + ky**2))
    return np.fft.ifft2(np.fft.fft2(image) * beam_amp).real
