"""
spectra.py — 2-D noise power-spectrum builders
==============================================

Every function returns a 2-D array P(ky, kx) on the standard FFT grid
(numpy.fft.fftfreq ordering, cycles/pixel), in the UNNORMALIZED-FFT power
convention used throughout noise_lib:

    E[ |numpy.fft.fft2(noise)|^2 ] = P(ky, kx).

Conventions follow the legacy modules exactly:

* Spectral slopes are the exponent of |k| (1-D convention):
  ``P ~ A^2 / |k|^alpha``.  For the isotropic term this means
  ``A^2 / (kx^2 + ky^2)^(alpha/2)``.
* DC / zero-frequency regularization: any vanishing frequency is replaced
  by the smallest non-zero frequency of the corresponding grid — the same
  "soft" convention as strict_1overf_noise and
  generate_anisotropic_1overf_noise in the legacy scripts.  The DC mode is
  therefore assigned a LARGE finite power (it is noisy), which is the
  physically safe choice highlighted by the "DC-zero cautionary tale"
  (paper Sec. 4.4): a mode must never be noiseless for the noise model
  while carrying signal.

New relative to the legacy code
-------------------------------
``psd_scan_powerlaw`` accepts an orientation angle, enabling the Tier-1
"fixed anisotropy, random orientation per image" noise model: the scan term
becomes ``A^2 / |k_par|^alpha`` with ``k_par = kx cos(theta) + ky sin(theta)``,
i.e. stripes rotated by theta.  ``theta = 0`` reproduces the legacy
``|kx|``-only spectrum identically.
"""

import numpy as np


def freq_grids(shape):
    """Return broadcastable frequency grids (fy, fx) in cycles/pixel.

    fy has shape (ny, 1), fx has shape (1, nx); both follow the standard
    numpy.fft.fftfreq ordering [0, 1/N, ..., -1/N].
    """
    ny, nx = shape
    fy = np.fft.fftfreq(ny).reshape(-1, 1)
    fx = np.fft.fftfreq(nx).reshape(1, -1)
    return fy, fx


def _regularize(arr):
    """Replace zeros of a non-negative array by its smallest positive value.

    Identical to the legacy convention ``f[0,0] = np.min(f[f>0])`` — the
    divergent power-law bins are capped at the fundamental-frequency power
    instead of producing 1/0.
    """
    pos_min = float(np.min(arr[arr > 0]))
    return np.where(arr > 0, arr, pos_min)


def psd_isotropic_powerlaw(shape, slope, amplitude):
    """Isotropic power-law spectrum  P = amplitude^2 / |k|^slope.

    Parameters
    ----------
    shape     : (ny, nx) map shape in pixels
    slope     : spectral index alpha of |k| (3.0 = steep red '1/f^3' noise)
    amplitude : A_iso prefactor (P has units amplitude^2)

    Returns
    -------
    P : (ny, nx) ndarray, FFT-ordered, positive everywhere (DC regularized).
    """
    fy, fx = freq_grids(shape)
    k2 = _regularize(fx**2 + fy**2)
    return amplitude**2 / (k2 ** (slope / 2.0))


def psd_scan_powerlaw(shape, slope, amplitude, angle=0.0):
    """Scan-direction (stripe) power-law spectrum  P = amplitude^2 / |k_par|^slope.

    ``k_par`` is the frequency component along the stripe NORMAL:
    ``k_par = kx cos(angle) + ky sin(angle)``.  With angle = 0 this is the
    legacy ``A_scan^2 / |kx|^alpha_scan`` term (noise correlated along x,
    horizontal stripes).  A non-zero angle rotates the anisotropy direction:
    stripes lie along the direction (-sin(angle), cos(angle)).

    Regularization: |k_par| is FLOORED at the fundamental frequency along
    the rotated direction, f_min = sqrt((cos(angle)/nx)^2 + (sin(angle)/ny)^2)
    (= 1/nx for a square map at any angle).  At angle = 0 this reproduces
    the legacy convention exactly — all non-zero |kx| are >= 1/nx, so only
    the kx = 0 column is affected — while remaining numerically safe at
    arbitrary angles, where floating-point residues and near-resonant
    (kx, ky) combinations can otherwise produce absurdly small |k_par|
    values and blow up the power law.

    Parameters
    ----------
    shape     : (ny, nx)
    slope     : alpha_scan
    amplitude : A_scan
    angle     : orientation angle in RADIANS (0 = legacy fixed direction)

    Returns
    -------
    P : (ny, nx) ndarray.
    """
    ny, nx = shape
    fy, fx = freq_grids(shape)
    k_par = np.abs(fx * np.cos(angle) + fy * np.sin(angle))
    f_min = np.sqrt((np.cos(angle) / nx)**2 + (np.sin(angle) / ny)**2)
    k_par = np.maximum(k_par, f_min)
    return amplitude**2 / (k_par ** slope)


def psd_two_component(shape, slope_scan, amplitude_scan,
                      slope_iso, amplitude_iso, scan_angle=0.0):
    """Two-component anisotropic spectrum (legacy generate_anisotropic_1overf_noise):

        P(kx, ky) = A_scan^2 / |k_par|^alpha_scan
                  + A_iso^2  / (kx^2 + ky^2)^(alpha_iso / 2)

    with the scan term optionally rotated by ``scan_angle`` (radians).
    ``amplitude_scan = 0`` reduces exactly to the isotropic spectrum; either
    term can be switched off by setting its amplitude to zero.

    Returns
    -------
    P : (ny, nx) ndarray.
    """
    P = psd_isotropic_powerlaw(shape, slope_iso, amplitude_iso) \
        if amplitude_iso != 0 else np.zeros(shape)
    if amplitude_scan != 0:
        P = P + psd_scan_powerlaw(shape, slope_scan, amplitude_scan,
                                  angle=scan_angle)
    if np.all(P == 0):
        raise ValueError("Both spectral amplitudes are zero — empty spectrum.")
    return P
