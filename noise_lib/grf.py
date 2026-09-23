"""
grf.py — Gaussian random field generation
=========================================

Canonical generator for all Gaussian noise components, replacing both legacy
constructions:

1. The random-phase generator (``generate_1overf_noise``) fixed |F(k)| =
   sqrt(P) exactly and randomized only the phases.  That is NOT a Gaussian
   random field: the per-realization power spectrum does not fluctuate
   (verified in an internal development note, May 2026).  It is not used here.

2. The Hermitian-projection generator (``strict_1overf_noise`` and
   ``generate_anisotropic_1overf_noise``) drew independent complex Gaussians
   with E[|F|^2] = P and then projected onto the Hermitian subspace,
   F_sym = (F(k) + conj(F(-k))) / 2.  The projection UNIFORMLY HALVES the
   power: for generic modes Var(Re F_sym) = Var(Im F_sym) = P/4 so
   E|F_sym|^2 = P/2, and at self-conjugate modes (DC, Nyquist combinations)
   F_sym = Re F ~ N(0, P/2) as well.  The result is a perfectly valid GRF —
   but with spectrum P/2, i.e. an effective amplitude a/sqrt(2).

The transfer-function method used here (M4 Sec. 4.4, ``grf_from_psd``)
filters a real white field in Fourier space.  It is exactly Hermitian by
construction, every mode carries the correct Gaussian statistics, and the
delivered spectrum is exactly P:

    E[ |fft2(noise)|^2 ] = P(k)     (unnormalized-FFT convention).

AMPLITUDE CONVENTION CHANGE (deliberate, agreed for the paper pipeline):
at equal amplitude parameter, maps produced here have sqrt(2) LARGER rms
than legacy strict_1overf_noise output.  To reproduce a legacy noise level
exactly, divide the legacy amplitude by sqrt(2).  The regression suite
(tests/regression_phase_a.py) verifies the factor.
"""

import numpy as np


def white_noise(shape, amplitude, rng):
    """White Gaussian noise with per-pixel standard deviation ``amplitude``.

    Identical statistics to the legacy generate_white_noise, but drawn from
    an explicit Generator instead of the global numpy RNG.
    """
    if amplitude == 0:
        return np.zeros(shape)
    return amplitude * rng.standard_normal(size=shape)


def grf_from_psd(P, rng):
    """Draw one real Gaussian random field with target 2-D spectrum ``P``.

    Transfer-function method: a real white field w ~ N(0, 1) has
    E[|fft2(w)|^2] = npix, so filtering with H = sqrt(P / npix) gives a real
    field with E[|fft2(n)|^2] = P exactly.  Hermitian symmetry is inherited
    from the real input (the filter is real and symmetric on the FFT grid
    whenever P is, which all noise_lib spectra are), so no projection — and
    no power loss — occurs.

    Parameters
    ----------
    P   : (ny, nx) ndarray — target spectrum, FFT-ordered (see spectra.py);
          must be non-negative and even under k -> -k (true for all spectra
          built from |k|, |k_par| or k^2).
    rng : numpy.random.Generator

    Returns
    -------
    noise : (ny, nx) float64 ndarray, zero-mean up to the (noisy) DC mode.
            NOTE: no mean subtraction is applied — the DC mode carries the
            power P(0,0) assigned by the spectrum, consistent with the
            "never zero the DC channel" rule (paper Sec. 4.4).
    """
    ny, nx = P.shape
    npix = ny * nx
    w = rng.standard_normal(size=(ny, nx))
    H = np.sqrt(P / npix)
    return np.fft.ifft2(np.fft.fft2(w) * H).real
