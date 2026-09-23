"""
artifacts.py — structured (non-Gaussian) noise components
=========================================================

The three sparse-structure injectors of the Tier-2 program, ported from the
legacy modules with their per-event latent variables EXPOSED (positions,
amplitudes, counts) so the drivers can store them in the HDF5 output — they
are needed downstream for the oracle matched filter, the per-image
conditional Gaussianization surrogate (eta_T1), and the Campbell-model
analytic benchmarks.

Campbell-model view (eta memo Sec. 7): glitches and scan crossings are
marked point processes in position space; PCA leaked modes are a marked
point process in mode space.  The latent lists returned here are exactly
the marks/positions of those processes.

Sign conventions for scan crossings
-----------------------------------
The two legacy implementations differ:

* ``scanning_residuals.py`` draws amp = amp_scale * |t_df| and shifts each
  along-scan profile positive-definite (min -> 0, peak -> amp): one-signed
  events => the map acquires SKEWNESS; bispectrum-led non-Gaussianity.
* ``algorithm_scanning_artifact.py`` draws amp = amp_scale * t_df (both
  signs) and rms-normalizes the profile without any shift: sign-symmetric
  events => odd cumulants vanish; trispectrum-led non-Gaussianity (the
  configuration assumed by eta memo Sec. 21.2, whose quadratic-rung ~ 1
  null test only applies here).

Both are retained behind ``sign_convention = 'symmetric' | 'positive'``
(default 'symmetric'), each bit-compatible with its legacy source given the
same Generator state.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import erfc


# ----------------------------------------------------------------------------------
# Scan-crossing residuals
# ----------------------------------------------------------------------------------

def scan_crossing_artifacts(shape, n_crossings_mean=8,
                            along_scan_corr=30.0,
                            beam_sigma_rows=2.0,
                            amp_scale=0.5, amp_df=2.5,
                            sign_convention='symmetric',
                            rng=None):
    """Sparse, beam-coherent scan-crossing residual artifacts.

    Physical model (see legacy docstrings for the full picture): a Poisson
    number of imperfectly-filtered scan crossings, each occupying a single
    row spread across ~beam_sigma_rows adjacent rows by the cross-scan beam,
    carrying a residual correlated along the scan (correlation length
    ``along_scan_corr``) with a heavy-tailed Student-t amplitude.

    Parameters
    ----------
    shape            : (ny, nx); scan direction = x (along rows)
    n_crossings_mean : Poisson mean number of crossings per map
    along_scan_corr  : along-scan correlation length (pixels)
    beam_sigma_rows  : cross-scan beam sigma (pixels)
    amp_scale        : amplitude scale
    amp_df           : Student-t degrees of freedom
    sign_convention  : 'symmetric' — signed amplitudes, rms-normalized
                       profiles (trispectrum-led; eta memo Sec. 21.2)
                       'positive'  — |t| amplitudes, min->0-shifted
                       positive-definite profiles (bispectrum-led; legacy
                       scanning_residuals.py behavior)
    rng              : numpy.random.Generator

    Returns
    -------
    artifact : (ny, nx) ndarray — summed crossing residual map
    n_cross  : int — number of crossings drawn
    info     : list of {'row0': int, 'amp': float} per crossing
    """
    if rng is None:
        rng = np.random.default_rng()
    if sign_convention not in ('symmetric', 'positive'):
        raise ValueError(f"Unknown sign_convention '{sign_convention}'.")

    ny, nx = shape
    artifact = np.zeros((ny, nx))

    n_cross = rng.poisson(n_crossings_mean)
    if n_cross == 0:
        return artifact, int(n_cross), []

    # Along-scan correlation filter (real-FFT frequencies), as in both legacy codes
    k = np.fft.rfftfreq(nx)
    along_filter = np.exp(-0.5 * (k * along_scan_corr)**2)

    info = []
    for _ in range(n_cross):
        row0 = int(rng.integers(0, ny))

        if sign_convention == 'positive':
            amp = amp_scale * float(np.abs(rng.standard_t(amp_df)))
        else:
            amp = amp_scale * float(rng.standard_t(amp_df))

        # Correlated 1-D Gaussian profile along the scan
        raw = rng.standard_normal(nx)
        prof = np.fft.irfft(np.fft.rfft(raw) * along_filter, n=nx)

        if sign_convention == 'positive':
            # Positive-definite residual baseline: shift min -> 0, peak -> amp
            prof = prof - prof.min()
            peak = prof.max()
            if peak > 0:
                prof *= amp / peak
        else:
            # Zero-mean signed residual, rms -> |amp| (legacy algorithm file)
            prof *= amp / (prof.std() + 1e-9)

        line = np.zeros((ny, nx))
        line[row0, :] = prof
        # Cross-scan beam coherence: smooth ONLY across rows (axis 0)
        line = gaussian_filter1d(line, sigma=beam_sigma_rows,
                                 axis=0, mode='nearest')
        artifact += line
        info.append({'row0': row0, 'amp': amp})

    return artifact, int(n_cross), info


# ----------------------------------------------------------------------------------
# Sub-threshold cosmic-ray glitch residuals
# ----------------------------------------------------------------------------------

def glitch_profile_1d(x, x0, tau_pix, sigma_beam):
    """Beam-convolved causal exponential along the scan direction (EMG profile).

    Closed-form convolution of exp(-(x-x0)/tau) H(x-x0) with a Gaussian PSF
    of sigma = sigma_beam, normalized to unit peak.  Bit-exact port of the
    legacy function (see its docstring for the pre-x0 leading-edge physics).
    """
    dx = x - x0
    raw = (np.exp(sigma_beam**2 / (2.0 * tau_pix**2) - dx / tau_pix)
           * erfc((sigma_beam**2 - tau_pix * dx)
                  / (np.sqrt(2.0) * sigma_beam * tau_pix)))
    peak = raw.max()
    return raw / peak if peak > 0 else raw


def add_glitch_residuals(image, n_glitches, tau_pix, amp_scale,
                         sigma_beam=2.0, sigma_cross=2.0, rng=None):
    """Inject sub-threshold glitch residuals (positive-definite stamps).

    Each stamp is the outer product of a beam-convolved causal exponential
    along x and a Gaussian along y, with Pareto amplitudes
    A = amp_scale * (Pareto(a=2.5) + 1)  (mean ~ 1.67 amp_scale).
    Bit-exact port of the legacy injector, with the per-event latents
    returned instead of discarded.  ``image`` is modified in-place.

    Returns
    -------
    image : the modified array (same object)
    info  : list of {'x0': float, 'y0': float, 'amp': float} per glitch
    """
    if rng is None:
        rng = np.random.default_rng()

    ny, nx = image.shape
    x_arr = np.arange(nx, dtype=float)
    y_arr = np.arange(ny, dtype=float)

    info = []
    for _ in range(n_glitches):
        x0 = float(rng.uniform(0.0, nx))
        y0 = float(rng.uniform(0.0, ny))
        A = amp_scale * (float(rng.pareto(a=2.5)) + 1.0)

        prof_x = glitch_profile_1d(x_arr, x0, tau_pix, sigma_beam)
        prof_y = np.exp(-0.5 * ((y_arr - y0) / sigma_cross)**2)
        image += A * np.outer(prof_y, prof_x)
        info.append({'x0': x0, 'y0': y0, 'amp': A})

    return image, info


# ----------------------------------------------------------------------------------
# PCA leaked-mode residuals
# ----------------------------------------------------------------------------------

def pca_leaked_modes(shape, n_leaked_mean, leak_efficiency,
                     A_iso, leak_df,
                     mode_corr_scan, mode_corr_cross,
                     rng=None):
    """PCA leaked-mode residual artifacts (Campbell model in mode space).

    A Poisson number of smooth, anisotropically-correlated 2-D modes leak
    through imperfect PCA subtraction, each with a both-signed heavy-tailed
    amplitude  a = leak_efficiency * A_iso * t_df  (Student-t; Davis-Kahan /
    Tracy-Widom motivation in the legacy docstring).  Sign-symmetric =>
    excess kurtosis without skewness.  Bit-exact port of the legacy
    generator (same draw order: amplitude, then mode field, per mode).

    Returns
    -------
    artifact : (ny, nx) ndarray
    n_leaked : int
    info     : list of {'amp': float} per leaked mode
    """
    if rng is None:
        rng = np.random.default_rng()

    ny, nx = shape
    artifact = np.zeros((ny, nx))

    n_leaked = rng.poisson(n_leaked_mean)
    if n_leaked == 0:
        return artifact, int(n_leaked), []

    kx = np.fft.fftfreq(nx).reshape(1, -1)
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    mode_filter = np.exp(-0.5 * ((kx * mode_corr_scan)**2 +
                                 (ky * mode_corr_cross)**2))

    info = []
    for _ in range(n_leaked):
        amp = leak_efficiency * A_iso * float(rng.standard_t(leak_df))

        raw = rng.standard_normal((ny, nx))
        mode = np.fft.ifft2(np.fft.fft2(raw) * mode_filter).real

        mode_std = mode.std()
        if mode_std > 0:
            mode *= amp / mode_std

        artifact += mode
        info.append({'amp': amp})

    return artifact, int(n_leaked), info
