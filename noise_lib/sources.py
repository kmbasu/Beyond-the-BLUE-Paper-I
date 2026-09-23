"""
sources.py — central source models, flux priors, and map-domain templates
=========================================================================

The paper's standard source is a beta model at the map center,

    I(r) = I0 * [1 + (r/rc)^2]^(-3/2),

with core radius rc = 7 px and the map convolved with a Gaussian beam of
FWHM = 5 px downstream (the beam lives in processing.beam_smooth and is
applied by the drivers to signal + correlated noise, never to white noise).
The profile and the flux priors reproduce the legacy
generate_beta_model_image* functions exactly, with two extensions:

* an explicit ``rng`` argument (reproducibility; legacy used np.random.*);
* ``source_type='point'``: an unresolved source — a delta function of
  amplitude I0 at the center pixel — which after the downstream beam
  convolution becomes a beam-shaped compact source.  This is the "compact"
  member of the template pair used by the eta-ceiling analysis and by the
  confusion-noise study.

``make_template`` builds the corresponding NOISELESS map-domain template
(source profile convolved with the beam, unit peak amplitude) for use by
the matched-filter and eta pipelines.
"""

import numpy as np

from .processing import beam_smooth


def beta_model(size, rc, I0=1.0, center=None):
    """Beta-model image I(r) = I0 * [1 + (r/rc)^2]^(-1.5) on a size x size grid.

    ``center`` defaults to (size//2, size//2), matching the legacy scripts
    and the matched-filter template convention.
    """
    if center is None:
        center = (size // 2, size // 2)
    y, x = np.indices((size, size))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    return I0 * (1.0 + (r / rc)**2) ** (-1.5)


def point_source(size, I0=1.0, center=None):
    """Delta-function source: I0 at the center pixel, zero elsewhere.

    The physical compact source is this delta convolved with the beam; the
    convolution is applied downstream by the driver pipeline exactly as for
    the beta model, so ``I0`` is the PRE-BEAM amplitude.
    """
    if center is None:
        center = (size // 2, size // 2)
    img = np.zeros((size, size))
    img[center[0], center[1]] = I0
    return img


def _draw_rc(rng, rc_mean, rc_sigma):
    """Positive-truncated normal core-radius draw (legacy convention)."""
    if rc_sigma == 0:
        return float(rc_mean)
    rc = rng.normal(loc=rc_mean, scale=rc_sigma)
    while rc <= 0:
        rc = rng.normal(loc=rc_mean, scale=rc_sigma)
    return float(rc)


def _draw_I0(rng, flux_mode, I0_range, I0_fixed, I0_mean, I0_sigma):
    """Flux draw under the three legacy priors: 'lin' | 'log' | 'gauss'."""
    if I0_fixed is not None:
        return float(I0_fixed)
    if flux_mode == 'lin':
        return float(rng.uniform(*I0_range))
    if flux_mode == 'log':
        log_min, log_max = np.log10(I0_range[0]), np.log10(I0_range[1])
        return float(10.0 ** rng.uniform(log_min, log_max))
    if flux_mode == 'gauss':
        I0 = rng.normal(loc=I0_mean, scale=I0_sigma)
        while I0 <= 0:
            I0 = rng.normal(loc=I0_mean, scale=I0_sigma)
        return float(I0)
    raise ValueError(f"Unknown flux_mode '{flux_mode}'.")


def draw_source(rng, size, source_type='beta', flux_mode='lin',
                rc_mean=7.0, rc_sigma=0.0,
                I0_range=(0.0, 5.0), I0_fixed=None,
                I0_mean=1.0, I0_sigma=0.3,
                is_empty=False):
    """Draw one (pre-beam) source image plus its latent parameters.

    Consolidates the three legacy generate_beta_model_image* functions and
    the empty-image branch of the legacy main loops into one call.

    Parameters
    ----------
    rng         : numpy.random.Generator (per-image child generator)
    size        : map size in pixels
    source_type : 'beta' (default) | 'point' | 'none'
    flux_mode   : 'lin' | 'log' | 'gauss'  — flux prior (legacy semantics)
    rc_mean, rc_sigma : core-radius prior (beta model only)
    I0_range, I0_fixed, I0_mean, I0_sigma : flux-prior parameters
    is_empty    : force a source-free image (I0 = 0), e.g. for eta-ceiling
                  noise-only ensembles or zero_fraction handling

    Returns
    -------
    img : (size, size) ndarray — pre-beam source map (zeros if empty)
    I0  : float — drawn amplitude (0.0 if empty)
    rc  : float — drawn core radius (0.0 if empty or point source)
    """
    if is_empty or source_type == 'none' or I0_fixed == 0:
        return np.zeros((size, size)), 0.0, 0.0

    I0 = _draw_I0(rng, flux_mode, I0_range, I0_fixed, I0_mean, I0_sigma)

    if source_type == 'beta':
        rc = _draw_rc(rng, rc_mean, rc_sigma)
        return beta_model(size, rc, I0), I0, rc
    if source_type == 'point':
        return point_source(size, I0), I0, 0.0
    raise ValueError(f"Unknown source_type '{source_type}'.")


def make_template(size, source_type='beta', rc=7.0, beam_fwhm=5.0,
                  normalize='peak'):
    """Noiseless map-domain template: unit source convolved with the beam.

    * 'beta'  : beta model (core radius rc) x beam  — the paper's standard
                extended template (rc = 7 px, FWHM = 5 px)
    * 'point' : delta x beam = the beam profile itself — the compact template

    Parameters
    ----------
    normalize : 'peak' (max = 1, matched-filter convention of the legacy
                template code) | 'none' (raw convolution of a unit-amplitude
                pre-beam source)

    Returns
    -------
    tau : (size, size) ndarray, source centered at (size//2, size//2)
    """
    if source_type == 'beta':
        pre = beta_model(size, rc, 1.0)
    elif source_type == 'point':
        pre = point_source(size, 1.0)
    else:
        raise ValueError(f"Unknown source_type '{source_type}'.")
    tau = beam_smooth(pre, beam_fwhm)
    if normalize == 'peak':
        tau = tau / tau.max()
    elif normalize != 'none':
        raise ValueError(f"Unknown normalize option '{normalize}'.")
    return tau
