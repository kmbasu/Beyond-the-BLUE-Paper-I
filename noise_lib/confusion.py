"""
confusion.py — extragalactic point-source confusion noise (Campbell model)
==========================================================================

The Tier-2 ``T2_CONFUSION`` rung of the eta advantage-ceiling campaign: a
marked Poisson (compound-Poisson) superposition of beam-shaped point
sources, drawn from a truncated differential number-count model dN/dS.
This is the classical P(D) / confusion field of Scheuer (1957) and Condon
(1974), and the Campbell model of `eta_advantage_ceiling_memo_v2.md` Sec. 7
with the profile psi identified with the BEAM ITSELF.

Why this model earns a rung of its own
--------------------------------------
Every other Tier-2 model in the campaign injects structure whose spectrum
differs from the template's, so the matched filter retains spectral
leverage and the nonlinear headroom is a correction.  Here the confusion
sources ARE beam-shaped, so for the COMPACT template (tau = beam)

    P_conf(k) = N <a^2> |B(k)|^2      and      |tau~(k)|^2 / P_conf(k) = const,

i.e. the per-mode signal-to-noise is exactly FLAT: the matched filter has
no spectral leverage whatsoever and the entire discrimination between "one
source of amplitude A" and "a superposition of many faint ones" is
higher-order.  That is the maximal-advantage configuration of the memo
(Sec. 21.4), and it is the only Tier-2 case whose non-Gaussianity is
additive, stationary, and free of any Tier-1 (latent-covariance) confound.

The flatness is EXACT on the discrete grid, not approximate, provided the
marks are deposited on pixel centres and convolved with the SAME kernel
that builds the template:

    d(x)  = sum_i a_i delta_{x, p_i}      (p_i uniform on the pixel lattice)
    =>  E|fft2(d)|^2 = N <a^2>            (white: lattice shot noise)
    =>  P_conf = N <a^2> |B~|^2 ,  |tau~_compact|^2 = |B~|^2 .

No pixel-window correction and no sub-pixel placement are required; both
would only complicate an identity that already holds.  Verified to 0.3%
against the analytic N<a^2> on 400 maps (tests_and_results/).

A WARNING THAT COST US TIME ELSEWHERE: DO NOT MEAN-SUBTRACT
------------------------------------------------------------
The reference implementation this module is ported from
(`GPD_Lensing_Cluster_Mass/analysis_modules/simulate_maps.py`) mean-subtracts
each map, which is right for P(D) work and WRONG here.  Mean subtraction
sets fft2(map)[0,0] = 0 in every image, so ``whitening.estimate_psd2d``
clamps P_hat[0,0] to its 1e-20 floor while the compact template has
tau~[0,0] = 1; ``sigma_mf_from_psd`` then divides by 1e-20 and collapses to
~1e-10.  Every other model in the campaign escapes this only because
``spectra._regularize`` hands the DC mode a large FINITE power.  The maps
produced here therefore keep their (Poisson-fluctuating) DC mode, exactly
as a physical confusion map does.

Band regularisation: read this before quoting a compact-template number
-----------------------------------------------------------------------
Memo v2 Sec. 21.4 calls this model "intrinsically band-regularised, so no
floor is needed".  Flat is not the same as convergent.  With
|tau~|^2/P = const the compact template's Fisher weight is UNIFORM over the
whole k-plane, so

    sigma_MF^-2 = sum_k |tau~|^2 / P = N_modes / (N <a^2>)

and the answer is set by how many modes are retained.  Nothing DIVERGES
(a real improvement on the red-background models, whose weight rises as
k^alpha), but the total still tracks the band, and on a 128^2 grid at
FWHM = 5 px the physical power spans ~40 decades, so finite precision
decides where the band ends: P_hat/|B~|^2 stays flat to k ~ 0.63 in float64
but only to k ~ 0.43 in float32, and sigma_MF differs by 24% between the
two.  Consequences, all enforced or flagged downstream:

  * this rung must be generated AND stored in float64
    (``ETA_MAPS_FLOAT64=1``; the registry row asserts it);
  * the production B^2 > 1e-8 mask must NOT be applied unchanged --- it
    would discard 67% of the compact template's genuine Fisher weight;
  * a beam-shaped Gaussian companion does NOT regularise the band, because
    P = (P_g + N<a^2>)|B~|^2 is still flat.  Only an UNSMOOTHED white floor
    does, which is what the ``T2_CONFUSION_WN`` registry row supplies.

Clustering: deliberately absent
-------------------------------
Real sub-mm sources are clustered, and the reference implementation has a
lognormal/Gaussian modulation mode for it (memo `full_simulation_summary_v2.md`
Secs. 10-11).  It is deliberately NOT ported, because it would destroy the
property this rung exists to demonstrate.  For a Cox process of rate
lambda_bar (1 + delta),

    P_conf(k) = |B~(k)|^2 [ lambda_bar <a^2> + lambda_bar^2 <a>^2 P_delta(k) ],

so the bracket acquires the k-dependence of the source-overdensity power
spectrum (a red tilt, P_delta ~ l^-1.2 for the CIB): the per-mode SNR stops
being flat and the matched filter recovers some spectral leverage.
Clustering is therefore a SEVERITY AXIS that interpolates away from the
maximal-advantage corner, not a refinement of it, and the honest baseline
for a ceiling demonstration is the unclustered Poisson field.  Anyone
reproducing this should know that the omission is a choice with a sign:
including clustering can only LOWER the compact-template ceiling.

Provenance
----------
The counts model and its fitted parameters come from the CIB-lensing / GPD
project (`GPD_Lensing_Cluster_Mass/`): `Simulations/counts_350um.py`
Sec. "The adopted fits" (Bethermin et al. 2012 Table 3 at 350 um, fitted
over S = 6.0-94.6 mJy, GOODS-N stacked points excluded) and
`analysis_modules/counts.py` for the (phi*/S*) normalisation convention.
The moment integrals reproduce `counts.BaseCounts.moment`; the confusion
statistics reproduce `counts.BaseCounts.confusion_stats`.  Nothing is
imported across projects: the ~40 lines needed are copied here so that the
other project's cached ensembles stay bit-for-bit reproducible whatever we
do to this file.

Units
-----
Flux density S : mJy.  dN/dS : mJy^-1 deg^-2.  Angles : arcsec.
Maps : mJy/beam (a source of flux S has PEAK S), which is what makes
sigma_c comparable to the published confusion limits.

References
----------
Scheuer (1957), Proc. Camb. Phil. Soc. 53, 764 -- the P(D) formalism.
Condon (1974), ApJ 188, 279 -- confusion noise variance from the counts.
Bethermin et al. (2012), A&A 542, A58 -- the 350 um counts fitted here.
"""

import numpy as np

from .processing import beam_smooth

DEG2_TO_ARCSEC2 = 3600.0 ** 2
FWHM_TO_SIGMA = 1.0 / 2.355          # noise_lib convention (see processing.beam_smooth)

# --- map geometry adopted for the campaign -----------------------------------------
# Chosen so that the paper's fixed pixel convention (beam FWHM = 5 px on a 128^2
# grid) corresponds to a physical configuration in which the fitted counts are
# calibrated: 20" beam at 4"/pixel = 5 px/FWHM, an 8.53' field.
PIX_ARCSEC = 4.0
BEAM_FWHM_ARCSEC = 20.0

# --- the adopted counts model (GPD project, Simulations/counts_350um.py) -----------
SCHECHTER_350UM = dict(alpha=-1.8896771723921697,
                       phistar_deg2=7013.971101909618,
                       sstar_mJy=19.01179957743825)

# Baseline severity.  S_MIN is the faint-end extrapolation limit (the counts are
# not calibrated below 6 mJy; see the GPD memo Sec. 3.1) and S_CUT is the
# bright-source removal threshold -- the "pipeline depth" axis of memo v2 Sec. 21.4.
S_MIN = 0.1
S_CUT = 100.0


def schechter_dnds(S, alpha=None, phistar_deg2=None, sstar_mJy=None):
    """Schechter differential counts  dN/dS = (phi*/S*) (S/S*)^alpha e^(-S/S*).

    Parameters
    ----------
    S            : flux density [mJy], array-like
    alpha, phistar_deg2, sstar_mJy : counts parameters; default to the fitted
        350 um values of ``SCHECHTER_350UM``.

    Returns
    -------
    dN/dS in mJy^-1 deg^-2.

    Note the (phi*/S*) Jacobian: it is the convention that reproduces the
    measured counts (GPD `analysis_modules/counts.py`, "IMPORTANT normalization
    convention"), not the form as typeset in Fujimoto et al. (2023) Eq. (4).
    """
    p = SCHECHTER_350UM
    alpha = p['alpha'] if alpha is None else alpha
    phistar = p['phistar_deg2'] if phistar_deg2 is None else phistar_deg2
    sstar = p['sstar_mJy'] if sstar_mJy is None else sstar_mJy
    S = np.asarray(S, dtype=float)
    return (phistar / sstar) * (S / sstar) ** alpha * np.exp(-S / sstar)


def flux_moment(k, s_lo=S_MIN, s_cut=S_CUT, n_grid=4096, **counts):
    """Flux moment  q_k = int_{s_lo}^{s_cut} S^k (dN/dS) dS   [mJy^k deg^-2].

    Log-space trapezoid (dS = S dlnS), matching the reference implementation's
    ``counts.BaseCounts.moment`` so the two projects' numbers are comparable.
    k = 0 is the source surface density, k = 1 the integrated flux (the
    background monopole), k = 2 sets the confusion variance, and k = 3, 4 the
    leading non-Gaussian cumulants.
    """
    g = np.geomspace(s_lo, s_cut, n_grid)
    return float(np.trapezoid(schechter_dnds(g, **counts) * g ** (k + 1),
                              np.log(g)))


def beam_solid_angle_deg2(beam_fwhm_arcsec=BEAM_FWHM_ARCSEC):
    """Gaussian-beam solid angle Omega_beam = 2 pi sigma_b^2 [deg^2]."""
    sig_b = beam_fwhm_arcsec * FWHM_TO_SIGMA
    return 2.0 * np.pi * sig_b ** 2 / DEG2_TO_ARCSEC2


def confusion_stats(beam_fwhm_arcsec=BEAM_FWHM_ARCSEC, s_lo=S_MIN, s_cut=S_CUT,
                    **counts):
    """Analytic one-point statistics of the confusion field (Campbell / P(D)).

    For a peak-normalised Gaussian beam the beam-solid-angle moments are
    Omega_n = int B^n d^2r = Omega_beam / n, so the cumulants of a pixel are
    kappa_n = Omega_n q_n and the standardised cumulants are
    gamma_{n-2} = kappa_n / kappa_2^(n/2) -- the CLT convergence diagnostics of
    the project's `memo_confusion_noise.md` Sec. 3.

    Returns a dict with
      sigma_c   : confusion rms [mJy/beam],  sigma_c^2 = (Omega_beam/2) q_2
      mean      : mean pixel flux Omega_beam q_1 [mJy/beam]
      n_beam    : sources per beam solid angle, Omega_beam q_0
      n_src_deg2, q0..q4
      skew      : gamma_1 = kappa_3 / kappa_2^1.5
      exkurt    : gamma_2 = kappa_4 / kappa_2^2
    These are CONTINUUM formulae (pixel much smaller than the beam); at
    5 px/FWHM the discrete corrections are sub-percent, and the validation
    script checks the realised map variance against sigma_c.
    """
    om = beam_solid_angle_deg2(beam_fwhm_arcsec)
    q = {k: flux_moment(k, s_lo, s_cut, **counts) for k in range(5)}
    kap = {n: (om / n) * q[n] for n in (1, 2, 3, 4)}
    return dict(sigma_c=float(np.sqrt(kap[2])), mean=float(kap[1]),
                n_beam=float(om * q[0]), n_src_deg2=float(q[0]),
                omega_beam_deg2=float(om),
                q0=q[0], q1=q[1], q2=q[2], q3=q[3], q4=q[4],
                skew=float(kap[3] / kap[2] ** 1.5),
                exkurt=float(kap[4] / kap[2] ** 2))


def mark_sampler(s_lo=S_MIN, s_cut=S_CUT, n_grid=4096, **counts):
    """Inverse-CDF sampler for the mark distribution p(S) ~ dN/dS on [s_lo, s_cut].

    Returns a callable ``draw(rng, n) -> ndarray`` of n fluxes in mJy.  The CDF
    is tabulated on a log grid and inverted by linear interpolation in log S,
    which is exact to the grid resolution; with 4096 nodes over ~3 decades the
    induced error on q_2 is far below the Monte-Carlo error of any ensemble we
    run (the validation script checks the realised <S^2> against ``flux_moment``).
    """
    g = np.geomspace(s_lo, s_cut, n_grid)
    pdf = schechter_dnds(g, **counts) * g          # dN/dlnS
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1])
                                           * np.diff(np.log(g)))))
    cdf /= cdf[-1]
    log_g = np.log(g)

    def draw(rng, n):
        return np.exp(np.interp(rng.random(n), cdf, log_g))
    return draw


def n_sources_mean(shape, pix_arcsec=PIX_ARCSEC, s_lo=S_MIN, s_cut=S_CUT,
                   **counts):
    """Expected number of injected sources in one map: q_0 times the map area."""
    ny, nx = shape
    area_deg2 = (ny * pix_arcsec) * (nx * pix_arcsec) / DEG2_TO_ARCSEC2
    return float(flux_moment(0, s_lo, s_cut, **counts) * area_deg2)


def beam_corr_sigma(shape, rho, pix_arcsec=PIX_ARCSEC,
                    beam_fwhm_arcsec=BEAM_FWHM_ARCSEC,
                    s_lo=S_MIN, s_cut=S_CUT, **counts):
    """Pre-beam white amplitude of a BEAM-CORRELATED Gaussian companion.

    ``rho`` is the companion-to-confusion power ratio.  Because the companion
    is smoothed by the same beam, that ratio is k-independent and is equally
    the ratio of the two components' map variances, so rho = f_N^2 with the
    f_N = sigma_N/sigma_c convention used for the unsmoothed floor:
    rho = 0.25 puts the same 0.5 sigma_c of Gaussian noise in the map as
    T2_CONFUSION_WN, differing ONLY in whether it shares the beam.

        sigma_w0^2 = rho * lambda_pix * <S^2> * (2 pi sigma_b,pix^2)^2

    Adding this white field BEFORE ``beam_smooth`` keeps
    P_tot = (1 + rho) N<a^2> |B|^2 exactly proportional to |B|^2, which is what
    preserves the i.i.d. structure of the whitened field (see the module note
    on T2_CONFUSION_ATM in eta_pipeline/models.py).
    """
    if rho <= 0:
        return 0.0
    ny, nx = shape
    sig_pix = (beam_fwhm_arcsec / pix_arcsec) * FWHM_TO_SIGMA
    lam_pix = n_sources_mean(shape, pix_arcsec, s_lo, s_cut, **counts) / (ny * nx)
    s2 = (flux_moment(2, s_lo, s_cut, **counts)
          / flux_moment(0, s_lo, s_cut, **counts))
    return float(np.sqrt(rho * lam_pix * s2) * 2.0 * np.pi * sig_pix ** 2)


def mark_distribution(s_lo=S_MIN, s_cut=S_CUT, n_grid=4096, **counts):
    """(support, weights) of the mark distribution, for the exact-eta reference.

    Log-spaced flux grid with weights proportional to dN/dlnS -- the same
    tabulation ``mark_sampler`` inverts, handed to
    ``eta_pipeline.analytic1d.compound_poisson_eta_1d``.
    """
    g = np.geomspace(s_lo, s_cut, n_grid)
    return g, schechter_dnds(g, **counts) * g


def confusion_map(shape, rng, pix_arcsec=PIX_ARCSEC,
                  beam_fwhm_arcsec=BEAM_FWHM_ARCSEC,
                  s_lo=S_MIN, s_cut=S_CUT, draw=None, lam=None,
                  sigma_pre_beam=0.0, **counts):
    """One noise-only confusion realisation, in mJy/beam.

    Sources are Poisson in number, uniform on the PIXEL LATTICE, with marks
    drawn from the truncated counts; each is deposited as a delta of amplitude
    S * (2 pi sigma_b,pix^2) so that after the flux-preserving
    ``processing.beam_smooth`` its peak is exactly S (the mJy/beam convention).

    NOT mean-subtracted, deliberately -- see the module docstring.

    Parameters
    ----------
    shape  : (ny, nx)
    rng    : numpy.random.Generator (one child generator per image)
    draw   : optional pre-built mark sampler from ``mark_sampler`` (pass it in
             a loop; rebuilding the CDF per image is the only real cost here)
    lam    : optional pre-computed mean source count (as ``n_sources_mean``)

    Returns
    -------
    (ny, nx) float64 map [mJy/beam], plus the realised source count via the
    ``latents`` machinery of the caller if wanted.
    """
    beam_fwhm_pix = beam_fwhm_arcsec / pix_arcsec
    sig_pix = beam_fwhm_pix * FWHM_TO_SIGMA
    peak_to_grid = 2.0 * np.pi * sig_pix ** 2
    if draw is None:
        draw = mark_sampler(s_lo, s_cut, **counts)
    if lam is None:
        lam = n_sources_mean(shape, pix_arcsec, s_lo, s_cut, **counts)

    n = int(rng.poisson(lam))
    raw = np.zeros(shape, dtype=np.float64)
    if n:
        iy = rng.integers(0, shape[0], n)
        ix = rng.integers(0, shape[1], n)
        np.add.at(raw, (iy, ix), draw(rng, n) * peak_to_grid)
    if sigma_pre_beam:
        #  the beam-correlated Gaussian companion: white BEFORE the beam, so it
        #  ends up sharing the beam exactly (see beam_corr_sigma)
        raw = raw + rng.normal(0.0, sigma_pre_beam, shape)
    return beam_smooth(raw, beam_fwhm_pix), n


def confusion_psd(shape, pix_arcsec=PIX_ARCSEC,
                  beam_fwhm_arcsec=BEAM_FWHM_ARCSEC,
                  s_lo=S_MIN, s_cut=S_CUT, rho_beam=0.0, **counts):
    """Analytic P_conf(k) = N <a^2> |B(k)|^2, unnormalised-FFT convention.

    The generative spectrum of the confusion field, for the complete-data
    bound (Method E) and for the null tests.  Exact for the discrete
    construction of ``confusion_map`` -- no continuum approximation enters.
    """
    beam_fwhm_pix = beam_fwhm_arcsec / pix_arcsec
    sig_pix = beam_fwhm_pix * FWHM_TO_SIGMA
    peak_to_grid = 2.0 * np.pi * sig_pix ** 2
    lam = n_sources_mean(shape, pix_arcsec, s_lo, s_cut, **counts)
    s2 = flux_moment(2, s_lo, s_cut, **counts) / flux_moment(0, s_lo, s_cut,
                                                             **counts)
    ny, nx = shape
    fy = np.fft.fftfreq(ny).reshape(-1, 1)
    fx = np.fft.fftfreq(nx).reshape(1, -1)
    b2 = np.exp(-4.0 * np.pi ** 2 * sig_pix ** 2 * (fx ** 2 + fy ** 2))
    return (1.0 + rho_beam) * lam * s2 * peak_to_grid ** 2 * b2
