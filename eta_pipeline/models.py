"""
models.py — the noise-model registry for the η campaign (Phase B2)
==================================================================

One entry per noise model of the paper, each carrying everything the
campaign orchestrator needs:

* ``gen(n, seed)`` — in-memory, seeded generator of the NOISE-ONLY
  ensemble at the paper conventions (128², beam FWHM 5), mirroring the
  Phase-A driver pipelines exactly (same noise_lib calls, same order).
  Ensembles are reproducible from (model, seed, n): the container's cheap
  runs and the M3's variational runs operate on IDENTICAL maps and splits.
* ``tier`` and per-method configuration: variational rung set, Tier-1
  quadrature grid builders, complete-data-bound background spectra,
  cumulant-estimator validity, and component generators for the event
  diagnostics (structured power fraction f_s, per-event MF significance ξ
  — the severity coordinates of the memo's two-regime law, §11).

Baseline severities are the Phase-A driver defaults (A = 5 convention).
Cheap severity sweeps (quadrature/analytic) reuse these builders with
modified parameters — see run_eta_campaign.py.

Registry rows (name → tier):
  T0_WHITE, T0_RED, T0_RED_REAL, T0_ANISO          — Gaussian controls (η = 1)
  T1_PSRAND, T1_RANDOR                             — conditionally Gaussian
  T1_PSRAND_SLOPE, T1_PSRAND_AMP                   — cheap-only variants
                                                     (quadrature closure tests)
  T2_MEDIAN, T2_CROSS_SYM, T2_CROSS_POS,
  T2_GLITCH, T2_PCA                                — genuinely non-Gaussian
"""

import numpy as np

import noise_lib as nl

from .quadrature import truncnorm_grid

SIZE = 128
BEAM_FWHM = 5.0
TEMPLATE_RC = 7.0
SHAPE = (SIZE, SIZE)

# baseline severities = Phase-A driver defaults
P_ISO_ARGS = (3.0, 5.0)                    # slope, amplitude
P_ANISO_ARGS = (3.0, 0.5, 3.0, 5.0)        # slope_scan, A_scan, slope_iso, A_iso
WHITE_REAL = 1.0
PSRAND_ARGS = (3.0, 0.5, 5.0, 0.8)         # slope_mean/sigma, amp_mean/sigma
CROSS_ARGS = dict(n_crossings_mean=8, along_scan_corr=30.0, beam_sigma_rows=2.0,
                  amp_scale=0.5, amp_df=2.5)
GLITCH_ARGS = dict(n_glitches=25, tau_pix=5.0, amp_scale=0.3,
                   sigma_beam=2.0, sigma_cross=2.0)
PCA_ARGS = dict(n_leaked_mean=3, leak_efficiency=0.15, A_iso=5.0, leak_df=2.5,
                mode_corr_scan=20.0, mode_corr_cross=40.0)

# --- white-noise floor (the *_WN registry variants) --------------------------------
# An unsmoothed detector floor.  Without one, a compact template on beam-smoothed
# red noise has Fisher weight |tau~|^2/P = B^2/(B^2 P_red) ~ k^alpha: the beam
# cancels, the information integral diverges, and every compact-template quantity
# reports the mode cutoff rather than the noise (see WN_Floor_Implementation_Plan.md
# and eta_advantage_ceiling_memo_v2.md Sec. 23).  The floor is PER MODEL, calibrated
# by tools/calibrate_wn_floor.py so that the white/red knee sits one octave above
# the EXTENDED template's Fisher-band centroid:
#
#     sigma_w^2 = P_background_ring(2 * exp<log k>_f) / n_pix .
#
# Verified consequence: the extended template keeps >98% of its Fisher weight below
# the knee (its ceiling shifts by a few percent), while the compact template's
# sigma_MF mask drift falls from 36-47% to 0.00%.  The floor is computed from the
# GAUSSIAN BACKGROUND only, so changing amp_scale / leak_efficiency never moves it.
WN_SEED_OFFSET = 606000     # independent RNG stream; no collision with +991
                            # (components), +555000 (hp), +900001 (closure)
WN_KNEE_FACTOR = 2.0        # k_knee = WN_KNEE_FACTOR * exp<log k>_f (extended)

# Calibrated 2026-08-31 by tools/calibrate_wn_floor.py (KNEE_FACTOR = 2.0,
# n_ens = 1500).  Verified: extended Fisher weight above the knee 0.6-1.9%,
# extended sigma_MF mask drift 0.00% with and without the floor, compact drift
# 36-47% -> 0.00%.  Re-run the tool if a background spectrum or the template
# pair changes.
WN_FLOORS = {
    'T1_PSRAND_WN':       0.27,     # k_ext 0.0600  knee 0.1200
    'T1_PSRAND_SLOPE_WN': 0.27,     # k_ext 0.0598  knee 0.1196 (shared with above
                                    #   on purpose: the closure test compares them)
    'T1_RANDOR_WN':       1.04,     # k_ext 0.0457  knee 0.0914 (aniso background
                                    #   is loud at the knee, hence the large floor)
    'T2_MEDIAN_WN':       1.06,     # k_ext 0.0460  knee 0.0920
    'T2_CROSS_SYM_WN':    0.51,     # k_ext 0.0502  knee 0.1005
    'T2_CROSS_POS_WN':    0.48,     # k_ext 0.0510  knee 0.1021
    'T2_GLITCH_WN':       0.86,     # k_ext 0.0491  knee 0.0982
    'T2_PCA_WN':          0.10,     # k_ext 0.0742  knee 0.1485 (leaked modes push
                                    #   the band up, so little floor is needed)
}


def beam2(shape=SHAPE, fwhm=BEAM_FWHM):
    """Squared beam transfer B²(k) (power convention) on the FFT grid."""
    ny, nx = shape
    sigma = fwhm / 2.355
    fy = np.fft.fftfreq(ny).reshape(-1, 1)
    fx = np.fft.fftfreq(nx).reshape(1, -1)
    return np.exp(-4.0 * np.pi**2 * sigma**2 * (fx**2 + fy**2))


def p_white(sigma_w, shape=SHAPE):
    """White-noise PSD in the unnormalized-FFT convention: E|fft2|^2 = n_pix*sigma^2."""
    return np.full(shape, float(shape[0]) * float(shape[1]) * float(sigma_w)**2)


def make_templates(size=SIZE):
    """The paper's template pair: extended (rc=7 β⊛beam) and compact (beam)."""
    return {
        'extended': nl.make_template(size, 'beta', TEMPLATE_RC, BEAM_FWHM,
                                     normalize='none'),
        'compact': nl.make_template(size, 'point', beam_fwhm=BEAM_FWHM,
                                    normalize='none'),
    }


def _child_rngs(seed, n):
    ss = np.random.SeedSequence(seed)
    return [np.random.default_rng(c) for c in ss.spawn(n)]


def map_dtype():
    """Storage dtype of generated maps: float32 (campaign default) or float64.

    Set the environment variable ETA_MAPS_FLOAT64=1 to store maps in float64.
    Why this exists (2026-09-03): on FLOORLESS red-background models the
    physical spectrum at high k is beam-suppressed by up to e^-40, so the
    float32 mantissa rounding of the stored map (absolute level ~1e-7 x |map|,
    white in k) dominates the map's own content there.  Whitening by the
    data-estimated P-hat then amplifies that rounding band to unit variance,
    and its per-image amplitude tracks the map envelope -- a heteroscedastic
    side channel with anomalous signal-to-noise that a nonlinear test
    function can read.  Generating in float64 (the variational code casts to
    float32 only AFTER whitening) removes the channel; comparing the two is
    the A/B test.  Floored models (P_w >> rounding power) are immune.
    """
    import os
    return np.float64 if os.environ.get('ETA_MAPS_FLOAT64') == '1' else np.float32


def _ensemble(n, seed, one_map):
    """Stack n maps from per-image child generators (float32 by default;
    see map_dtype())."""
    out = np.empty((n, SIZE, SIZE), dtype=map_dtype())
    for i, rng in enumerate(_child_rngs(seed, n)):
        out[i] = one_map(rng)
    return out


def _with_floor(gen, sigma_w):
    """Add an unsmoothed white floor to every map of an existing generator.

    The floor is drawn from an INDEPENDENT seed stream, so calling this wrapper and
    the base generator with the SAME seed gives a paired pair of ensembles: the
    same correlated-noise realization, with and without a floor.

    NOTE (2026-08-31): the campaign does NOT get that pairing.  Each registry row
    carries its own seed, and run_eta_campaign.prepare_model calls the generator
    with the *_WN row's seed, so a *_WN campaign ensemble is an INDEPENDENT DRAW
    from its floorless namesake, not the same draw plus a floor.  For light-tailed
    models that is immaterial; for the heavy-tailed ones it is not (T2_PCA's sample
    excess kurtosis measured 288.6 at one seed and 44.6 at another, on ensembles
    whose floor differs by a factor 10^-3 in variance -- the statistic is nowhere
    near converged with t_2.5 marks).  Any floorless-vs-floored comparison on a
    heavy-tailed model must therefore be run with matched seeds before it is
    attributed to the floor.

    Valid only where the floor commutes with everything the base generator does --
    true for all additive models.  T2_MEDIAN is NOT one of them (a row-wise median
    is nonlinear and the Phase-A pipeline order puts white noise BEFORE it), so it
    has its own generator, _gen_median_floor.
    """
    def wrapped(n, seed):
        maps, lat = gen(n, seed)
        if sigma_w:
            for i, rng in enumerate(_child_rngs(seed + WN_SEED_OFFSET, n)):
                maps[i] += nl.white_noise(SHAPE, sigma_w, rng).astype(np.float32)
        return maps, lat
    wrapped.__name__ = getattr(gen, '__name__', 'gen') + '_wn'
    return wrapped


def _components_with_floor(comp, sigma_w):
    """Floor variant of a (background, structure, counts) component splitter.

    The floor belongs to the IRREDUCIBLE GAUSSIAN BACKGROUND g, never to the
    structured component: that is what makes the complete-data bound
    eta <= [sum |tau~|^2/P_g] / [sum |tau~|^2/P_tot] come out right, and it is
    what lowers f_s and the per-event significance xi (README_PhaseB2a Sec. 5
    attributed xi_rms = 15-50 to the missing floor).
    """
    def wrapped(n, seed):
        bg, st, counts = comp(n, seed)
        if sigma_w:
            for i, rng in enumerate(_child_rngs(seed + WN_SEED_OFFSET, n)):
                bg[i] += nl.white_noise(SHAPE, sigma_w, rng).astype(np.float32)
        return bg, st, counts
    return wrapped


def _gen_median_floor(sigma_w):
    """T2_MEDIAN with a floor -- the one model where placement is not commutative.

    Phase-A pipeline order is  correlated noise -> beam -> UNSMOOTHED WHITE ->
    artifacts -> row median.  A row-wise median is nonlinear, so white noise added
    after it is a different model from white noise added before it; the whole point
    of this case is that nonlinearity, so the order matters and the floor goes in
    before the median.
    """
    def gen(n, seed):
        P = nl.psd_two_component(SHAPE, *P_ANISO_ARGS)
        out = np.empty((n, SIZE, SIZE), dtype=np.float32)
        rngs_w = _child_rngs(seed + WN_SEED_OFFSET, n)
        for i, rng in enumerate(_child_rngs(seed, n)):
            m = nl.beam_smooth(nl.grf_from_psd(P, rng), BEAM_FWHM)
            m = m + nl.white_noise(SHAPE, sigma_w, rngs_w[i])
            out[i] = nl.row_median_removal(m, axis=1)[0]
        return out, {}
    gen.__name__ = 'gen_t2_median_wn'
    return gen


# ----------------------------------------------------------------------------------
# Generators (noise-only; pipeline order identical to the Phase-A drivers)
# ----------------------------------------------------------------------------------

def gen_t0_white(n, seed):
    return _ensemble(n, seed, lambda rng: nl.white_noise(SHAPE, 1.0, rng)), {}


def gen_t0_red(n, seed):
    P = nl.psd_isotropic_powerlaw(SHAPE, *P_ISO_ARGS)
    f = lambda rng: nl.beam_smooth(nl.grf_from_psd(P, rng), BEAM_FWHM)
    return _ensemble(n, seed, f), {}


def gen_t0_red_real(n, seed):
    P = nl.psd_isotropic_powerlaw(SHAPE, *P_ISO_ARGS)
    f = lambda rng: (nl.beam_smooth(nl.grf_from_psd(P, rng), BEAM_FWHM)
                     + nl.white_noise(SHAPE, WHITE_REAL, rng))
    return _ensemble(n, seed, f), {}


def gen_t0_aniso(n, seed):
    P = nl.psd_two_component(SHAPE, *P_ANISO_ARGS)
    f = lambda rng: nl.beam_smooth(nl.grf_from_psd(P, rng), BEAM_FWHM)
    return _ensemble(n, seed, f), {}


def _gen_psrand(n, seed, slope_sigma, amp_sigma):
    sm, _, am, _ = PSRAND_ARGS
    maps = np.empty((n, SIZE, SIZE), dtype=map_dtype())
    slopes, amps = np.empty(n), np.empty(n)
    for i, rng in enumerate(_child_rngs(seed, n)):
        n1f, lat = nl.psrand_1overf_noise(SHAPE, sm, slope_sigma, am, amp_sigma,
                                          rng=rng)
        maps[i] = nl.beam_smooth(n1f, BEAM_FWHM)
        slopes[i], amps[i] = lat['slope'], lat['amplitude']
    return maps, {'slope': slopes, 'amplitude': amps}


def gen_t1_psrand(n, seed):
    return _gen_psrand(n, seed, PSRAND_ARGS[1], PSRAND_ARGS[3])


def gen_t1_psrand_slope(n, seed):
    return _gen_psrand(n, seed, PSRAND_ARGS[1], 0.0)


def gen_t1_psrand_amp(n, seed):
    return _gen_psrand(n, seed, 0.0, PSRAND_ARGS[3])


def gen_t1_randor(n, seed):
    maps = np.empty((n, SIZE, SIZE), dtype=np.float32)
    angles = np.empty(n)
    ss_sc, a_sc, ss_iso, a_iso = P_ANISO_ARGS
    for i, rng in enumerate(_child_rngs(seed, n)):
        th = float(rng.uniform(0.0, np.pi))
        P = nl.psd_two_component(SHAPE, ss_sc, a_sc, ss_iso, a_iso,
                                 scan_angle=th)
        maps[i] = nl.beam_smooth(nl.grf_from_psd(P, rng), BEAM_FWHM)
        angles[i] = th
    return maps, {'angle': angles}


def gen_t2_median(n, seed):
    P = nl.psd_two_component(SHAPE, *P_ANISO_ARGS)

    def f(rng):
        m = nl.beam_smooth(nl.grf_from_psd(P, rng), BEAM_FWHM)
        return nl.row_median_removal(m, axis=1)[0]
    return _ensemble(n, seed, f), {}


def _gen_cross(n, seed, sign):
    P = nl.psd_isotropic_powerlaw(SHAPE, *P_ISO_ARGS)

    def f(rng):
        m = nl.beam_smooth(nl.grf_from_psd(P, rng), BEAM_FWHM)
        art, _, _ = nl.scan_crossing_artifacts(SHAPE, sign_convention=sign,
                                               rng=rng, **CROSS_ARGS)
        return m + art
    return _ensemble(n, seed, f), {}


def gen_t2_cross_sym(n, seed):
    return _gen_cross(n, seed, 'symmetric')


def gen_t2_cross_pos(n, seed):
    return _gen_cross(n, seed, 'positive')


def gen_t2_glitch(n, seed):
    P = nl.psd_two_component(SHAPE, *P_ANISO_ARGS)

    def f(rng):
        m = nl.beam_smooth(nl.grf_from_psd(P, rng), BEAM_FWHM)
        m, _ = nl.add_glitch_residuals(m, GLITCH_ARGS['n_glitches'],
                                       GLITCH_ARGS['tau_pix'],
                                       GLITCH_ARGS['amp_scale'],
                                       sigma_beam=GLITCH_ARGS['sigma_beam'],
                                       sigma_cross=GLITCH_ARGS['sigma_cross'],
                                       rng=rng)
        return m
    return _ensemble(n, seed, f), {}


def gen_t2_pca(n, seed):
    P = nl.psd_isotropic_powerlaw(SHAPE, *P_ISO_ARGS)

    def f(rng):
        m = nl.beam_smooth(nl.grf_from_psd(P, rng), BEAM_FWHM)
        art, _, _ = nl.pca_leaked_modes(SHAPE, rng=rng, **PCA_ARGS)
        return m + art
    return _ensemble(n, seed, f), {}


# ----------------------------------------------------------------------------------
# Component generators (event diagnostics: f_s and per-event ξ)
# ----------------------------------------------------------------------------------

def _components_cross(n, seed, sign):
    P = nl.psd_isotropic_powerlaw(SHAPE, *P_ISO_ARGS)
    bg = np.empty((n, SIZE, SIZE), dtype=np.float32)
    st = np.empty((n, SIZE, SIZE), dtype=np.float32)
    counts = np.empty(n)
    for i, rng in enumerate(_child_rngs(seed, n)):
        bg[i] = nl.beam_smooth(nl.grf_from_psd(P, rng), BEAM_FWHM)
        art, c, _ = nl.scan_crossing_artifacts(SHAPE, sign_convention=sign,
                                               rng=rng, **CROSS_ARGS)
        st[i], counts[i] = art, c
    return bg, st, counts


def _components_glitch(n, seed):
    P = nl.psd_two_component(SHAPE, *P_ANISO_ARGS)
    bg = np.empty((n, SIZE, SIZE), dtype=np.float32)
    st = np.empty((n, SIZE, SIZE), dtype=np.float32)
    for i, rng in enumerate(_child_rngs(seed, n)):
        bg[i] = nl.beam_smooth(nl.grf_from_psd(P, rng), BEAM_FWHM)
        z = np.zeros(SHAPE)
        z, _ = nl.add_glitch_residuals(z, GLITCH_ARGS['n_glitches'],
                                       GLITCH_ARGS['tau_pix'],
                                       GLITCH_ARGS['amp_scale'],
                                       sigma_beam=GLITCH_ARGS['sigma_beam'],
                                       sigma_cross=GLITCH_ARGS['sigma_cross'],
                                       rng=rng)
        st[i] = z
    counts = np.full(n, GLITCH_ARGS['n_glitches'], dtype=float)
    return bg, st, counts


def _components_pca(n, seed):
    P = nl.psd_isotropic_powerlaw(SHAPE, *P_ISO_ARGS)
    bg = np.empty((n, SIZE, SIZE), dtype=np.float32)
    st = np.empty((n, SIZE, SIZE), dtype=np.float32)
    counts = np.empty(n)
    for i, rng in enumerate(_child_rngs(seed, n)):
        bg[i] = nl.beam_smooth(nl.grf_from_psd(P, rng), BEAM_FWHM)
        art, c, _ = nl.pca_leaked_modes(SHAPE, rng=rng, **PCA_ARGS)
        st[i], counts[i] = art, c
    return bg, st, counts


# ----------------------------------------------------------------------------------
# Tier-1 quadrature grid builders
# ----------------------------------------------------------------------------------

def quad_psrand(slope_sigma=PSRAND_ARGS[1], amp_sigma=PSRAND_ARGS[3],
                n_slope=61, n_amp=201):
    """Runner closure for the per-image PSRAND quadrature.

    Uses the FACTORIZED scaled quadrature (P = a²·P_base(slope)): the slope
    grid is swept at 1-D cost and the amplitude grid is effectively
    continuous (201 nodes for free), removing the amplitude-quantization
    bias found in the B2a validation.  Spectra include the beam; the
    k-mask keeps modes with B² > 1e-8 (rounding-noise guard, B1 lesson).
    Returns run(maps_eval, tau) -> dict(eta, err, ...).
    """
    from .quadrature import eta_marg_quadrature_scaled
    sm, _, am, _ = PSRAND_ARGS
    B2 = beam2()
    k_mask = B2 > 1e-8
    s_grid, s_w = ((np.array([sm]), np.array([1.0])) if slope_sigma == 0 else
                   truncnorm_grid(sm, slope_sigma, 1.0, n=n_slope))
    a_grid, a_w = ((np.array([am]), np.array([1.0])) if amp_sigma == 0 else
                   truncnorm_grid(am, amp_sigma, 1e-6, n=n_amp))

    def run(maps_eval, tau):
        base = np.stack([nl.psd_isotropic_powerlaw(SHAPE, s, 1.0) * B2
                         for s in s_grid])
        eta, err = eta_marg_quadrature_scaled(maps_eval, base, s_w,
                                              a_grid, a_w, tau, k_mask=k_mask)
        return {'eta': eta, 'err': err, 'n_grid': int(len(s_w) * len(a_w)),
                'method': 'scaled quadrature (slope x amp factorized)'}
    return run


def quad_randor(n_theta=180):
    """Runner closure for the random-orientation quadrature (θ grid over
    [0, π), midpoint nodes; spectrum has period π)."""
    from .quadrature import eta_marg_quadrature
    B2 = beam2()
    k_mask = B2 > 1e-8
    ss_sc, a_sc, ss_iso, a_iso = P_ANISO_ARGS
    thetas = (np.arange(n_theta) + 0.5) * np.pi / n_theta

    def run(maps_eval, tau):
        P_grid = np.stack([nl.psd_two_component(SHAPE, ss_sc, a_sc, ss_iso,
                                                a_iso, scan_angle=t) * B2
                           for t in thetas])
        eta, err = eta_marg_quadrature(maps_eval, P_grid,
                                       np.full(n_theta, 1.0 / n_theta), tau,
                                       k_mask=k_mask)
        return {'eta': eta, 'err': err, 'n_grid': int(n_theta),
                'method': 'theta-grid quadrature'}
    return run


def quad_psrand_wn(sigma_w, slope_sigma=PSRAND_ARGS[1], amp_sigma=PSRAND_ARGS[3],
                   n_slope=61, n_amp=201):
    """Runner closure for the FLOORED PSRAND quadrature.

    The floor is additive, so P = a^2*P_base(s) + P_w does not factorize and the
    fast scaled path of quad_psrand is unavailable.  eta_marg_quadrature_floor
    sweeps the (s, a) grid explicitly while keeping the validated resolution
    (n_amp ~ 200: the amplitude posterior is far sharper than a 41-node grid can
    represent -- the quantization bias found in the B2a validation).

    The k-mask is retained as a cheap guard, but with a floor present it should be
    inert; null test N8 asserts exactly that.
    """
    from .quadrature import eta_marg_quadrature_floor
    sm, _, am, _ = PSRAND_ARGS
    B2 = beam2()
    Pw = p_white(sigma_w)
    k_mask = B2 > 1e-8
    s_grid, s_w = ((np.array([sm]), np.array([1.0])) if slope_sigma == 0 else
                   truncnorm_grid(sm, slope_sigma, 1.0, n=n_slope))
    a_grid, a_w = ((np.array([am]), np.array([1.0])) if amp_sigma == 0 else
                   truncnorm_grid(am, amp_sigma, 1e-6, n=n_amp))

    def run(maps_eval, tau):
        base = np.stack([nl.psd_isotropic_powerlaw(SHAPE, s, 1.0) * B2
                         for s in s_grid])
        eta, err = eta_marg_quadrature_floor(maps_eval, base, s_w, a_grid, a_w,
                                             tau, Pw, k_mask=k_mask)
        return {'eta': eta, 'err': err, 'n_grid': int(len(s_w) * len(a_w)),
                'sigma_w': float(sigma_w),
                'method': f'full 2-D quadrature (slope x amp), floor sigma_w={sigma_w}'}
    return run


def quad_randor_wn(sigma_w, n_theta=180):
    """Runner closure for the FLOORED random-orientation quadrature.

    Cheap by comparison: the theta grid is already explicit, so the floor is one
    addition per node.
    """
    from .quadrature import eta_marg_quadrature
    B2 = beam2()
    Pw = p_white(sigma_w)
    k_mask = B2 > 1e-8
    ss_sc, a_sc, ss_iso, a_iso = P_ANISO_ARGS
    thetas = (np.arange(n_theta) + 0.5) * np.pi / n_theta

    def run(maps_eval, tau):
        P_grid = np.stack([nl.psd_two_component(SHAPE, ss_sc, a_sc, ss_iso,
                                                a_iso, scan_angle=t) * B2 + Pw
                           for t in thetas])
        eta, err = eta_marg_quadrature(maps_eval, P_grid,
                                       np.full(n_theta, 1.0 / n_theta), tau,
                                       k_mask=k_mask)
        return {'eta': eta, 'err': err, 'n_grid': int(n_theta),
                'sigma_w': float(sigma_w),
                'method': f'theta-grid quadrature, floor sigma_w={sigma_w}'}
    return run


# ----------------------------------------------------------------------------------
# Complete-data-bound background spectra (Tier-2 event models)
# ----------------------------------------------------------------------------------

def bg_iso():
    return nl.psd_isotropic_powerlaw(SHAPE, *P_ISO_ARGS) * beam2()


def bg_aniso():
    return nl.psd_two_component(SHAPE, *P_ANISO_ARGS) * beam2()


def bg_with_floor(bg_fn, sigma_w):
    """Complete-data-bound background spectrum of a floored model: P_g + P_white."""
    return lambda: bg_fn() + p_white(sigma_w)


# ----------------------------------------------------------------------------------
# Confusion noise (appended 2026-09-03) -- the Campbell / compound-Poisson rung
# ----------------------------------------------------------------------------------
# Physical configuration: the paper's fixed pixel convention (128^2, beam FWHM 5 px)
# read as a 20" beam at 4"/pixel -- an 8.53' field -- so that the 350 um counts fitted
# in the CIB-lensing/GPD project apply unchanged.  See noise_lib/confusion.py for the
# construction, the provenance of the counts, and the two warnings that matter
# (no mean subtraction; the band is NOT self-regularising).
#
# Baseline severity S_min = 0.1 mJy, S_cut = 100 mJy gives analytically
#     sigma_c = 6.449 mJy/beam,  N_beam = 28.25,  S_cut/sigma_c = 15.5,
#     skewness 2.120, excess kurtosis 9.091,  ~16 340 sources per map.
# N_beam = 28 is far below the 200-per-beam threshold at which the reference
# implementation substitutes a Gaussian for the faint end, so no faint-end
# substitution is used and that systematic does not arise at this geometry.
#
# THE FLOOR.  The WN_FLOORS k-space convention above (knee = 2 exp<log k>_f of the
# EXTENDED template) does not transfer to this model: the confusion field's extended
# Fisher band sits at exp<log k> = 0.020, so that formula would demand sigma_w = 6.5
# sigma_c and drown the model.  The floor is instead set physically, at the
# instrument-noise ratio f_N = sigma_N/sigma_c = 0.5 adopted for Herschel and CCAT in
# the source project.  Verified to deliver what the convention actually exists for:
# the compact Fisher band moves to exp<log k> = 0.109 with 0.0% of its weight above
# the production mask edge, and the complete-data bound is stable to 0.4% across
# B^2 cuts 1e-4...1e-14 (null test N8).
CONFUSION_ARGS = dict(s_lo=nl.confusion.S_MIN, s_cut=nl.confusion.S_CUT)
CONFUSION_STATS = nl.confusion.confusion_stats(**CONFUSION_ARGS)
CONFUSION_SIGMA_C = CONFUSION_STATS['sigma_c']
CONFUSION_F_N = 0.5
CONFUSION_FLOOR = CONFUSION_F_N * CONFUSION_SIGMA_C      # 3.2244 mJy/beam


def _gen_confusion(n, seed, sigma_w=0.0):
    """Confusion ensemble, ALWAYS float64 -- deliberately ignoring map_dtype().

    Every other model may be stored in float32 because its spectrum is either
    white-floored or shallow.  Here P_conf = N<a^2>|B|^2 spans ~39 decades across
    the grid, and float32 storage moves the compact template's sigma_MF by 24%
    (P_hat/|B|^2 leaves the flat plateau at k ~ 0.43 instead of k ~ 0.63).  The
    variational code casts to float32 only AFTER whitening, so returning float64
    here is sufficient and needs no environment variable.

    sigma_w > 0 adds an UNSMOOTHED white floor from an independent seed stream,
    in float64 (the shared _with_floor wrapper casts its floor to float32, which
    is harmless for the other models but pointless here).
    """
    draw = nl.confusion.mark_sampler(**CONFUSION_ARGS)
    lam = nl.confusion.n_sources_mean(SHAPE, **CONFUSION_ARGS)
    maps = np.empty((n, SIZE, SIZE), dtype=np.float64)
    n_src = np.empty(n)
    rngs_w = _child_rngs(seed + WN_SEED_OFFSET, n) if sigma_w else None
    for i, rng in enumerate(_child_rngs(seed, n)):
        m, ns = nl.confusion.confusion_map(SHAPE, rng, draw=draw, lam=lam,
                                           **CONFUSION_ARGS)
        if sigma_w:
            m = m + nl.white_noise(SHAPE, sigma_w, rngs_w[i])
        maps[i], n_src[i] = m, ns
    return maps, {'n_src': n_src}


def gen_t2_confusion(n, seed):
    return _gen_confusion(n, seed, 0.0)


def gen_t2_confusion_wn(n, seed):
    return _gen_confusion(n, seed, CONFUSION_FLOOR)


def _components_confusion(n, seed, sigma_w=0.0):
    """(background, structure, counts) for the f_s / xi severity diagnostics.

    The ENTIRE confusion field is the structured component and the irreducible
    Gaussian background is the white floor alone -- zeros in the floorless case,
    where f_s is therefore 1 by construction.  That is not a degenerate corner
    but the defining statement of this model: there is no Gaussian part to hide
    behind, which is also why the floorless row carries no complete-data bound
    (perfect removal of every source leaves an empty map, so the envelope is
    infinite rather than large).
    """
    draw = nl.confusion.mark_sampler(**CONFUSION_ARGS)
    lam = nl.confusion.n_sources_mean(SHAPE, **CONFUSION_ARGS)
    bg = np.zeros((n, SIZE, SIZE), dtype=np.float64)
    st = np.empty((n, SIZE, SIZE), dtype=np.float64)
    counts = np.empty(n)
    rngs_w = _child_rngs(seed + WN_SEED_OFFSET, n) if sigma_w else None
    for i, rng in enumerate(_child_rngs(seed, n)):
        st[i], counts[i] = nl.confusion.confusion_map(SHAPE, rng, draw=draw,
                                                      lam=lam, **CONFUSION_ARGS)
        if sigma_w:
            bg[i] = nl.white_noise(SHAPE, sigma_w, rngs_w[i])
    return bg, st, counts


# --- the beam-correlated companion (T2_CONFUSION_ATM, appended 2026-09-03) ---------
# The THIRD configuration, and the only one in the campaign with an exactly known eta.
#
# T2_CONFUSION_WN adds an UNSMOOTHED white floor: P_tot = C|B|^2 + P_w.  That breaks
# P ~ |B|^2, regularises the compact band, and makes eta template-DEPENDENT.
# T2_CONFUSION_ATM adds the same amount of Gaussian noise (rho = f_N^2 = 0.25, i.e.
# 0.5 sigma_c in the map) but BEAM-CORRELATED, so P_tot = (1 + rho) C|B|^2 stays exactly
# proportional to |B|^2.  Whitening then returns
#
#     x = (d + w) * const ,   d = the marked-point lattice field,  w white Gaussian,
#
# i.e. i.i.d. PIXELS.  Three exact consequences follow, and each is a test:
#   (a) eta is TEMPLATE-INDEPENDENT (J = I_1 Id; null test N4);
#   (b) eta is FINITE and computable in closed form -- the Gaussian removes the atom
#       that makes the floorless model's Fisher information diverge --- namely
#       eta = Var[Y] I(Y) for Y = D + W, evaluated by
#       eta_pipeline.analytic1d.compound_poisson_eta_1d;
#   (c) the complete-data bound is exactly (1 + rho)/rho = 5, for ANY template, since
#       both spectra are proportional.
# At the campaign configuration this gives eta = 3.569 (converged to 5e-5), a value
# the variational ladder must reproduce on BOTH templates.  Nothing else in the
# battery calibrates the ABSOLUTE scale of an eta > 1 at production map size: N3 does
# it on 32^2 i.i.d. pixels with a single-pixel template.
#
# This row does NOT regularise the band (P is still ~|B|^2, so the compact template's
# Fisher weight is still uniform and still precision-limited).  It is a null test, not
# a production row -- and the gap between its measured and exact eta is a direct,
# calibrated measurement of how much the rounding side-channel costs.
CONFUSION_RHO_ATM = CONFUSION_F_N ** 2          # 0.25: same map-level sigma_N as _WN


def _gen_confusion_atm(n, seed, rho=CONFUSION_RHO_ATM):
    """Confusion + a BEAM-CORRELATED Gaussian companion; always float64."""
    draw = nl.confusion.mark_sampler(**CONFUSION_ARGS)
    lam = nl.confusion.n_sources_mean(SHAPE, **CONFUSION_ARGS)
    sig0 = nl.confusion.beam_corr_sigma(SHAPE, rho, **CONFUSION_ARGS)
    maps = np.empty((n, SIZE, SIZE), dtype=np.float64)
    n_src = np.empty(n)
    for i, rng in enumerate(_child_rngs(seed, n)):
        maps[i], n_src[i] = nl.confusion.confusion_map(
            SHAPE, rng, draw=draw, lam=lam, sigma_pre_beam=sig0,
            **CONFUSION_ARGS)
    return maps, {'n_src': n_src}


def _components_confusion_atm(n, seed, rho=CONFUSION_RHO_ATM):
    """Structure = the confusion field; background = the beam-correlated Gaussian.

    Drawn from the SAME child generators as _gen_confusion_atm, but the two
    components are produced separately, so bg + st is a valid realisation of the
    model only in distribution, not per image -- which is all the f_s / xi
    diagnostics use.
    """
    draw = nl.confusion.mark_sampler(**CONFUSION_ARGS)
    lam = nl.confusion.n_sources_mean(SHAPE, **CONFUSION_ARGS)
    sig0 = nl.confusion.beam_corr_sigma(SHAPE, rho, **CONFUSION_ARGS)
    bg = np.empty((n, SIZE, SIZE), dtype=np.float64)
    st = np.empty((n, SIZE, SIZE), dtype=np.float64)
    counts = np.empty(n)
    for i, rng in enumerate(_child_rngs(seed, n)):
        st[i], counts[i] = nl.confusion.confusion_map(
            SHAPE, rng, draw=draw, lam=lam, **CONFUSION_ARGS)
        bg[i] = nl.beam_smooth(rng.normal(0.0, sig0, SHAPE), BEAM_FWHM)
    return bg, st, counts


def bg_confusion_atm(rho=CONFUSION_RHO_ATM):
    """Complete-data-bound background: the beam-correlated companion alone.

    P_g = rho C |B|^2, so the bound is exactly (1 + rho)/rho for every template.
    """
    return lambda: (rho / (1.0 + rho)) * nl.confusion.confusion_psd(
        SHAPE, rho_beam=rho, **CONFUSION_ARGS)


def bg_confusion(sigma_w):
    """Complete-data-bound background: the white floor alone."""
    return lambda: p_white(sigma_w)


# ----------------------------------------------------------------------------------
# The registry
# ----------------------------------------------------------------------------------

RUNGS_GAUSS = ('linear', 'quadratic', 'cnn')
RUNGS_T1 = ('linear', 'quadratic', 'cnn')
RUNGS_T2 = ('linear', 'quadratic', 'cubic', 'cnn')

REGISTRY = {
    'T0_WHITE': dict(
        seed=777000, floor=0.0,
        tier='T0', gen=gen_t0_white, rungs=RUNGS_GAUSS, cumulant_ok=True,
        note='white Gaussian benchmark; production η=1 null'),
    'T0_RED': dict(
        seed=777017, floor=0.0,
        tier='T0', gen=gen_t0_red, rungs=RUNGS_GAUSS, cumulant_ok=True,
        note='isotropic 1/f³ + beam; production η=1 null'),
    'T0_RED_REAL': dict(
        seed=777034, floor=WHITE_REAL,
        tier='T0', gen=gen_t0_red_real, rungs=RUNGS_GAUSS, cumulant_ok=True,
        note='realistic: beamed 1/f³ + unsmoothed white; η=1 null'),
    'T0_ANISO': dict(
        seed=777051, floor=0.0,
        tier='T0', gen=gen_t0_aniso, rungs=RUNGS_GAUSS, cumulant_ok=True,
        note='fixed-direction anisotropic Gaussian; η=1 with 2-D-PSD whitening'),
    'T1_PSRAND': dict(
        seed=777068, floor=0.0,
        tier='T1', gen=gen_t1_psrand, rungs=RUNGS_T1, cumulant_ok=False,
        quad=quad_psrand,
        note='per-image slope+amplitude randomization (flagship §5.1); '
             'quadrature exact; cumulants unreliable (variance-mixture tails)'),
    'T1_PSRAND_SLOPE': dict(
        seed=777085, floor=0.0,
        tier='T1', gen=gen_t1_psrand_slope, rungs=None, cumulant_ok=False,
        quad=lambda: quad_psrand(amp_sigma=0.0),
        note='slope-only variant (cheap-only): quadrature closure vs the '
             'M4 §5.2 dispersion scaling'),
    'T1_PSRAND_AMP': dict(
        seed=777102, floor=0.0,
        tier='T1', gen=gen_t1_psrand_amp, rungs=None, cumulant_ok=False,
        quad=lambda: quad_psrand(slope_sigma=0.0),
        note='amplitude-only variant (cheap-only): η_marg = E[σ²]E[σ⁻²] '
             'closure (erratum 1)'),
    'T1_RANDOR': dict(
        seed=777119, floor=0.0,
        tier='T1', gen=gen_t1_randor, rungs=RUNGS_T1, cumulant_ok=False,
        quad=quad_randor,
        note='random-orientation anisotropy (§5.2); quadrature exact over θ'),
    'T2_MEDIAN': dict(
        seed=777136, floor=0.0,
        tier='T2', gen=gen_t2_median, rungs=RUNGS_T2, cumulant_ok=True,
        note='fixed anisotropy + row-median removal; light tails, cumulants OK; '
             'no complete-data bound (structure is processing-induced, not additive)'),
    'T2_CROSS_SYM': dict(
        seed=777153, floor=0.0,
        tier='T2', gen=gen_t2_cross_sym, rungs=RUNGS_T2, cumulant_ok=False,
        bound_bg=bg_iso, components=lambda n, s: _components_cross(n, s, 'symmetric'),
        note='scan crossings, sign-symmetric (trispectrum-led; quadratic-rung '
             'null expected §21.2); t_2.5 marks ⇒ cumulants unreliable'),
    'T2_CROSS_POS': dict(
        seed=777170, floor=0.0,
        tier='T2', gen=gen_t2_cross_pos, rungs=RUNGS_T2, cumulant_ok=False,
        bound_bg=bg_iso, components=lambda n, s: _components_cross(n, s, 'positive'),
        note='scan crossings, positive-definite (bispectrum-led legacy variant)'),
    'T2_GLITCH': dict(
        seed=777187, floor=0.0,
        tier='T2', gen=gen_t2_glitch, rungs=RUNGS_T2, cumulant_ok=True,
        bound_bg=bg_aniso, components=lambda n, s: _components_glitch(n, s),
        note='sub-threshold glitches (Pareto a=2.5 marks: finite 4th moment, '
             'cumulants heavy but defined); Campbell analytics in B2c'),
    'T2_PCA': dict(
        seed=777204, floor=0.0,
        tier='T2', gen=gen_t2_pca, rungs=RUNGS_T2, cumulant_ok=False,
        bound_bg=bg_iso, components=lambda n, s: _components_pca(n, s),
        note='PCA leaked modes (t_2.5 marks ⇒ cumulants unreliable); '
             'tier factorization surrogate in B2c'),

    # --- white-noise-floor variants (appended 2026-08-31) -------------------------
    # APPEND ONLY.  Seeds were positional before this revision (SEED_BASE+17*index);
    # they are now explicit, but keep new rows at the end anyway so the two agree.
    # Floors are per model, from tools/calibrate_wn_floor.py -- see WN_FLOORS.
    'T1_PSRAND_WN': dict(
        seed=777221, floor=WN_FLOORS['T1_PSRAND_WN'],
        tier='T1', gen=_with_floor(gen_t1_psrand, WN_FLOORS['T1_PSRAND_WN']),
        rungs=RUNGS_T1, cumulant_ok=False,
        quad=lambda: quad_psrand_wn(WN_FLOORS['T1_PSRAND_WN']),
        note='PSRAND + unsmoothed white floor: the band-regularized flagship. '
             'Quadrature is the non-factorized 2-D form (the floor breaks '
             'P = a^2 P_base).  No AMP-only twin exists by design -- see note '
             'on T1_PSRAND_AMP.'),
    'T1_PSRAND_SLOPE_WN': dict(
        seed=777238, floor=WN_FLOORS['T1_PSRAND_SLOPE_WN'],
        tier='T1', gen=_with_floor(gen_t1_psrand_slope,
                                   WN_FLOORS['T1_PSRAND_SLOPE_WN']),
        rungs=None, cumulant_ok=False,
        quad=lambda: quad_psrand_wn(WN_FLOORS['T1_PSRAND_SLOPE_WN'],
                                    amp_sigma=0.0),
        note='slope-only floored variant (cheap-only): the closure test against '
             'the EFFECTIVE lever arm u_eff = (P_red/P_tot) log(k/k_piv), which '
             'only applies once a non-tilting component exists'),
    'T1_RANDOR_WN': dict(
        seed=777255, floor=WN_FLOORS['T1_RANDOR_WN'],
        tier='T1', gen=_with_floor(gen_t1_randor, WN_FLOORS['T1_RANDOR_WN']),
        rungs=RUNGS_T1, cumulant_ok=False,
        quad=lambda: quad_randor_wn(WN_FLOORS['T1_RANDOR_WN']),
        note='random-orientation anisotropy + white floor; the incoherent-channel '
             'exemplar with a regularized compact band'),
    'T2_MEDIAN_WN': dict(
        seed=777272, floor=WN_FLOORS['T2_MEDIAN_WN'],
        tier='T2', gen=_gen_median_floor(WN_FLOORS['T2_MEDIAN_WN']),
        rungs=RUNGS_T2, cumulant_ok=True,
        note='row-median removal + white floor.  NOTE: the floor is applied BEFORE '
             'the median (Phase-A pipeline order); a row median is nonlinear, so '
             'this is not the same model as median-then-floor.  No complete-data '
             'bound (structure is processing-induced, not additive)'),
    'T2_CROSS_SYM_WN': dict(
        seed=777289, floor=WN_FLOORS['T2_CROSS_SYM_WN'],
        tier='T2', gen=_with_floor(gen_t2_cross_sym, WN_FLOORS['T2_CROSS_SYM_WN']),
        rungs=RUNGS_T2, cumulant_ok=False,
        bound_bg=bg_with_floor(bg_iso, WN_FLOORS['T2_CROSS_SYM_WN']),
        components=_components_with_floor(
            lambda n, s: _components_cross(n, s, 'symmetric'),
            WN_FLOORS['T2_CROSS_SYM_WN']),
        note='scan crossings (sign-symmetric) + white floor'),
    'T2_CROSS_POS_WN': dict(
        seed=777306, floor=WN_FLOORS['T2_CROSS_POS_WN'],
        tier='T2', gen=_with_floor(gen_t2_cross_pos, WN_FLOORS['T2_CROSS_POS_WN']),
        rungs=RUNGS_T2, cumulant_ok=False,
        bound_bg=bg_with_floor(bg_iso, WN_FLOORS['T2_CROSS_POS_WN']),
        components=_components_with_floor(
            lambda n, s: _components_cross(n, s, 'positive'),
            WN_FLOORS['T2_CROSS_POS_WN']),
        note='scan crossings (positive-definite) + white floor'),
    'T2_GLITCH_WN': dict(
        seed=777323, floor=WN_FLOORS['T2_GLITCH_WN'],
        tier='T2', gen=_with_floor(gen_t2_glitch, WN_FLOORS['T2_GLITCH_WN']),
        rungs=RUNGS_T2, cumulant_ok=True,
        bound_bg=bg_with_floor(bg_aniso, WN_FLOORS['T2_GLITCH_WN']),
        components=_components_with_floor(_components_glitch,
                                          WN_FLOORS['T2_GLITCH_WN']),
        note='sub-threshold glitches + white floor.  The floor also lowers the '
             'per-event significance xi, which README_PhaseB2a Sec. 5 traced to '
             'its absence -- recompute f_s and xi_rms before touching amp_scale'),
    'T2_PCA_WN': dict(
        seed=777340, floor=WN_FLOORS['T2_PCA_WN'],
        tier='T2', gen=_with_floor(gen_t2_pca, WN_FLOORS['T2_PCA_WN']),
        rungs=RUNGS_T2, cumulant_ok=False,
        bound_bg=bg_with_floor(bg_iso, WN_FLOORS['T2_PCA_WN']),
        components=_components_with_floor(_components_pca,
                                          WN_FLOORS['T2_PCA_WN']),
        note='PCA leaked modes + white floor; the flagship Tier-2 case'),
    # --- confusion noise (appended 2026-09-03) ------------------------------------
    # APPEND ONLY, as above.  Seeds continue the +17 stride: index 21 -> 777357,
    # index 22 -> 777374, so the explicit seeds and the positional fallback agree.
    'T2_CONFUSION': dict(
        seed=777357, floor=0.0,
        tier='T2', gen=gen_t2_confusion, rungs=RUNGS_T2, cumulant_ok=False,
        components=lambda n, s: _components_confusion(n, s, 0.0),
        note='confusion noise, unclustered Poisson, NO floor.  The compact '
             'template has |tau~|^2/P exactly flat, so the MF has zero spectral '
             'leverage -- but flat is not convergent: the Fisher sum is '
             'mode-count-limited and ends up decided by floating point, not by '
             'physics.  Kept as the diagnostic that establishes this; the '
             'quotable numbers come from T2_CONFUSION_WN.  cumulant_ok is '
             'False, but NOT for the usual reason: the marks are perfectly '
             'well behaved (a dN/dS truncated at S_cut has finite q_3 and q_4 '
             'by construction, unlike the t_2.5 marks of CROSS/PCA).  Method B '
             'fails here because the field is nowhere near weakly '
             'non-Gaussian -- pixel skewness 2.13, excess kurtosis 9.14, and '
             'the measured ||k4(t^)||^2 is O(10^4) where the perturbative '
             'expansion needs <<1 -- and because the estimator subtracts O(M) '
             'disconnected pieces at M = 16384 (memo v2 Sec. 17).  Measured '
             'eta_pert came out -12052 +/- 8719: consistent with anything.'),
    'T2_CONFUSION_WN': dict(
        seed=777374, floor=CONFUSION_FLOOR,
        tier='T2', gen=gen_t2_confusion_wn, rungs=RUNGS_T2, cumulant_ok=False,
        bound_bg=bg_confusion(CONFUSION_FLOOR),
        components=lambda n, s: _components_confusion(n, s, CONFUSION_FLOOR),
        note='confusion + unsmoothed white floor at f_N = 0.5 (the production '
             'row).  Stationary, additive, no Tier-1 confound, both templates '
             'band-interior, bound stable: the best-conditioned Tier-2 case in '
             'the taxonomy.  Complete-data bounds 41.7 (compact) / 201.9 '
             '(extended) -- by far the largest in the campaign, and they are the '
             'formal ceiling for perfect cataloguing of every source to S_min.  '
             'cumulant_ok False for the same reason as the floorless row: the '
             'marks are fine, the field is simply far outside the perturbative '
             'regime and Method B is variance-limited at this pixel count.'),
    'T2_CONFUSION_ATM': dict(
        seed=777391, floor=0.0,
        tier='T2', gen=_gen_confusion_atm, rungs=RUNGS_T2, cumulant_ok=False,
        bound_bg=bg_confusion_atm(),
        components=_components_confusion_atm,
        note='confusion + a BEAM-CORRELATED Gaussian companion at rho = 0.25 '
             '(the same 0.5 sigma_c of Gaussian noise as T2_CONFUSION_WN, '
             'differing only in whether it shares the beam).  The exactly '
             'solvable reference: whitening returns i.i.d. pixels, so eta is '
             'template-independent and equal to Var[Y] I(Y) for Y = D + W, '
             '= 3.569 at this configuration, with a complete-data bound of '
             'exactly 5.  A NULL TEST, not a production row: the band is not '
             'regularised (P is still ~|B|^2), and the gap between the measured '
             'and exact eta measures what the rounding side-channel costs.  '
             'floor = 0.0 is correct -- the companion is not an unsmoothed '
             'floor and must not be reported as one.'),
}

# the floorless / floored partition, for --models selection and reporting
BASE_MODELS = [k for k in REGISTRY if not k.endswith('_WN')]
WN_MODELS = [k for k in REGISTRY if k.endswith('_WN')]

SEED_BASE = 777000


def model_seed(name):
    """Per-model ensemble seed.

    Historically this was SEED_BASE + 17*index, i.e. derived from the entry's
    POSITION in REGISTRY -- so inserting a row anywhere but the end silently
    changed the seed, and therefore the whole ensemble, of every model after it
    while the stored results still looked valid.  Every entry now carries an
    explicit ``seed``; the positional formula survives only as a fallback, and
    tests/regression_phase_a.py asserts the two agree for the original models.
    """
    entry = REGISTRY[name]
    if entry.get('seed') is not None:
        return int(entry['seed'])
    return SEED_BASE + 17 * list(REGISTRY).index(name)


# models with a variational card (rungs not None): the M3 campaign set
VARIATIONAL_MODELS = [k for k, v in REGISTRY.items() if v['rungs']]
