"""
quadrature.py — Method D: exact Tier-1 latent quadrature (memo §19)
===================================================================

For conditionally-Gaussian covariance mixtures with a low-dimensional
latent α — per-image PSRAND (α = slope and/or amplitude) and random
orientation (α = angle) — the marginal score is exact:

    p(n) = Σ_α w_α N(0, N_α)   ⇒   s(n) = E[N_α⁻¹ | n] · n,

so with per-image posterior weights W_iα ∝ w_α exp(logL_iα),

    logL_iα = −½ Σ_k ( |ñ_i|²/P_α + ln P_α ),
    τ†s(n_i) = Σ_α W_iα Σ_k Re(τ̃* ñ_i)/P_α ,
    η_marg   = σ_MF² · E_i[ (τ†s(n_i))² ],

with σ_MF² the MF variance under the MARGINAL spectrum P̄ = Σ_α w_α P_α
(the true marginal covariance for a mixture of zero-mean stationary
GRFs).  The full-k sums are exact for real fields (Hermitian pairs carry
half the power each; −½Σ_all reproduces the true Gaussian log-likelihood
up to α-independent constants).

A subtlety worth stating (it surfaced in the Phase-B1 null tests): for
AMPLITUDE-ONLY randomization the η memo §21.5 expected η_marg = 1 "by
scale-freeness".  The scale-freeness theorem (M4 §5.2) is about the
ensemble-MF-vs-oracle-MF gap — the MF estimator is invariant under
N → cN, so that gap is exactly zero.  The marginal-CRLB ceiling computed
here is a different object: with a near-perfectly-inferable scale latent
it converges to η_marg → E[σ²]·E[σ⁻²] ≥ 1 (Cauchy–Schwarz), the
inverse-variance-weighting headroom of a MARGINALLY-unbiased (but
conditionally-biased-given-σ) estimator, attained near A ≈ 0 and diluted
at larger A.  The quadrature therefore returns > 1 for amplitude-only
mixtures — see the null-test driver for the numerical confrontation and
README_PhaseB1 for the discussion.
"""

import numpy as np


def eta_marg_quadrature(images, P_grid, weights, tau, P_marg=None,
                        k_mask=None, n_boot=2000, seed=0,
                        return_posterior=False, chunk=1000):
    """Exact marginal-score η for a covariance-mixture ensemble.

    Parameters
    ----------
    images  : (B, H, W) noise-only maps (use SPLIT_EVAL; the method has no
              fitting step, but keep the split discipline for reporting)
    P_grid  : (n_alpha, H, W) spectra on the latent grid, unnormalized-FFT
              convention (build from noise_lib.spectra with the generative
              parameters; grid must cover the latent prior's support)
    weights : (n_alpha,) prior weights w_α (normalized internally)
    tau     : (H, W) template
    P_marg  : (H, W) marginal spectrum for σ_MF; default Σ_α w_α P_α
    k_mask  : optional boolean (H, W) mask of modes to INCLUDE.  Required
              whenever P_grid contains strong analytic suppressions (e.g.
              a beam factor B²(k) ~ e^{-40} at high k): the maps' FFT
              values there are floating-point rounding noise, and dividing
              it by the analytically tiny P produces garbage (caught by
              null test N5).  Estimated-from-data spectra do not have this
              problem.  Default: all modes.
    return_posterior : also return the (B, n_alpha) posterior weights
                       (diagnostics: latent inferability)

    Returns
    -------
    eta, boot_err [, W_post]
    """
    images = np.asarray(images)
    P_grid = np.asarray(P_grid)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    if P_marg is None:
        P_marg = np.tensordot(w, P_grid, axes=(0, 0))
    if k_mask is None:
        k_mask = np.ones(P_marg.shape, dtype=bool)
    msk = k_mask.astype(float)
    tau_F = np.fft.fft2(tau)
    sigma_mf2 = 1.0 / np.sum(msk * np.abs(tau_F)**2 / P_marg)

    # log-likelihood on the grid; constant ln-det per alpha.  Images are
    # independent, so the computation is chunked (memory: the full-ensemble
    # FFT stack at high-precision ensemble sizes would be several GB).
    logdet = 0.5 * (msk * np.log(P_grid)).sum(axis=(-2, -1))     # (n_alpha,)
    contribs, posts = [], []
    for i0 in range(0, len(images), chunk):
        F = np.fft.fft2(images[i0:i0 + chunk], axes=(-2, -1))    # (b, H, W)
        absF2 = np.abs(F)**2
        chi2 = 0.5 * np.tensordot(absF2, msk / P_grid, axes=([1, 2], [1, 2]))
        logL = -(chi2 + logdet[None, :]) + np.log(w)[None, :]
        logL -= logL.max(axis=1, keepdims=True)
        W_post = np.exp(logL)
        W_post /= W_post.sum(axis=1, keepdims=True)

        # per-alpha matched-filter forms: mf_ialpha = Σ_k Re(τ̃* ñ_i)/P_α
        mf = np.tensordot(np.conj(tau_F)[None, :, :] * F,
                          msk / P_grid, axes=([1, 2], [1, 2])).real
        score_proj = (W_post * mf).sum(axis=1)                   # (b,)
        contribs.append(sigma_mf2 * score_proj**2)
        if return_posterior:
            posts.append(W_post)
    contrib = np.concatenate(contribs)
    W_post = np.concatenate(posts) if return_posterior else None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(contrib), size=(n_boot, len(contrib)))
    boots = contrib[idx].mean(axis=1)
    out = (float(contrib.mean()), float(boots.std()))
    return out + (W_post,) if return_posterior else out


def truncnorm_grid(mean, sigma, lower, n=41, n_sigma=4.0):
    """Latent grid + weights for the clipped-normal priors of noise_lib.psrand.

    noise_lib draws α = max(lower, N(mean, sigma)) — a point mass at
    `lower` plus a truncated normal above it.  Returns (grid, weights)
    with the point mass folded into the first node.  sigma = 0 returns the
    degenerate single-node grid.
    """
    if sigma == 0:
        return np.array([mean]), np.array([1.0])
    lo = max(lower, mean - n_sigma * sigma)
    hi = mean + n_sigma * sigma
    grid = np.linspace(lo, hi, n)
    z = (grid - mean) / sigma
    wts = np.exp(-0.5 * z**2)
    # clipped mass below `lower` sits at the boundary node
    from math import erf, sqrt
    p_below = 0.5 * (1.0 + erf((lower - mean) / (sqrt(2.0) * sigma)))
    wts = wts / wts.sum() * (1.0 - p_below)
    wts[0] += p_below
    return grid, wts


def eta_marg_quadrature_scaled(images, base_grid, base_weights,
                               amp_grid, amp_weights, tau, k_mask=None,
                               n_boot=2000, seed=0, chunk=400):
    """Exact quadrature for mixtures with an AMPLITUDE latent: P = a²·P_base(s).

    For amplitude scaling the per-image likelihood factorizes,

        chi2(i; s, a) = C_i(s) / a²,     C_i(s) = ½ Σ_k msk |ñ_i|²/P_base(s),
        logdet(s, a)  = ½ (M_msk ln a² + D(s)),
        mf(i; s, a)   = mf_i(s) / a²,

    so the full (s, a) grid costs only the 1-D sweep over the shape latent
    s — the amplitude grid can be made arbitrarily fine for free.  This
    removes the amplitude-quantization bias of the generic grid (the
    posterior is far sharper than any affordable 2-D grid; found in the
    B2a validation, where a 41-node amplitude grid overshot the
    complete-data limit E[a²]E[a⁻²]).

    base_grid/base_weights: (n_s, H, W) spectra at UNIT amplitude + prior;
    amp_grid/amp_weights: amplitude prior nodes (use 200+ nodes);
    marginal spectrum for σ_MF: Σ_s w_s P_base(s) · Σ_a w_a a².
    """
    images = np.asarray(images)
    base_grid = np.asarray(base_grid)
    ws = np.asarray(base_weights, float); ws /= ws.sum()
    a = np.asarray(amp_grid, float)
    wa = np.asarray(amp_weights, float); wa /= wa.sum()

    if k_mask is None:
        k_mask = np.ones(base_grid.shape[-2:], dtype=bool)
    msk = k_mask.astype(float)
    M_msk = float(msk.sum())

    P_marg = np.tensordot(ws, base_grid, axes=(0, 0)) * np.sum(wa * a**2)
    tau_F = np.fft.fft2(tau)
    sigma_mf2 = 1.0 / np.sum(msk * np.abs(tau_F)**2 / P_marg)

    D = (msk * np.log(base_grid)).sum(axis=(-2, -1))          # (n_s,)
    log_prior = np.log(ws)[:, None] + np.log(wa)[None, :]      # (n_s, n_a)
    ldet = 0.5 * (D[:, None] + M_msk * np.log(a**2)[None, :])  # (n_s, n_a)

    contribs = []
    for i0 in range(0, len(images), chunk):
        F = np.fft.fft2(images[i0:i0 + chunk], axes=(-2, -1))
        absF2 = np.abs(F)**2
        C = 0.5 * np.tensordot(absF2, msk / base_grid,
                               axes=([1, 2], [1, 2]))          # (B, n_s)
        mf_s = np.tensordot(np.conj(tau_F)[None] * F, msk / base_grid,
                            axes=([1, 2], [1, 2])).real        # (B, n_s)
        # logL[i, s, a] = -C[i,s]/a² - ldet[s,a] + log_prior[s,a]
        logL = (-C[:, :, None] / (a**2)[None, None, :]
                - ldet[None] + log_prior[None])
        logL -= logL.max(axis=(1, 2), keepdims=True)
        W = np.exp(logL)
        W /= W.sum(axis=(1, 2), keepdims=True)
        score = (W * (mf_s[:, :, None] / (a**2)[None, None, :])).sum(axis=(1, 2))
        contribs.append(sigma_mf2 * score**2)
    contrib = np.concatenate(contribs)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(contrib), size=(n_boot, len(contrib)))
    return float(contrib.mean()), float(contrib[idx].mean(axis=1).std())


def eta_marg_quadrature_floor(images, base_grid, base_weights, amp_grid,
                              amp_weights, tau, P_floor, k_mask=None,
                              n_boot=2000, seed=0, chunk=400):
    """Exact quadrature for P = a²·P_base(s) + P_floor  (ADDITIVE white floor).

    An unsmoothed detector floor breaks the factorization that
    ``eta_marg_quadrature_scaled`` exploits: chi2 no longer scales as 1/a² and
    the log-determinant no longer separates, so the (s, a) grid has to be swept
    explicitly.  Materializing it is unnecessary, though — for each shape node
    the whole amplitude grid is one matmul — so this routine keeps the VALIDATED
    grid resolution (n_a ~ 200; a 41-node amplitude grid was shown in the B2a
    validation to quantize the very sharp amplitude posterior) at a memory cost
    of one (n_a, H, W) block instead of (n_s·n_a, H, W).

    Cost: n_s BLAS calls of (B, M)·(M, n_a) per image chunk.

    base_grid/base_weights : (n_s, H, W) spectra at UNIT amplitude + prior
    amp_grid/amp_weights   : amplitude prior nodes (use ~200)
    P_floor                : (H, W) or scalar white PSD, unnormalized-FFT
                             convention (n_pix·sigma_w²)
    """
    images = np.asarray(images)
    base_grid = np.asarray(base_grid, dtype=np.float64)
    ws = np.asarray(base_weights, float); ws /= ws.sum()
    a = np.asarray(amp_grid, float)
    wa = np.asarray(amp_weights, float); wa /= wa.sum()
    n_s, H, W = base_grid.shape
    n_a = len(a)

    P_floor = np.broadcast_to(np.asarray(P_floor, dtype=np.float64), (H, W))
    if k_mask is None:
        k_mask = np.ones((H, W), dtype=bool)
    msk = k_mask.astype(np.float64)

    # marginal spectrum for sigma_MF: E[a²]·E_s[P_base] + P_floor
    P_marg = np.tensordot(ws, base_grid, axes=(0, 0)) * np.sum(wa * a**2) + P_floor
    tau_F = np.fft.fft2(tau)
    sigma_mf2 = 1.0 / np.sum(msk * np.abs(tau_F)**2 / P_marg)

    log_prior = np.log(ws)[:, None] + np.log(wa)[None, :]          # (n_s, n_a)

    # log-determinant term: independent of the images, so precompute once
    ldet = np.empty((n_s, n_a))
    for si in range(n_s):
        Pa = a[:, None, None]**2 * base_grid[si] + P_floor         # (n_a, H, W)
        ldet[si] = 0.5 * (np.log(Pa) * msk).sum(axis=(-2, -1))

    tau_c = np.conj(tau_F)
    contribs = []
    for i0 in range(0, len(images), chunk):
        F = np.fft.fft2(images[i0:i0 + chunk], axes=(-2, -1))
        B = F.shape[0]
        absF2 = (np.abs(F)**2).reshape(B, -1)
        cross = (tau_c[None] * F).real.reshape(B, -1)
        logL = np.empty((B, n_s, n_a))
        mf = np.empty((B, n_s, n_a))
        for si in range(n_s):
            Pa = a[:, None, None]**2 * base_grid[si] + P_floor
            invP = (msk / Pa).reshape(n_a, -1)                     # (n_a, M)
            logL[:, si, :] = -0.5 * (absF2 @ invP.T) - ldet[si][None] \
                             + log_prior[si][None]
            mf[:, si, :] = cross @ invP.T
        logL -= logL.max(axis=(1, 2), keepdims=True)
        Wp = np.exp(logL)
        Wp /= Wp.sum(axis=(1, 2), keepdims=True)
        score = (Wp * mf).sum(axis=(1, 2))
        contribs.append(sigma_mf2 * score**2)
    contrib = np.concatenate(contribs)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(contrib), size=(n_boot, len(contrib)))
    return float(contrib.mean()), float(contrib[idx].mean(axis=1).std())
