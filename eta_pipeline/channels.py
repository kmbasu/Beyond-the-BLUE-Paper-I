"""
channels.py — two-channel Tier-1 analysis, σ_norm baselines, Method-E curve
===========================================================================

Implements the v5-memo (CNN_vs_MF_theory_v5.md) analysis layer for the
conditionally-Gaussian (Tier-1) models, on top of the latent spectra grids
already used by the exact quadrature (Method D).

Per-image log-spectrum fluctuation, relative to the marginal spectrum:

    δ_z(k) = log P_z(k) − log P̄(k),        P̄ = Σ_z w_z P_z ,

split relative to a fixed template's Fisher weights f_k ∝ |τ̃|²/P̄ into a
band-mean (COHERENT / band-amplitude) part and a within-band residual
(INCOHERENT / shape) part (v5 §5.2.1):

    η − 1  ≈  Var_z[⟨δ⟩_f]  +  E_z[Var_f(δ)]          (second order).

The exactly-computable per-latent variance ratio of the ensemble filter,

    c(z) = ⟨P_z/P̄⟩_f = σ²_MF(ens filter | z) / σ²_MF(ens filter, marginal),

satisfies E_z[c] = 1 identically and drives everything nonperturbative:

* the inverse-variance-weighting (coherent) gain E[c]·E[c⁻¹] = E[c⁻¹]
  (v5 §5.2.5 at A = 0; erratum-1 channel);
* the Method-E attainability curve — the best marginally-unbiased rescaled
  MF, Â' = b(z)·Â_MF with E[b] = 1, has (minimizing over b at each A)

      V_E(A) = 1 / E_z[ 1/(σ²_MF·c(z) + A²) ]  −  A²,

  which starts at σ²_MF/E[c⁻¹] (the full coherent gain) and relaxes to
  σ²_MF as A → ∞.  Every point is attainable by an estimator that is
  unbiased at every A (given a well-inferable latent), so the curve is the
  honest "what a CNN can actually deliver vs A" companion to the flat
  σ²_MF/η ceiling (v5 §5.2.5 "new deliverable");
* the σ_norm baselines (v5 §4.3.1): a scalar weight u(z) measurable per
  image (map RMS, or the template's own band power) defines the rescaled
  estimator b ∝ 1/u with E[b] = 1 and realized variance ratio E[b²c].
  With u = c this is the full coherent gain; with u = total map power it
  is what a hit/weight map buys — v5's caution that "a scalar weight
  cannot equalise a spectrum" is then a computable number, not a remark.

All functions take (P_grid, weights) — the same latent spectra the
quadrature builders construct — plus a template and the standard k-mask.

NOTE on inferability: these formulas treat the latent as read off the
image exactly.  For our T1 models the Method-D posteriors are extremely
sharp (thousands of informative modes), so the idealization is at the
percent level; where it matters the quadrature η is the exact statement.
"""

import numpy as np


# ----------------------------------------------------------------------------------
# Core weights
# ----------------------------------------------------------------------------------

def fisher_weights(tau, P, k_mask=None):
    """Normalized Fisher weights f_k = (|τ̃|²/P) / Σ(|τ̃|²/P) on the FFT grid."""
    P = np.asarray(P)
    tau_F2 = np.abs(np.fft.fft2(tau))**2
    f = tau_F2 / P
    if k_mask is not None:
        f = f * k_mask
    return f / f.sum()


def marginal_spectrum(P_grid, weights):
    w = np.asarray(weights, float)
    w = w / w.sum()
    return np.tensordot(w, np.asarray(P_grid), axes=(0, 0)), w


# ----------------------------------------------------------------------------------
# Two-channel decomposition (v5 §5.2.1–5.2.3)
# ----------------------------------------------------------------------------------

def two_channel_decomposition(P_grid, weights, tau, k_mask=None):
    """Coherent/incoherent split of the Tier-1 ceiling for one template.

    Returns a dict with:
      coh_var    Var_z[⟨δ⟩_f]           — coherent (band-amplitude) term
      inc_var    E_z[Var_f(δ)]          — incoherent (shape) term
      eta_2nd    1 + coh_var + inc_var  — second-order forecast
      eta_ivw    E_z[c]·E_z[c⁻¹]        — nonperturbative coherent (IVW) gain
      sd_band    sd_z[⟨δ⟩_f]            — severity coordinate (log band power)
      rms_shape  E_z[Var_f(δ)]^½        — severity coordinate (in-band shape)
      c_grid     per-latent variance ratios c(z) (for Method-E / σ_norm)
    """
    P_grid = np.asarray(P_grid)
    P_bar, w = marginal_spectrum(P_grid, weights)
    f = fisher_weights(tau, P_bar, k_mask)
    msk = f > 0

    logratio = np.zeros_like(P_grid)
    logratio[:, msk] = np.log(P_grid[:, msk] / P_bar[msk])
    d_mean = (f * logratio).sum(axis=(-2, -1))                    # ⟨δ_z⟩_f
    d_var = (f * logratio**2).sum(axis=(-2, -1)) - d_mean**2      # Var_f(δ_z)

    coh_var = max(0.0, float(np.sum(w * d_mean**2) - np.sum(w * d_mean)**2))
    inc_var = max(0.0, float(np.sum(w * d_var)))

    c = (P_grid[:, msk] / P_bar[msk] * f[msk]).sum(axis=-1)       # c(z); E[c]=1
    eta_ivw = float(np.sum(w * c) * np.sum(w / c))

    return {
        'coh_var': coh_var, 'inc_var': inc_var,
        'eta_2nd': 1.0 + coh_var + inc_var,
        'eta_ivw': eta_ivw,
        'sd_band': float(np.sqrt(coh_var)),
        'rms_shape': float(np.sqrt(inc_var)),
        'c_grid': c, 'weights': w,
    }


# ----------------------------------------------------------------------------------
# Method-E attainability curve (v5 §5.2.5)
# ----------------------------------------------------------------------------------

def method_e_curve(c, w, a_grid):
    """V_E(A)/σ²_MF on a grid of a = A/σ_MF.

    V_E(A) = 1/E[1/(σ²c + A²)] − A²  in units of σ²_MF:
        v(a) = 1/E_z[1/(c + a²)] − a².
    Returns (v, eta_eff) with eta_eff(a) = 1/v(a) — the effective
    advantage still available at amplitude A = a·σ_MF (eta_eff(0) = E[c⁻¹];
    eta_eff → 1 as a → ∞).
    """
    c = np.asarray(c, float)
    w = np.asarray(w, float); w = w / w.sum()
    a2 = np.asarray(a_grid, float)**2
    inv = np.array([np.sum(w / (c + x)) for x in a2])
    v = 1.0 / inv - a2
    return v, 1.0 / v


# ----------------------------------------------------------------------------------
# σ_norm baselines (v5 §4.3.1, §5.2.7 item 3)
# ----------------------------------------------------------------------------------

def sigma_norm_analytic(P_grid, weights, tau, k_mask=None):
    """Variance ratios of scalar-rescaled ensemble MFs, from the latent grid.

    For a measurable scalar level u(z), the marginally-unbiased rescaled
    estimator b(z) = u(z)⁻¹/E[u⁻¹] has variance ratio E[b²c] against the
    ensemble MF.  Computed for:
      'band'  u = c(z)          — the template's own band power (the best any
                                  scalar normalization can do; = 1/E[c⁻¹])
      'rms'   u = total power   — map-variance / hit-map style weight (what a
                                  real weight map measures)
    Returns dict of variance ratios V_b/σ²_MF (≤ 1) and the corresponding
    η re-normalizations η_vs_norm = η_raw · ratio.
    """
    P_grid = np.asarray(P_grid)
    P_bar, w = marginal_spectrum(P_grid, weights)
    f = fisher_weights(tau, P_bar, k_mask)
    msk = f > 0
    c = (P_grid[:, msk] / P_bar[msk] * f[msk]).sum(axis=-1)

    if k_mask is None:
        tot_mask = np.ones(P_bar.shape, bool)
    else:
        tot_mask = np.asarray(k_mask, bool)
    tot = P_grid[:, tot_mask].sum(axis=-1)
    tot = tot / np.sum(w * tot)                                   # E[u]=1

    out = {}
    for key, u in (('band', c), ('rms', tot)):
        b = (1.0 / u) / np.sum(w / u)
        out[f'ratio_{key}'] = float(np.sum(w * b**2 * c))
    out['note'] = ('ratio_* = V(b-rescaled ens MF)/V(ens MF) at A=0; '
                   'multiply a raw eta by ratio_* to re-baseline against '
                   'sigma_norm (band = best scalar; rms = weight-map style)')
    return out


def two_channel_scaled(base_grid, base_weights, amp_grid, amp_weights, tau,
                       k_mask=None):
    """Two-channel decomposition for amplitude-scaled mixtures P = a²·P_base(s)
    WITHOUT forming the (s, a) product grid of spectra (memory: the product
    grid at quadrature resolution would be tens of GB; the scaling latent
    enters ⟨δ⟩_f as an additive log a² and drops out of Var_f(δ) exactly).

    Returns the same dict as two_channel_decomposition, with c_grid/weights
    flattened over the (a, s) product (small vectors, not spectra).
    """
    base_grid = np.asarray(base_grid)
    ws = np.asarray(base_weights, float); ws = ws / ws.sum()
    a = np.asarray(amp_grid, float)
    wa = np.asarray(amp_weights, float); wa = wa / wa.sum()

    Ea2 = float(np.sum(wa * a**2))
    P_bar = np.tensordot(ws, base_grid, axes=(0, 0)) * Ea2
    f = fisher_weights(tau, P_bar, k_mask)
    msk = f > 0

    logratio = np.zeros_like(base_grid)
    logratio[:, msk] = np.log(base_grid[:, msk] / P_bar[msk])
    d_mean_s = (f * logratio).sum(axis=(-2, -1))              # s-part of ⟨δ⟩_f
    d_var_s = (f * logratio**2).sum(axis=(-2, -1)) - d_mean_s**2

    la = np.log(a**2)
    d_mean = la[:, None] + d_mean_s[None, :]                  # (n_a, n_s)
    w2 = wa[:, None] * ws[None, :]
    mu = np.sum(w2 * d_mean)
    coh_var = max(0.0, float(np.sum(w2 * (d_mean - mu)**2)))
    inc_var = max(0.0, float(np.sum(ws * d_var_s)))           # a-independent

    c_s = (base_grid[:, msk] / P_bar[msk] * f[msk]).sum(axis=-1)
    c = (a**2)[:, None] * c_s[None, :]                        # E[c] = 1
    eta_ivw = float(np.sum(w2 * c) * np.sum(w2 / c))

    return {
        'coh_var': coh_var, 'inc_var': inc_var,
        'eta_2nd': 1.0 + coh_var + inc_var,
        'eta_ivw': eta_ivw,
        'sd_band': float(np.sqrt(coh_var)),
        'rms_shape': float(np.sqrt(inc_var)),
        'c_grid': c.ravel(), 'weights': w2.ravel(),
    }


def sigma_norm_scaled(base_grid, base_weights, amp_grid, amp_weights, tau,
                      k_mask=None):
    """σ_norm variance ratios for amplitude-scaled mixtures (see
    sigma_norm_analytic); total power factorizes as a²·tot(s)."""
    base_grid = np.asarray(base_grid)
    ws = np.asarray(base_weights, float); ws = ws / ws.sum()
    a = np.asarray(amp_grid, float)
    wa = np.asarray(amp_weights, float); wa = wa / wa.sum()
    Ea2 = float(np.sum(wa * a**2))
    P_bar = np.tensordot(ws, base_grid, axes=(0, 0)) * Ea2
    f = fisher_weights(tau, P_bar, k_mask)
    msk = f > 0
    if k_mask is None:
        tot_mask = np.ones(P_bar.shape, bool)
    else:
        tot_mask = np.asarray(k_mask, bool)

    c_s = (base_grid[:, msk] / P_bar[msk] * f[msk]).sum(axis=-1)
    t_s = base_grid[:, tot_mask].sum(axis=-1)
    w2 = (wa[:, None] * ws[None, :]).ravel()
    c = ((a**2)[:, None] * c_s[None, :]).ravel()
    tot = ((a**2)[:, None] * t_s[None, :]).ravel()
    tot = tot / np.sum(w2 * tot)

    out = {}
    for key, u in (('band', c), ('rms', tot)):
        b = (1.0 / u) / np.sum(w2 / u)
        out[f'ratio_{key}'] = float(np.sum(w2 * b**2 * c))
    out['note'] = ('ratio_* = V(b-rescaled ens MF)/V(ens MF) at A=0; '
                   'multiply a raw eta by ratio_* to re-baseline against '
                   'sigma_norm (band = best scalar; rms = weight-map style)')
    return out


def sigma_norm_empirical(maps_eval, tau, P_bar, k_mask=None, n_boot=2000,
                         seed=0):
    """Image-based validation of the σ_norm ratios (no latent grid needed).

    y_i    : ensemble-filter MF outputs (noise-only ⇒ zero-mean)
    u_rms  : per-image map variance (weight-map analogue)
    u_band : per-image measured Fisher-band power  Σ f_k |x̃|²_k / P̄_k
    For each u: b_i ∝ 1/u_i (E[b] = 1) and the realized variance ratio
    E[b²y²]/E[y²].  b and y share the image, so the estimate carries an
    O(1/n_modes) inference-coupling bias — quote alongside the analytic
    version, not instead of it.
    """
    maps_eval = np.asarray(maps_eval)
    P_bar = np.asarray(P_bar)
    f = fisher_weights(tau, P_bar, k_mask)
    msk = f > 0
    tau_F = np.fft.fft2(tau)
    inv_norm = np.sum(np.where(msk, np.abs(tau_F)**2 / P_bar, 0.0))

    F = np.fft.fft2(maps_eval, axes=(-2, -1))
    y = (np.conj(tau_F)[None] * F / P_bar)[:, msk].sum(axis=-1).real / inv_norm
    u_rms = maps_eval.var(axis=(-2, -1))
    u_band = (np.abs(F)**2 / P_bar * f)[:, msk].sum(axis=-1)

    rng = np.random.default_rng(seed)
    out = {'sigma_emp2': float(np.mean(y**2))}
    for key, u in (('band', u_band), ('rms', u_rms)):
        b = (1.0 / u) / np.mean(1.0 / u)
        ratio = np.mean(b**2 * y**2) / np.mean(y**2)
        idx = rng.integers(0, len(y), size=(n_boot, len(y)))
        boots = (np.mean(b[idx]**2 * y[idx]**2, axis=1)
                 / np.mean(y[idx]**2, axis=1))
        out[f'ratio_{key}'] = float(ratio)
        out[f'ratio_{key}_err'] = float(boots.std())
    return out
