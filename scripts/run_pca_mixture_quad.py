#!/usr/bin/env python
"""
run_pca_mixture_quad.py — exact η for the PCA leaked-mode model as a Gaussian scale mixture
==========================================================================================

Why this exists (2026-09-03).  Reading noise_lib.artifacts.pca_leaked_modes shows
that every leaked mode is a FRESH Gaussian random field with one fixed anisotropic
spectrum (Gaussian filter, correlation lengths 20 px along scan, 40 px across),
rescaled to unit pixel rms and multiplied by a heavy-tailed amplitude
a = leak_efficiency · A_iso · t_ν  (ν = 2.5);  the number of modes is Poisson(3).
Conditioned on the amplitudes the artifact is Gaussian with covariance
(Σ_i a_i²)·C_mode, so the whole model is a ONE-PARAMETER Gaussian scale mixture:

    P_c(k) = P_red(k)·B²(k) + c·P_unit(k) [+ P_w],      c = Σ_i a_i² ≥ 0,

with P_unit the unit-pixel-variance spectrum of the mode filter.  That is a
Tier-1 (conditionally Gaussian) model by construction — the "tier-boundary case"
of the campaign was never a boundary case — and its exact ceiling follows from
the same latent quadrature used for PSRAND/RANDOR, on a 1-D grid in c.

(The only approximation: each mode is normalised by its own sample rms, i.e. the
field is projected onto a sphere of ~130 effective dof; a single linear functional
of such a vector is Gaussian to O(1/N_eff) — negligible here.)

What is computed, for both templates and both configurations
(floorless T2_PCA, seed 777204; floored T2_PCA_WN, σ_w = 0.10, seed 777340):
  * the prior on c by Monte Carlo (10^6 draws of the compound Poisson-t law),
    binned on a log grid plus the c = 0 atom (Poisson zero: e^-3 ≈ 5 %);
  * a spectrum check: the model's marginal spectrum E_c[P_c] against the
    campaign's data-estimated P̂ (radial ratio) and the stored σ_MF;
  * the exact marginal-score η on the campaign EVAL split (n = 1500, paired
    with every stored cheap number) AND on the full 6000-map ensemble (the
    quadrature has no fitting step, so this is legitimate and 4× more precise);
  * the well-inferable limit η_wi = σ²_MF·E_c[I_c], the rank-1 IVW gain
    E[c']E[1/c'], and the two-channel decomposition + Method-E curve, i.e. the
    full Tier-1 treatment PSRAND/RANDOR already have;
  * the per-image posterior sd(log c | n) — how well the latent is inferred.

How to read it against the ladder:
  * floored: reweight 1.655 vs η_exact → the attained fraction (Pass-A style);
  * floorless: the CNN's VAL 2.3–2.7 vs η_exact.  Below η_exact ⇒ genuine Tier-1
    headroom the band-diagonal reweight basis cannot express;  above ⇒ impossible
    for a physical channel ⇒ the float32 rounding side channel.

Output: eta_results/pca_mixture_quadrature.json.   Runtime ~5–10 min on the M3
(dominated by generating 2 × 6000 maps).   Usage: python scripts/run_pca_mixture_quad.py
"""

#%% 1. setup
import json
import sys
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
for _p in (_here, *_here.parents):
    if (_p / 'eta_pipeline').is_dir():
        sys.path.insert(0, str(_p))
        break

import noise_lib as nl                                   # noqa: E402
import eta_pipeline as ep                                # noqa: E402
from eta_pipeline import models as em                    # noqa: E402
from eta_pipeline import channels as ch                  # noqa: E402
from eta_pipeline.quadrature import eta_marg_quadrature  # noqa: E402

OUT = _p / 'results' / 'eta_results'
SPLIT_SEED = 12345
N_ENS = 6000
SHAPE = em.SHAPE
NPIX = SHAPE[0] * SHAPE[1]
B2 = em.beam2()
K_MASK = B2 > 1e-8
TEMPLATES = em.make_templates()
PA = em.PCA_ARGS


#%% 2. the model's spectra and the latent prior
def mode_unit_spectrum():
    """Unnormalised-FFT spectrum of ONE leaked mode at unit pixel variance.

    pca_leaked_modes filters unit white noise with the amplitude filter
    F(k) = exp(-½[(k_x L_s)² + (k_y L_c)²]) and rescales the result to unit rms,
    so its spectrum is  P_unit = NPIX² F² / Σ_k F²  (pixel variance
    = Σ_k P / NPIX² = 1).
    """
    kx = np.fft.fftfreq(SHAPE[1]).reshape(1, -1)
    ky = np.fft.fftfreq(SHAPE[0]).reshape(-1, 1)
    F2 = np.exp(-((kx * PA['mode_corr_scan'])**2 + (ky * PA['mode_corr_cross'])**2))
    return NPIX**2 * F2 / F2.sum()


def c_prior(n_mc=1_000_000, n_grid=320, seed=20260903):
    """Monte-Carlo prior of c = Σ_{i≤N} a_i², N ~ Poisson(λ), a = ε A t_ν.

    Returns (c_grid, weights, stats).  Node 0 is the c = 0 atom (N = 0).  The
    remaining nodes are a log grid between the 1e-5 and 1-1e-5 quantiles of the
    positive draws; the top bin absorbs the tail (weights sum to one).
    """
    rng = np.random.default_rng(seed)
    N = rng.poisson(PA['n_leaked_mean'], n_mc)
    scale = PA['leak_efficiency'] * PA['A_iso']
    tot = N.sum()
    a2 = (scale * rng.standard_t(PA['leak_df'], tot))**2
    c = np.zeros(n_mc)
    np.add.at(c, np.repeat(np.arange(n_mc), N), a2)
    pos = c[c > 0]
    lo, hi = np.quantile(pos, [1e-5, 1 - 1e-5])
    edges = np.geomspace(lo, hi, n_grid + 1)
    hist, _ = np.histogram(np.clip(pos, lo, hi * 0.999999), bins=edges)
    grid = np.sqrt(edges[1:] * edges[:-1])
    w = hist.astype(float)
    p0 = float(np.mean(c == 0))
    c_grid = np.concatenate([[0.0], grid])
    weights = np.concatenate([[p0 * n_mc], w])
    weights /= weights.sum()
    stats = {'p_zero': p0, 'mean_c': float(c.mean()), 'median_c': float(np.median(c)),
             'q90_c': float(np.quantile(c, 0.9)), 'q99_c': float(np.quantile(c, 0.99)),
             'grid_lo': float(lo), 'grid_hi': float(hi), 'n_grid': int(len(c_grid)),
             'mean_log_c_pos': float(np.mean(np.log(pos))), 'sd_log_c_pos': float(np.std(np.log(pos)))}
    return c_grid, weights, stats


P_RED = nl.psd_isotropic_powerlaw(SHAPE, *em.P_ISO_ARGS) * B2
P_UNIT = mode_unit_spectrum()
t0 = time.time()
C_GRID, C_W, C_STATS = c_prior()
print(f"latent prior: P(c=0) = {C_STATS['p_zero']:.4f}, E[c] = {C_STATS['mean_c']:.3f}, "
      f"median {C_STATS['median_c']:.3f}, q99 {C_STATS['q99_c']:.1f}, "
      f"{C_STATS['n_grid']} nodes ({time.time() - t0:.1f} s)")


def spectrum_grid(sigma_w):
    Pw = em.p_white(sigma_w) if sigma_w else 0.0
    return P_RED[None] + C_GRID[:, None, None] * P_UNIT[None] + Pw


def radial(P2d, nbins=24):
    fy = np.fft.fftfreq(SHAPE[0]).reshape(-1, 1); fx = np.fft.fftfreq(SHAPE[1]).reshape(1, -1)
    k = np.sqrt(fx**2 + fy**2).ravel()
    edges = np.linspace(0, 0.5, nbins + 1)
    idx = np.clip(np.digitize(k, edges) - 1, 0, nbins - 1)
    prof = np.bincount(idx, weights=P2d.ravel(), minlength=nbins) / np.maximum(np.bincount(idx, minlength=nbins), 1)
    return 0.5 * (edges[1:] + edges[:-1]), prof


#%% 3. exact quadrature, both configurations, both templates
results = {'latent_prior': C_STATS, 'pca_args': PA, 'k_mask': 'B^2 > 1e-8'}
for label, name in (('floorless', 'T2_PCA'), ('floored', 'T2_PCA_WN')):
    sigma_w = float(em.REGISTRY[name].get('floor', 0.0))
    t0 = time.time()
    maps, _ = em.REGISTRY[name]['gen'](N_ENS, em.model_seed(name))
    sp = ep.make_splits(N_ENS, seed=SPLIT_SEED)
    P_hat = ep.estimate_psd2d(maps[sp['psd']])
    print(f'\n=== {label} ({name}, sigma_w = {sigma_w}): generated {N_ENS} maps in {time.time() - t0:.0f} s')
    P_grid = spectrum_grid(sigma_w)
    P_bar = np.tensordot(C_W, P_grid, axes=(0, 0))
    # spectrum check: model marginal vs data P-hat (radial ratio inside the mask)
    kc, pm = radial(P_bar * K_MASK); _, ph = radial(P_hat * K_MASK)
    ratio = (pm / np.maximum(ph, 1e-300))
    stored = json.loads((OUT / f'{name}__extended.json').read_text())
    entry = {'model': name, 'sigma_w': sigma_w, 'n_ens': N_ENS,
             'spectrum_check': {'k': kc.tolist(), 'model_over_data': ratio.tolist(),
                                'note': 'E_c[P_c] / P_hat, radial, masked; should be ~1 where the mask is on'}}
    print('  model/data spectrum ratio (radial):', ' '.join(f'{r:.2f}' for r in ratio[:12]), '...')
    for tname, tau in TEMPLATES.items():
        d = {}
        # sigma_MF from the model's marginal spectrum vs the campaign's stored value
        _, s_model = ep.whitened_template(tau, P_bar)
        d['sigma_mf_model'] = float(s_model)
        d['sigma_mf_stored_pred'] = json.loads((OUT / f'{name}__{tname}.json').read_text())['sigma_mf']['pred']
        # exact quadrature: EVAL split (paired with the cheap layer) and full ensemble
        t1 = time.time()
        eta_e, err_e, W_post = eta_marg_quadrature(maps[sp['eval']], P_grid, C_W, tau,
                                                   k_mask=K_MASK, return_posterior=True)
        eta_f, err_f = eta_marg_quadrature(maps, P_grid, C_W, tau, k_mask=K_MASK)
        d['eta_exact_eval1500'] = {'eta': eta_e, 'err': err_e}
        d['eta_exact_full6000'] = {'eta': eta_f, 'err': err_f}
        # posterior width on log c (positive nodes only) per image
        lc = np.log(np.where(C_GRID > 0, C_GRID, np.nan))
        Wp = W_post[:, 1:]; Wp = Wp / np.maximum(Wp.sum(axis=1, keepdims=True), 1e-300)
        m1 = Wp @ lc[1:]; m2 = Wp @ lc[1:]**2
        d['post_sd_log_c_mean'] = float(np.nanmean(np.sqrt(np.maximum(m2 - m1**2, 0))))
        d['prior_sd_log_c'] = C_STATS['sd_log_c_pos']
        # well-inferable limit, two channels, Method-E
        tau_F2 = np.abs(np.fft.fft2(tau))**2 * K_MASK
        I_c = (tau_F2[None] / P_grid).sum(axis=(-2, -1))
        d['eta_well_inferable'] = float(np.sum(C_W * I_c) * s_model**2)
        dec = ch.two_channel_decomposition(P_grid, C_W, tau, K_MASK)
        c_ratio, wz = dec.pop('c_grid'), dec.pop('weights')
        a_over_sig = np.concatenate([[0.0], np.geomspace(0.05, 8.0, 40)])
        v_curve, eta_eff = ch.method_e_curve(c_ratio, wz, a_over_sig)
        d['two_channel'] = dec
        d['method_e'] = {'a_over_sigma': a_over_sig.tolist(), 'eta_eff': eta_eff.tolist()}
        nrm = ch.sigma_norm_analytic(P_grid, C_W, tau, K_MASK)
        d['sigma_norm'] = nrm
        # ladder values on record, for the comparison
        var2 = json.loads((OUT / f'{name}__{tname}.json').read_text()).get('variational2', {}).get('rungs', {})
        d['ladder_on_record'] = {k: v['eta'] for k, v in var2.items()}
        d['runtime_s'] = round(time.time() - t1, 1)
        entry[tname] = d
        print(f"  {tname:9s} sigma_MF model {s_model:.3f} (stored {d['sigma_mf_stored_pred']:.3f}) | "
              f"eta exact EVAL {eta_e:.3f} ± {err_e:.3f}, full {eta_f:.3f} ± {err_f:.3f} | "
              f"well-inf {d['eta_well_inferable']:.3f}  IVW {dec['eta_ivw']:.3f}  "
              f"coh {dec['coh_var']:.3f} inc {dec['inc_var']:.3f} | sd(log c|n) {d['post_sd_log_c_mean']:.3f} "
              f"(prior {C_STATS['sd_log_c_pos']:.3f}) | ladder {d['ladder_on_record']}")
    results[label] = entry
    del maps

(OUT / 'pca_mixture_quadrature.json').write_text(json.dumps(results, indent=1, default=float))
print(f"\nwritten {OUT / 'pca_mixture_quadrature.json'}")
