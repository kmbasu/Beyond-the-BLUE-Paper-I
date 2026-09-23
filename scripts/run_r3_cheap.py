#!/usr/bin/env python
"""
run_r3_cheap.py — the three cheap confirmations of TODO §P2 (Phase-B close-out)
===============================================================================

Runs on the M3 (CPU only, no PyTorch), in parallel with the GPU-cluster ladder array.
Everything here reuses the campaign's own functions (eta_pipeline, noise_lib)
so the numbers are directly comparable with the stored eta_results/ JSONs.
Spyder cell mode: each ``#%%`` cell is independent after cell 1.

Sections
--------
  2. P2.2  LATENT INFERABILITY — is the 18% over-prediction of the closed-form
           tilt ceiling on the floored extended template a posterior-width
           effect?  For the slope-only PSRAND model, floorless
           (T1_PSRAND_SLOPE) and floored (T1_PSRAND_SLOPE_WN), both templates:
             * exact quadrature η on the campaign EVAL split (reproduces the
               stored 9.46 / 7.76 values as a self-check);
             * the exact WELL-INFERABLE limit on the same grid,
               η_wi = σ²_MF(P̄) · E_s[Σ_k |τ̃_k|²/P_s(k)]  (= E_s[I_s]·σ²_MF),
               which needs no pivot or lever-arm approximation;
             * the per-image posterior sd(α | n) from
               eta_marg_quadrature(..., return_posterior=True), and the rms
               error of the posterior mean against the generating slope.
           If the floored posterior is materially wider AND η_meas/η_wi drops
           from ~1 to ~0.8, the residual is latent inferability (Cohen's
           convexity loss), and memo §4.5 can state it.
  3. P2.3  σ_norm BASELINES + METHOD-E on the white-floor Tier-1 rows
           (T1_PSRAND_WN, T1_PSRAND_SLOPE_WN, T1_RANDOR_WN) through the GENERIC
           two_channel_decomposition / sigma_norm_analytic (the *_scaled fast
           paths assume P = a²P_base and are invalid with a floor), plus the
           empirical σ_norm ratios on the T1_PSRAND_WN campaign EVAL split.
           Figure: eta_results/r3_method_e_wn.png (floorless vs floored).
  4. P2.1  MATCHED-SEED DIAGNOSTIC on a heavy-tailed Tier-2 model (T2_PCA):
           four ensembles — base@seed_base, base@seed_WN, WN@seed_base,
           WN@seed_WN — so that the floor effect (paired, same draw) can be
           separated from the draw effect (same config, different seed) in
           σ_MF, the moments and the complete-data bound.

Outputs (eta_results/):  r3_inferability.json, r3_channels_wn.json,
r3_method_e_wn.png, r3_matched_seed.json.  Nothing existing is modified.

Usage (from the repository root):
    python scripts/run_r3_cheap.py                # all sections, ~10-20 min on the M3
    python scripts/run_r3_cheap.py --only 2       # one section (2, 3 or 4)
"""

#%% 1. setup
import argparse
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
from eta_pipeline.quadrature import (truncnorm_grid,     # noqa: E402
                                     eta_marg_quadrature)

OUT = _p / 'results' / 'eta_results'
OUT.mkdir(exist_ok=True)
SPLIT_SEED = 12345
N_ENS = 6000                       # the frozen campaign ensemble size
B2 = em.beam2()
K_MASK = B2 > 1e-8
TEMPLATES = em.make_templates()
SM, SS, AM, AS = em.PSRAND_ARGS    # 3.0, 0.5, 5.0, 0.8

if __name__ == '__main__' and '__file__' in globals():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=int, default=None, choices=[2, 3, 4])
    ONLY = ap.parse_args().only
else:
    ONLY = None


def _want(section):
    return ONLY is None or ONLY == section


def campaign_ensemble(name, n=N_ENS):
    """Regenerate a registry model's frozen campaign ensemble + splits + P̂."""
    t0 = time.time()
    maps, lat = em.REGISTRY[name]['gen'](n, em.model_seed(name))
    sp = ep.make_splits(n, seed=SPLIT_SEED)
    P_hat = ep.estimate_psd2d(maps[sp['psd']])
    print(f'  [{name}] regenerated {n} maps in {time.time() - t0:.0f} s')
    return maps, lat, sp, P_hat


def psrand_wn_grid(sigma_w, slope_sigma, amp_sigma, n_slope=61, n_amp=41):
    """Explicit (s × a) spectrum grid  P = a²·P_base(s) + P_w  and prior weights.

    n_amp = 41 (not the quadrature's 201) is enough for Fisher-band statistics,
    which are smooth in a; the 201-node grid mattered only for the very sharp
    amplitude POSTERIOR inside the quadrature.
    """
    s_grid, s_w = ((np.array([SM]), np.array([1.0])) if slope_sigma == 0 else
                   truncnorm_grid(SM, slope_sigma, 1.0, n=n_slope))
    a_grid, a_w = ((np.array([AM]), np.array([1.0])) if amp_sigma == 0 else
                   truncnorm_grid(AM, amp_sigma, 1e-6, n=n_amp))
    Pw = em.p_white(sigma_w) if sigma_w else 0.0
    base = np.stack([nl.psd_isotropic_powerlaw(em.SHAPE, s, 1.0) * B2
                     for s in s_grid])                              # (n_s, H, W)
    P = (a_grid[None, :, None, None]**2 * base[:, None] + Pw)       # (n_s, n_a, H, W)
    w = (s_w[:, None] * a_w[None, :])
    return P.reshape(-1, *em.SHAPE), w.ravel() / w.sum(), (s_grid, s_w, a_grid, a_w)


def randor_wn_grid(sigma_w, n_theta=180):
    ss_sc, a_sc, ss_iso, a_iso = em.P_ANISO_ARGS
    th = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    Pw = em.p_white(sigma_w) if sigma_w else 0.0
    P = np.stack([nl.psd_two_component(em.SHAPE, ss_sc, a_sc, ss_iso, a_iso,
                                       scan_angle=t) * B2 + Pw for t in th])
    return P, np.full(n_theta, 1.0 / n_theta)


#%% 2. P2.2 — latent inferability (slope-only PSRAND, floorless vs floored)
if _want(2):
    print('=== P2.2 latent inferability: slope-only PSRAND, floorless vs floored ===')
    infer = {}
    for label, name in (('floorless', 'T1_PSRAND_SLOPE'),
                        ('floored', 'T1_PSRAND_SLOPE_WN')):
        sigma_w = em.REGISTRY[name].get('floor', 0.0)
        maps, lat, sp, P_hat = campaign_ensemble(name)
        ev = sp['eval']
        s_grid, s_w = truncnorm_grid(SM, SS, 1.0, n=61)
        Pw = em.p_white(sigma_w) if sigma_w else 0.0
        P_grid = np.stack([AM**2 * nl.psd_isotropic_powerlaw(em.SHAPE, s, 1.0) * B2 + Pw
                           for s in s_grid])
        P_bar = np.tensordot(s_w / s_w.sum(), P_grid, axes=(0, 0))
        # prior moments of the (truncated) slope
        wn = s_w / s_w.sum()
        prior_mean = float(np.sum(wn * s_grid))
        prior_sd = float(np.sqrt(np.sum(wn * s_grid**2) - prior_mean**2))
        infer[label] = {'model': name, 'sigma_w': float(sigma_w),
                        'prior_sd_slope': prior_sd, 'n_eval': int(len(ev))}
        for tname, tau in TEMPLATES.items():
            t0 = time.time()
            eta, err, W_post = eta_marg_quadrature(maps[ev], P_grid, s_w, tau,
                                                   k_mask=K_MASK,
                                                   return_posterior=True)
            # exact well-inferable limit on the same grid: E_s[I_s] · σ²_MF(P̄)
            tau_F2 = np.abs(np.fft.fft2(tau))**2 * K_MASK
            I_s = (tau_F2[None] / P_grid).sum(axis=(-2, -1))       # (n_s,)
            sigma_mf2 = 1.0 / (tau_F2 / P_bar).sum()
            eta_wi = float(np.sum(wn * I_s) * sigma_mf2)
            # posterior statistics
            pm = W_post @ s_grid
            psd = np.sqrt(np.maximum(W_post @ s_grid**2 - pm**2, 0.0))
            truth = lat['slope'][ev]
            rms_err = float(np.sqrt(np.mean((pm - truth)**2)))
            infer[label][tname] = {
                'eta_quadrature': eta, 'err': err,
                'eta_well_inferable': eta_wi,
                'ratio_meas_over_wi': eta / eta_wi,
                'post_sd_mean': float(psd.mean()),
                'post_sd_median': float(np.median(psd)),
                'post_mean_rms_error': rms_err,
                'inferability_index': float(1.0 - (psd.mean() / prior_sd)**2),
                'runtime_s': round(time.time() - t0, 1)}
            print(f"  {label:9s} {tname:9s} eta {eta:7.3f} ± {err:.3f} | "
                  f"well-inferable {eta_wi:7.3f}  ratio {eta / eta_wi:.3f} | "
                  f"sd(a|n) {psd.mean():.4f} (prior {prior_sd:.3f})  "
                  f"rms err {rms_err:.4f}")
        del maps
    (OUT / 'r3_inferability.json').write_text(json.dumps(infer, indent=1,
                                                         default=float))
    print(f"written {OUT / 'r3_inferability.json'}")

#%% 3. P2.3 — two-channel decomposition, sigma_norm, Method-E on the *_WN rows
if _want(3):
    print('=== P2.3 channels / sigma_norm / Method-E on the white-floor Tier-1 rows ===')
    WN_ROWS = {
        'T1_PSRAND_WN':       lambda: psrand_wn_grid(em.WN_FLOORS['T1_PSRAND_WN'], SS, AS)[:2],
        'T1_PSRAND_SLOPE_WN': lambda: psrand_wn_grid(em.WN_FLOORS['T1_PSRAND_SLOPE_WN'], SS, 0.0)[:2],
        'T1_RANDOR_WN':       lambda: randor_wn_grid(em.WN_FLOORS['T1_RANDOR_WN']),
    }
    chan = {}
    a_over_sig = np.concatenate([[0.0], np.geomspace(0.05, 8.0, 40)])
    for name, build in WN_ROWS.items():
        t0 = time.time()
        P_grid, w = build()
        chan[name] = {'sigma_w': float(em.WN_FLOORS[name]), 'n_grid': int(len(w))}
        for tname, tau in TEMPLATES.items():
            dec = ch.two_channel_decomposition(P_grid, w, tau, K_MASK)
            nrm = ch.sigma_norm_analytic(P_grid, w, tau, K_MASK)
            c, wz = dec.pop('c_grid'), dec.pop('weights')
            v_curve, eta_eff = ch.method_e_curve(c, wz, a_over_sig)
            chan[name][tname] = {
                'decomposition': dec, 'sigma_norm': nrm,
                'method_e': {'a_over_sigma': a_over_sig.tolist(),
                             'v_over_sigma2': v_curve.tolist(),
                             'eta_eff': eta_eff.tolist()}}
            print(f"  {name:19s} {tname:9s} coh {dec['coh_var']:6.3f}  inc {dec['inc_var']:6.4f}  "
                  f"eta_2nd {dec['eta_2nd']:6.3f}  eta_ivw {dec['eta_ivw']:6.3f}  "
                  f"norm ratio band/rms {nrm['ratio_band']:.3f}/{nrm['ratio_rms']:.3f}")
        del P_grid
        print(f'     ({time.time() - t0:.0f} s)')

    # empirical sigma_norm on the floored flagship's campaign EVAL split
    maps, _, sp, P_hat = campaign_ensemble('T1_PSRAND_WN')
    for tname, tau in TEMPLATES.items():
        emp = ch.sigma_norm_empirical(maps[sp['eval']], tau, P_hat, K_MASK)
        chan['T1_PSRAND_WN'][tname]['sigma_norm_empirical'] = emp
        ana = chan['T1_PSRAND_WN'][tname]['sigma_norm']
        print(f"  T1_PSRAND_WN {tname:9s} empirical band {emp['ratio_band']:.3f} ± "
              f"{emp['ratio_band_err']:.3f} (analytic {ana['ratio_band']:.3f}) | "
              f"rms {emp['ratio_rms']:.3f} ± {emp['ratio_rms_err']:.3f} "
              f"(analytic {ana['ratio_rms']:.3f})")
    del maps
    (OUT / 'r3_channels_wn.json').write_text(json.dumps(chan, indent=1, default=float))
    print(f"written {OUT / 'r3_channels_wn.json'}")

    # figure: Method-E floorless (b2b_channels.json, if present) vs floored
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b2b_file = OUT / 'b2b_channels.json'
    b2b = json.loads(b2b_file.read_text()) if b2b_file.exists() else {}
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    pairs = [('T1_PSRAND', 'T1_PSRAND_WN', 'C0'), ('T1_RANDOR', 'T1_RANDOR_WN', 'C3')]
    for base, wn, col in pairs:
        for tname, ls in (('extended', '-'), ('compact', '--')):
            if base in b2b and tname in b2b[base]:
                me = b2b[base][tname]['method_e']
                ax.plot(me['a_over_sigma'], me['eta_eff'], color=col, ls=ls, alpha=0.35,
                        label=f'{base} / {tname} (floorless)')
            me = chan[wn][tname]['method_e']
            ax.plot(me['a_over_sigma'], me['eta_eff'], color=col, ls=ls, lw=2,
                    label=f'{wn} / {tname}')
    ax.axvspan(0, 2, color='0.9', zorder=0)
    ax.text(1.0, 1.05, 'outside certified band\n(A < 2σ$_{MF}$)', ha='center',
            fontsize=8, color='0.35')
    ax.set_xlabel(r'$A/\sigma_{\rm MF}$')
    ax.set_ylabel(r'attainable advantage  $\sigma_{\rm MF}^2/\min_b V_b(A)$')
    ax.set_yscale('log'); ax.set_xlim(0, 8)
    ax.legend(fontsize=7.5)
    ax.set_title('Method-E attainability: floorless (faint) vs white-floor (bold)')
    fig.tight_layout(); fig.savefig(OUT / 'r3_method_e_wn.png', dpi=150)
    print(f"figure written: {OUT / 'r3_method_e_wn.png'}")

#%% 4. P2.1 — matched-seed diagnostic on a heavy-tailed Tier-2 model
if _want(4):
    print('=== P2.1 matched-seed diagnostic (T2_PCA) ===')
    MATCH = [('T2_PCA', 'T2_PCA_WN')]        # add ('T2_GLITCH', 'T2_GLITCH_WN') if wanted
    matched = {}
    for base, wn in MATCH:
        seeds = {'seed_base': em.model_seed(base), 'seed_wn': em.model_seed(wn)}
        gens = {'base': em.REGISTRY[base], 'wn': em.REGISTRY[wn]}
        res = {'seeds': seeds, 'floor_wn': float(em.REGISTRY[wn].get('floor', 0.0)),
               'cells': {}}
        sp = ep.make_splits(N_ENS, seed=SPLIT_SEED)
        for cfg, entry in gens.items():
            for skey, seed in seeds.items():
                key = f'{cfg}@{skey}'
                t0 = time.time()
                maps, _ = entry['gen'](N_ENS, seed)
                P_hat = ep.estimate_psd2d(maps[sp['psd']])
                mom = nl.diagnostics.moment_summary(maps[sp['eval']])
                cell = {'moments': mom}
                for tname, tau in TEMPLATES.items():
                    _, s_pred = ep.whitened_template(tau, P_hat)
                    A_hat = ep.whitening.mf_amplitudes(maps[sp['eval']], tau, P_hat)
                    d = {'sigma_mf_pred': float(s_pred), 'sigma_mf_emp': float(A_hat.std())}
                    if entry.get('bound_bg') is not None:
                        d['bound_complete_data'] = float(
                            ep.complete_data_bound(tau, entry['bound_bg'](), P_hat))
                    cell[tname] = d
                res['cells'][key] = cell
                print(f"  {key:16s} kurt {mom['excess_kurtosis']:8.2f}  "
                      f"sigMF ext/cmp {cell['extended']['sigma_mf_emp']:.3f}/"
                      f"{cell['compact']['sigma_mf_emp']:.3f}  bound ext/cmp "
                      f"{cell['extended'].get('bound_complete_data', float('nan')):.3f}/"
                      f"{cell['compact'].get('bound_complete_data', float('nan')):.3f}"
                      f"   ({time.time() - t0:.0f} s)")
                del maps
        # paired floor effect vs seed effect
        C = res['cells']

        def delta(a, b, path):
            x, y = C[a], C[b]
            for k in path:
                x, y = x[k], y[k]
            return float(y - x)
        eff = {}
        for tname in TEMPLATES:
            eff[tname] = {
                'floor_effect_paired@seed_base': {
                    'sigma_mf_emp': delta('base@seed_base', 'wn@seed_base', (tname, 'sigma_mf_emp')),
                    'bound': delta('base@seed_base', 'wn@seed_base', (tname, 'bound_complete_data'))},
                'floor_effect_paired@seed_wn': {
                    'sigma_mf_emp': delta('base@seed_wn', 'wn@seed_wn', (tname, 'sigma_mf_emp')),
                    'bound': delta('base@seed_wn', 'wn@seed_wn', (tname, 'bound_complete_data'))},
                'seed_effect_floorless': {
                    'sigma_mf_emp': delta('base@seed_base', 'base@seed_wn', (tname, 'sigma_mf_emp')),
                    'bound': delta('base@seed_base', 'base@seed_wn', (tname, 'bound_complete_data'))},
                'seed_effect_floored': {
                    'sigma_mf_emp': delta('wn@seed_base', 'wn@seed_wn', (tname, 'sigma_mf_emp')),
                    'bound': delta('wn@seed_base', 'wn@seed_wn', (tname, 'bound_complete_data'))},
                'campaign_comparison_unpaired': {
                    'sigma_mf_emp': delta('base@seed_base', 'wn@seed_wn', (tname, 'sigma_mf_emp')),
                    'bound': delta('base@seed_base', 'wn@seed_wn', (tname, 'bound_complete_data'))},
            }
        eff['excess_kurtosis'] = {
            'floor_effect_paired@seed_base': delta('base@seed_base', 'wn@seed_base', ('moments', 'excess_kurtosis')),
            'floor_effect_paired@seed_wn': delta('base@seed_wn', 'wn@seed_wn', ('moments', 'excess_kurtosis')),
            'seed_effect_floorless': delta('base@seed_base', 'base@seed_wn', ('moments', 'excess_kurtosis')),
            'seed_effect_floored': delta('wn@seed_base', 'wn@seed_wn', ('moments', 'excess_kurtosis'))}
        res['effects'] = eff
        matched[base] = res
        print('  effects (differences, second minus first):')
        for k, v in eff['excess_kurtosis'].items():
            print(f"    kurtosis {k:32s} {v:+9.2f}")
        for tname in TEMPLATES:
            for k, v in eff[tname].items():
                print(f"    {tname:9s} {k:32s} d sigma_MF {v['sigma_mf_emp']:+.4f}   d bound {v['bound']:+.4f}")
    (OUT / 'r3_matched_seed.json').write_text(json.dumps(matched, indent=1, default=float))
    print(f"written {OUT / 'r3_matched_seed.json'}")

print('\nR3 pass complete.')
