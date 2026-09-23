#!/usr/bin/env python
"""
run_b2b_cheap.py — Phase B2b cheap analysis: v5 add-ons + master table v2
=========================================================================

The container-side half of Phase B2b (CNN_vs_MF_theory_v5.md alignment).
Runs in minutes; no PyTorch required.  Sections (Spyder cells):

  1. σ_α = 0.1 (and 0.2) perturbative CLOSURE TEST for slope-only PSRAND
     (v5 §5.2.3 action item): exact quadrature vs the second-order
     two-channel forecast and its lognormal (IVW) continuation.
  2. TWO-CHANNEL DECOMPOSITION for every Tier-1 row × template
     (coherent Var[⟨δ⟩_f], incoherent E[Var_f δ]) — the invariant
     severity coordinates of v5 §5.2.7 — plus the analytic σ_norm
     baselines (band-level and map-RMS scalar normalizations).
  3. EMPIRICAL σ_norm validation on the flagship rows (regenerated
     campaign ensembles, same seeds/splits as B2a — paired by design).
  4. METHOD-E attainability curves η_eff(A) (v5 §5.2.5): the best
     marginally-unbiased rescaled MF at each source amplitude; figure
     eta_results/b2b_method_e.png.
  5. MASTER TABLE v2 (v5 terminology): σ_empirical, σ_norm ratios,
     η ceiling + kind, σ_CRLB/σ_MF = 1/√η, severity coordinates;
     writes eta_results/master_table_v2.csv.
  6. CROSS-CHECK summary → eta_results/b2b_crosschecks.json.

Outputs merge into eta_results/ alongside the B2a campaign JSONs.
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
from eta_pipeline.quadrature import (truncnorm_grid,     # noqa: E402
                                     eta_marg_quadrature_scaled)

OUT = _p / 'results' / 'eta_results'
OUT.mkdir(exist_ok=True)
SPLIT_SEED = 12345
SEED_BASE = 777000

B2 = em.beam2()
K_MASK = B2 > 1e-8
TEMPLATES = em.make_templates()
SM, SS, AM, AS = em.PSRAND_ARGS                          # 3.0, 0.5, 5.0, 0.8


def psrand_grids(slope_sigma, amp_sigma, n_slope=61, n_amp=201):
    """(base spectra on the slope grid incl. beam, slope wts, amp grid, amp wts)."""
    s_grid, s_w = ((np.array([SM]), np.array([1.0])) if slope_sigma == 0 else
                   truncnorm_grid(SM, slope_sigma, 1.0, n=n_slope))
    a_grid, a_w = ((np.array([AM]), np.array([1.0])) if amp_sigma == 0 else
                   truncnorm_grid(AM, amp_sigma, 1e-6, n=n_amp))
    base = np.stack([nl.psd_isotropic_powerlaw(em.SHAPE, s, 1.0) * B2
                     for s in s_grid])
    return base, s_w, a_grid, a_w


def randor_grid(n_theta=180):
    ss_sc, a_sc, ss_iso, a_iso = em.P_ANISO_ARGS
    th = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    P = np.stack([nl.psd_two_component(em.SHAPE, ss_sc, a_sc, ss_iso, a_iso,
                                       scan_angle=t) * B2 for t in th])
    return P, np.full(n_theta, 1.0 / n_theta)


#%% 2. closure test: slope-only PSRAND at sigma_alpha = 0.1 and 0.2
import os
FORCE = bool(os.environ.get('B2B_FORCE'))

print('=== closure test (v5 §5.2.3) ===')
closure = {}
if (OUT / 'b2b_closure.json').exists() and not FORCE:
    closure = json.loads((OUT / 'b2b_closure.json').read_text())
    print('  loaded existing b2b_closure.json (set B2B_FORCE=1 to recompute)')
for sig_a in () if closure else (0.1, 0.2):
    base, s_w, a_grid, a_w = psrand_grids(sig_a, 0.0)
    n_maps, seed = 4000, SEED_BASE + 900001 + int(1000 * sig_a)
    t0 = time.time()
    maps, _ = em._gen_psrand(n_maps, seed, sig_a, 0.0)
    gen_s = time.time() - t0
    entry = {'sigma_alpha': sig_a, 'n_maps': n_maps, 'seed': seed}
    for tname, tau in TEMPLATES.items():
        dec = ch.two_channel_scaled(base, s_w, a_grid, a_w, tau, K_MASK)
        t0 = time.time()
        eta, err = eta_marg_quadrature_scaled(maps, base, s_w, a_grid, a_w,
                                              tau, k_mask=K_MASK)
        entry[tname] = {
            'eta_quad': eta, 'err': err,
            'eta_2nd_forecast': dec['eta_2nd'],
            'eta_ivw_forecast': dec['eta_ivw'],
            'coh_var': dec['coh_var'], 'inc_var': dec['inc_var'],
            'quad_runtime_s': round(time.time() - t0, 1),
        }
        print(f"  sigma_a={sig_a}  {tname:9s} quad {eta:.4f} ± {err:.4f} | "
              f"2nd-order {dec['eta_2nd']:.4f}  IVW {dec['eta_ivw']:.4f}")
    entry['gen_runtime_s'] = round(gen_s, 1)
    closure[f'{sig_a}'] = entry
(OUT / 'b2b_closure.json').write_text(json.dumps(closure, indent=1,
                                                 default=float))

#%% 3. two-channel decomposition + analytic sigma_norm for all T1 rows
print('=== two-channel decomposition + sigma_norm (analytic) ===')
T1_GRIDS = {
    'T1_PSRAND':       ('scaled', psrand_grids(SS, AS)),
    'T1_PSRAND_SLOPE': ('scaled', psrand_grids(SS, 0.0)),
    'T1_PSRAND_AMP':   ('scaled', psrand_grids(0.0, AS)),
    'T1_RANDOR':       ('plain', randor_grid()),
}
channels_res = {}
for name, (kind, grids) in T1_GRIDS.items():
    channels_res[name] = {}
    for tname, tau in TEMPLATES.items():
        if kind == 'scaled':
            base, s_w, a_grid, a_w = grids
            dec = ch.two_channel_scaled(base, s_w, a_grid, a_w, tau, K_MASK)
            nrm = ch.sigma_norm_scaled(base, s_w, a_grid, a_w, tau, K_MASK)
        else:
            P_grid, w = grids
            dec = ch.two_channel_decomposition(P_grid, w, tau, K_MASK)
            nrm = ch.sigma_norm_analytic(P_grid, w, tau, K_MASK)
        c, wz = dec.pop('c_grid'), dec.pop('weights')
        a_over_sig = np.concatenate([[0.0], np.geomspace(0.05, 8.0, 40)])
        v_curve, eta_eff = ch.method_e_curve(c, wz, a_over_sig)
        channels_res[name][tname] = {
            'decomposition': dec, 'sigma_norm': nrm,
            'method_e': {'a_over_sigma': a_over_sig.tolist(),
                         'v_over_sigma2': v_curve.tolist(),
                         'eta_eff': eta_eff.tolist()},
        }
        print(f"  {name:16s} {tname:9s} coh {dec['coh_var']:7.3f}  "
              f"inc {dec['inc_var']:6.4f}  eta_ivw {dec['eta_ivw']:7.3f}  "
              f"norm ratio band/rms {nrm['ratio_band']:.3f}/{nrm['ratio_rms']:.3f}")

#%% 4. empirical sigma_norm validation on the flagship rows
print('=== empirical sigma_norm (flagship rows, campaign ensembles) ===')
for name in ('T1_PSRAND', 'T1_RANDOR'):
    seed = em.model_seed(name)
    t0 = time.time()
    maps, _ = em.REGISTRY[name]['gen'](6000, seed)
    sp = ep.make_splits(6000, seed=SPLIT_SEED)
    P_hat = ep.estimate_psd2d(maps[sp['psd']])
    print(f'  [{name}] regenerated 6000 maps in {time.time()-t0:.0f} s')
    for tname, tau in TEMPLATES.items():
        emp = ch.sigma_norm_empirical(maps[sp['eval']], tau, P_hat, K_MASK)
        channels_res[name][tname]['sigma_norm_empirical'] = emp
        ana = channels_res[name][tname]['sigma_norm']
        print(f"    {tname:9s} ratio band emp {emp['ratio_band']:.3f} ± "
              f"{emp['ratio_band_err']:.3f} (analytic {ana['ratio_band']:.3f})"
              f" | rms emp {emp['ratio_rms']:.3f} ± {emp['ratio_rms_err']:.3f}"
              f" (analytic {ana['ratio_rms']:.3f})")
    del maps
(OUT / 'b2b_channels.json').write_text(json.dumps(channels_res, indent=1,
                                                  default=float))

#%% 5. high-precision Tier-1 quadratures (fresh, larger ensembles)
# The quadrature has NO fitting step, so the campaign's EVAL-split
# restriction was convention, not necessity — and on heavy-tailed mixtures
# n_eval = 1500 leaves the ceiling with optimistic-looking errors (the B2a
# PSRAND-compact value sat ~2σ low of a fresh 6000-map run, briefly
# *below* the attainable IVW gain E[c]E[1/c], which is impossible for the
# true η).  Table values therefore come from fresh high-precision runs.
print('=== high-precision T1 quadratures ===')
HP_N = {'T1_PSRAND': 20000, 'T1_PSRAND_SLOPE': 20000,
        'T1_PSRAND_AMP': 20000, 'T1_RANDOR': 12000}
hp_file = OUT / 'b2b_quad_hp.json'
quad_hp = json.loads(hp_file.read_text()) if hp_file.exists() else {}
for name, n_hp in HP_N.items():
    if name in quad_hp and not FORCE:
        print(f'  [{name}] exists — skipped'); continue
    idx = list(em.REGISTRY).index(name)
    seed = SEED_BASE + 555000 + idx     # hp stream: kept positional deliberately,
                                        # so existing hp ensembles reproduce
    t0 = time.time()
    maps, _ = em.REGISTRY[name]['gen'](n_hp, seed)
    print(f'  [{name}] generated {n_hp} maps in {time.time()-t0:.0f} s')
    runner = (em.REGISTRY[name]['quad'])()
    quad_hp[name] = {'n': n_hp, 'seed': seed}
    for tname, tau in TEMPLATES.items():
        t0 = time.time()
        q = runner(maps, tau)
        quad_hp[name][tname] = {'eta': q['eta'], 'err': q['err'],
                                'runtime_s': round(time.time() - t0, 1)}
        print(f"    {tname:9s} eta = {q['eta']:.4f} ± {q['err']:.4f} "
              f"({quad_hp[name][tname]['runtime_s']} s)")
    hp_file.write_text(json.dumps(quad_hp, indent=1, default=float))
    del maps

#%% 6. Method-E figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.2, 5.0))
styles = {('T1_PSRAND', 'extended'): ('C0', '-'),
          ('T1_PSRAND', 'compact'): ('C0', '--'),
          ('T1_RANDOR', 'extended'): ('C3', '--'),
          ('T1_RANDOR', 'compact'): ('C3', '-')}
for (name, tname), (col, ls) in styles.items():
    me = channels_res[name][tname]['method_e']
    a = np.array(me['a_over_sigma']); e = np.array(me['eta_eff'])
    ax.plot(a, e, color=col, ls=ls, label=f'{name} / {tname}')
    ax.axhline(e[0], color=col, ls=':', lw=0.8, alpha=0.5)
ax.axvspan(0, 2, color='0.9', zorder=0)
ax.text(1.0, 1.05, 'outside certified band\n(A < 2σ$_{MF}$)', ha='center',
        fontsize=8, color='0.35')
ax.set_xlabel(r'$A/\sigma_{\rm MF}$')
ax.set_ylabel(r'attainable advantage  $\sigma_{\rm MF}^2/\min_b V_b(A)$')
ax.set_yscale('log')
ax.set_xlim(0, 8)
ax.legend(fontsize=9)
ax.set_title('Method-E: coherent-channel attainability vs source amplitude\n'
             '(dotted: A=0 ceiling of each curve; v5 §5.2.5)')
fig.tight_layout()
fig.savefig(OUT / 'b2b_method_e.png', dpi=150)
print(f"figure written: {OUT / 'b2b_method_e.png'}")

#%% 7. master table v2 (v5 terminology) + cross-checks
rows, checks = [], {}
for p in sorted(OUT.glob('T*__*.json')):
    r = json.loads(p.read_text())
    name, tname, tier = r['model'], r['template'], r['tier']
    quad = r.get('quadrature', {})
    bound = r.get('bound_complete_data')
    hp = quad_hp.get(name, {}).get(tname)
    if tier == 'T0':
        eta, kind = 1.0, 'null (Gaussian)'
    elif hp:
        eta, kind = hp['eta'], 'exact quadrature (hp)'
    elif quad:
        eta, kind = quad.get('eta'), 'exact quadrature'
    elif bound is not None:
        eta, kind = bound, 'complete-data bound'
    else:
        eta, kind = None, 'pending variational2'
    chn = channels_res.get(name, {}).get(tname, {})
    dec = chn.get('decomposition', {})
    nrm = chn.get('sigma_norm', {})
    var2 = r.get('variational2', {}).get('rungs', {})
    eta_v2 = (max(v['eta'] for v in var2.values()) if var2 else None)
    rows.append({
        'model': name, 'template': tname, 'tier': tier,
        'sigma_empirical': r.get('sigma_mf', {}).get('emp'),
        'eta_ceiling': eta, 'ceiling_kind': kind,
        'sigma_CRLB_over_MF': (1.0 / np.sqrt(eta)
                               if eta and kind != 'complete-data bound'
                               else None),
        'coh_sd_band': dec.get('sd_band'),
        'inc_rms_shape': dec.get('rms_shape'),
        'norm_ratio_band': nrm.get('ratio_band'),
        'norm_ratio_rms': nrm.get('ratio_rms'),
        'eta_vs_norm_band': (eta * nrm['ratio_band']
                             if eta and nrm.get('ratio_band') else None),
        'f_s': r.get('diagnostics', {}).get('f_s'),
        'xi_rms': r.get('diagnostics', {}).get('xi_rms'),
        'eta_var2_best': eta_v2,
    })

import csv
with open(OUT / 'master_table_v2.csv', 'w', newline='') as fcsv:
    wr = csv.DictWriter(fcsv, fieldnames=list(rows[0]))
    wr.writeheader(); wr.writerows(rows)
print(f"written {OUT / 'master_table_v2.csv'}")


def _num(v, n=3):
    return '—' if v is None else f'{v:.{n}f}'


hdr = (f"{'model':16s} {'tmpl':9s} {'eta_ceil':>9s} {'kind':>20s} "
       f"{'coh_sd':>7s} {'inc_rms':>8s} {'eta_vs_norm':>12s}")
print(hdr); print('-' * len(hdr))
for r in rows:
    print(f"{r['model']:16s} {r['template']:9s} {_num(r['eta_ceiling'], 3):>9s} "
          f"{r['ceiling_kind']:>20s} {_num(r['coh_sd_band']):>7s} "
          f"{_num(r['inc_rms_shape'], 4):>8s} {_num(r['eta_vs_norm_band']):>12s}")

# cross-checks
for sig_a, entry in closure.items():
    for tname in ('extended', 'compact'):
        e = entry[tname]
        pull = (e['eta_quad'] - e['eta_2nd_forecast']) / max(e['err'], 1e-12)
        checks[f'closure_{sig_a}_{tname}'] = {
            'quad': e['eta_quad'], 'err': e['err'],
            'forecast_2nd': e['eta_2nd_forecast'],
            'forecast_ivw': e['eta_ivw_forecast'], 'pull_vs_2nd': pull,
            'pass_2sigma': bool(abs(pull) < 2.0)}
amp = json.loads((OUT / 'T1_PSRAND_AMP__extended.json').read_text())
checks['amp_analytic'] = {
    'quad': amp['quadrature']['eta'],
    'analytic': amp['quadrature'].get('analytic_E2Einv2'),
}
for name in T1_GRIDS:
    for tname in ('extended', 'compact'):
        f = OUT / f'{name}__{tname}.json'
        if not f.exists():
            continue
        q = json.loads(f.read_text()).get('quadrature', {})
        hp = quad_hp.get(name, {}).get(tname, {})
        d = channels_res[name][tname]['decomposition']
        checks[f'ivw_vs_quad_{name}_{tname}'] = {
            'eta_quad_b2a': q.get('eta'), 'eta_quad_hp': hp.get('eta'),
            'eta_hp_err': hp.get('err'), 'eta_ivw_attainable': d['eta_ivw'],
            'eta_2nd_forecast': d['eta_2nd'],
            'ivw_below_hp': (bool(d['eta_ivw'] <= hp['eta'] + 2 * hp['err'])
                             if hp else None)}
(OUT / 'b2b_crosschecks.json').write_text(json.dumps(checks, indent=1,
                                                     default=float))
print(f"cross-checks written: {OUT / 'b2b_crosschecks.json'}")
print('\nB2b cheap pass complete.')
