#!/usr/bin/env python
"""
check_bound_dc.py — is the DC mode inflating the confusion rows' complete-data bounds?
======================================================================================

Hypothesis (2026-09-04, consolidation review).  The confusion maps keep their
(large, Poisson-fluctuating) DC mode deliberately, so the data-estimated P̂[0,0]
is dominated by mean²·n_pix², whereas the ANALYTIC background spectrum used in the
numerator of the complete-data bound has P_g[0,0] = plateau (the Poisson field's
DC fluctuation power).  The numerator therefore contains a large DC term
|τ̃(0)|²/P_g(0) that the denominator |τ̃(0)|²/P̂(0) lacks, and the bound is biased
HIGH — most for the extended template, whose τ̃(0) = Σ τ is its largest single
Fourier coefficient.  This would explain the `_ATM` extended bound of 5.30 against
its exact value (1+ρ)/ρ = 5.000 (6 % high) while the compact one reads 4.97.
Every other campaign model is zero-mean with a regularised analytic DC that matches
the data, so the effect is confusion-specific.

Computes, for T2_CONFUSION_WN and T2_CONFUSION_ATM, both templates, on the
campaign's own PSD split (regenerated from the registry seed):
  (a) the bound as stored (all modes, analytic P_g vs data P̂);
  (b) the bound with the DC mode excluded from both sums;
  (c) the bound with P_g(0) replaced by P̂(0)  (DC treated consistently);
  (d) for _ATM only, the fully analytic bound (P_tot = (1+ρ)P_conf), which must be 5.
Writes eta_results/confusion_bound_dc_check.json.
Usage: python scripts/tools/check_bound_dc.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
for _p in (_here, *_here.parents):
    if (_p / 'eta_pipeline').is_dir():
        sys.path.insert(0, str(_p)); break
import eta_pipeline as ep                                # noqa: E402
from eta_pipeline import models as em                    # noqa: E402

OUT = _p / 'results' / 'eta_results'
N_ENS, SPLIT_SEED = 6000, 12345
TEMPLATES = em.make_templates()


def bound(tau, Pg, Pt, drop_dc=False):
    t2 = np.abs(np.fft.fft2(tau))**2
    if drop_dc:
        t2 = t2.copy(); t2[0, 0] = 0.0
    return float(np.sum(t2 / np.maximum(Pg, 1e-20)) / np.sum(t2 / np.maximum(Pt, 1e-20)))


res = {}
for name in ('T2_CONFUSION_WN', 'T2_CONFUSION_ATM'):
    entry = em.REGISTRY[name]
    t0 = time.time()
    maps, _ = entry['gen'](N_ENS, em.model_seed(name))
    sp = ep.make_splits(N_ENS, seed=SPLIT_SEED)
    P_hat = ep.estimate_psd2d(maps[sp['psd']])
    Pg = entry['bound_bg']()
    print(f'=== {name}: {N_ENS} maps in {time.time() - t0:.0f} s;  P_hat(0) = {P_hat[0, 0]:.3e}, '
          f'P_g(0) = {Pg[0, 0]:.3e}, P_hat at k_min = {P_hat[0, 1]:.3e}')
    res[name] = {'P_hat_dc': float(P_hat[0, 0]), 'P_g_dc': float(Pg[0, 0]), 'P_hat_kmin': float(P_hat[0, 1])}
    Pg_fix = Pg.copy(); Pg_fix[0, 0] = P_hat[0, 0]
    for tname, tau in TEMPLATES.items():
        d = {'stored_style_all_modes': bound(tau, Pg, P_hat),
             'dc_excluded': bound(tau, Pg, P_hat, drop_dc=True),
             'dc_consistent': bound(tau, Pg_fix, P_hat)}
        if name == 'T2_CONFUSION_ATM':
            rho = em.CONFUSION_RHO_ATM
            d['fully_analytic'] = bound(tau, Pg, Pg * (1 + rho) / rho)
            d['exact_expected'] = (1 + rho) / rho
        res[name][tname] = d
        print(f"  {tname:9s} " + '  '.join(f'{k} {v:.3f}' for k, v in d.items()))
    del maps
(OUT / 'confusion_bound_dc_check.json').write_text(json.dumps(res, indent=1))
print(f"written {OUT / 'confusion_bound_dc_check.json'}")
