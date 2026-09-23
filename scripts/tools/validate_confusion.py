#!/usr/bin/env python
"""
validate_confusion.py — the acceptance battery for the T2_CONFUSION rung
========================================================================

Provenance and acceptance script for ``T2_CONFUSION`` / ``T2_CONFUSION_WN``
(registry rows in ``eta_pipeline/models.py``, generator in
``noise_lib/confusion.py``).  Run it before any eta produced with these rows is
quoted, exactly as ``run_null_tests.py`` gates the rest of the campaign.

Eight checks, in dependence order:

  V1  Campbell closure, one-point.  The realised ensemble's mean, variance,
      skewness and excess kurtosis against the analytic kappa_n = Omega_n q_n
      with Omega_n = Omega_beam / n.  This is Method C (memo v2 Sec. 18) used
      as a validator rather than a predictor, and it is the check no other
      Tier-2 model can run -- for the artifact models the mark distribution has
      no closed-form moments worth trusting.

  V2  Campbell closure, two-point.  P_hat against the analytic
      P_conf = N<a^2>|B|^2 of ``confusion_psd``.  Tests the DFT normalisation
      that memo v2 Sec. 18 says must be "fixed once against one simulated
      ensemble" -- the prefactor is fixed by construction here, so this is a
      falsifiable test rather than a fit.

  V3  Flatness.  |tau~_compact|^2 / P_hat constant across the band: the defining
      property of the model (no MF spectral leverage).  Also reports the same
      ratio for the extended template, which is NOT flat and should not be.

  V4  Mode-count scaling, FLOORLESS.  sigma_MF vs the number of retained modes
      under nested B^2 masks.  Flat per-mode SNR predicts
      sigma_MF^-2 = N_modes / (N<a^2>) exactly, i.e. sigma_MF * sqrt(N_modes)
      = const.  Confirming that scaling IS the demonstration that the floorless
      configuration does not converge: there is no band, only a mode count.
      (Memo v2 Sec. 21.4 claims the opposite; see the conclusion memo.)

  V5  Band stability, FLOORED.  The same scan plus ``bounds.bound_stability``.
      Null test N8: an unstable bound means the band is unregularised.  The
      floored row must pass; the floorless row is expected to fail, and its
      failure is the result.

  V6  Precision sensitivity.  sigma_MF from float32-stored vs float64-stored
      maps.  P_conf spans ~39 decades across a 128^2 grid at FWHM 5 px, so the
      stored mantissa decides where the flat plateau ends.  A large drift here
      is why the registry generator ignores map_dtype() and always returns
      float64.

  V7  sigma_MF, predicted vs empirical, under FULL SPLIT DISCIPLINE (P_hat from
      SPLIT_PSD only).  The in-sample version of this check is biased low; the
      split version is the one that certifies the conventions.

  V8  Tail diagnostics of the MF amplitude.  The compact template's MF output
      inherits the P(D) skewness, so its EVAL-split errors are heavy-tailed.
      Reported so that later bootstrap intervals are read with that in mind.

Usage
-----
    python scripts/tools/validate_confusion.py [--n-ens 4000] [--out DIR]

Writes ``confusion_validation_<date>.json`` and a human-readable ``.log`` to
``tests_and_results/`` (override with --out).  Exit status is non-zero if any
check that is expected to pass does not.

torch is NOT required: the script loads the numpy-only members of
``eta_pipeline`` (whitening, bounds, cumulants) directly if importing the
package fails because torch is absent, so it runs in a bare container as well
as on the M3.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]                                     # repository root
sys.path.insert(0, str(_ROOT))

import noise_lib as nl                                       # noqa: E402


def _torchless_package():
    """Make ``eta_pipeline`` importable without torch.

    ``eta_pipeline/__init__`` pulls in ``variational``, hence torch, which this
    script does not need: whitening, bounds and models are numpy-only.  Putting
    a stub package in ``sys.modules`` with the right ``__path__`` bypasses the
    package body while leaving submodule imports -- including the relative
    ``from .quadrature import ...`` inside models.py -- fully functional.
    Harmless where torch IS present, but we only do it if it is not.
    """
    import types
    pkg = types.ModuleType('eta_pipeline')
    pkg.__path__ = [str(_ROOT / 'eta_pipeline')]
    sys.modules['eta_pipeline'] = pkg


try:
    import torch                                             # noqa: F401,E402
except Exception:
    _torchless_package()

from eta_pipeline import whitening, bounds                    # noqa: E402
from eta_pipeline import models as _models_mod                # noqa: E402


# %% Configuration -------------------------------------------------------------------
SIZE, FWHM, RC = 128, 5.0, 7.0
SHAPE = (SIZE, SIZE)
NPIX = SIZE * SIZE
TOL = dict(v1=0.03, v2=0.01, v3=0.02, v5_drift=0.02, v7=0.05)


def psd_chunked(images, chunk=400):
    """Ensemble mean periodogram, computed in blocks.

    ``whitening.estimate_psd2d`` transforms the whole stack at once, which for
    a 12k x 128^2 float64 ensemble allocates ~3 GB of complex128 and is what
    killed the first full-size run of this script.  Numerically identical to
    the library routine (same mean of |fft2|^2, same 1e-20 floor); the chunking
    only bounds peak memory.
    """
    images = np.asarray(images)
    acc = np.zeros(images.shape[-2:], dtype=np.float64)
    for i in range(0, len(images), chunk):
        blk = images[i:i + chunk]
        acc += (np.abs(np.fft.fft2(blk, axes=(-2, -1))) ** 2).sum(axis=0)
    return np.maximum(acc / len(images), 1e-20)


def geometry():
    """Frequency grid, |B|^2, and the template pair at the paper conventions."""
    fy = np.fft.fftfreq(SIZE).reshape(-1, 1)
    fx = np.fft.fftfreq(SIZE).reshape(1, -1)
    sig = FWHM / 2.355
    b2 = np.exp(-4.0 * np.pi ** 2 * sig ** 2 * (fx ** 2 + fy ** 2))
    tau = {'compact': nl.make_template(SIZE, 'point', beam_fwhm=FWHM,
                                       normalize='none'),
           'extended': nl.make_template(SIZE, 'beta', RC, FWHM,
                                        normalize='none')}
    return np.hypot(fx, fy), b2, tau


# %% V1-V2: Campbell closure ---------------------------------------------------------
def v1_one_point(maps, stats):
    """Realised one-point cumulants against kappa_n = Omega_n q_n."""
    z = (maps - maps.mean()) / maps.std()
    got = dict(mean=float(maps.mean()), rms=float(maps.std()),
               skew=float((z ** 3).mean()), exkurt=float((z ** 4).mean() - 3.0))
    want = dict(mean=stats['mean'], rms=stats['sigma_c'],
                skew=stats['skew'], exkurt=stats['exkurt'])
    rel = {k: abs(got[k] / want[k] - 1.0) for k in got}
    return dict(measured=got, analytic=want, rel_dev=rel,
                passed=bool(max(rel.values()) < TOL['v1']))


def v2_two_point(P_hat, P_analytic, k, n_ens):
    """P_hat / P_conf over the trustworthy band, against periodogram noise."""
    m = (k > 0) & (k < 0.32)
    ratio = P_hat[m] / P_analytic[m]
    exp_scatter = 1.0 / np.sqrt(n_ens)
    return dict(mean=float(ratio.mean()), scatter=float(ratio.std()),
                expected_scatter=float(exp_scatter),
                passed=bool(abs(ratio.mean() - 1.0) < TOL['v2']))


# %% V3: flatness --------------------------------------------------------------------
def v3_flatness(tau, P_hat, k, floored):
    """Relative spread of |tau~|^2/P over the band.

    FLOORLESS: the compact ratio must be flat to within the periodogram noise
    (that is the model's defining property) while the extended ratio must NOT
    be -- it inherits |beta~|^2 and is strongly peaked at low k.

    FLOORED: flatness is deliberately BROKEN.  P = N<a^2>|B|^2 + n_pix sigma_w^2
    rolls the compact ratio off beyond the knee, which is the entire purpose of
    the floor, so a large spread here is the pass condition, not a failure.
    """
    out = {}
    for name, t in tau.items():
        w = np.abs(np.fft.fft2(t)) ** 2 / P_hat
        m = (k > 0) & (k < 0.32)
        out[name] = dict(rel_spread=float(w[m].std() / w[m].mean()),
                         mean=float(w[m].mean()))
    if floored:
        out['expectation'] = 'compact ratio rolls off past the knee'
        out['passed'] = bool(out['compact']['rel_spread'] > 0.1)
    else:
        out['expectation'] = 'compact ratio flat to periodogram noise'
        out['passed'] = bool(out['compact']['rel_spread'] < TOL['v3']
                             and out['extended']['rel_spread'] > 1.0)
    return out


# %% V4-V5: the band ------------------------------------------------------------------
def band_scan(tau, P_hat, b2, cuts=(1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-14, 1e-20)):
    """sigma_MF and the retained mode count under nested B^2 > cut masks.

    For a flat per-mode SNR, sigma_MF^-2 is exactly proportional to the number
    of retained modes, so ``sigma_MF * sqrt(n_modes)`` is invariant.  That
    product is the diagnostic: constant => no band, only a mode count.
    """
    tf = np.abs(np.fft.fft2(tau)) ** 2
    rows = []
    for c in cuts:
        m = b2 > c
        s = 1.0 / np.sqrt((tf[m] / P_hat[m]).sum())
        rows.append(dict(cut=float(c), n_modes=int(m.sum()),
                         sigma_mf=float(s),
                         invariant=float(s * np.sqrt(m.sum()))))
    inv = np.array([r['invariant'] for r in rows])
    smf = np.array([r['sigma_mf'] for r in rows])
    n8 = np.array([r['sigma_mf'] for r in rows
                   if r['cut'] in (1e-4, 1e-8, 1e-14)])
    return dict(rows=rows,
                invariant_drift=float((inv.max() - inv.min()) / inv.min()),
                sigma_mf_drift=float((smf.max() - smf.min()) / smf.min()),
                sigma_mf_drift_n8=float((n8.max() - n8.min()) / n8.min()))


# %% V6: precision --------------------------------------------------------------------
def v6_precision(maps64, tau, b2, cut=1e-8):
    out = {}
    P = {}
    for name, dt in (('float64', np.float64), ('float32', np.float32)):
        #  chunked, and the float32 cast is done per block so the full
        #  downcast copy is never materialised
        acc = np.zeros(SHAPE); n = len(maps64)
        for i in range(0, n, 400):
            blk = maps64[i:i + 400].astype(dt, copy=False)
            acc += (np.abs(np.fft.fft2(blk, axes=(-2, -1))) ** 2).sum(axis=0)
        P[name] = np.maximum(acc / n, 1e-20)
    for tname, t in tau.items():
        vals = {n: float(whitening.sigma_mf_from_psd(t, P[n])) for n in P}
        out[tname] = dict(sigma_mf=vals,
                          rel_dev=abs(vals['float32'] / vals['float64'] - 1.0))
    return out


# %% V7-V8: split discipline and tails -------------------------------------------------
def v7_sigma_mf(maps, tau):
    sp = whitening.make_splits(len(maps))
    P_hat = psd_chunked(maps[sp['psd']])
    out = {}
    for name, t in tau.items():
        pred = float(whitening.sigma_mf_from_psd(t, P_hat))
        amp = whitening.mf_amplitudes(maps[sp['eval']], t, P_hat)
        emp = float(amp.std())
        ex = float(((amp - amp.mean()) ** 4).mean() / emp ** 4 - 3.0)
        #  The sample standard deviation of n draws with excess kurtosis g2 has
        #  relative standard error sqrt((g2 + 2) / (4n)).  For the COMPACT
        #  template on confusion noise the MF output inherits the P(D) tail
        #  (g2 ~ 10-70), so a fixed few-percent tolerance would fail on Monte
        #  Carlo alone.  Judge the deviation in units of ITS OWN error instead.
        se = float(np.sqrt((ex + 2.0) / (4.0 * len(amp))))
        out[name] = dict(predicted=pred, empirical=emp, ratio=emp / pred,
                         n_eval=int(len(amp)), rel_stderr=se,
                         n_sigma=float(abs(emp / pred - 1.0) / max(se, 1e-12)),
                         skew=float(((amp - amp.mean()) ** 3).mean() / emp ** 3),
                         exkurt=ex)
    out['passed'] = bool(all(out[n]['n_sigma'] < 3.0 for n in tau))
    return out, P_hat


# %% Driver ----------------------------------------------------------------------------
def run(n_ens, out_dir):
    k, b2, tau = geometry()
    em = _models_mod
    args = em.CONFUSION_ARGS
    stats = nl.confusion.confusion_stats(**args)
    P_analytic = nl.confusion.confusion_psd(SHAPE, **args)
    floor = em.CONFUSION_FLOOR

    res = {'generated': datetime.now().isoformat(timespec='seconds'),
           'n_ens': n_ens, 'config': dict(args), 'analytic': stats,
           'floor': floor, 'f_N': em.CONFUSION_F_N}

    for row, sigma_w in (('T2_CONFUSION', 0.0), ('T2_CONFUSION_WN', floor)):
        maps, lat = em.REGISTRY[row]['gen'](n_ens, em.model_seed(row))
        P_hat = psd_chunked(maps)
        P_gen = P_analytic + (NPIX * sigma_w ** 2 if sigma_w else 0.0)
        r = {'dtype': str(maps.dtype), 'n_src_mean': float(lat['n_src'].mean())}
        if sigma_w == 0.0:
            r['V1_one_point'] = v1_one_point(maps, stats)
        r['V2_two_point'] = v2_two_point(P_hat, P_gen, k, n_ens)
        r['V3_flatness'] = v3_flatness(tau, P_hat, k, bool(sigma_w))
        r['V4V5_band'] = {n: band_scan(tau[n], P_hat, b2) for n in tau}
        if sigma_w:
            P_w = em.p_white(sigma_w)
            r['V5_bound'] = {
                n: dict(bound=bounds.complete_data_bound(tau[n], P_w, P_gen),
                        **bounds.bound_stability(tau[n], P_w, P_gen, b2))
                for n in tau}
        r['V6_precision'] = v6_precision(maps, tau, b2)
        r['V7_sigma_mf'], _ = v7_sigma_mf(maps, tau)
        res[row] = r
        del maps

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    (out_dir / f'confusion_validation_{stamp}.json').write_text(
        json.dumps(res, indent=1, default=float))
    return res, out_dir / f'confusion_validation_{stamp}.json'


def report(res):
    L = []
    a = res['analytic']
    L.append(f"config: S_min={res['config']['s_lo']} S_cut={res['config']['s_cut']} mJy;"
             f" n_ens={res['n_ens']}")
    L.append(f"analytic: sigma_c={a['sigma_c']:.4f} mJy/beam  N_beam={a['n_beam']:.2f}"
             f"  skew={a['skew']:.4f}  exkurt={a['exkurt']:.4f}"
             f"  floor sigma_w={res['floor']:.4f} (f_N={res['f_N']})")
    for row in ('T2_CONFUSION', 'T2_CONFUSION_WN'):
        r = res[row]
        L.append(f"\n=== {row}  (dtype {r['dtype']}, <N_src>={r['n_src_mean']:.0f}) ===")
        if 'V1_one_point' in r:
            v = r['V1_one_point']
            L.append("  V1 Campbell one-point closure "
                     f"[{'PASS' if v['passed'] else 'FAIL'}]")
            for key in ('mean', 'rms', 'skew', 'exkurt'):
                L.append(f"       {key:8s} measured {v['measured'][key]:11.5f}"
                         f"   analytic {v['analytic'][key]:11.5f}"
                         f"   dev {100*v['rel_dev'][key]:6.3f} %")
        v = r['V2_two_point']
        L.append(f"  V2 P_hat/P_conf = {v['mean']:.5f} +/- {v['scatter']:.4f}"
                 f"  (periodogram noise {v['expected_scatter']:.4f})"
                 f" [{'PASS' if v['passed'] else 'FAIL'}]")
        v = r['V3_flatness']
        L.append(f"  V3 |tau~|^2/P relative spread: compact "
                 f"{v['compact']['rel_spread']:.4f}, extended "
                 f"{v['extended']['rel_spread']:.3f}"
                 f" [{'PASS' if v['passed'] else 'FAIL'}]")
        for tn in ('compact', 'extended'):
            b = r['V4V5_band'][tn]
            L.append(f"  V4 band scan, {tn}: sigma_MF drifts "
                     f"{100*b['sigma_mf_drift']:8.2f} % across all B^2 cuts "
                     f"({100*b['sigma_mf_drift_n8']:.2f} % across the N8 triple "
                     f"1e-4/1e-8/1e-14); sigma_MF*sqrt(N_modes) drifts "
                     f"{100*b['invariant_drift']:7.3f} %")
            L.append("        " + " ".join(
                f"[{x['cut']:.0e}: N={x['n_modes']:5d} s={x['sigma_mf']:.5g}]"
                for x in b['rows']))
        if 'V5_bound' in r:
            for tn, b in r['V5_bound'].items():
                L.append(f"  V5 complete-data bound, {tn}: {b['bound']:10.3f}"
                         f"   mask drift {100*b['max_rel_drift']:.3f} %"
                         f" [{'STABLE' if b['stable'] else 'UNSTABLE'}]")
        for tn, v in r['V6_precision'].items():
            L.append(f"  V6 precision, {tn}: sigma_MF float64 "
                     f"{v['sigma_mf']['float64']:.5g} vs float32 "
                     f"{v['sigma_mf']['float32']:.5g}  -> {100*v['rel_dev']:.2f} %")
        v = r['V7_sigma_mf']
        L.append(f"  V7 sigma_MF (split discipline) "
                 f"[{'PASS' if v['passed'] else 'FAIL'}]")
        for tn in ('compact', 'extended'):
            d = v[tn]
            L.append(f"       {tn:9s} predicted {d['predicted']:11.5f}"
                     f"  empirical {d['empirical']:11.5f}  ratio {d['ratio']:.4f}"
                     f"  = {d['n_sigma']:.2f} sigma (rel. s.e. "
                     f"{100*d['rel_stderr']:.2f} % from V8 exkurt {d['exkurt']:.2f},"
                     f" skew {d['skew']:.2f}, n_eval {d['n_eval']})")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--n-ens', type=int, default=4000)
    ap.add_argument('--out', default=str(_ROOT / 'results' / 'tests_and_results'))
    a = ap.parse_args()
    res, path = run(a.n_ens, a.out)
    txt = report(res)
    print(txt)
    Path(str(path).replace('.json', '.log')).write_text(txt + "\n")
    print(f"\nwrote {path}\n      {str(path).replace('.json', '.log')}")
    ok = all(res[r].get(kk, {}).get('passed', True)
             for r in ('T2_CONFUSION', 'T2_CONFUSION_WN')
             for kk in ('V1_one_point', 'V2_two_point', 'V3_flatness',
                        'V7_sigma_mf'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
