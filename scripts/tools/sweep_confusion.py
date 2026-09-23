#!/usr/bin/env python
"""
sweep_confusion.py — analytic severity sweeps for the T2_CONFUSION rung
=======================================================================

The sweeps that memo v2 Sec. 21.4 asks for -- "S_cut/sigma_conf (the pipeline-
depth axis) and template extension r_c (two beta sizes, one smaller and one
larger than the beam)" -- plus the instrument-noise axis f_N that the band
analysis forced on us.

Everything here is ANALYTIC.  The confusion field's generative spectrum is
known in closed form, P_conf = N<a^2>|B|^2 (``noise_lib.confusion.confusion_psd``,
verified against 8000 simulated maps to 0.07% in tools/validate_confusion.py),
so sigma_MF, the Fisher band and the complete-data bound follow without
generating a single ensemble.  A sweep that would cost hours of Monte Carlo for
any other Tier-2 model costs a second here -- which is itself an argument for
this model as the paper's severity-sweep exemplar.

Four sweeps:

  S1  Pipeline depth.  S_cut = 25...200 mJy at fixed S_min.  Bright-source
      removal truncates the mark distribution, so it moves the higher
      cumulants strongly and sigma_c hardly at all.

  S2  CLT depth.  S_min = 0.1...6 mJy at fixed S_cut.  This is the axis that
      moves N_beam (28 -> 0.3) and therefore the non-Gaussianity, while
      leaving sigma_c nearly fixed -- the two severity knobs are close to
      orthogonal in sigma_c, which is a convenient accident of the Schechter's
      faint-end convergence.

  S3  Instrument noise.  f_N = sigma_N/sigma_c = 0.05...2.  The axis that
      decides whether the model is well posed at all, and the one that
      answers a specific question left open by null test N4b: that test notes
      that the "compact template wins" corner of the overlap law "requires a
      background that shares the event spectrum (pure confusion)".  Confusion
      shares it exactly -- until an unsmoothed white floor is added, which is
      the only thing that makes the compact band converge.  S3 measures the
      price: the bound ratio compact/extended as a function of f_N.

  S4  Template extension.  r_c = 1...14 px against a 5 px beam, i.e. the two
      beta sizes of the task card and then some, at the baseline severity.

Usage
-----
    python scripts/tools/sweep_confusion.py [--f-n 0.5] [--out DIR]

Writes ``confusion_sweeps_<date>.json`` and ``.log`` to tests_and_results/.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]          # repository root
sys.path.insert(0, str(_ROOT))

try:
    import torch                                              # noqa: F401
except Exception:                                             # see validate_confusion
    import types
    _pkg = types.ModuleType('eta_pipeline')
    _pkg.__path__ = [str(_ROOT / 'eta_pipeline')]
    sys.modules['eta_pipeline'] = _pkg

import noise_lib as nl                                        # noqa: E402
from eta_pipeline import bounds, whitening                    # noqa: E402

SIZE, FWHM = 128, 5.0
SHAPE = (SIZE, SIZE)
NPIX = SIZE * SIZE
SIG_PIX = FWHM / 2.355


def grids():
    fy = np.fft.fftfreq(SIZE).reshape(-1, 1)
    fx = np.fft.fftfreq(SIZE).reshape(1, -1)
    b2 = np.exp(-4.0 * np.pi ** 2 * SIG_PIX ** 2 * (fx ** 2 + fy ** 2))
    return np.hypot(fx, fy), b2


K, B2 = grids()


def template(kind, rc=7.0):
    if kind == 'compact':
        return nl.make_template(SIZE, 'point', beam_fwhm=FWHM, normalize='none')
    return nl.make_template(SIZE, 'beta', rc, FWHM, normalize='none')


def cell(tau, s_lo, s_cut, f_n):
    """All analytic quantities for one (severity, template, floor) cell."""
    st = nl.confusion.confusion_stats(s_lo=s_lo, s_cut=s_cut)
    P_c = nl.confusion.confusion_psd(SHAPE, s_lo=s_lo, s_cut=s_cut)
    sw = f_n * st['sigma_c']
    P_w = np.full(SHAPE, NPIX * sw ** 2) if sw > 0 else None
    P_t = P_c + (P_w if P_w is not None else 0.0)

    w = np.abs(np.fft.fft2(tau)) ** 2 / P_t
    w[0, 0] = 0.0
    wn = w / w.sum()
    logk = float(np.exp((wn * np.log(np.maximum(K, 1e-6))).sum()))
    out = dict(sigma_c=st['sigma_c'], n_beam=st['n_beam'], skew=st['skew'],
               exkurt=st['exkurt'], sigma_w=float(sw),
               sigma_mf=float(1.0 / np.sqrt(w.sum())), k_band=logk,
               w_above_mask=float(wn[K > 0.321].sum()))
    if sw > 0:
        out['k_knee'] = float(np.sqrt(
            max(np.log(P_c[0, 0] / (NPIX * sw ** 2)), 0.0)
            / (4.0 * np.pi ** 2 * SIG_PIX ** 2)))
        out['bound'] = bounds.complete_data_bound(tau, P_w, P_t)
        stab = bounds.bound_stability(tau, P_w, P_t, B2)
        out['bound_drift'] = stab['max_rel_drift']
        out['bound_stable'] = stab['stable']
    return out


def sweep(name, axis, values, f_n, s_lo=None, s_cut=None, rc=7.0):
    rows = []
    for v in values:
        kw = dict(s_lo=s_lo if s_lo is not None else nl.confusion.S_MIN,
                  s_cut=s_cut if s_cut is not None else nl.confusion.S_CUT,
                  f_n=f_n)
        this_rc = rc
        if axis == 's_cut':
            kw['s_cut'] = v
        elif axis == 's_lo':
            kw['s_lo'] = v
        elif axis == 'f_n':
            kw['f_n'] = v
        elif axis == 'rc':
            this_rc = v
        r = {axis: v}
        for tn in ('compact', 'extended'):
            r[tn] = cell(template(tn, this_rc), **kw)
        if 'bound' in r['compact'] and 'bound' in r['extended']:
            r['bound_ratio_compact_over_extended'] = (
                r['compact']['bound'] / r['extended']['bound'])
        rows.append(r)
    return dict(name=name, axis=axis, rows=rows)


def fmt(sw):
    L = [f"\n### {sw['name']}  (axis: {sw['axis']})"]
    head = (f"  {sw['axis']:>8} | {'sigma_c':>8} {'N_beam':>8} {'skew':>6} "
            f"{'exkurt':>7} | {'sMF_cmp':>9} {'band_cmp':>8} {'bnd_cmp':>9} | "
            f"{'sMF_ext':>8} {'band_ext':>8} {'bnd_ext':>9} | {'cmp/ext':>7}")
    L.append(head); L.append("  " + "-" * (len(head) - 2))
    for r in sw['rows']:
        c, e = r['compact'], r['extended']
        v = r[sw['axis']]
        L.append(
            f"  {v:8.3g} | {c['sigma_c']:8.3f} {c['n_beam']:8.2f} "
            f"{c['skew']:6.3f} {c['exkurt']:7.3f} | {c['sigma_mf']:9.3f} "
            f"{c['k_band']:8.4f} {c.get('bound', float('nan')):9.3f} | "
            f"{e['sigma_mf']:8.3f} {e['k_band']:8.4f} "
            f"{e.get('bound', float('nan')):9.3f} | "
            f"{r.get('bound_ratio_compact_over_extended', float('nan')):7.3f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--f-n', type=float, default=0.5)
    ap.add_argument('--out', default=str(_ROOT / 'results' / 'tests_and_results'))
    a = ap.parse_args()

    res = {'generated': datetime.now().isoformat(timespec='seconds'),
           'f_n_baseline': a.f_n, 'geometry': dict(
               size=SIZE, beam_fwhm_pix=FWHM,
               pix_arcsec=nl.confusion.PIX_ARCSEC,
               beam_fwhm_arcsec=nl.confusion.BEAM_FWHM_ARCSEC)}
    res['S1_pipeline_depth'] = sweep(
        'S1  pipeline depth: bright-source removal threshold S_cut [mJy]',
        's_cut', [25.0, 50.0, 100.0, 200.0, 400.0], a.f_n)
    res['S2_clt_depth'] = sweep(
        'S2  CLT depth: faint-end limit S_min [mJy] (moves N_beam)',
        's_lo', [0.03, 0.1, 0.3, 1.0, 3.0, 6.0], a.f_n)
    res['S3_instrument_noise'] = sweep(
        'S3  instrument noise: f_N = sigma_N / sigma_c',
        'f_n', [0.05, 0.1, 0.25, 0.5, 1.0, 2.0], a.f_n)
    res['S4_template_extension'] = sweep(
        'S4  template extension: beta core radius r_c [px] vs a 5 px beam',
        'rc', [1.0, 2.0, 3.5, 5.0, 7.0, 10.0, 14.0], a.f_n)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    (out / f'confusion_sweeps_{stamp}.json').write_text(
        json.dumps(res, indent=1, default=float))
    txt = ("confusion severity sweeps -- all analytic, from "
           "noise_lib.confusion.confusion_psd\n"
           f"baseline: S_min={nl.confusion.S_MIN} mJy, "
           f"S_cut={nl.confusion.S_CUT} mJy, f_N={a.f_n}, "
           f"{SIZE}^2 at {nl.confusion.PIX_ARCSEC}\"/px, "
           f"beam {nl.confusion.BEAM_FWHM_ARCSEC}\" = {FWHM} px\n"
           "'band' is exp<log k>_f, the Fisher-band centroid in cyc/px; "
           "'bnd' is the complete-data bound.\n"
           "In S4 the 'compact' columns are r_c-independent by construction "
           "and are repeated as a reference.\n"
           "\nWHY THE BOUND AND THE BAND ARE CONSTANT DOWN S1 AND S2.  Not a "
           "bug: it is the\ncleanest structural property this model has.  The "
           "confusion spectrum is\nP_conf = N<a^2>|B|^2 with amplitude "
           "proportional to sigma_c^2, and the floor is\nquoted as sigma_w = "
           "f_N sigma_c, so BOTH spectra carry the same sigma_c^2 and the\n"
           "counts model cancels out of every second-order quantity.  Changing "
           "S_cut or\nS_min therefore moves the NON-GAUSSIANITY (S1: skewness "
           "1.31 -> 2.18, excess\nkurtosis 2.76 -> 10.14; S2: N_beam 84 -> "
           "0.33) while leaving sigma_MF/sigma_c,\nthe Fisher band and the "
           "complete-data bound exactly invariant.  The two axes of\nthe "
           "memo's two-regime law are thus exactly orthogonal here -- 'how much "
           "room'\nis a pure function of f_N and the template, 'how much of it "
           "is reachable' is a\npure function of the counts truncation.  No "
           "other model in the campaign\nseparates them.")
    for key in ('S1_pipeline_depth', 'S2_clt_depth', 'S3_instrument_noise',
                'S4_template_extension'):
        txt += "\n" + fmt(res[key])
    print(txt)
    (out / f'confusion_sweeps_{stamp}.log').write_text(txt + "\n")
    print(f"\nwrote {out / f'confusion_sweeps_{stamp}.json'}")


if __name__ == '__main__':
    main()
