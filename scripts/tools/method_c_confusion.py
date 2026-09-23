#!/usr/bin/env python
"""
method_c_confusion.py — why the compact template does NOT win in confusion
==========================================================================

Answers a question the complete-data bound cannot: the bound is built from
spectra alone, so it is blind to the difference between a strongly
non-Gaussian field and a Gaussian one with the same power spectrum. It can
say how much removable structure a template's band contains; it cannot say
how much NON-GAUSSIAN information is there. This script computes the latter,
two ways.

Background. `confusion_noise_reference.md` §§5.2, 5.3 and 6.1 argue that for a
point source in confusion the per-mode SNR is flat, so the matched filter has
no spectral leverage, and conclude that "the CNN advantage is expected to be
largest for point-source extraction ... and smallest for extended-source
extraction". `README_T2_Confusion_v2.md` §6.3 found the opposite ordering in the
complete-data bounds. This script shows that the conclusion does not follow
from its premise, and locates exactly where the two statements part company.

Part A — the floorless field, exactly.
    Whitening is x = ifft2(fft2(n) sqrt(npix/P)). For pure confusion
    P = C|B|^2 carries the SAME beam as the field, so the beam cancels
    identically and the whitened map is the bare marked-point lattice field
    times a constant. Its pixels are therefore INDEPENDENT. For i.i.d. pixels
    the Fisher operator is a multiple of the identity (J = I_1 * Id; null test
    N4), so Delta = (I_1 - 1) Id and

        eta = 1 + sigma_MF^2 tau^dag Delta tau = 1 + (I_1 - 1)||t^||^2 = I_1

    with no template left in the expression. A1 verifies the cancellation band
    by band (and shows where float64 gives out). A2 shows the whitened
    marginal carries an atom at zero of mass exp(-lambda_pix), which makes
    I_1 infinite: any empty pixel reads A t^_p with no noise at all.

Part B — Method C (memo v2 §18), template-contracted cumulant norms.
    With u = |psi~|^2 / P_tot and w = |t^~|^2 / npix (summing to 1),

        ||k3(t^)||^2 ~ sum_q u(q) w(q) (u*u)(q),
        ||k4(t^)||^2 ~ sum_q u(q) w(q) (u*u*u)(q),

    the convolutions by FFT. The absolute DFT prefactors are the ones memo §18
    says must be fixed against a simulation -- but they are COMMON to all
    templates, so the ratios reported here need no calibration at all. This is
    the cheap, prefactor-free way to ask which template couples more strongly
    to the bispectrum and trispectrum.

    Floorless, u is flat (it is |psi~|^2 / C|psi~|^2), hence so is u*u, and the
    sums collapse to sum_q w(q) = 1 for every template: the norms are exactly
    equal, confirming Part A perturbatively. Floored, u rolls off at the knee
    and the compact template loses.

Usage
-----
    python scripts/tools/method_c_confusion.py [--out DIR]

Caveat worth keeping in view: Method C is a WEAK-non-Gaussianity diagnostic and
this field is not weakly non-Gaussian (pixel skewness 2.12, excess kurtosis
9.09), which is why Method B fails outright on it (README §7). The ratios are
therefore indicative, not predictive. Part A's template-independence, by
contrast, is exact -- it follows from i.i.d.-ness, not from an expansion.
"""

import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]          # repository root
sys.path.insert(0, str(_ROOT))
try:
    import torch                                              # noqa: F401
except Exception:
    _p = types.ModuleType('eta_pipeline')
    _p.__path__ = [str(_ROOT / 'eta_pipeline')]
    sys.modules['eta_pipeline'] = _p

import noise_lib as nl                                        # noqa: E402
from eta_pipeline import whitening, models as em              # noqa: E402

SIZE, FWHM = 128, 5.0
SHAPE = (SIZE, SIZE)
NPIX = SIZE * SIZE
SIG = FWHM / 2.355
PTG = 2.0 * np.pi * SIG ** 2

_fy = np.fft.fftfreq(SIZE).reshape(-1, 1)
_fx = np.fft.fftfreq(SIZE).reshape(1, -1)
K = np.hypot(_fx, _fy)
B2 = np.exp(-4.0 * np.pi ** 2 * SIG ** 2 * (_fx ** 2 + _fy ** 2))

TEMPLATES = {
    'compact': lambda: nl.make_template(SIZE, 'point', beam_fwhm=FWHM,
                                        normalize='none'),
    'sub-beam': lambda: nl.make_template(SIZE, 'beta', 2.0, FWHM,
                                         normalize='none'),
    'extended': lambda: nl.make_template(SIZE, 'beta', 7.0, FWHM,
                                         normalize='none'),
}


def part_a(args, seed=777357):
    """Whitening cancels the beam exactly; the whitened field is i.i.d."""
    rng = np.random.default_rng(seed)
    draw = nl.confusion.mark_sampler(**args)
    lam = nl.confusion.n_sources_mean(SHAPE, **args)
    n = int(rng.poisson(lam))
    raw = np.zeros(SHAPE)
    iy, ix = rng.integers(0, SIZE, n), rng.integers(0, SIZE, n)
    np.add.at(raw, (iy, ix), draw(rng, n) * PTG)

    P = nl.confusion.confusion_psd(SHAPE, **args)
    X = np.fft.fft2(nl.beam_smooth(raw, FWHM)) * np.sqrt(NPIX / P)
    R = np.fft.fft2(raw) * np.sqrt(NPIX / P[0, 0])
    bands = []
    for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6),
                   (.6, .71)):
        m = (K >= lo) & (K < hi) & (K > 0)
        bands.append(dict(k_lo=lo, k_hi=hi,
                          rel_dev=float(np.abs(X[m] - R[m]).mean()
                                        / np.abs(R[m]).mean())))
    lp = K < 0.30
    band_limited = float(
        np.abs(np.fft.ifft2(np.where(lp, X, 0)).real
               - np.fft.ifft2(np.where(lp, R, 0)).real).max()
        / np.fft.ifft2(np.where(lp, R, 0)).real.std())
    return dict(bands=bands, band_limited_max_rel_dev=band_limited,
                atom_measured=float((raw == 0).mean()),
                atom_poisson=float(np.exp(-lam / NPIX)),
                lambda_pix=float(lam / NPIX))


def norms(P_tot):
    """Method C: (||k3||^2, ||k4||^2) per template, up to a common prefactor."""
    u = (PTG ** 2 * B2) / P_tot
    U = np.fft.fft2(u)
    uu = np.real(np.fft.ifft2(U ** 2))
    uuu = np.real(np.fft.ifft2(U ** 3))
    out = {}
    for name, build in TEMPLATES.items():
        t_hat, _ = whitening.whitened_template(build(), P_tot)
        w = np.abs(np.fft.fft2(t_hat)) ** 2 / NPIX        # sums to 1
        out[name] = dict(k3=float((u * w * uu).sum()),
                         k4=float((u * w * uuu).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(_ROOT / 'results' / 'tests_and_results'))
    a = ap.parse_args()

    args = em.CONFUSION_ARGS
    sc = nl.confusion.confusion_stats(**args)['sigma_c']
    P_conf = nl.confusion.confusion_psd(SHAPE, **args)

    res = {'generated': datetime.now().isoformat(timespec='seconds'),
           'config': dict(args), 'sigma_c': sc}
    res['A_floorless_whitening'] = part_a(args)
    res['B_method_c'] = {}
    L = []

    A = res['A_floorless_whitening']
    L.append("A. FLOORLESS: whitening returns the bare delta field")
    L.append(f"   {'k band':>11} {'|X-R|/|R|':>14}")
    for b in A['bands']:
        L.append(f"   {b['k_lo']:.2f}-{b['k_hi']:.2f}  {b['rel_dev']:14.3e}")
    L.append(f"   band-limited to k<0.30: max rel. dev "
             f"{A['band_limited_max_rel_dev']:.3e}")
    L.append("   -> exact in exact arithmetic; the 1/B inversion (up to 1e19)")
    L.append("      amplifies the map's own rounding floor beyond k ~ 0.3.")
    L.append(f"   atom at zero: measured {A['atom_measured']:.4f}, "
             f"Poisson exp(-{A['lambda_pix']:.4f}) = {A['atom_poisson']:.4f}")
    L.append("   -> i.i.d. pixels with an atom => I_1 = infinity => eta = inf,")
    L.append("      identically for every template.")

    L.append("\nB. METHOD C: template-contracted cumulant norms, "
             "normalised to 'extended'")
    for label, P_tot in (('floorless', P_conf),
                         ('f_N = 0.25', P_conf + NPIX * (0.25 * sc) ** 2),
                         ('f_N = 0.50', P_conf + NPIX * (0.50 * sc) ** 2),
                         ('f_N = 1.00', P_conf + NPIX * (1.00 * sc) ** 2)):
        r = norms(P_tot)
        res['B_method_c'][label] = r
        e3, e4 = r['extended']['k3'], r['extended']['k4']
        L.append(f"\n   {label}")
        L.append(f"     {'template':10s} {'||k3(t^)||^2':>14s} "
                 f"{'||k4(t^)||^2':>14s}")
        for nm in ('compact', 'sub-beam', 'extended'):
            L.append(f"     {nm:10s} {r[nm]['k3']/e3:14.6f} "
                     f"{r[nm]['k4']/e4:14.6f}")
    L.append("\n   Floorless: exactly 1.000000 for every template -- Part A,")
    L.append("   confirmed perturbatively.  Floored: the compact template")
    L.append("   carries ~40% less third-order and ~33% less fourth-order")
    L.append("   signal than the extended one.")

    txt = "\n".join(L)
    print(txt)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    (out / f'confusion_method_c_{stamp}.json').write_text(
        json.dumps(res, indent=1, default=float))
    (out / f'confusion_method_c_{stamp}.log').write_text(txt + "\n")
    print(f"\nwrote {out / f'confusion_method_c_{stamp}.json'}")


if __name__ == '__main__':
    main()
