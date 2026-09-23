#!/usr/bin/env python
"""
eta_exact_confusion.py — the exactly-solvable confusion reference (T2_CONFUSION_ATM)
====================================================================================

`T2_CONFUSION_ATM` is confusion plus a BEAM-CORRELATED Gaussian companion at
rho = Var[companion]/Var[confusion] = 0.25 — the same 0.5 sigma_c of Gaussian
noise that `T2_CONFUSION_WN` puts in the map, differing only in whether it
shares the beam.  Because it does, P_tot = (1+rho) C |B|^2 stays exactly
proportional to |B|^2 and whitening returns

    x = (d + w) * const,    d = the marked-point lattice field, w white Gaussian

i.e. **i.i.d. pixels**.  Hence eta = Var[Y] I(Y) for the scalar Y = D + W: it is
finite (the Gaussian removes the atom that makes the floorless model diverge),
computable in closed form, and identical for every template.

WHAT THIS ROW IS FOR.  It is a calibration instrument for the FORECASTING layer,
not a physics case and not a statement about source extraction.  The campaign
currently has no test that asks whether the variational ladder can recover a
KNOWN eta > 1 through the full production path — N1/N2 only check eta = 1, and
N3 checks a known eta on 32^2 i.i.d. pixels with a single-pixel template, which
exercises none of the 128^2 whitening, split and template machinery.  This row
supplies exactly that, with three independent exact predictions:

    E1  eta = 3.569 at the campaign configuration (converged to 5e-5)
    E2  eta_compact = eta_extended, exactly (i.i.d. => J = I_1 Id; null test N4)
    E3  complete-data bound = (1 + rho)/rho = 5 exactly, for every template

It does NOT regularise the band — P is still proportional to |B|^2, so the
compact template's Fisher weight is still uniform and still precision-limited.
That is deliberate: the GAP between the measured and the exact eta is then a
calibrated measurement of what the float-rounding side channel costs, which is
otherwise only visible as a float32-vs-float64 difference with no ground truth.
Section D below predicts that gap from first principles.

Usage
-----
    python scripts/tools/eta_exact_confusion.py [--n-ens 6000] [--out DIR]

Writes confusion_exact_<date>.{json,log} to tests_and_results/.
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
from eta_pipeline import whitening, bounds, analytic1d as a1  # noqa: E402
from eta_pipeline import models as em                         # noqa: E402

SIZE, FWHM = 128, 5.0
SHAPE = (SIZE, SIZE)
NPIX = SIZE * SIZE
_fy = np.fft.fftfreq(SIZE).reshape(-1, 1)
_fx = np.fft.fftfreq(SIZE).reshape(1, -1)
K = np.hypot(_fx, _fy)
B2 = np.exp(-4.0 * np.pi ** 2 * (FWHM / 2.355) ** 2 * (_fx ** 2 + _fy ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-ens', type=int, default=6000)
    ap.add_argument('--out', default=str(_ROOT / 'results' / 'tests_and_results'))
    a = ap.parse_args()

    ARGS = em.CONFUSION_ARGS
    rho = em.CONFUSION_RHO_ATM
    marks, wts = nl.confusion.mark_distribution(**ARGS)
    lam_pix = nl.confusion.n_sources_mean(SHAPE, **ARGS) / NPIX
    sc = nl.confusion.confusion_stats(**ARGS)['sigma_c']
    P_gen = nl.confusion.confusion_psd(SHAPE, rho_beam=rho, **ARGS)
    tau = {'compact': nl.make_template(SIZE, 'point', beam_fwhm=FWHM,
                                       normalize='none'),
           'extended': nl.make_template(SIZE, 'beta', 7.0, FWHM,
                                        normalize='none'),
           'sub-beam': nl.make_template(SIZE, 'beta', 2.0, FWHM,
                                        normalize='none')}
    res = {'generated': datetime.now().isoformat(timespec='seconds'),
           'config': dict(ARGS), 'rho': rho, 'lambda_pix': lam_pix,
           'sigma_c': sc, 'n_ens': a.n_ens}
    L = [f"T2_CONFUSION_ATM   rho = {rho}   lambda_pix = {lam_pix:.5f}"
         f"   sigma_c = {sc:.4f} mJy/beam"]

    # --- E1: the exact eta, with its convergence ---------------------------------
    eta = a1.compound_poisson_eta_1d(marks, wts, lam_pix, rho)
    conv = {}
    for key, val in (('n', 1 << 19), ('n', 1 << 21), ('n_sd', 30.0),
                     ('p_floor', 1e-12)):
        conv[f'{key}={val}'] = a1.compound_poisson_eta_1d(
            marks, wts, lam_pix, rho, **{key: val})
    res['E1_eta_exact'] = eta
    res['E1_convergence'] = conv
    L.append(f"\nE1  EXACT eta = {eta:.6f}")
    L.append("    convergence: " + ", ".join(
        f"{k} -> {v:.6f} ({100*(v/eta-1):+.4f}%)" for k, v in conv.items()))

    # --- E3: the complete-data bound, exactly (1+rho)/rho for every template -----
    P_g = em.REGISTRY['T2_CONFUSION_ATM']['bound_bg']()
    L.append(f"\nE3  complete-data bound, predicted (1+rho)/rho = {(1+rho)/rho:.4f}")
    res['E3_bounds'] = {}
    for nm, t in tau.items():
        v = bounds.complete_data_bound(t, P_g, P_gen)
        st = bounds.bound_stability(t, P_g, P_gen, B2)
        res['E3_bounds'][nm] = dict(bound=v, drift=st['max_rel_drift'],
                                    stable=st['stable'])
        L.append(f"    {nm:9s} {v:10.5f}   dev {100*(v/((1+rho)/rho)-1):+7.4f} %"
                 f"   mask drift {100*st['max_rel_drift']:.4f} %")
    L.append(f"    eta / bound = {eta/((1+rho)/rho):.4f}  "
             f"(the model uses {100*eta/((1+rho)/rho):.0f}% of its envelope)")

    # --- the ensemble: is the whitened field really (d + w) x const? -------------
    gen = em.REGISTRY['T2_CONFUSION_ATM']['gen']
    maps, lat = gen(a.n_ens, em.model_seed('T2_CONFUSION_ATM'))
    L.append(f"\nA   ensemble: {a.n_ens} maps, dtype {maps.dtype}, "
             f"<N_src> {lat['n_src'].mean():.0f}")
    L.append(f"    map rms {maps.std():.4f}  vs  sigma_c sqrt(1+rho) = "
             f"{sc*np.sqrt(1+rho):.4f}   ({100*(maps.std()/(sc*np.sqrt(1+rho))-1):+.3f} %)")

    #  Two whitenings, measuring two different things.
    #  (i) by the ANALYTIC P the whitened field SHOULD be (d + w) x const
    #      exactly.  It is not, and spectacularly so: P ~ |B|^2 spans 39 decades,
    #      so the 1/B inversion (up to 1e19) turns the map's own rounding floor
    #      into the dominant content of the corner modes and nothing normalises
    #      it back down.  The sharpest statement of the band pathology anywhere
    #      in the campaign -- and NOT what the pipeline does.
    #  (ii) by the ESTIMATED P-hat (what the pipeline does) the contaminated
    #      modes are normalised back to unit variance, so the cost is not blow-up
    #      but DILUTION: those modes carry rounding instead of signal.  Section D
    #      measures that from the standardised cumulants.
    seed = em.model_seed('T2_CONFUSION_ATM')
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    draw = nl.confusion.mark_sampler(**ARGS)
    lam = nl.confusion.n_sources_mean(SHAPE, **ARGS)
    sig0 = nl.confusion.beam_corr_sigma(SHAPE, rho, **ARGS)
    ptg = 2 * np.pi * (FWHM / 2.355) ** 2
    nsrc = int(rng.poisson(lam)); pre = np.zeros(SHAPE)
    iy, ix = rng.integers(0, SIZE, nsrc), rng.integers(0, SIZE, nsrc)
    np.add.at(pre, (iy, ix), draw(rng, nsrc) * ptg)
    pre = pre + rng.normal(0.0, sig0, SHAPE)
    x_an = whitening.whiten_maps(maps[:1], P_gen)[0]
    x_exact = pre * np.sqrt(NPIX / P_gen[0, 0])
    eps_an = float(np.var(x_an - x_exact) / np.var(x_an))
    lp = K < 0.30
    xa = np.fft.ifft2(np.where(lp, np.fft.fft2(x_an), 0)).real
    xe = np.fft.ifft2(np.where(lp, np.fft.fft2(x_exact), 0)).real
    eps_lo = float(np.var(xa - xe) / np.var(xe))
    res['A_whiten_by_analytic_P'] = dict(full_band=eps_an, k_below_0p30=eps_lo)
    L.append("    whitened by the ANALYTIC P, against the exact (d + w) x const:")
    L.append(f"      spurious variance fraction, full band   {eps_an:.4f}")
    L.append(f"      the same, restricted to k < 0.30        {eps_lo:.3e}")
    L.append("      -> exact where the beam has not underflowed, destroyed above.")

    # --- E2: template independence, three ways ----------------------------------
    P_hat = whitening.estimate_psd2d(maps[whitening.make_splits(a.n_ens)['psd']])
    L.append("\nE2  template independence (exact prediction: all equal)")
    L.append(f"    {'template':10s} {'sigma_MF':>12s} {'||k3(t^)||^2':>14s}"
             f" {'||k4(t^)||^2':>14s}")
    u = (ptg ** 2 * B2) / P_gen
    U = np.fft.fft2(u)
    uu, uuu = np.real(np.fft.ifft2(U ** 2)), np.real(np.fft.ifft2(U ** 3))
    res['E2'] = {}
    for nm, t in tau.items():
        t_hat, smf = whitening.whitened_template(t, P_hat)
        w = np.abs(np.fft.fft2(t_hat)) ** 2 / NPIX
        k3, k4 = float((u * w * uu).sum()), float((u * w * uuu).sum())
        res['E2'][nm] = dict(sigma_mf=float(smf), k3=k3, k4=k4)
        L.append(f"    {nm:10s} {smf:12.5f} {k3:14.6e} {k4:14.6e}")
    k3s = [res['E2'][n]['k3'] for n in tau]
    L.append(f"    spread of ||k3||^2 across templates: "
             f"{100*(max(k3s)/min(k3s)-1):.6f} %   (exact prediction: 0)")
    L.append("    sigma_MF differs between templates -- it must; eta does not.")

    # --- the standardised cumulants of the whitened field ------------------------
    m1 = float((wts * marks).sum() / wts.sum())
    mk = [float((wts * marks ** j).sum() / wts.sum()) for j in range(1, 5)]
    k2 = lam_pix * mk[1] * (1 + rho)
    sk = lam_pix * mk[2] / k2 ** 1.5
    ek = lam_pix * mk[3] / k2 ** 2
    xs = whitening.whiten_maps(maps[:2000], P_hat)
    z = (xs - xs.mean()) / xs.std()
    L.append(f"\nB   whitened marginal, measured vs the exact Y = D + W cumulants")
    L.append(f"    skewness        {float((z**3).mean()):10.5f}  exact {sk:10.5f}")
    L.append(f"    excess kurtosis {float((z**4).mean()-3):10.5f}  exact {ek:10.5f}")
    sk_m, ek_m = float((z ** 3).mean()), float((z ** 4).mean() - 3)
    res['B_cumulants'] = dict(skew_measured=sk_m, skew_exact=sk,
                              exkurt_measured=ek_m, exkurt_exact=ek)

    # --- D: what the rounding channel costs -------------------------------------
    #  If a fraction eps of the whitened unit variance is independent Gaussian,
    #  the standardised cumulants scale as (1-eps)^{3/2} and (1-eps)^2.  Two
    #  independent routes to the same eps is the check that "extra independent
    #  Gaussian" is the right description of what the rounding does.
    eps3 = 1.0 - (sk_m / sk) ** (2.0 / 3.0)
    eps4 = 1.0 - (ek_m / ek) ** 0.5
    eps = 0.5 * (eps3 + eps4)
    rho_eff = rho + (eps / (1.0 - eps)) * (1.0 + rho)
    eta_eff = a1.compound_poisson_eta_1d(marks, wts, lam_pix, rho_eff)
    res['D'] = dict(eps_from_skew=eps3, eps_from_kurtosis=eps4, eps=eps,
                    rho_eff=rho_eff, eta_expected_from_pipeline=eta_eff)
    L.append("")
    L.append("D   what the rounding channel costs, from the dilution it causes")
    L.append(f"    eps from the skewness, (1-eps)^1.5 : {eps3:.4f}")
    L.append(f"    eps from the kurtosis, (1-eps)^2   : {eps4:.4f}")
    L.append(f"    the two agree to {100*abs(eps3-eps4)/eps:.1f} %  ->  eps = {eps:.4f}")
    L.append(f"    rho_eff = rho + eps/(1-eps)(1+rho) = {rho_eff:.5f}")
    L.append(f"    => the LADDER should recover eta = {eta_eff:.4f}, not {eta:.4f}"
             f"   (a {100*(1-eta_eff/eta):.1f} % shortfall)")
    L.append("    Landing on the FIRST number would mean the rounding channel is")
    L.append("    harmless here; on the SECOND, that this dilution model is right")
    L.append("    and the channel costs exactly that much.  Both are informative,")
    L.append("    which is what a calibrated reference is for.")

    txt = "\n".join(L)
    print(txt)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    (out / f'confusion_exact_{stamp}.json').write_text(
        json.dumps(res, indent=1, default=float))
    (out / f'confusion_exact_{stamp}.log').write_text(txt + "\n")
    print(f"\nwrote {out / f'confusion_exact_{stamp}.json'}")


if __name__ == '__main__':
    main()
