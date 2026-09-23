#!/usr/bin/env python
"""
diag_rounding_band.py — is float32 mantissa rounding a side channel in the floorless models?
=============================================================================================

Hypothesis (2026-09-03, from the GPU-cluster re-run).  On the FLOORLESS T2_PCA and
T2_MEDIAN models the CNN rung finds a large, restart-stable VAL signal
(PCA: 5/6 restarts at 2.3-2.7; MEDIAN: 5/6 at 2.0-2.1) that vanishes
completely under a white floor of only sigma_w = 0.10 (10^-3 of the map
variance).  Every physical component of these maps is beam-suppressed at
high k by up to e^-40, so beyond |k| ~ 0.35 cyc/px the stored float32 map
contains nothing but its own mantissa rounding: absolute level ~ |map| x 2^-24,
white in k, and HETEROSCEDASTIC -- its local amplitude tracks the local
map envelope.  The data-estimated P-hat includes this band, so whitening
amplifies it to unit variance; a nonlinear test function can then read the
per-image artifact envelope from a channel in which no Gaussian model noise
competes with it.  A floor of sigma_w = 0.1 (P_w = n_pix x 0.01 = 164) buries
it.  This is the third member of the "unmodelled channel with anomalous
signal-to-noise" family (DC zeroing; missing floor in the Fisher band).

What this script measures (CPU only, ~2 min on the M3):
  1. For n maps of each floorless model, the map in float64 and its float32
     cast; the rounding residual r = float32(m) - m; radial spectra of m and
     r, and the crossover k above which P_r > P_m.
  2. Per-image power in the high-k band (|k| > K_HI) of the float32 maps vs
     (a) the per-image artifact energy (PCA: sum art^2; MEDIAN: map variance)
     and (b) the map RMS -- Pearson/Spearman correlations.  A correlation
     near 1 means the rounding band is an envelope readout.
  3. The same band-power correlation for the corresponding *_WN model (the
     control: with a floor it should be ~0).
  4. Fraction of the WHITENED variance (P-hat from float32 maps) lying in
     the rounding band -- what the CNN actually sees.

Writes eta_results/r3_rounding_band.json and prints a summary.
Usage:  python scripts/tools/diag_rounding_band.py [--n 600]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
for _p in (_here, *_here.parents):
    if (_p / 'eta_pipeline').is_dir():
        sys.path.insert(0, str(_p)); break
import noise_lib as nl                                   # noqa: E402
import eta_pipeline as ep                                # noqa: E402
from eta_pipeline import models as em                    # noqa: E402

OUT = _p / 'results' / 'eta_results'
K_HI = 0.35
SHAPE = em.SHAPE
B2 = em.beam2()


def gen64(name, n, seed):
    """Regenerate a floorless model in float64 (same seeds/draw order as the campaign)."""
    if name == 'T2_PCA':
        P = nl.psd_isotropic_powerlaw(SHAPE, *em.P_ISO_ARGS)
        maps, arts = np.empty((n, *SHAPE)), np.empty((n, *SHAPE))
        for i, rng in enumerate(em._child_rngs(seed, n)):
            m = nl.beam_smooth(nl.grf_from_psd(P, rng), em.BEAM_FWHM)
            art, _, _ = nl.pca_leaked_modes(SHAPE, rng=rng, **em.PCA_ARGS)
            maps[i], arts[i] = m + art, art
        return maps, (arts**2).sum(axis=(-2, -1))
    if name == 'T2_MEDIAN':
        P = nl.psd_two_component(SHAPE, *em.P_ANISO_ARGS)
        maps = np.empty((n, *SHAPE))
        for i, rng in enumerate(em._child_rngs(seed, n)):
            m = nl.beam_smooth(nl.grf_from_psd(P, rng), em.BEAM_FWHM)
            maps[i] = nl.row_median_removal(m, axis=1)[0]
        return maps, maps.var(axis=(-2, -1))
    raise ValueError(name)


def radial(P2d, nbins=40):
    fy = np.fft.fftfreq(SHAPE[0]).reshape(-1, 1); fx = np.fft.fftfreq(SHAPE[1]).reshape(1, -1)
    k = np.sqrt(fx**2 + fy**2).ravel()
    edges = np.linspace(0, 0.5 * np.sqrt(2), nbins + 1)
    idx = np.clip(np.digitize(k, edges) - 1, 0, nbins - 1)
    prof = np.bincount(idx, weights=P2d.ravel(), minlength=nbins) / np.maximum(np.bincount(idx, minlength=nbins), 1)
    return 0.5 * (edges[1:] + edges[:-1]), prof


def band_power(maps, kmask):
    F = np.fft.fft2(maps, axes=(-2, -1))
    return (np.abs(F)**2 * kmask).sum(axis=(-2, -1))


def corr(a, b):
    from scipy.stats import spearmanr
    return float(np.corrcoef(a, b)[0, 1]), float(spearmanr(a, b).correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=600)
    args = ap.parse_args()
    fy = np.fft.fftfreq(SHAPE[0]).reshape(-1, 1); fx = np.fft.fftfreq(SHAPE[1]).reshape(1, -1)
    kk = np.sqrt(fx**2 + fy**2)
    hi = kk > K_HI
    res = {'n': args.n, 'k_hi': K_HI}
    for base, wn in (('T2_PCA', 'T2_PCA_WN'), ('T2_MEDIAN', 'T2_MEDIAN_WN')):
        seed = em.model_seed(base)
        m64, energy = gen64(base, args.n, seed)
        m32 = m64.astype(np.float32).astype(np.float64)
        r = m32 - m64                                           # exact rounding residual
        P_m = np.mean(np.abs(np.fft.fft2(m64, axes=(-2, -1)))**2, axis=0)
        P_r = np.mean(np.abs(np.fft.fft2(r, axes=(-2, -1)))**2, axis=0)
        P_32 = np.mean(np.abs(np.fft.fft2(m32, axes=(-2, -1)))**2, axis=0)
        kc, pm = radial(P_m); _, pr = radial(P_r)
        cross = kc[np.argmax(pr > pm)] if np.any(pr > pm) else None
        # per-MODE criterion (the radial average is misleading when a single line of
        # modes -- MEDIAN's k_x = 0 row-offset line -- dominates an annulus): fraction
        # of all Fourier modes whose mean float32-rounding power exceeds the physical
        # power, overall and restricted to |k| > K_HI
        dom = P_r > P_m
        frac_modes_dom = float(dom.mean()); frac_modes_dom_hi = float(dom[hi].mean())
        # float64 rounding (relative 2^-53) for the same maps, by scaling
        P_r64 = P_r * (2.0**-53 / 2.0**-24)**2
        frac_modes_dom64 = float((P_r64 > P_m).mean())
        # per-image band powers
        bp32 = band_power(m32, hi); bp64 = band_power(m64, hi); bpr = band_power(r, hi)
        rms = m64.std(axis=(-2, -1))
        c_e32 = corr(np.log(bp32), np.log(energy + 1e-30)); c_r32 = corr(np.log(bp32), np.log(rms))
        c_e64 = corr(np.log(bp64 + 1e-300), np.log(energy + 1e-30))
        # whitened-variance fraction in the rounding band (P-hat from the float32 maps)
        frac_w = float(hi.sum() / hi.size)                      # by construction, whitening equalises modes
        frac_r_in_32 = float((P_r * hi).sum() / (P_32 * hi).sum())
        # control: the floored model
        mw, _ = em.REGISTRY[wn]['gen'](args.n, seed)            # paired with m32 (same seed)
        mw = mw.astype(np.float64)
        bpw = band_power(mw, hi)
        c_ew = corr(np.log(bpw), np.log(energy + 1e-30)); c_rw = corr(np.log(bpw), np.log(rms))
        P_w = np.mean(np.abs(np.fft.fft2(mw, axes=(-2, -1)))**2, axis=0)
        res[base] = {
            'rounding_rms_per_pixel': float(r.std()), 'map_rms': float(m64.std()),
            'crossover_k_rounding_exceeds_physical': (float(cross) if cross is not None else None),
            'radial_k': kc.tolist(), 'P_physical': pm.tolist(), 'P_rounding': pr.tolist(),
            'hi_band_mode_fraction': frac_w,
            'frac_modes_rounding_dominated_float32': frac_modes_dom,
            'frac_modes_rounding_dominated_float32_hiband': frac_modes_dom_hi,
            'frac_modes_rounding_dominated_float64_estimate': frac_modes_dom64,
            'hi_band_power_fraction_from_rounding_float32': frac_r_in_32,
            'hi_band_mean_power': {'float64': float((P_m * hi).sum() / hi.sum()),
                                   'rounding': float((P_r * hi).sum() / hi.sum()),
                                   'floored_WN': float((P_w * hi).sum() / hi.sum())},
            'corr_log_hiband_vs_log_energy': {'float32': c_e32, 'float64': c_e64, 'floored_WN': c_ew},
            'corr_log_hiband_vs_log_rms': {'float32': c_r32, 'floored_WN': c_rw},
        }
        print(f'=== {base}  (n={args.n})')
        print(f'  map rms {m64.std():.3f}   rounding rms/pixel {r.std():.2e}')
        print(f'  rounding exceeds physical spectrum (radial average) above k ~ {cross}')
        print(f'  per-mode: rounding-dominated modes = {frac_modes_dom:.3f} of all (float32), '
              f'{frac_modes_dom_hi:.3f} of |k|>{K_HI}; float64 estimate {frac_modes_dom64:.4f}')
        print(f"  high-k band (|k|>{K_HI}) mean power: physical {res[base]['hi_band_mean_power']['float64']:.3e}   "
              f"rounding {res[base]['hi_band_mean_power']['rounding']:.3e}   floored {res[base]['hi_band_mean_power']['floored_WN']:.3e}")
        print(f'  fraction of float32 high-k power that is rounding: {frac_r_in_32:.3f}   (mode fraction of the band: {frac_w:.3f})')
        print(f'  corr[log hi-band power, log artifact energy]: float32 {c_e32[0]:+.3f}/{c_e32[1]:+.3f} (Pearson/Spearman)   '
              f'float64 {c_e64[0]:+.3f}/{c_e64[1]:+.3f}   floored {c_ew[0]:+.3f}/{c_ew[1]:+.3f}')
        print(f'  corr[log hi-band power, log map rms]:         float32 {c_r32[0]:+.3f}/{c_r32[1]:+.3f}   floored {c_rw[0]:+.3f}/{c_rw[1]:+.3f}')
    (OUT / 'r3_rounding_band.json').write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwritten {OUT / 'r3_rounding_band.json'}")


if __name__ == '__main__':
    main()
