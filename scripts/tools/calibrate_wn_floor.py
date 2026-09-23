#!/usr/bin/env python
"""
calibrate_wn_floor.py — set the per-model white-noise floor from the Fisher band
================================================================================

Provenance script for the ``floor`` values hard-coded in
``eta_pipeline/models.py``.  Re-run it if a model's background spectrum or the
template pair changes; the printed table is what belongs in the registry.

THE CRITERION (see WN_Floor_Implementation_Plan.md §1)
-----------------------------------------------------
Without an unsmoothed white floor the compact template's Fisher weight is

    f_k  ~  |tau~|^2 / P  =  B^2 / (B^2 * P_red)  =  1 / P_red  ~  k^alpha ,

because the beam cancels between template and background.  The information
integral then diverges and every compact-template quantity reports whatever
mode cutoff happens to be in force.  A real instrument always has a detector
floor that is NOT beam-smoothed, which terminates the per-mode SNR beyond the
beam scale and makes the integral converge.

We therefore choose, per model, the SMALLEST floor that regularises the band
while leaving the extended template's physics alone:

    sigma_w^2 = P_bg_ring(k_knee) / n_pix ,   k_knee = KNEE_FACTOR * k_ext ,

with k_ext = exp<log k>_f the geometric-mean centroid of the EXTENDED
template's (floorless) Fisher band, P_bg_ring the azimuthal mean of the model's
Gaussian BACKGROUND spectrum (severity-independent by construction: the
structured components are deliberately excluded so that changing amp_scale or
leak_efficiency never changes the floor), and KNEE_FACTOR = 2.

Reading: the white/red knee is placed one octave above the extended template's
band centre.  The extended template then keeps ~99% of its Fisher weight below
the knee and its ceiling shifts by only a few percent, while the compact
template — whose weight piles up against the cutoff — becomes well-defined.

Run from the repository root:  python scripts/tools/calibrate_wn_floor.py
"""

import sys
import types
import importlib
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_root = _here.parent.parent                 # repository root
sys.path.insert(0, str(_root))
import noise_lib as nl                                              # noqa: E402

# import models without eta_pipeline/__init__ (no torch needed for this script)
if 'eta_pipeline' not in sys.modules:
    _pkg = types.ModuleType('eta_pipeline')
    _pkg.__path__ = [str(_root / 'eta_pipeline')]
    sys.modules['eta_pipeline'] = _pkg
M = importlib.import_module('eta_pipeline.models')

KNEE_FACTOR = 2.0
N_ENS = 1500
SEED = 777000

SIZE = M.SIZE
B2 = M.beam2()
_fy = np.fft.fftfreq(SIZE).reshape(-1, 1)
_fx = np.fft.fftfreq(SIZE).reshape(1, -1)
KK = np.sqrt(_fx**2 + _fy**2)
NPIX = SIZE * SIZE
MASK = (B2 > 1e-8) & (KK > 0)          # the meaningful band, floorless


def ring_mean(P, k0, width=0.01):
    """Azimuthal mean of P in the annulus |k| in [k0-w, k0+w] (handles anisotropy)."""
    sel = (KK > k0 - width) & (KK < k0 + width)
    if not sel.any():
        sel = np.abs(KK - k0) < 2 * width
    return float(P[sel].mean())


def k_ext_centroid(P_bar, tau_ext):
    """Geometric-mean centroid exp<log k>_f of the extended template's Fisher band."""
    w = np.where(MASK, np.abs(np.fft.fft2(tau_ext))**2 / P_bar, 0.0)
    f = w / w.sum()
    return float(np.exp((f * np.log(np.where(KK > 0, KK, 1.0))).sum()))


def calibrate(gen, bg_fn, tau_ext, n_ens=N_ENS, seed=SEED):
    maps, _ = gen(n_ens, seed)
    P_bar = (np.abs(np.fft.fft2(maps.astype(np.float64), axes=(-2, -1)))**2).mean(0)
    k_ext = k_ext_centroid(P_bar, tau_ext)
    k_knee = KNEE_FACTOR * k_ext
    sigma_w = np.sqrt(ring_mean(bg_fn(), k_knee) / NPIX)
    return k_ext, k_knee, sigma_w


# (registry name of the floorless parent, generator, background spectrum)
CASES = [
    ('T1_PSRAND',       M.gen_t1_psrand,       M.bg_iso),
    ('T1_PSRAND_SLOPE', M.gen_t1_psrand_slope, M.bg_iso),
    ('T1_RANDOR',       M.gen_t1_randor,       M.bg_aniso),
    ('T2_MEDIAN',       M.gen_t2_median,       M.bg_aniso),
    ('T2_CROSS_SYM',    M.gen_t2_cross_sym,    M.bg_iso),
    ('T2_CROSS_POS',    M.gen_t2_cross_pos,    M.bg_iso),
    ('T2_GLITCH',       M.gen_t2_glitch,       M.bg_aniso),
    ('T2_PCA',          M.gen_t2_pca,          M.bg_iso),
]

def verify(gen, sigma_w, templates):
    """Check the criterion delivered what it promises, per template."""
    maps, _ = gen(N_ENS, SEED)
    P0 = (np.abs(np.fft.fft2(maps.astype(np.float64), axes=(-2, -1)))**2).mean(0)
    Pw = NPIX * sigma_w**2
    m8 = (B2 > 1e-8) & (KK > 0)
    m14 = (B2 > 1e-14) & (KK > 0)
    out = {}
    for tname, tau in templates.items():
        t2 = np.abs(np.fft.fft2(tau))**2
        row = {}
        for label, P in (('none', P0), ('floor', P0 + Pw)):
            s8 = 1 / np.sqrt((t2[m8] / P[m8]).sum())
            s14 = 1 / np.sqrt((t2[m14] / P[m14]).sum())
            row[label] = (s8, 100 * abs(s8 - s14) / s8)
        f = np.where(m8, t2 / (P0 + Pw), 0.0); f /= f.sum()
        row['above_knee'] = float(f[m8 & (KK > KNEE_FACTOR * k_ext_centroid(P0, templates['extended']))].sum())
        out[tname] = row
    return out


if __name__ == '__main__':
    tau_ext = M.make_templates()['extended']
    print(f'KNEE_FACTOR = {KNEE_FACTOR};  n_ens = {N_ENS};  mask = B^2 > 1e-8\n')
    print(f"{'parent model':18s} {'k_ext':>7s} {'k_knee':>7s} {'sigma_w':>9s} {'rounded':>8s}")
    print('-' * 54)
    results = []
    for name, gen, bg in CASES:
        k_ext, k_knee, sw = calibrate(gen, bg, tau_ext)
        print(f'{name:18s} {k_ext:7.4f} {k_knee:7.4f} {sw:9.4f} {round(sw, 2):8.2f}')
        results.append((name, gen, round(sw, 2)))

    tpl = M.make_templates()
    print('\n--- verification: does the criterion do what it claims? ---')
    print(f"{'parent model':18s} {'sig_w':>6s} | "
          f"{'ext drift0':>10s} {'ext driftF':>10s} {'ext>knee':>9s} | "
          f"{'cmp drift0':>10s} {'cmp driftF':>10s}")
    print('-' * 84)
    for name, gen, sw in results:
        v = verify(gen, sw, tpl)
        e, c = v['extended'], v['compact']
        print(f'{name:18s} {sw:6.2f} | {e["none"][1]:9.2f}% {e["floor"][1]:9.2f}% '
              f'{e["above_knee"]:9.3f} | {c["none"][1]:9.2f}% {c["floor"][1]:9.2f}%')
