#!/usr/bin/env python
"""
run_null_tests.py — mandatory η-pipeline validation battery (memo §15)
======================================================================

No η number from this pipeline is to be trusted before this battery
passes.  All DFT-normalization, whitening, split-discipline and estimator
conventions are exercised here on synthetic ensembles generated on the fly
(no HDF5 inputs needed).  Maps are 64² (32² for the i.i.d. tests) — the
conventions being validated are size-independent; production runs use 128².

The battery (results table printed at the end):

  N1  Gaussian WHITE null      — every method must return η = 1 / κ-norms = 0
  N2  Gaussian RED null        — same, through the estimated-PSD whitening
                                 path (split discipline, beam-smoothed 1/f³)
  N3  Known-η validation       — i.i.d. two-component Gaussian scale
                                 mixture, single-pixel template: variational
                                 bound vs exact 1-D quadrature vs the weak
                                 formula γ₄²/6
  N4  Template dependence on i.i.d. noise — exact theory says η is
                                 TEMPLATE-INDEPENDENT for i.i.d. pixels
                                 (score is separable; 𝒥 = I_1d·𝕀), so the
                                 memo §15.2 expectation of CLT suppression
                                 is corrected here (see README_PhaseB1);
                                 the pipeline must reproduce the equality.
  N4b Overlap-law ordering     — correlated Campbell mini-model (smoothed
                                 sparse positive spikes on a WHITE
                                 background): the whitened departure is
                                 LOW-k concentrated (the white noise owns
                                 high k; the event profile g~ cuts off the
                                 structured cumulants above its own scale),
                                 so the EXTENDED template must see the
                                 larger eta.  The compact-template-wins
                                 configuration requires a background that
                                 shares the event spectrum (pure confusion
                                 — a Phase-B2 case).
  N5  Tier-1 quadrature nulls  — per-image amplitude-only randomization:
                                 η_marg → E[σ²]E[σ⁻²] (NOT 1; memo §21.5
                                 corrected — see quadrature.py docstring),
                                 confronted with its analytic prediction;
                                 slope-only randomization: η_marg > 1.
  N6  Ladder ≤ ceiling (theorem) — audit of the STORED campaign results
                                 (eta_results/): every physical variational
                                 rung must lie at or below the exact ceiling
                                 (Tier-1 / PCA quadrature, _ATM exact value)
                                 or the complete-data bound, within 2σ.
                                 Rung ORDERING is deliberately NOT required:
                                 each rung is an independent lower bound and
                                 the conv rung's sensitivity floor (~1.15) is
                                 worse than the cubic's (~1.05), so cubic >
                                 conv is legitimate.  (Added 2026-09-07.)
  N7  Known-η recovery on production-size maps — T2_CONFUSION_ATM: the exact
                                 η = I_1 (compound_poisson_eta_1d), the same
                                 for every template by theorem, must be (a)
                                 below its complete-data bound (1+ρ)/ρ, (b)
                                 recovered by the production ladder with the
                                 template-adapted basis to ≥ 0.8 of the
                                 rounding-diluted value on BOTH templates,
                                 (c) not exceeded by any rung, (d) template-
                                 independent within 2σ.  This replaces the
                                 circular 'tier factorisation' N7 of earlier
                                 drafts, and it is the test that exposed the
                                 pooling-basis flaw (Fourier basis: 63 % / 0 %).
                                 (Added 2026-09-07.)
  N8  Band regularization      — sigma_MF and the complete-data bound stable
                                 against the mode mask for every floored cell.

Run all:      python scripts/run_null_tests.py
Run subset:   python scripts/run_null_tests.py N1 N3
(or execute the #%% cells in Spyder)
"""

#%% ================================================================================
# === IMPORTS AND PATH SETUP ===
# ==================================================================================

import sys
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
for _p in (_here, *_here.parents):
    if (_p / 'eta_pipeline').is_dir():
        sys.path.insert(0, str(_p))
        break
# noise_lib and eta_pipeline live at the repository root
import torch

import noise_lib as nl
import eta_pipeline as ep
from eta_pipeline import variational as ev

RESULTS = []


def record(test, name, ok, detail):
    RESULTS.append((test, name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {test} {name}  ({detail})")


def close(a, b, tol):
    return abs(a - b) <= tol


#%% ================================================================================
# === COMMON PIECES ===
# ==================================================================================

SIZE = 64
N_ENS = 2400
SEED = 20260827
TEMPLATE_RC, BEAM_FWHM = 7.0, 5.0
RUNGS = ('linear', 'quadratic', 'cnn')
EPOCHS = 100   # good default on mps/cuda; on a weak CPU reduce to ~25


def beam2(shape, fwhm):
    """Squared beam transfer B²(k) on the FFT grid (power convention)."""
    ny, nx = shape
    sigma = fwhm / 2.355
    fy = np.fft.fftfreq(ny).reshape(-1, 1)
    fx = np.fft.fftfreq(nx).reshape(1, -1)
    return np.exp(-4.0 * np.pi**2 * sigma**2 * (fx**2 + fy**2))


def prepare(maps, tau, seed=1):
    """splits -> P̂ (PSD split) -> whitened fit/val/eval + t̂ + σ_MF.

    Also seeds torch: model initializations otherwise draw from torch's
    UNSEEDED global RNG, making trained-rung values wander run to run by
    more than their bootstrap errors (which cover evaluation statistics
    only) — the mechanism behind the intermittent N4b verdicts observed
    on repeated runs.  Residual non-determinism from mps/cuda kernels is
    absorbed by restarts and the paired N4b criterion."""
    torch.manual_seed(SEED)
    sp = ep.make_splits(len(maps), seed=seed)
    P_hat = ep.estimate_psd2d(maps[sp['psd']])
    t_hat, sigma_mf = ep.whitened_template(tau, P_hat)
    out = {k: ep.whiten_maps(maps[sp[k]], P_hat) for k in ('fit', 'val', 'eval')}
    return out, P_hat, t_hat, sigma_mf, sp


def gaussian_null(test, maps, tau, note):
    """Run the full method suite on a Gaussian ensemble; everything -> null."""
    t0 = time.time()
    W, P_hat, t_hat, sig_mf, sp = prepare(maps, tau)

    # sigma_MF prediction vs empirical MF scatter (convention check)
    A_hat = ep.whitening.mf_amplitudes(maps[sp['eval']], tau, P_hat)
    ratio = A_hat.std() / sig_mf
    record(test, 'sigma_MF predicted vs empirical', close(ratio, 1.0, 0.08),
           f'emp/pred = {ratio:.3f}')

    # variational ladder
    ladder = ev.run_ladder(W['fit'], W['val'], W['eval'], t_hat,
                           rungs=RUNGS, epochs=EPOCHS, patience=5, verbose=False)
    # run_ladder returns (eta, err, restart_etas) per rung — the third slot was
    # added by the Phase-B1 restarts update; unpack defensively.
    for name, (eta, err, *_) in ladder.items():
        tol = max(3.0 * err, 0.03)
        record(test, f'variational {name} rung = 1',
               close(eta, 1.0, tol), f'eta = {eta:.4f} ± {err:.4f}')

    # cumulant estimators -> 0 (within their own bootstrap errors)
    half = len(W['eval']) // 2
    _, _, (k3, k3e), (k4, k4e) = ep.cumulants.eta_perturbative(
        W['eval'][:half], W['eval'][half:], t_hat)
    record(test, 'cumulant k3-norm = 0', abs(k3) < 3.0 * k3e,
           f'k3 = {k3:+.2f} ± {k3e:.2f}')
    record(test, 'cumulant k4-norm = 0', abs(k4) < 3.0 * k4e,
           f'k4 = {k4:+.2f} ± {k4e:.2f}')
    print(f'--- {note}: {time.time()-t0:.0f} s')
    return ladder


#%% ================================================================================
# === N1: GAUSSIAN WHITE NULL ===
# ==================================================================================

def test_N1():
    rng = np.random.default_rng(SEED)
    maps = rng.standard_normal((N_ENS, SIZE, SIZE))
    tau = nl.make_template(SIZE, 'beta', TEMPLATE_RC, BEAM_FWHM, normalize='none')
    gaussian_null('N1', maps, tau, 'white Gaussian null')


#%% ================================================================================
# === N2: GAUSSIAN RED NULL (estimated-PSD whitening path) ===
# ==================================================================================

def test_N2():
    rng = np.random.default_rng(SEED + 1)
    P = nl.psd_isotropic_powerlaw((SIZE, SIZE), 3.0, 5.0)
    maps = np.stack([nl.beam_smooth(nl.grf_from_psd(P, rng), BEAM_FWHM)
                     for _ in range(N_ENS)])
    tau = nl.make_template(SIZE, 'beta', TEMPLATE_RC, BEAM_FWHM, normalize='none')
    gaussian_null('N2', maps, tau, 'red (1/f^3 + beam) Gaussian null')


#%% ================================================================================
# === N3: KNOWN-η VALIDATION (i.i.d. scale mixture, single-pixel template) ===
# ==================================================================================

def test_N3(eps=0.1, R=6.0, n_ens=8000, size=32):
    t0 = time.time()
    rng = np.random.default_rng(SEED + 2)
    maps = ep.analytic1d.sample_scale_mixture((n_ens, size, size), eps, R, rng)

    eta_exact = ep.scale_mixture_eta_1d(eps, R)
    g3, g4 = ep.analytic1d.scale_mixture_moments(eps, R)
    eta_weak = ep.eta_weak_1d(g3, g4)
    print(f'    exact 1-D eta = {eta_exact:.4f}   weak formula = {eta_weak:.4f} '
          f'(gamma4 = {g4:.3f})')

    tau = np.zeros((size, size)); tau[size // 2, size // 2] = 1.0
    W, P_hat, t_hat, sig, sp = prepare(maps, tau)

    # analytic-score plug-in: f*(x) = sum_p t_hat_p s_1d(x_p) evaluated on
    # EVAL must reproduce eta exactly (validates the bound evaluation
    # independently of any training)
    s1, s2 = ep.analytic1d.scale_mixture_params(eps, R)
    def score_1d(v):
        g1 = np.exp(-0.5*(v/s1)**2)/(np.sqrt(2*np.pi)*s1)
        g2 = np.exp(-0.5*(v/s2)**2)/(np.sqrt(2*np.pi)*s2)
        p  = (1-eps)*g1 + eps*g2
        dp = (1-eps)*g1*(-v/s1**2) + eps*g2*(-v/s2**2)
        return -dp/np.maximum(p, 1e-300)
    def dscore_1d(v, h=1e-4):
        return (score_1d(v+h) - score_1d(v-h))/(2*h)
    xe = W['eval']
    f_star = np.einsum('pq,ipq->i', t_hat, score_1d(xe))
    proj   = np.einsum('pq,ipq->i', t_hat**2, dscore_1d(xe))
    terms  = 2.0*proj - f_star**2
    m_pl, e_pl = ep.whitening.bootstrap_mean(terms)
    record('N3', 'analytic-score plug-in reproduces exact eta',
           close(m_pl, eta_exact, max(3*e_pl, 0.02)),
           f'bound = {m_pl:.4f} ± {e_pl:.4f} vs exact {eta_exact:.4f}')

    ladder = ev.run_ladder(W['fit'], W['val'], W['eval'], t_hat,
                           rungs=('linear', 'pixelwise'), epochs=80,
                           lr=2e-2, patience=20, batch=1024, verbose=False)
    e_lin, err_lin = ladder['linear'][:2]          # (eta, err, restart_etas)
    e_pix, err_pix = ladder['pixelwise'][:2]
    record('N3', 'linear rung = 1 (MF anchor)', close(e_lin, 1.0, max(3*err_lin, 0.03)),
           f'eta = {e_lin:.4f} ± {err_lin:.4f}')
    frac = (e_pix - 1.0) / (eta_exact - 1.0)
    record('N3', 'pixelwise bound recovers exact eta',
           (e_pix <= eta_exact + 3 * err_pix) and frac > 0.5,
           f'eta = {e_pix:.4f} ± {err_pix:.4f} vs exact {eta_exact:.4f} '
           f'(recovered {100*frac:.0f}% of eta-1)')

    # weak-formula line: REPORT only (gamma4 = 3 is far outside the weak
    # regime, and the pair-trick kappa4 estimator is MC-hungry at this M —
    # memo Sec. 17 caveat)
    half = len(W['eval']) // 2
    e_pert, e_err, _, _ = ep.cumulants.eta_perturbative(
        W['eval'][:half], W['eval'][half:], t_hat)
    print(f'    [report] cumulant eta_pert = {e_pert:.2f} ± {e_err:.2f}  '
          f'(weak formula {eta_weak:.3f}, exact {eta_exact:.4f}; strong-NG '
          f'regime — no PASS criterion)')

    # kappa4 estimator validated in ITS regime: tiny maps (M=16), raw
    # unit-variance i.i.d. maps, delta template; averaged over independent
    # repetitions (per-rep scatter is large and two-batch-correlated):
    # ||kappa4(t_hat)||^2 = gamma4^2
    rng2 = np.random.default_rng(SEED + 20)
    tau4 = np.zeros((4, 4)); tau4[2, 2] = 1.0
    reps = []
    for _ in range(12):
        tiny = ep.analytic1d.sample_scale_mixture((6000, 4, 4), eps, R, rng2)
        v, _ = ep.cumulants.k4_norm(tiny[:3000], tiny[3000:], tau4, n_boot=10)
        reps.append(v)
    k4m, k4sem = np.mean(reps), np.std(reps) / np.sqrt(len(reps))
    record('N3', 'kappa4 pair-trick = gamma4^2 (tiny-map regime, 12 reps)',
           close(k4m, g4**2, 4 * k4sem),
           f'k4 = {k4m:.2f} ± {k4sem:.2f} vs gamma4^2 = {g4**2:.2f}')
    print(f'--- N3: {time.time()-t0:.0f} s')
    return eta_exact


#%% ================================================================================
# === N4: TEMPLATE (IN)DEPENDENCE ON i.i.d. NOISE ===
# ==================================================================================
# Exact statement: for i.i.d. pixels the joint score is separable,
# s_p(x) = s_1d(x_p), so 𝒥 = I_1d·𝕀 and η = I_1d for EVERY unit template —
# no CLT suppression (the sum statistic Gaussianizes, but the optimal
# estimator applies the nonlinearity per pixel BEFORE averaging).  This
# corrects memo §15.2's expected behaviour; the contracted-cumulant norms
# are likewise exactly template-independent (‖κ̃₃(t̂)‖² = γ₃²·‖t̂‖⁴ etc.).

def test_N4(eps=0.1, R=6.0, n_ens=8000, size=32):
    t0 = time.time()
    rng = np.random.default_rng(SEED + 3)
    maps = ep.analytic1d.sample_scale_mixture((n_ens, size, size), eps, R, rng)
    eta_exact = ep.scale_mixture_eta_1d(eps, R)

    etas = {}
    for label, tau in [
            ('pixel', None),
            ('extended (beta rc=7 * beam)',
             nl.make_template(size, 'beta', TEMPLATE_RC, BEAM_FWHM,
                              normalize='none'))]:
        if tau is None:
            tau = np.zeros((size, size)); tau[size // 2, size // 2] = 1.0
        W, P_hat, t_hat, sig, sp = prepare(maps, tau)
        ladder = ev.run_ladder(W['fit'], W['val'], W['eval'], t_hat,
                               rungs=('pixelwise',), epochs=80, lr=2e-2,
                               patience=20, batch=1024)
        etas[label] = ladder['pixelwise'][:2]       # drop restart_etas
        print(f'    {label:28s} eta = {etas[label][0]:.4f} ± {etas[label][1]:.4f}')

    (e1, s1), (e2, s2) = etas.values()
    record('N4', 'eta template-independent on i.i.d. noise (exact theory)',
           close(e1, e2, max(3 * np.hypot(s1, s2), 0.05 * (eta_exact - 1))),
           f'pixel {e1:.4f} vs extended {e2:.4f} (exact {eta_exact:.4f})')
    print(f'--- N4: {time.time()-t0:.0f} s')


#%% ================================================================================
# === N4b: SOURCE-DEPARTURE OVERLAP-LAW ORDERING (Campbell mini-model) ===
# ==================================================================================
# Sparse positive spikes (event FWHM 2 px, per-event MF-SNR ~ 2, ~30 events
# per map) on a UNIT WHITE background.  Where does the departure live?  The
# structured cumulants carry the event profile g~(k) on every leg, so they
# cut off above the event scale, while the white background owns the high-k
# power: after whitening, the non-Gaussian information is LOW-k
# concentrated, and by the overlap law (M4 §5.5) the EXTENDED template must
# see the larger eta.  (The naive reading "compact events => compact
# template wins" holds only when the background shares the event spectrum —
# flat whitened SNR, as in pure confusion noise — which is a Phase-B2
# case.)  This test therefore checks that the pipeline reproduces the
# overlap-law ordering as derived for THIS model.

def test_N4b(n_ens=4500, size=32, rate=0.03, amp=6.0):
    t0 = time.time()
    rng = np.random.default_rng(SEED + 4)
    maps = []
    for _ in range(n_ens):
        spikes = amp * (rng.random((size, size)) < rate)
        m = nl.beam_smooth(spikes, 2.0) + rng.standard_normal((size, size))
        maps.append(m)
    maps = np.stack(maps)

    # Both templates share the SAME maps and splits, so the ordering is
    # tested with a PAIRED per-image bootstrap of the difference: the
    # correlated part of the evaluation fluctuations cancels, which the
    # unpaired 2·hypot criterion ignored (it straddled 2 sigma and flipped
    # verdicts between runs whose values agreed within errors — observed in
    # repeated mps runs, 2026-08-28).  Restarts absorb training stochasticity.
    etas, best_terms = {}, {}
    for label, tau in [('compact (delta)', None),
                       ('extended (beta rc=7 * beam)',
                        nl.make_template(size, 'beta', TEMPLATE_RC, BEAM_FWHM,
                                         normalize='none'))]:
        if tau is None:
            tau = np.zeros((size, size)); tau[size // 2, size // 2] = 1.0
        W, P_hat, t_hat, sig, sp = prepare(maps, tau)
        ladder, terms = ev.run_ladder(W['fit'], W['val'], W['eval'], t_hat,
                                      rungs=('quadratic', 'pixelwise', 'cnn'),
                                      epochs=40, patience=10, lr=1e-2,
                                      reg_to_init=1e-3, batch=1024,
                                      restarts=2, return_terms=True)
        best_name = max(ladder, key=lambda k: ladder[k][0])
        etas[label] = ladder[best_name][:2]         # drop restart_etas
        best_terms[label] = terms[best_name]
        print(f'    {label:28s} best eta = {etas[label][0]:.4f} ± '
              f'{etas[label][1]:.4f}  (' +
              '  '.join(f'{k} {v[0]:.3f}' for k, v in ladder.items()) + ')')

    (ec, sc), (ee, se) = etas.values()
    diff = best_terms['extended (beta rc=7 * beam)'] - best_terms['compact (delta)']
    d_mean, d_err = ep.whitening.bootstrap_mean(diff)
    record('N4b', 'overlap-law ordering: extended > compact (low-k departure)',
           d_mean > 2.0 * d_err,
           f'paired Delta = {d_mean:.4f} ± {d_err:.4f} '
           f'(extended {ee:.4f}, compact {ec:.4f})')
    print(f'--- N4b: {time.time()-t0:.0f} s')


#%% ================================================================================
# === N5: TIER-1 QUADRATURE NULLS (per-image PSRAND) ===
# ==================================================================================

def test_N5(n_ens=1500, size=64):
    # NOTE: maps deliberately NOT beam-smoothed here, so the analytic P_grid
    # is exact.  (With a beam, the analytic grid's B^2 ~ e^{-40} suppression
    # at high k divides floating-point rounding noise and explodes — the
    # failure this battery caught; production quadrature on beamed data
    # must pass k_mask or use estimated spectra.  See quadrature.py.)
    t0 = time.time()
    shape = (size, size)
    tau = nl.make_template(size, 'beta', TEMPLATE_RC, BEAM_FWHM, normalize='none')

    # ---- amplitude-only randomization ----
    rng = np.random.default_rng(SEED + 5)
    maps = np.stack([nl.psrand_1overf_noise(shape, 3.0, 0.0, 5.0, 0.8,
                                            rng=rng)[0] for _ in range(n_ens)])

    grid, wts = ep.quadrature.truncnorm_grid(5.0, 0.8, 1e-6, n=41)
    P_grid = np.stack([nl.psd_isotropic_powerlaw(shape, 3.0, a) for a in grid])
    eta_q, err_q = ep.eta_marg_quadrature(maps, P_grid, wts, tau)

    # analytic near-perfect-inference limit: E[sig^2] E[sig^-2] over the prior
    # (sig^2 = squared amplitude; Cauchy-Schwarz makes this >= 1)
    eta_pred = np.sum(wts * grid**2) * np.sum(wts / grid**2)
    record('N5', 'amplitude-only quadrature matches E[s2]E[1/s2] '
           '(NOT the memo-expected 1; see README)',
           close(eta_q, eta_pred, max(3 * err_q, 0.02)),
           f'eta_marg = {eta_q:.4f} ± {err_q:.4f} vs analytic {eta_pred:.4f}')

    # ---- slope-only randomization ----
    rng = np.random.default_rng(SEED + 6)
    maps = np.stack([nl.psrand_1overf_noise(shape, 3.0, 0.5, 5.0, 0.0,
                                            rng=rng)[0] for _ in range(n_ens)])
    grid_s, wts_s = ep.quadrature.truncnorm_grid(3.0, 0.5, 1.0, n=41)
    P_grid = np.stack([nl.psd_isotropic_powerlaw(shape, s, 5.0) for s in grid_s])
    eta_s, err_s = ep.eta_marg_quadrature(maps, P_grid, wts_s, tau)
    record('N5', 'slope-only quadrature gives eta > 1 (Tier-1 headroom)',
           eta_s - 1.0 > 3 * err_s, f'eta_marg = {eta_s:.4f} ± {err_s:.4f}')

    # ---- N5c: the non-factorized floored path reduces to the validated one ----
    # eta_marg_quadrature_floor exists because an additive white floor breaks
    # P = a^2 P_base(s).  At P_floor = 0 the two must agree to machine precision;
    # this is the only check that the new sweep is arithmetically the same object.
    from eta_pipeline.quadrature import (eta_marg_quadrature_scaled,
                                         eta_marg_quadrature_floor)
    rng = np.random.default_rng(SEED + 7)
    maps_a = np.stack([nl.psrand_1overf_noise(shape, 3.0, 0.0, 5.0, 0.8,
                                              rng=rng)[0] for _ in range(400)])
    base = np.stack([nl.psd_isotropic_powerlaw(shape, 3.0, 1.0)])
    bw = np.array([1.0])
    a_g, a_w = ep.quadrature.truncnorm_grid(5.0, 0.8, 1e-6, n=101)
    e_sc, _ = eta_marg_quadrature_scaled(maps_a, base, bw, a_g, a_w, tau)
    e_fl, _ = eta_marg_quadrature_floor(maps_a, base, bw, a_g, a_w, tau, 0.0)
    record('N5', 'floored quadrature == scaled quadrature at zero floor',
           close(e_sc, e_fl, 1e-6 * max(abs(e_sc), 1.0)),
           f'scaled {e_sc:.8f} vs floor-path {e_fl:.8f}')

    # and with a real floor it must still be a legitimate ceiling (>= 1)
    Pw = float(size * size) * 0.27**2
    e_wn, err_wn = eta_marg_quadrature_floor(maps_a, base, bw, a_g, a_w, tau, Pw)
    record('N5', 'floored quadrature returns a legitimate ceiling (eta >= 1)',
           e_wn > 1.0 - 3 * err_wn, f'eta_marg(sigma_w=0.27) = {e_wn:.4f} ± {err_wn:.4f}')
    print(f'--- N5: {time.time()-t0:.0f} s')


#%% ================================================================================
# === N6: LADDER <= CEILING — theorem audit of the stored campaign results ===
# ==================================================================================

def _load_results():
    import json
    from pathlib import Path
    out = _p / 'results' / 'eta_results'
    res = {}
    for p in sorted(out.glob('T*__*.json')):
        try:
            res[p.stem] = json.loads(p.read_text())
        except ValueError:
            pass
    return out, res


def _ceiling_for(cell, res, out):
    """(value, kind) of the one-sided ceiling a cell's rungs may not exceed."""
    import json
    name, tname = cell['model'], cell['template']
    if name in ('T2_PCA', 'T2_PCA_WN'):
        f = out / 'pca_mixture_quadrature.json'
        if f.exists():
            q = json.loads(f.read_text())
            key = 'floorless' if name == 'T2_PCA' else 'floored'
            return q[key][tname]['eta_exact_full6000']['eta'], 'exact (scale-mixture quadrature)'
    if name == 'T2_CONFUSION_ATM':
        return 3.5688442215901395, 'exact (compound-Poisson I_1)'
    if 'quadrature' in cell and cell['quadrature'].get('eta') is not None:
        return cell['quadrature']['eta'], 'exact (latent quadrature)'
    b = cell.get('bound_complete_data_dc_excluded', cell.get('bound_complete_data'))
    if b is not None:
        return b, 'complete-data bound'
    if cell.get('tier') == 'T0':
        return 1.0, 'theorem (Gaussian)'
    return None, None


def test_N6():
    """Every physical variational rung <= its ceiling (2 sigma).  No ordering."""
    t0 = time.time()
    out, res = _load_results()
    n_cells = 0
    for stem, cell in res.items():
        rungs = cell.get('variational2', {}).get('rungs')
        if not rungs:
            continue
        ceil, kind = _ceiling_for(cell, res, out)
        if ceil is None:
            print(f"    [report] {stem}: no ceiling on record (ladder-only cell) — not testable")
            continue
        worst = None
        for rname, r in rungs.items():
            if not r.get('physical', True):
                continue                        # stamped rounding-channel rungs
            excess = (r['eta'] - 2.0 * r['err']) - ceil
            if worst is None or excess > worst[1]:
                worst = (rname, excess, r['eta'], r['err'])
        if worst is None:
            continue
        n_cells += 1
        rname, excess, eta, err = worst
        # T0 cells: the ceiling is 1 exactly; allow the EVAL-split normalisation
        # scatter of the anchor (~4 %) on top of the 2 sigma
        slack = 0.04 if kind.startswith('theorem') else 0.0
        record('N6', f'{stem}: all rungs <= {kind} {ceil:.3f}',
               excess <= slack,
               f"highest rung '{rname}' = {eta:.3f} +- {err:.3f}")
    record('N6', 'ordering NOT required (independent lower bounds; conv floor > cubic floor)',
           True, f'{n_cells} cells audited')
    print(f'--- N6: {time.time()-t0:.0f} s')


#%% ================================================================================
# === N7: KNOWN-eta RECOVERY THROUGH THE PRODUCTION LADDER (T2_CONFUSION_ATM) ===
# ==================================================================================

def test_N7(min_fraction=0.8):
    """The exactly solvable production-size row, recovered by the production path."""
    import json
    t0 = time.time()
    from eta_pipeline import models as em
    from eta_pipeline import analytic1d as a1
    import noise_lib as nl

    ARGS = em.CONFUSION_ARGS
    rho = em.CONFUSION_RHO_ATM
    marks, wts = nl.confusion.mark_distribution(**ARGS)
    lam_pix = nl.confusion.n_sources_mean(em.SHAPE, **ARGS) / (em.SIZE * em.SIZE)
    eta_exact = a1.compound_poisson_eta_1d(marks, wts, lam_pix, rho)
    bound = (1.0 + rho) / rho
    record('N7', 'exact eta <= complete-data bound (1+rho)/rho',
           eta_exact <= bound, f'eta = {eta_exact:.4f}, bound = {bound:.3f}')

    # rounding dilution of the far-corner float64 modes, as calibrated by
    # tools/eta_exact_confusion.py (eps = 0.041, two independent estimates)
    out, res = _load_results()
    eps = 0.041
    for f in sorted((out.parent / 'tests_and_results').glob('confusion_exact_*.json')):
        try:
            eps = json.loads(f.read_text())['D']['eps']
        except (KeyError, ValueError):
            pass
    # use the stored pipeline expectation when available (same calibration file)
    exp = None
    for f in sorted((out.parent / 'tests_and_results').glob('confusion_exact_*.json')):
        try:
            exp = json.loads(f.read_text())['D']['eta_expected_from_pipeline']
        except (KeyError, ValueError):
            pass
    eta_diluted = exp if exp is not None else eta_exact
    print(f"    exact eta = {eta_exact:.4f}; rounding-diluted expectation = {eta_diluted:.4f} (eps = {eps:.4f})")

    got = {}
    for tname in ('extended', 'compact'):
        cell = res.get(f'T2_CONFUSION_ATM__{tname}')
        if cell is None or 'variational2' not in cell:
            record('N7', f'{tname}: production ladder on record', False, 'missing'); continue
        v2 = cell['variational2']
        pool = v2.get('config', {}).get('pool', 'fourier')
        best = max(v2['rungs'].values(), key=lambda r: r['eta'])
        got[tname] = (best['eta'], best['err'], pool)
        record('N7', f'{tname}: ladder ({pool} basis) recovers >= {min_fraction:.0%} of the known eta',
               best['eta'] >= min_fraction * eta_diluted,
               f"best rung {best['eta']:.3f} +- {best['err']:.3f} = "
               f"{best['eta']/eta_exact:.0%} of exact, {best['eta']/eta_diluted:.0%} of diluted")
        record('N7', f'{tname}: no rung exceeds the exact eta (2 sigma)',
               best['eta'] - 2 * best['err'] <= eta_exact,
               f"{best['eta']:.3f} +- {best['err']:.3f} vs {eta_exact:.4f}")
        # the historical failure, for the record
        for sup in cell.get('variational2_superseded', []):
            sec = sup.get('section', {})
            if sec.get('config', {}).get('pool', 'fourier') == 'fourier' and 'cnn' in sec.get('rungs', {}):
                e = sec['rungs']['cnn']['eta']
                print(f"    [report] {tname}: Fourier-basis ladder recovered {e:.3f} = {e/eta_exact:.0%} (superseded)")
                break
    if len(got) == 2:
        (e1, s1, _), (e2, s2, _) = got['extended'], got['compact']
        d = abs(e1 - e2); sd = np.hypot(s1, s2)
        record('N7', 'template independence (theorem for i.i.d. whitened pixels), 2 sigma',
               d <= 2 * sd, f'|ext - cmp| = {d:.3f} +- {sd:.3f}')
    print(f'--- N7: {time.time()-t0:.0f} s')


#%% ================================================================================
# === N8: BAND REGULARIZATION (is the number the noise, or the mode cutoff?) ===
# ==================================================================================

def test_N8(n_ens=400):
    """Every Fisher-weighted quantity must be stable against the mode mask.

    Without an unsmoothed white floor a COMPACT template on beam-smoothed red
    noise has Fisher weight |tau~|^2/P = B^2/(B^2 P_red) ~ k^alpha: the beam
    cancels, the integral diverges, and sigma_MF and the complete-data bound
    report the k-mask rather than the noise.

    Two quantities behave differently and are tested differently:

    * The complete-data bound divides by the ANALYTIC background P_g, which is
      beam-suppressed for every model, so a floorless compact bound is unstable
      universally -- assert that directly.
    * sigma_MF divides by the ESTIMATED total spectrum, and a model whose
      structured component carries unsmoothed high-k power (glitches above all)
      partially self-regularises.  So per-model instability is NOT universal
      here; assert instead the two statements that are: every *_WN cell is
      stable, and the floor never makes stability worse.  A battery-level check
      then confirms the diagnostic is still live, i.e. that at least one
      floorless compact cell does drift.
    """
    t0 = time.time()
    from eta_pipeline import models as em

    templates = em.make_templates()
    B2 = em.beam2()
    kk = np.sqrt(np.fft.fftfreq(em.SIZE).reshape(1, -1)**2
                 + np.fft.fftfreq(em.SIZE).reshape(-1, 1)**2)
    m8 = (B2 > 1e-8) & (kk > 0)
    m14 = (B2 > 1e-14) & (kk > 0)
    TOL = 0.02          # the instabilities we detect are 13%-1700%

    pairs = [('T2_CROSS_SYM', 'T2_CROSS_SYM_WN'), ('T2_GLITCH', 'T2_GLITCH_WN'),
             ('T2_PCA', 'T2_PCA_WN'), ('T1_PSRAND', 'T1_PSRAND_WN'),
             ('T2_CONFUSION', 'T2_CONFUSION_WN')]      # added 2026-09-07 (README v3 §4)
    drifts = {}
    for base_name, wn_name in pairs:
        for name in (base_name, wn_name):
            entry = em.REGISTRY[name]
            maps, _ = entry['gen'](n_ens, em.model_seed(name))
            P_hat = ep.estimate_psd2d(maps)
            floored = entry.get('floor', 0.0) > 0
            for tname, tau in templates.items():
                t2 = np.abs(np.fft.fft2(tau))**2
                s8 = 1.0 / np.sqrt((t2[m8] / P_hat[m8]).sum())
                s14 = 1.0 / np.sqrt((t2[m14] / P_hat[m14]).sum())
                drifts[(name, tname)] = abs(s8 - s14) / s8
                if floored:
                    record('N8', f'{name}|{tname} sigma_MF stable vs k-mask',
                           drifts[(name, tname)] < TOL,
                           f'drift = {drifts[(name, tname)]:.2%}')
                else:
                    print(f"    [report] {name}|{tname} sigma_MF drift = "
                          f"{drifts[(name, tname)]:.2%}")
            if entry.get('bound_bg') is not None:
                for tname, tau in templates.items():
                    st = ep.bound_stability(tau, entry['bound_bg'](), P_hat, B2,
                                            tol=TOL)
                    expect_stable = floored or tname == 'extended'
                    record('N8', f'{name}|{tname} complete-data bound stable'
                                 f"{'' if expect_stable else ' (expected UNSTABLE)'}",
                           st['stable'] == expect_stable,
                           f"drift = {st['max_rel_drift']:.2%}, "
                           f"values = {[round(v, 3) for v in st['values'].values()]}")
        # the floor must never make stability worse
        for tname in templates:
            d0, d1 = drifts[(base_name, tname)], drifts[(wn_name, tname)]
            record('N8', f'{base_name}|{tname} floor does not worsen sigma_MF stability',
                   d1 <= max(d0, TOL) + 1e-12, f'{d0:.2%} -> {d1:.2%}')

    # is the diagnostic still live?  (it would be worthless if nothing drifted)
    live = [k for k, v in drifts.items()
            if not k[0].endswith('_WN') and k[1] == 'compact' and v >= TOL]
    record('N8', 'k-mask diagnostic is live (a floorless compact cell does drift)',
           len(live) > 0,
           f"{len(live)} unstable: " + ', '.join(f'{n} {drifts[(n, t)]:.1%}'
                                                 for n, t in live))
    print(f'--- N8: {time.time()-t0:.0f} s')


#%% ================================================================================
# === MAIN ===
# ==================================================================================

def main(names=None):
    all_tests = {'N1': test_N1, 'N2': test_N2, 'N3': test_N3,
                 'N4': test_N4, 'N4b': test_N4b, 'N5': test_N5,
                 'N6': test_N6, 'N7': test_N7, 'N8': test_N8}
    names = names or list(all_tests)
    for n in names:
        print(f'\n===== {n} =====')
        all_tests[n]()
    n_pass = sum(ok for *_, ok, _ in RESULTS)
    print('\n' + '=' * 64)
    print(f'Null-test battery: {n_pass}/{len(RESULTS)} checks passed')
    for test, name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {test:4s} {name}: {detail}")
    return n_pass == len(RESULTS)


if __name__ == '__main__':
    ok = main(sys.argv[1:] or None)
    sys.exit(0 if ok else 1)
