#!/usr/bin/env python
"""
teaser_eval.py -- adjudication of a ResNet teaser checkpoint against the
eta-campaign reference lines (Paper I Sec. 6.1)
=======================================================================

What it does
------------
For one checkpoint written by train_teaser_ddp.py and the NOISE-ONLY evaluation
file written by make_teaser_dataset.py:

1. Builds the A-grid design.  For each of n_grid amplitudes A_j in [0, L] and
   each EVAL noise map n_i (the maps after the PSD split of the eval file,
   default all ~4000) it forms d_ij = A_j * tau_ext + n_i, applies the
   checkpoint's own normalisation (global min/max, float32 -- identical to the
   training path) and records the network output A_hat_ij.  NO clipping of the
   output.  The same noise realisations are used at every A (common random
   numbers), which makes g(A) smooth and g'(A) well determined.

2. Per grid point:  g(A) = E[A_hat|A] - A,  V(A) = Var(A_hat|A) with MC errors;
   g'(A) from the 1/SE-weighted smoothing spline of biasfixed_diagnostics;
   the bias-consistent region and the strict g'~0 band exactly as
   biasfixed_diagnostics.find_locally_unbiased_band defines them (bias gate
   |g| < max(2 SE, 0.1 sigma_MF), slope gate |g'| < eps_slope);
   the shrinkage-cleaned ratio R(A) = V / ([1 + g'(A)]^2 sigma_MF^2);
   band averages of V/sigma_MF^2 and R with a bootstrap over MAPS (the
   per-A estimates share the noise realisations, so a bootstrap over the map
   index is the correct error).

3. The paired matched-filter baseline on the SAME maps: the ensemble MF is
   linear, so A_hat_MF(A) = A + MF(n_i) with MF(n_i) stored in the eval file
   (built from the exact mean spectrum P_bar where it is known, else from
   P_hat).  Its variance is the empirical sigma_MF^2 on exactly these maps; the
   ratio V_CNN(A) / V_MF is reported next to V_CNN(A) / sigma_MF,exact^2.

4. Reference lines for the figure / table: the flat eta = 1 floor, the
   biased-CRLB envelope [1 + g'(A)]^2, and for T1_PSRAND_WN the band-level
   sigma_norm^2 = 0.224, the reweight rung 1/7.119, the exact ceiling 1/10.18
   and the Method-E envelope V_E(A) (from r3_channels_wn.json).

Outputs (in --outdir, prefixed by the checkpoint stem):
   <stem>_eval.json     all curves, band limits, band averages, references,
                        provenance (seeds, sigma_MF values, normalisation)
   <stem>_grid.npz      the raw A_hat_ij matrix (n_grid, n_real) + A grid
   <stem>_diag.png      biasfixed_diagnostics figure (per-run)
   <stem>_teaser.png    a two-panel diagnostic figure for this run
   <stem>_runlog.md     a Markdown summary block of the run

Which sigma_MF
--------------
sigma_MF is the matched-filter error under the ensemble-mean spectrum.  The
primary reference is the EXACT value (analytic P_bar; T0_RED_REAL 0.8630,
T1_PSRAND_WN 1.3285).  The eval file also carries sigma_MF(P_hat) from its PSD
split and the paired empirical value; all three are reported.  (The campaign
JSONs quote 0.864 / 1.300 from their PSD splits and 0.854 / 1.205 empirical --
the PSRAND spread is EVAL-split scatter of a covariance mixture, not a
discrepancy.)

Usage
-----
    python teaser_eval.py --checkpoint TEASER_T0_RED_REAL_mse.pth \
        --eval-h5 TEASER_T0_RED_REAL_evalnoise6000_s990034.h5 --outdir eval_out
    # options: --n-grid 41 --n-real 4000 --batch 256 --device cuda|cpu|mps
    #          --sigma-ref exact|phat|paired   --eps-slope 0.05   --n-boot 500

Requires torch, numpy, scipy, matplotlib, h5py; utils_teaser_ddp.py and
biasfixed_diagnostics.py next to this file (or on PYTHONPATH).
"""

#%% ================================================================================
# === IMPORTS ===
# ==================================================================================

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

_here = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
sys.path.insert(0, str(_here))
import biasfixed_diagnostics as bd                       # noqa: E402
from utils_teaser_ddp import ResNet, ResidualBlock       # noqa: E402

# reference lines per model (eta-campaign values; see results/eta_results)
REFERENCE_LINES = {
    'T0_RED_REAL': {'eta=1 floor (theorem)': 1.0},
    'T1_PSRAND_WN': {
        'eta=1 floor': 1.0,
        'sigma_norm^2 (band weight map)': 0.224,
        'reweight rung 1/7.119': 1.0 / 7.119,
        'exact ceiling 1/10.18': 1.0 / 10.18,
    },
}


#%% ================================================================================
# === MODEL / DATA LOADING ===
# ==================================================================================

def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get('config', {})
    if cfg.get('is_heteroscedastic', False):
        raise SystemExit("heteroscedastic checkpoints are not supported by the teaser evaluation")
    model = ResNet(ResidualBlock, [3, 4, 6, 3], width_multiplier=1.0, heteroscedastic=False,
                   negative_slope=cfg.get('leaky_relu_slope', 0.1)).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    teaser = ck.get('teaser')
    if teaser is None:
        raise SystemExit("checkpoint has no 'teaser' block (was it trained with train_teaser_ddp.py?)")
    return model, ck, teaser


def load_eval(path, n_real):
    with h5py.File(path, 'r') as f:
        n_psd = int(f.attrs['n_psd'])
        noise = f['noise'][n_psd:]
        tau = f['template_ext'][:]
        out = dict(
            model=str(f.attrs['model']), L=float(f.attrs['L']), n_psd=n_psd,
            master_seed=int(f.attrs['master_seed']),
            sigma_mf_exact=float(f.attrs.get('sigma_mf_exact', np.nan)),
            sigma_mf_phat=float(f.attrs['sigma_mf_pred']),
            sigma_mf_paired_phat=float(f.attrs['sigma_mf_emp']),
            sigma_mf_paired_exact=float(f.attrs.get('sigma_mf_emp_exactfilter', np.nan)),
            campaign_sigma_mf_pred=float(f.attrs.get('campaign_sigma_mf_pred', np.nan)),
            campaign_sigma_mf_emp=float(f.attrs.get('campaign_sigma_mf_emp', np.nan)),
        )
        if 'mf_amplitudes_eval_exact' in f:
            a_mf = f['mf_amplitudes_eval_exact'][:]
            out['mf_filter'] = 'exact P_bar'
        else:
            a_mf = f['mf_amplitudes_eval'][:]
            out['mf_filter'] = 'P_hat (PSD split)'
    if n_real is not None and n_real < len(noise):
        noise, a_mf = noise[:n_real], a_mf[:n_real]
    return noise.astype(np.float32), tau.astype(np.float32), a_mf.astype(np.float64), out


#%% ================================================================================
# === THE A-GRID FORWARD PASS ===
# ==================================================================================

@torch.no_grad()
def predict_grid(model, noise, tau, A_grid, norm_min, norm_max, device, batch=256):
    """A_hat[j, i] = network(normalise(A_j tau + n_i)).  Normalisation identical
    to utils_teaser_ddp.HDF5ImageDataset.normalize (float32, global constants)."""
    n_real = len(noise)
    out = np.empty((len(A_grid), n_real), dtype=np.float64)
    scale = np.float32(norm_max - norm_min) if norm_max > norm_min else np.float32(1.0)
    tau_t = torch.from_numpy(tau).to(device)
    t0 = time.time()
    for j, A in enumerate(A_grid):
        for i0 in range(0, n_real, batch):
            n = torch.from_numpy(noise[i0:i0 + batch]).to(device)
            d = (n + np.float32(A) * tau_t - np.float32(norm_min)) / scale
            out[j, i0:i0 + batch] = model(d.unsqueeze(1)).float().cpu().numpy()
        if j % 10 == 0:
            print(f"  A = {A:.3f}  ({j + 1}/{len(A_grid)}, {time.time() - t0:.0f} s)", flush=True)
    return out


#%% ================================================================================
# === STATISTICS ON THE GRID ===
# ==================================================================================

def grid_stats(A_grid, Ahat):
    """stats dict in the biasfixed_diagnostics format, one 'bin' per grid point."""
    n = Ahat.shape[1]
    resid = Ahat - A_grid[:, None]
    g = resid.mean(axis=1)
    V = Ahat.var(axis=1, ddof=1)
    dA = A_grid[1] - A_grid[0]
    edges = np.concatenate([[A_grid[0] - dA / 2], 0.5 * (A_grid[1:] + A_grid[:-1]),
                            [A_grid[-1] + dA / 2]])
    return {
        'bin_centers': A_grid.copy(), 'bin_edges': edges,
        'cond_mean_pred': Ahat.mean(axis=1), 'cond_bias': g, 'cond_var': V,
        'cond_bias_sq': g**2, 'cond_mse': (resid**2).mean(axis=1),
        'cond_bias_err': resid.std(axis=1, ddof=1) / np.sqrt(n),
        'cond_var_err': V * np.sqrt(2.0 / (n - 1)),
        'counts': np.full(len(A_grid), n, dtype=int), 'valid': np.ones(len(A_grid), bool),
    }


def band_average(values, mask):
    return float(np.mean(values[mask])) if mask.any() else float('nan')


def bootstrap_band(A_grid, Ahat, gprime, sigma, region_mask, strict_mask, n_boot, seed=0):
    """Bootstrap over MAPS of the band-averaged V/sigma^2 and R (g' held fixed)."""
    rng = np.random.default_rng(seed)
    n = Ahat.shape[1]
    env = (1.0 + gprime)**2 * sigma**2
    res = {k: [] for k in ('V_region', 'R_region', 'V_strict', 'R_strict', 'g_region')}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Vb = Ahat[:, idx].var(axis=1, ddof=1)
        gb = Ahat[:, idx].mean(axis=1) - A_grid
        res['V_region'].append(band_average(Vb / sigma**2, region_mask))
        res['R_region'].append(band_average(Vb / env, region_mask))
        res['V_strict'].append(band_average(Vb / sigma**2, strict_mask))
        res['R_strict'].append(band_average(Vb / env, strict_mask))
        res['g_region'].append(band_average(gb, region_mask))
    out = {}
    for k, v in res.items():
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        out[k] = ((float(v[ok].mean()), float(v[ok].std(ddof=1))) if ok.sum() > 1
                  else (float('nan'), float('nan')))
    return out


def crossover(A_grid, ratio, level=1.0):
    """First A (interpolated) at which ratio(A) rises through `level` --
    the point where the network's gain over the MF fades (T2 prediction A ~ sigma_MF)."""
    below = ratio < level
    if not below.any() or below.all():
        return None
    for j in range(1, len(A_grid)):
        if below[j - 1] and not below[j]:
            f = (level - ratio[j - 1]) / (ratio[j] - ratio[j - 1])
            return float(A_grid[j - 1] + f * (A_grid[j] - A_grid[j - 1]))
    return None


#%% ================================================================================
# === FIGURES ===
# ==================================================================================

def teaser_figure(A_grid, st, gprime, band, sigma, model, refs, method_e, mf_ratio, title, path):
    """Two panels: g(A) with +-2 SE and the bands; V/sigma_MF^2 with the [1+g']^2
    envelope and the reference lines.  Style close to make_figures/figcommon.py."""
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'font.family': 'serif',
                         'mathtext.fontset': 'cm', 'axes.linewidth': 0.6, 'lines.linewidth': 1.2})
    C = dict(anchor='#7f7f7f', rung='#1f5fa8', exact='#2a9d4f', bound='#c8401f', cmp='#e08a1e',
             pending='#8e44ad')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x = A_grid / sigma
    g, se = st['cond_bias'], st['cond_bias_err']
    ax1.fill_between(x, -2 * se / sigma, 2 * se / sigma, color=C['anchor'], alpha=0.25, lw=0,
                     label=r'$\pm 2\,\mathrm{SE}$ null band')
    ax1.plot(x, g / sigma, color=C['rung'], label=r'$g(A)=\mathbb{E}[\hat A|A]-A$')
    for key, col, lab in (('region_mask', C['exact'], 'bias-consistent region'),
                          ('band_mask', C['cmp'], r"strict $g'\approx 0$ band")):
        m = band.get(key)
        if m is not None and np.any(m):
            lo, hi = x[m].min(), x[m].max()
            ax1.axvspan(lo, hi, color=col, alpha=0.12, lw=0, label=lab)
    ax1.axhline(0, color='k', lw=0.5)
    ax1.set_xlabel(r'$A/\sigma_{\rm MF}$'); ax1.set_ylabel(r'$g(A)/\sigma_{\rm MF}$')
    ax1.legend(loc='best', frameon=False)

    V = st['cond_var'] / sigma**2
    Verr = st['cond_var_err'] / sigma**2
    ax2.fill_between(x, V - Verr, V + Verr, color=C['rung'], alpha=0.2, lw=0)
    ax2.plot(x, V, color=C['rung'], label=r'$V(A)/\sigma_{\rm MF}^2$ (network)')
    ax2.plot(x, (1 + gprime)**2, color=C['bound'], ls='--',
             label=r"$[1+g'(A)]^2$ biased-CRLB envelope")
    styles = {'eta=1 floor (theorem)': ('k', '-'), 'eta=1 floor': ('k', '-'),
              'sigma_norm^2 (band weight map)': (C['cmp'], ':'),
              'reweight rung 1/7.119': (C['rung'], '-.'),
              'exact ceiling 1/10.18': (C['exact'], '--')}
    for lab, val in refs.items():
        col, ls = styles.get(lab, (C['pending'], ':'))
        ax2.axhline(val, color=col, ls=ls, lw=0.9, label=lab.replace('sigma_norm^2', r'$\sigma_{\rm norm}^2$')
                    .replace('eta=1', r'$\eta=1$'))
    if method_e is not None:
        ax2.plot(method_e['a_over_sigma'], method_e['v_over_sigma2'], color=C['exact'], lw=1.0,
                 alpha=0.8, label=r'Method-E envelope $V_E(A)$')
    if mf_ratio is not None:
        ax2.axhline(mf_ratio, color=C['anchor'], ls=':', lw=0.9,
                    label=r'paired MF on the same maps')
    for key, col in (('region_mask', C['exact']), ('band_mask', C['cmp'])):
        m = band.get(key)
        if m is not None and np.any(m):
            ax2.axvspan(x[m].min(), x[m].max(), color=col, alpha=0.12, lw=0)
    ax2.set_xlabel(r'$A/\sigma_{\rm MF}$'); ax2.set_ylabel(r'$V(A)/\sigma_{\rm MF}^2$')
    ymax = max(1.6, float(np.nanmax(V[np.isfinite(V)]) * 1.15)) if np.isfinite(V).any() else 1.6
    ax2.set_ylim(0, min(ymax, 3.0)); ax2.legend(loc='best', frameon=False, ncol=1)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


#%% ================================================================================
# === MAIN ===
# ==================================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--eval-h5', required=True, help='noise-only file from make_teaser_dataset.py')
    ap.add_argument('--outdir', default='eval_out')
    ap.add_argument('--n-grid', type=int, default=41)
    ap.add_argument('--n-real', type=int, default=None, help='EVAL maps to use (default: all)')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--sigma-ref', choices=['exact', 'phat', 'paired'], default='exact',
                    help='which sigma_MF normalises V(A) (all three are reported)')
    ap.add_argument('--eps-slope', type=float, default=0.05)
    ap.add_argument('--bias-floor-frac', type=float, default=0.10)
    ap.add_argument('--max-gap', type=int, default=1)
    ap.add_argument('--n-boot', type=int, default=500)
    ap.add_argument('--method-e-json', default=None,
                    help='r3_channels_wn.json for the Method-E envelope (default: next to script)')
    ap.add_argument('--label', default=None, help='run label for the log block, e.g. T1a')
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem
    print(f"device {device};  checkpoint {args.checkpoint}")

    model, ck, teaser = load_checkpoint(args.checkpoint, device)
    noise, tau, a_mf, info = load_eval(args.eval_h5, args.n_real)
    L = float(teaser.get('L', info['L']))
    if info['model'] != teaser.get('model', info['model']):
        print(f"WARNING: checkpoint model {teaser.get('model')} != eval file model {info['model']}")
    n_real = len(noise)
    print(f"model {info['model']}  L = {L}  eval maps {n_real}  norm [{teaser['norm_min']:.4f}, "
          f"{teaser['norm_max']:.4f}]  loss {ck['config'].get('loss_function')}")

    sig_exact, sig_phat = info['sigma_mf_exact'], info['sigma_mf_phat']
    sig_paired = float(a_mf.std(ddof=1))
    sigma = {'exact': sig_exact if np.isfinite(sig_exact) else sig_phat,
             'phat': sig_phat, 'paired': sig_paired}[args.sigma_ref]
    print(f"sigma_MF: exact {sig_exact:.4f}  P_hat {sig_phat:.4f}  paired ({info['mf_filter']}, "
          f"{n_real} maps) {sig_paired:.4f}  -> using {args.sigma_ref} = {sigma:.4f}")

    # ---- forward pass on the grid ----
    A_grid = np.linspace(0.0, L, args.n_grid)
    print(f"forward pass: {args.n_grid} x {n_real} maps ...")
    Ahat = predict_grid(model, noise, tau, A_grid, teaser['norm_min'], teaser['norm_max'],
                        device, args.batch)
    np.savez_compressed(outdir / f"{stem}_grid.npz", A_grid=A_grid, Ahat=Ahat.astype(np.float32),
                        mf_noise_amplitudes=a_mf)

    # ---- statistics, bands, ratios ----
    st = grid_stats(A_grid, Ahat)
    A_fine, mu_prime, gsp, _ = bd.compute_local_slope(st['bin_centers'], st['cond_bias'],
                                                     st['cond_bias_err'], st['valid'])
    band = bd.find_locally_unbiased_band(st, gsp, eps_slope=args.eps_slope, sigma_mf=sigma,
                                         bias_floor_frac=args.bias_floor_frac,
                                         max_gap=args.max_gap)
    gprime = np.asarray(band['g_prime'], dtype=float)
    y_true = np.repeat(A_grid, n_real); y_pred = Ahat.ravel()
    summ = bd.compute_band_summaries(st, band, y_true, y_pred, sigma_mf=sigma)
    scalar = bd.compute_scalar_summaries(st, sigma_mf=sigma)

    region_mask = np.asarray(band.get('region_mask', np.zeros(len(A_grid), bool)), bool)
    strict_mask = np.asarray(band.get('band_mask', np.zeros(len(A_grid), bool)), bool)
    env = (1.0 + gprime)**2 * sigma**2
    V_ratio = st['cond_var'] / sigma**2
    R = st['cond_var'] / env
    V_mf_paired = float(a_mf.var(ddof=1))
    V_ratio_paired = st['cond_var'] / V_mf_paired
    boot = bootstrap_band(A_grid, Ahat, gprime, sigma, region_mask, strict_mask, args.n_boot)

    # ---- reference lines / Method-E ----
    refs = REFERENCE_LINES.get(info['model'], {'eta=1 floor': 1.0})
    method_e = None
    mej = Path(args.method_e_json) if args.method_e_json else _here.parent / 'results' / 'eta_results' / 'r3_channels_wn.json'
    if info['model'] in ('T1_PSRAND_WN',) and mej.exists():
        me = json.load(open(mej))[info['model']]['extended']['method_e']
        method_e = {k: list(map(float, me[k])) for k in ('a_over_sigma', 'v_over_sigma2')}
    x_cross = crossover(A_grid, V_ratio, 1.0)

    # ---- report ----
    rg, stb = summ['region'], summ['strict']
    print("=" * 72)
    print(f"TEASER EVALUATION  {stem}   model {info['model']}   sigma_MF({args.sigma_ref}) = {sigma:.4f}")
    print("=" * 72)
    if rg is None:
        print("  bias-consistent region: NOT FOUND")
    else:
        print(f"  bias-consistent region: A in [{rg['A_lo']:.3f}, {rg['A_hi']:.3f}]  "
              f"({region_mask.sum()} grid points)  = [{rg['A_lo']/sigma:.2f}, {rg['A_hi']/sigma:.2f}] sigma_MF")
        print(f"    <V>/sigma_MF^2 = {boot['V_region'][0]:.4f} +- {boot['V_region'][1]:.4f}   "
              f"<R> = {boot['R_region'][0]:.4f} +- {boot['R_region'][1]:.4f}   "
              f"MACB = {rg['MACB']:.4f}   dg/dA = {rg['dg_dA']:+.4f} +- {rg['dg_dA_se']:.4f}")
    if stb is None:
        print("  strict g'~0 band: NOT FOUND")
    else:
        print(f"  strict g'~0 band:       A in [{stb['A_lo']:.3f}, {stb['A_hi']:.3f}]  "
              f"<V>/sigma^2 = {boot['V_strict'][0]:.4f} +- {boot['V_strict'][1]:.4f}   "
              f"<R> = {boot['R_strict'][0]:.4f} +- {boot['R_strict'][1]:.4f}")
    print(f"  paired MF on the same {n_real} maps: sigma = {sig_paired:.4f} "
          f"(= {sig_paired/sigma:.3f} x sigma_ref);  V_CNN/V_MF,paired at A = L/2: "
          f"{np.interp(L/2, A_grid, V_ratio_paired):.3f}")
    print(f"  V/sigma^2 at A = 0: {V_ratio[0]:.3f};  at L/2: {np.interp(L/2, A_grid, V_ratio):.3f};  "
          f"at L: {V_ratio[-1]:.3f};  crossover V/sigma^2 = 1 at A/sigma = "
          f"{'none' if x_cross is None else f'{x_cross/sigma:.2f}'}")
    print(f"  global g'(A) range: [{gprime.min():+.3f}, {gprime.max():+.3f}];  "
          f"min R over grid: {np.nanmin(R):.3f}")
    print("=" * 72)

    # ---- figures ----
    bd.plot_biasfixed_diagnostics(st, band, A_fine, mu_prime, summ, scalar, sigma_mf=sigma,
                                  title_prefix=stem, save_path=str(outdir / f"{stem}_diag.png"))
    plt.close('all')
    title = (f"{info['model']}, extended template, {ck['config'].get('loss_function')}, "
             f"L = {L:g}, n = {n_real} maps x {args.n_grid} amplitudes")
    teaser_figure(A_grid, st, gprime, band, sigma, info['model'], refs, method_e,
                  V_mf_paired / sigma**2, title, outdir / f"{stem}_teaser.png")

    # ---- JSON ----
    def f(x):
        return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x
    result = dict(
        checkpoint=str(args.checkpoint), eval_h5=str(args.eval_h5), model=info['model'], L=L,
        loss_function=ck['config'].get('loss_function'), n_real=n_real, n_grid=args.n_grid,
        sigma_ref=args.sigma_ref, sigma_mf=dict(used=sigma, exact=f(sig_exact), phat=sig_phat,
                                               paired=sig_paired, paired_filter=info['mf_filter'],
                                               campaign_pred=f(info['campaign_sigma_mf_pred']),
                                               campaign_emp=f(info['campaign_sigma_mf_emp'])),
        r_design=float(np.sqrt(12) * sigma / L),
        A_grid=A_grid.tolist(), g=st['cond_bias'].tolist(), g_se=st['cond_bias_err'].tolist(),
        V=st['cond_var'].tolist(), V_se=st['cond_var_err'].tolist(), g_prime=gprime.tolist(),
        V_over_sigma2=V_ratio.tolist(), R=R.tolist(), V_over_Vmf_paired=V_ratio_paired.tolist(),
        envelope=((1 + gprime)**2).tolist(),
        region=(None if rg is None else dict(A_lo=rg['A_lo'], A_hi=rg['A_hi'],
                                              n_points=int(region_mask.sum()), MACB=rg['MACB'],
                                              dg_dA=rg['dg_dA'], dg_dA_se=rg['dg_dA_se'],
                                              V_over_sigma2=boot['V_region'], R=boot['R_region'],
                                              g_mean=boot['g_region'])),
        strict=(None if stb is None else dict(A_lo=stb['A_lo'], A_hi=stb['A_hi'],
                                               n_points=int(strict_mask.sum()),
                                               V_over_sigma2=boot['V_strict'], R=boot['R_strict'])),
        crossover_A_over_sigma=(None if x_cross is None else x_cross / sigma),
        references=refs, method_e=method_e, gates=dict(eps_slope=args.eps_slope,
                                                       bias_floor_frac=args.bias_floor_frac,
                                                       k_se=2.0, max_gap=args.max_gap),
        training=dict(config=ck.get('config'), teaser={k: v for k, v in teaser.items()
                                                       if k != 'h5_attrs'},
                      epochs=ck.get('epoch'), final_val_loss=(ck.get('test_losses') or [None])[-1]),
        eval_seed=info['master_seed'], n_boot=args.n_boot,
    )
    json.dump(result, open(outdir / f"{stem}_eval.json", 'w'), indent=1, default=float)

    # ---- Markdown run-log block ----
    lab = args.label or stem
    tt = teaser
    hp = ck.get('config', {})
    lines = [f"### Run {lab} -- `{info['model']}`, extended, {hp.get('loss_function')}",
             f"- dataset: {hp.get('h5_file')} / master seed {tt.get('h5_attrs', {}).get('master_seed')} / "
             f"n_total {tt.get('h5_attrs', {}).get('n_samples')} (val_split {hp.get('val_split')}) / "
             f"L = {L:g} / r = {np.sqrt(12) * sigma / L:.3f}",
             f"- sigma_MF: exact {sig_exact:.4f}, P_hat {sig_phat:.4f}, paired {sig_paired:.4f} "
             f"(campaign {info['campaign_sigma_mf_pred']:.3f} / {info['campaign_sigma_mf_emp']:.3f}); used {args.sigma_ref}",
             f"- training: train_teaser_ddp.py / epochs {hp.get('num_epochs')} / lr {hp.get('initial_lr')} "
             f"(scaled {hp.get('scaled_lr')}) / batch {hp.get('batch_size')} x {hp.get('num_gpus')} GPU / "
             f"aug {tt.get('augmentation')} / bias bins {tt.get('bias_bins')}",
             f"- checkpoint: {args.checkpoint}",
             f"- eval: {args.eval_h5} (seed {info['master_seed']}), {n_real} maps x {args.n_grid} A; "
             f"outputs {outdir}/{stem}_eval.json, _grid.npz, _diag.png, _teaser.png"]
    if rg is not None:
        lines.append(f"- band [A_lo, A_hi] = [{rg['A_lo']:.3f}, {rg['A_hi']:.3f}] "
                     f"([{rg['A_lo']/sigma:.2f}, {rg['A_hi']/sigma:.2f}] sigma_MF); "
                     f"<V>/sigma_MF^2 = {boot['V_region'][0]:.3f} +- {boot['V_region'][1]:.3f}; "
                     f"<R>_band = {boot['R_region'][0]:.3f} +- {boot['R_region'][1]:.3f}; MACB_band = {rg['MACB']:.4f}")
    else:
        lines.append("- band: bias-consistent region NOT found")
    if stb is not None:
        lines.append(f"- strict g'~0 band [{stb['A_lo']:.3f}, {stb['A_hi']:.3f}]: <R> = {boot['R_strict'][0]:.3f} +- {boot['R_strict'][1]:.3f}")
    lines.append(f"- V/sigma^2 at A=0 / L/2 / L: {V_ratio[0]:.3f} / {np.interp(L/2, A_grid, V_ratio):.3f} / {V_ratio[-1]:.3f}; "
                 f"crossover (V/sigma^2 = 1) at A/sigma_MF = {'none' if x_cross is None else f'{x_cross/sigma:.2f}'}; "
                 f"V_CNN/V_MF,paired at L/2 = {np.interp(L/2, A_grid, V_ratio_paired):.3f}")
    if info['model'] == 'T1_PSRAND_WN':
        lines.append("- comparison with sigma_norm^2 (0.224), 1/7.12 (0.140), 1/10.18 (0.098): "
                     f"in-band <V>/sigma^2 = {boot['V_region'][0]:.3f}")
    lines.append("- notes:")
    (outdir / f"{stem}_runlog.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == '__main__':
    main()
