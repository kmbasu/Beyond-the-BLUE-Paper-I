#!/usr/bin/env python
"""
make_fig12_teaser.py -- Paper I Fig. 3 (Sec. 6.1, Sec. 7): the ResNet teaser figure
from the teaser_eval.py JSONs.

Layout: one row per model, two panels per row.
  left  : g(A)/sigma_MF, the +-2 SE null band, and the certified bias-consistent region of
          the primary run (shaded)
  right : V(A)/sigma_MF^2 with the biased-CRLB envelope [1+g'(A)]^2 of the primary run,
          the eta = 1 floor, and -- for T1_PSRAND_WN -- the band-level sigma_norm^2, the
          reweight rung, the exact ceiling and the Method-E envelope V_E(A)
Both loss variants of a model are drawn (solid = primary, dashed = secondary); the
curves come from the BEST-VALIDATION checkpoints unless other JSONs are passed.

Usage
-----
    python make_fig12_teaser.py --out fig12_resnet_teaser --figdir ../Figures \\
        eval_out_best/TEASER_T0_RED_REAL_L5_bc15.pth.best_eval.json \\
        eval_out_best/TEASER_T0_RED_REAL_L5_mse.pth.best_eval.json \\
        eval_out_best/TEASER_T1_PSRAND_WN_L7_bc15.pth.best_eval.json \\
        eval_out_best/TEASER_T1_PSRAND_WN_L7_mse.pth.best_eval.json
The first JSON given for a model is the primary (solid) run.  Tries to import the paper's
figcommon.py (palette / style / FIGDIR); falls back to an equivalent inline style.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('jsons', nargs='+')
ap.add_argument('--out', default='fig12_resnet_teaser')
ap.add_argument('--figdir', default=None)
ap.add_argument('--xmax', type=float, default=None, help='upper A/sigma_MF of the panels (default: L/sigma)')
args = ap.parse_args()

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper'))
    from figcommon import C, FIGDIR, display_name  # noqa
except Exception:
    C = {'anchor': '#7f7f7f', 'rung': '#1f5fa8', 'exact': '#2a9d4f', 'bound': '#c8401f',
         'cmp': '#e08a1e', 'pending': '#8e44ad'}
    FIGDIR = Path('.')
    _N = {'T0_RED_REAL': 'red + white noise', 'T1_PSRAND_WN': 'spectral-tilt mixture + floor',
          'T0_WHITE': 'white noise'}
    display_name = lambda m: _N.get(m, m)
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 7,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'font.family': 'serif',
                     'mathtext.fontset': 'cm', 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
                     'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
figdir = Path(args.figdir) if args.figdir else Path(FIGDIR)
figdir.mkdir(parents=True, exist_ok=True)

runs = [json.load(open(p)) for p in args.jsons]
models = []
for r in runs:
    if r['model'] not in models:
        models.append(r['model'])
LOSS_LABEL = {'mse': 'MSE loss', 'bias_corrected_mse': 'bias-corrected loss',
              'ema_bias_corrected_mse': 'bias-corrected loss (EMA)'}
PANEL = 'abcdefgh'

fig, axes = plt.subplots(len(models), 2, figsize=(7.1, 2.55 * len(models)), squeeze=False)
for row, model in enumerate(models):
    ax1, ax2 = axes[row]
    rs = [r for r in runs if r['model'] == model]
    sigma = rs[0]['sigma_mf']['used']
    L = rs[0]['L']
    xmax = args.xmax or L / sigma
    for k, r in enumerate(rs):
        ls = '-' if k == 0 else '--'
        lab = LOSS_LABEL.get(r['loss_function'], r['loss_function'])
        A = np.array(r['A_grid']); x = A / sigma
        g, se = np.array(r['g']), np.array(r['g_se'])
        if k == 0:
            ax1.fill_between(x, -2 * se / sigma, 2 * se / sigma, color=C['anchor'], alpha=0.25, lw=0,
                             label=r'$\pm 2\,$SE of $g$')
            if r.get('region'):
                lo, hi = r['region']['A_lo'] / sigma, r['region']['A_hi'] / sigma
                ax1.axvspan(lo, hi, color=C['exact'], alpha=0.10, lw=0, label='certified region')
                ax2.axvspan(lo, hi, color=C['exact'], alpha=0.10, lw=0)
            ax2.plot(x, np.array(r['envelope']), color=C['bound'], ls=':', lw=1.1,
                     label=r"$[1+g'(A)]^2$ (biased CRLB)")
        ax1.plot(x, g / sigma, color=C['rung'], ls=ls, label=lab)
        ax2.plot(x, np.array(r['V_over_sigma2']), color=C['rung'], ls=ls, label=lab)
    ax1.axhline(0, color='k', lw=0.5)
    ax2.axhline(1.0, color='k', lw=0.9, label=r'matched filter, $\eta=1$')
    refs = rs[0].get('references', {})
    for lab, val in refs.items():
        if 'eta=1' in lab:
            continue
        if 'sigma_norm' in lab:
            col, txt = C['cmp'], r'$\sigma_{\rm norm}^2$ (band weight map)'
        elif 'reweight' in lab:
            col, txt = C['rung'], 'reweight rung, $1/7.12$'
        else:
            col, txt = C['exact'], 'exact ceiling, $1/10.18$'
        ax2.axhline(val, color=col, ls='-.', lw=0.8, label=txt)
    me = rs[0].get('method_e')
    if me:
        xe, ye = np.array(me['a_over_sigma']), np.array(me['v_over_sigma2'])
        m = xe <= xmax * 1.02
        ax2.plot(xe[m], ye[m], color=C['exact'], lw=1.0, alpha=0.9, label=r'Method-E envelope $V_E(A)$')
    for ax in (ax1, ax2):
        ax.set_xlim(0, xmax)
    ax1.set_ylabel(r'$g(A)/\sigma_{\rm MF}$'); ax2.set_ylabel(r'$V(A)/\sigma_{\rm MF}^2$')
    ax1.set_title(f"({PANEL[2*row]}) {display_name(model)}, ext. template", loc='left')
    ax2.set_title(f"({PANEL[2*row+1]})", loc='left')
    ax2.set_ylim(0, 1.75 if model != 'T1_PSRAND_WN' else 1.95)
    ax1.legend(frameon=False, loc='lower left' if model != 'T1_PSRAND_WN' else 'upper right')
    ax2.legend(frameon=False, loc='upper left', ncol=2, handlelength=2.0, columnspacing=1.0)
for ax in axes[-1]:
    ax.set_xlabel(r'$A/\sigma_{\rm MF}$')
fig.tight_layout(h_pad=1.2)
for ext in ('pdf', 'png'):
    fig.savefig(figdir / f"{args.out}.{ext}")
print('wrote', figdir / f"{args.out}.pdf")
