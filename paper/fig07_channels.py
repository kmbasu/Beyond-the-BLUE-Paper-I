"""
fig07_channels.py -- FIG 7: the two channels of the covariance-mixture advantage.

Purpose
-------
For each floored covariance-mixture cell: the coherent (band-level) scatter
sd[<delta>_f], the incoherent (within-band shape) rms E[Var_f delta]^{1/2}, and the
resulting decomposition of the exact ceiling into the scalar-weighting gain
eta_B = E[c]E[1/c] (what a weight map can take) times the re-shaping remainder eta/eta_B.

Two panels: (a) the two severity coordinates per cell; (b) log eta split into the two
factors, with the exact eta on top.  PCA leakage uses the numbers of the exact
scale-mixture quadrature (v4 Sec. 6.2), which are typed in below.

Data
----
  eta_results/r3_channels_wn.json
  eta_results/<MODEL>__<template>.json (quadrature)

Usage
-----
    python fig07_channels.py
"""
from figcommon import *

ch = json.load(open(RESULTS / 'r3_channels_wn.json'))
CELLS = [('T1_PSRAND_WN', 'extended'), ('T1_PSRAND_WN', 'compact'),
         ('T1_RANDOR_WN', 'extended'), ('T1_RANDOR_WN', 'compact'),
         ('T2_PCA_WN', 'extended'), ('T2_PCA_WN', 'compact')]
PCA = {'extended': dict(eta=2.140, etaB=1.32, sd_band=0.22, rms_shape=0.36),
       'compact': dict(eta=1.088, etaB=1.02, sd_band=0.005, rms_shape=0.06)}

rows = []
for model, tpl in CELLS:
    if model in ch:
        d = ch[model][tpl]['decomposition']
        eta = load_cell(model, tpl)['quadrature']['eta']
        rows.append(dict(label=f'{display_name(model)}\n{tpl}', eta=eta, etaB=d['eta_ivw'],
                         sd_band=d['sd_band'], rms_shape=d['rms_shape'], tpl=tpl))
    else:
        p = PCA[tpl]
        rows.append(dict(label=f'{display_name(model)}\n{tpl}', tpl=tpl, **p))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw={'width_ratios': [1, 1.1]})
x = np.arange(len(rows)); w = 0.36
ax = axes[0]
ax.bar(x - w / 2, [r['sd_band'] for r in rows], w, color=C['exact'], label=r'coherent: sd$[\langle\delta\rangle_{\hat f}]$')
ax.bar(x + w / 2, [r['rms_shape'] for r in rows], w, color=C['rung'], label=r'incoherent: $\mathbb{E}[\mathrm{Var}_{\hat f}\delta]^{1/2}$')
ax.set_xticks(x); ax.set_xticklabels([r['label'] for r in rows], fontsize=7)
ax.set_ylabel('log band-power scatter')
ax.set_title('(a) the two severity coordinates', loc='left')
ax.legend(frameon=False, fontsize=7.5)

ax = axes[1]
lB = np.log([r['etaB'] for r in rows]); lR = np.log([r['eta'] / r['etaB'] for r in rows])
ax.bar(x, lB, 0.6, color=C['exact'], label=r'scalar weighting, $\log\eta_B$ (weight-map accessible)')
ax.bar(x, lR, 0.6, bottom=lB, color=C['rung'], label=r'filter re-shaping, $\log(\eta/\eta_B)$')
for xi, r in zip(x, rows):
    ax.text(xi, np.log(r['eta']) + 0.04, f"{r['eta']:.2f}", ha='center', fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([r['label'] for r in rows], fontsize=7)
ax.set_ylabel(r'$\log\eta$')
yt = [1, 1.5, 2, 3, 5, 10]; ax.set_yticks(np.log(yt)); ax.set_yticklabels([str(t) for t in yt])
ax.set_ylim(0, np.log(14))
ax.set_title(r'(b) the exact ceiling split into its two factors', loc='left')
ax.legend(frameon=False, fontsize=7.5, loc='upper right')
fig.subplots_adjust(wspace=0.25)
savefig(fig, 'fig07_channels')
