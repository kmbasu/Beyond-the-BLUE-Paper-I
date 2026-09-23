"""
fig06_methodE.py -- FIG 6: where the covariance-mixture advantage lives in amplitude.

Purpose
-------
The Method-E attainability envelope V_E(A) (Eq. methodE) for the floored tilt mixture
(extended and compact templates) and the floored random-orientation anisotropy, against
the flat Cramer-Rao floor 1/eta and the mixture-matched-filter rung, with the certified
band A >= 2 sigma_MF shaded.  All curves are in units of sigma_MF^2 against A/sigma_MF.

Data
----
  eta_results/r3_channels_wn.json   ['<MODEL>'][template]['method_e']{a_over_sigma, v_over_sigma2}
  eta_results/<MODEL>__<template>.json   quadrature eta (floor) and the reweight rung

Usage
-----
    python fig06_methodE.py
"""
from figcommon import *

ch = json.load(open(RESULTS / 'r3_channels_wn.json'))
PANELS = [('T1_PSRAND_WN', 'extended', 'spectral-tilt mixture, extended'),
          ('T1_PSRAND_WN', 'compact', 'spectral-tilt mixture, compact'),
          ('T1_RANDOR_WN', 'compact', 'random-orientation anisotropy, compact')]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=False)
for ax, (model, tpl, title) in zip(axes, PANELS):
    me = ch[model][tpl]['method_e']
    a = np.array(me['a_over_sigma']); v = np.array(me['v_over_sigma2'])
    cell = load_cell(model, tpl)
    eta = cell['quadrature']['eta']
    rw = (cell.get('variational2') or {}).get('rungs', {}).get('reweight', {})
    etaB = ch[model][tpl]['decomposition']['eta_ivw']
    ax.axvspan(2.0, a.max(), color=C['anchor'], alpha=0.10, lw=0)
    ax.axhline(1.0, color=C['anchor'], lw=0.8, label=r'ensemble MF, $\sigma_{\rm MF}^2$')
    ax.axhline(1.0 / eta, color=C['exact'], lw=1.0, ls='--', label=rf'floor $1/\eta = 1/{eta:.2f}$')
    ax.plot(a, v, color=C['ext'] if tpl == 'extended' else C['cmp'], lw=1.4,
            label=r'envelope $V_E(A)$ (scalar rescaling)')
    if rw and rw.get('eta') and rw['eta'] > 1.02:
        ax.axhline(1.0 / rw['eta'], color=C['rung'], lw=0.9, ls=':',
                   label=rf'mixture MF rung, $1/{rw["eta"]:.2f}$')
    ax.text(0.03, 0.05, rf'$\eta_B = {etaB:.2f}$', transform=ax.transAxes, fontsize=8)
    ax.text(2.1, 0.93, 'certified band', fontsize=7, color=C['anchor'], va='top')
    ax.set_yscale('log')
    ax.set_ylim(min(0.08, 0.7 / eta), 1.4)
    ax.set_xlim(0, a.max())
    ax.set_xlabel(r'$A/\sigma_{\rm MF}$')
    ax.set_title(title, loc='left')
    ax.legend(frameon=False, fontsize=7, loc='center right')
axes[0].set_ylabel(r'$V(A)/\sigma_{\rm MF}^{2}$')
fig.subplots_adjust(wspace=0.28)
savefig(fig, 'fig06_methodE')
