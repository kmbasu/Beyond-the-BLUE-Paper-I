"""
fig08_ladders.py -- FIG 8: the variational ladder for every cell of the realistic configuration.

Purpose
-------
The paper's main results figure. One panel per noise model (floored configuration),
both templates: the value of each rung of the Volterra ladder
(linear -> reweight -> quadratic -> cubic -> conv) with its EVAL bootstrap error,
against the model's ceiling -- an exact quadrature (green) where the model is a
covariance mixture, a complete-data upper bound (red) where the structure is
removable, nothing where neither exists (row-median residual). The linear anchor
sits at eta ~ 1 (grey band).

Reading rule (Sec. 10): every rung is a certified LOWER bracket on eta; the exact
line or the bound is the UPPER bracket. A rung at the anchor means the searched
class found nothing above the ladder's sensitivity floor (~1.05 cubic, ~1.15 conv),
not that the noise has nothing.

Data
----
  eta_results/<MODEL>__<template>.json          rungs: ['variational2']['rungs'][rung]['eta','err']
  eta_results/<MODEL>__<template>.json          exact : ['quadrature']['eta','err']   (T1)
  eta_results/pca_mixture_quadrature.json       exact : ['floored'][template]['eta_well_inferable'] (+ v4 table)
  eta_results/<MODEL>__<template>.json          bound : ['bound_complete_data']
  eta_results/confusion_bound_dc_check.json     DC-consistent confusion bounds (supersede the stored ones)
  T2_CONFUSION_ATM exact eta = 3.5688 (analytic; tools/eta_exact_confusion.py)

Only rungs that were actually run and stored are plotted. The beam-sharing-companion
panel has no 'reweight' point: that cell whitens to i.i.d. pixels by construction, so a
filter whose weights key on measured band powers has nothing to key on. It was in fact
run twice on this cell and returned the linear anchor to seven digits with zero restart
spread, but those values belong to earlier passes and are not spliced into the final one.

Usage
-----
    python fig08_ladders.py
"""
from figcommon import *

PANELS = [
    ('T1_PSRAND_WN',      'spectral-tilt mixture'),
    ('T1_RANDOR_WN',      'random-orientation anisotropy'),
    ('T2_PCA_WN',         'PCA leakage (scale mixture)'),
    ('T2_MEDIAN_WN',      'row-median residual'),
    ('T2_CROSS_SYM_WN',   'scan crossings (sym.)'),
    ('T2_CROSS_POS_WN',   'scan crossings (pos.)'),
    ('T2_GLITCH_WN',      'sub-threshold glitches'),
    ('T2_CONFUSION_WN',   'source confusion'),
    ('T2_CONFUSION_ATM',  'beam-sharing companion (exact test)'),
]
PCA_EXACT = {'extended': (2.140, 0.050), 'compact': (1.088, 0.020)}   # v4 §6.2
ATM_EXACT = 3.5688                                                     # analytic
YMIN = 0.7

dc = json.load(open(RESULTS / 'confusion_bound_dc_check.json')) \
     if (RESULTS / 'confusion_bound_dc_check.json').exists() else None

def dc_bound(model, template):
    """DC-consistent complete-data bound for the confusion rows, if the check has run."""
    if dc is None:
        return None
    try:
        return float(dc[model][template]['dc_consistent'])
    except (KeyError, TypeError):
        # tolerate other layouts of the check file
        for k, v in dc.items():
            if model in k and isinstance(v, dict):
                for kk, vv in v.items():
                    if template in kk and isinstance(vv, dict) and 'dc_consistent' in vv:
                        return float(vv['dc_consistent'])
        return None

def rung_values(cell):
    """(rung, eta, err) for the rungs present in variational2 (round-2 ladder)."""
    out = []
    v2 = (cell or {}).get('variational2') or {}
    rungs = v2.get('rungs') or {}
    for r in RUNG_ORDER:
        if r in rungs and rungs[r] is not None and rungs[r].get('eta') is not None:
            if rungs[r].get('physical') is False:
                continue
            out.append((r, float(rungs[r]['eta']), float(rungs[r].get('err') or 0.0)))
    return out

fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.6), sharex=True)
xpos = {r: i for i, r in enumerate(RUNG_ORDER)}
off = {'extended': -0.12, 'compact': +0.12}
col = {'extended': C['ext'], 'compact': C['cmp']}

for ax, (model, title) in zip(axes.ravel(), PANELS):
    ymax = 1.0
    ax.axhspan(0.9, 1.1, color=C['anchor'], alpha=0.12, lw=0)
    ax.axhline(1.0, color=C['anchor'], lw=0.6)
    for template in ('extended', 'compact'):
        cell = load_cell(model, template)
        vals = rung_values(cell)
        if vals:
            # Lay the rungs out on the full ladder and leave NaN where a rung was not run, so
            # that the connecting line BREAKS at the gap instead of stepping over it.  Without
            # this, a cell missing 'reweight' looks as though it climbs from the linear anchor.
            d = {r: (e, s) for r, e, s in vals}
            x = np.array([xpos[r] + off[template] for r in RUNG_ORDER])
            y = np.array([d[r][0] if r in d else np.nan for r in RUNG_ORDER])
            ye = np.array([d[r][1] if r in d else 0.0 for r in RUNG_ORDER])
            low = y < YMIN * 1.05                     # off-scale restarts (e.g. a negative-direction bound)
            ax.errorbar(x[~low], y[~low], yerr=ye[~low], fmt='o-', ms=4, color=col[template],
                        capsize=2, lw=0.9, label=template)
            if low.any():
                ax.plot(x[low], np.full(low.sum(), YMIN * 1.08), marker='v', ls='none',
                        color=col[template], ms=5)
            ymax = max(ymax, np.nanmax(y[~low] + ye[~low]))
        # ceilings
        exact = err = None
        if cell is not None and cell.get('quadrature'):
            exact, err = cell['quadrature']['eta'], cell['quadrature'].get('err', 0)
        if model == 'T2_PCA_WN':
            exact, err = PCA_EXACT[template]
        if model == 'T2_CONFUSION_ATM':
            exact, err = ATM_EXACT, 0.0
        bound = None
        if cell is not None and cell.get('bound_complete_data') is not None:
            bound = float(cell['bound_complete_data'])
        if 'CONFUSION' in model:
            b2 = dc_bound(model, template)
            if b2 is not None:
                bound = b2
        if exact is not None:
            ax.axhline(exact, color=C['exact'], lw=1.0, ls='-' if template == 'extended' else '--')
            if err:
                ax.axhspan(exact - err, exact + err, color=C['exact'], alpha=0.10, lw=0)
            ymax = max(ymax, exact + (err or 0))
            ax.text(4.45, exact, f'{exact:.2f}', color=C['exact'], fontsize=6.5,
                    va='bottom' if template == 'extended' else 'top', ha='left')
        if bound is not None and (model != 'T2_PCA_WN'):
            ax.axhline(bound, color=C['bound'], lw=1.0, ls='-' if template == 'extended' else '--')
            ymax = max(ymax, bound)
            ax.text(4.45, bound, f'{bound:.3g}', color=C['bound'], fontsize=6.5,
                    va='bottom' if template == 'extended' else 'top', ha='left')
        elif bound is not None and model == 'T2_PCA_WN':
            # loose bound for a scale mixture: draw faint, dotted (Sec. 7.3)
            ax.axhline(bound, color=C['bound'], lw=0.7, ls=':', alpha=0.6)
            ax.text(4.45, bound, f'{bound:.2f} (loose)', color=C['bound'], fontsize=6, va='center', ha='left', alpha=0.8)
            ymax = max(ymax, bound)
    ax.set_yscale('log')
    ax.set_ylim(YMIN, max(1.6, ymax * 1.6))
    ax.set_title(title, loc='left', fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels([RUNG_LABEL[r] for r in RUNG_ORDER], rotation=30)
    ax.set_xlim(-0.5, 5.4)
    ax.grid(axis='y', lw=0.3, alpha=0.4, which='both')

for ax in axes[:, 0]:
    ax.set_ylabel(r'$\eta$ (lower bracket per rung)')
# legend on the first panel
h_ext = plt.Line2D([], [], color=C['ext'], marker='o', ms=4, label='extended template')
h_cmp = plt.Line2D([], [], color=C['cmp'], marker='o', ms=4, label='compact template')
h_ex = plt.Line2D([], [], color=C['exact'], label='exact ceiling (solid ext, dashed cmp)')
h_bd = plt.Line2D([], [], color=C['bound'], label='complete-data bound')
fig.legend(handles=[h_ext, h_cmp, h_ex, h_bd], loc='lower center', ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(hspace=0.35, wspace=0.28, bottom=0.08)
savefig(fig, 'fig08_ladders')
