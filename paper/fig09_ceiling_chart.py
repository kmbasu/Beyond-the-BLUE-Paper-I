"""
fig09_ceiling_chart.py -- FIG 9: every ceiling of the realistic configuration on one axis.

Purpose
-------
The paper's summary figure: for each (model, template) cell of the floored set, the
upper bracket (exact ceiling or complete-data bound) and the lower bracket (best
attained rung), on a logarithmic eta axis with the observing-time reading on the
right-hand axis (eta = effective integration-time factor).  Gaussian controls are
shown as the theorem value 1.  Cells whose lower bracket is a ladder null on a compact
template are marked 'open' (Sec. 10).

Data: the per-cell JSONs, the PCA quadrature (typed), the DC-consistent confusion bounds.
Usage:  python fig09_ceiling_chart.py
"""
from figcommon import *

dc = json.load(open(RESULTS / 'confusion_bound_dc_check.json')) if (RESULTS / 'confusion_bound_dc_check.json').exists() else {}
PCA_EXACT = {'extended': 2.140, 'compact': 1.088}
ATM_EXACT = 3.5688

ROWS = [  # (model, label, kind) kind: 'gauss' | 'mixture' | 'bound' | 'ladder'
    ('T0_WHITE', 'white noise', 'gauss'), ('T0_RED_REAL', 'red + white', 'gauss'),
    ('T0_ANISO', 'fixed anisotropy', 'gauss'),
    ('T1_PSRAND_WN', 'spectral-tilt mixture', 'mixture'),
    ('T1_RANDOR_WN', 'random-orientation aniso.', 'mixture'),
    ('T2_PCA_WN', 'PCA leakage', 'mixture'),
    ('T2_MEDIAN_WN', 'row-median residual', 'ladder'),
    ('T2_CROSS_SYM_WN', 'scan crossings (sym.)', 'bound'),
    ('T2_CROSS_POS_WN', 'scan crossings (pos.)', 'bound'),
    ('T2_GLITCH_WN', 'sub-threshold glitches', 'bound'),
    ('T2_CONFUSION_WN', 'source confusion', 'bound'),
]

def best_rung(cell):
    """Best rung that moved AND sits significantly above the anchor (> anchor + 2 err and > 1.05)."""
    v2 = (cell.get('variational2') or {}).get('rungs') or {}
    lin = v2.get('linear') or {}
    anchor, aerr = lin.get('eta', 1.0), lin.get('err') or 0.0
    best, name = None, None
    for r in RUNG_ORDER[1:]:
        rr = v2.get(r)
        if not rr or rr.get('eta') is None or rr.get('physical') is False:
            continue
        tel = (rr.get('telemetry') or {}).get('restarts')
        moved = any(t.get('moved_off_init') for t in tel) if tel else True
        signif = rr['eta'] > max(1.05, anchor + 2.0 * max(aerr, rr.get('err') or 0.0))
        if moved and signif and (best is None or rr['eta'] > best):
            best, name = rr['eta'], r
    return best, name

fig, ax = plt.subplots(figsize=(11.5, 5.2))
y = 0
ylabels = []
for model, label, kind in ROWS:
    for tpl, col, dy in (('extended', C['ext'], +0.18), ('compact', C['cmp'], -0.18)):
        cell = load_cell(model, tpl)
        if cell is None:
            continue
        yy = y + dy
        upper = None; upper_kind = None
        if kind == 'gauss':
            upper, upper_kind = 1.0, 'theorem'
        elif kind == 'mixture':
            upper = PCA_EXACT[tpl] if model == 'T2_PCA_WN' else cell['quadrature']['eta']
            upper_kind = 'exact'
        elif kind == 'bound':
            upper = cell.get('bound_complete_data')
            if 'CONFUSION' in model:
                try:
                    upper = float(dc[model][tpl]['dc_consistent'])
                except (KeyError, TypeError):
                    pass
            upper_kind = 'bound'
        lower, rung = best_rung(cell)
        anchor = ((cell.get('variational2') or {}).get('rungs') or {}).get('linear', {}).get('eta', 1.0)
        lo = max(lower or 1.0, 1.0)
        if upper is not None and upper_kind != 'theorem':
            ax.plot([lo, upper], [yy, yy], color=col, lw=1.2, alpha=0.5)
            ax.plot(upper, yy, marker='|' if upper_kind == 'bound' else 'D', ms=9 if upper_kind == 'bound' else 5,
                    color=C['bound'] if upper_kind == 'bound' else C['exact'], ls='none')
        if kind == 'gauss':
            ax.plot(1.0, yy, marker='D', ms=5, color=C['exact'], ls='none')
        if lower is not None and lower > 1.02:
            ax.plot(lower, yy, marker='o', ms=5, color=col, ls='none')
            ax.text(lower * 0.97, yy, RUNG_LABEL[rung], fontsize=6, color=col, ha='right', va='center')
        else:
            ax.plot(1.0, yy, marker='o', ms=5, mfc='none', mec=col, ls='none')
            if model == 'T2_MEDIAN_WN' and tpl == 'compact':
                ax.text(1.03, yy, 'null (both bases)', fontsize=6.5, color=col, va='center')
    ylabels.append(label)
    y += 1
ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
ax.invert_yaxis()

# --- rows to which the observing-time reading does NOT apply ---------------
# eta is an equivalent integration-time factor only where the noise integrates
# down.  Confusion noise does not: there eta is a variance factor and nothing
# more.  Separate those rows, shade them, and dagger their labels so the top
# axis cannot be read across them.
NO_TIME_READING = {'T2_CONFUSION_WN'}
_nt = [i for i, (m, _, _) in enumerate(ROWS) if m in NO_TIME_READING]
if _nt:
    ax.axhline(min(_nt) - 0.5, color='0.35', lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.axhspan(min(_nt) - 0.5, max(_nt) + 0.5, color='0.93', zorder=0)
    ax.set_yticklabels([lab + r' $\dagger$' if i in _nt else lab
                        for i, lab in enumerate(ylabels)])
    ax.text(0.995, 0.015,
            r'$\dagger$ variance factor only: confusion noise does not integrate down',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5, color='0.25')
ax.set_xscale('log'); ax.set_xlim(0.85, 300)
ax.set_xlabel(r'advantage ceiling $\eta$  (lower bracket: best attained rung;  upper bracket: exact ceiling $\diamond$ or complete-data bound $|$)')
ax.grid(axis='x', lw=0.3, alpha=0.4, which='both')
ax.axvline(1.0, color=C['anchor'], lw=0.8)
ax2 = ax.twiny()
ax2.set_xscale('log'); ax2.set_xlim(ax.get_xlim())
ax2.set_xticks([1, 1.3, 2, 5, 10, 100]); ax2.set_xticklabels(['1×', '1.3×', '2×', '5×', '10×', '100×'])
ax2.set_xlabel('equivalent integration-time factor for an ideal estimator '
               '(unshaded rows only)')
h = [plt.Line2D([], [], color=C['ext'], marker='o', ls='none', label='extended template'),
     plt.Line2D([], [], color=C['cmp'], marker='o', ls='none', label='compact template'),
     plt.Line2D([], [], color=C['exact'], marker='D', ls='none', label='exact ceiling / theorem'),
     plt.Line2D([], [], color=C['bound'], marker='|', ms=9, ls='none', label='complete-data bound'),
     plt.Line2D([], [], color='k', marker='o', mfc='none', ls='none', label='no rung moved (null above the floor)')]
ax.legend(handles=h, frameon=False, fontsize=7.5, loc='upper right')
savefig(fig, 'fig09_ceiling_chart')
