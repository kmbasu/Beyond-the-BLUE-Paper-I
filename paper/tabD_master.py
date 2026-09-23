"""
tabD_master.py -- Appendix D: the master tables, generated from the result JSONs.

Purpose
-------
Writes two LaTeX table bodies into ../Figures/:
  tabD_master_floored.tex    every cell of the realistic (floored) configuration
  tabD_master_floorless.tex  the floorless set, with above-reweight variational values
                             stamped 'not physical' (Sec. 10)
plus tabD_codenames.tex, the lookup from the paper's descriptive names to the
registry code names.

Columns: model, template, floor sigma_w, sigma_MF (predicted from P_hat), exact eta
(quadrature; PCA from pca_mixture_quadrature / v4), complete-data bound (DC-consistent
for the confusion rows), rung values linear / reweight / quadratic / cubic / conv, and
a provenance flag.  Values that never exceeded their anchor within the patience window
are printed in grey.

Usage
-----
    python tabD_master.py          # then \input the .tex bodies from D_master_table.tex
"""
from figcommon import *

PCA_EXACT = {('T2_PCA_WN', 'extended'): (2.140, 0.050), ('T2_PCA_WN', 'compact'): (1.088, 0.020),
             ('T2_PCA', 'extended'): (2.074, 0.047), ('T2_PCA', 'compact'): (1.023, 0.018)}
ATM_EXACT = 3.5688
dc = json.load(open(RESULTS / 'confusion_bound_dc_check.json')) if (RESULTS / 'confusion_bound_dc_check.json').exists() else {}

def dc_bound(model, template):
    try:
        return float(dc[model][template]['dc_consistent'])
    except (KeyError, TypeError):
        return None

def cn(m):
    return m.replace('_', r'\_')

def fmt(v, e=None, grey=False, nd=3):
    if v is None:
        return '--'
    s = f'{v:.{nd}f}' if abs(v) < 100 else f'{v:.3g}'
    if e:
        s += f' $\\pm$ {e:.{nd}f}'
    return f'\\textcolor{{gray}}{{{s}}}' if grey else s

def row(model, template):
    cell = load_cell(model, template)
    if cell is None:
        return None
    entry = em.REGISTRY[model]
    floor = entry['floor']
    smf = cell.get('sigma_mf', {}).get('pred')
    exact = err = None
    if cell.get('quadrature'):
        exact, err = cell['quadrature']['eta'], cell['quadrature'].get('err')
    if (model, template) in PCA_EXACT:
        exact, err = PCA_EXACT[(model, template)]
    if model == 'T2_CONFUSION_ATM':
        exact, err = ATM_EXACT, None
    bound = cell.get('bound_complete_data')
    if 'CONFUSION' in model and dc_bound(model, template) is not None:
        bound = dc_bound(model, template)
    if model == 'T2_CONFUSION':
        bound = None
    v2 = (cell.get('variational2') or {}).get('rungs') or {}
    anchor = v2.get('linear', {}).get('eta') if v2 else None
    cols = []
    for r in RUNG_ORDER:
        rr = v2.get(r)
        if not rr or rr.get('eta') is None:
            cols.append('--'); continue
        eta, e = rr['eta'], rr.get('err')
        moved = True
        tel = (rr.get('telemetry') or {}).get('restarts')
        if r != 'linear' and tel:
            moved = any(t.get('moved_off_init') for t in tel)
        nonphys = rr.get('physical') is False
        s = fmt(eta, e, grey=(not moved) and r != 'linear')
        if nonphys:
            s += '$^{\\times}$'
        cols.append(s)
    units = 'mJy/b' if 'CONFUSION' in model else ''
    name = display_name(model)
    fl = f'{floor:.2f}' if floor else '0'
    return (f'{name} & \\code{{{cn(model)}}} & {template[:3]} & {fl} & {fmt(smf, nd=2)}{units and " " + units} & '
            f'{fmt(exact, err, nd=2)} & {fmt(bound, nd=2) if bound else "--"} & ' + ' & '.join(cols) + ' \\\\')

HEADER = ('model & code name & tpl & $\\sigma_w$ & $\\sigma_{\\rm MF}$ & exact $\\eta$ & bound & '
          'linear & reweight & quadratic & cubic & conv \\\\')

def write_table(models, fname):
    lines = ['\\begin{tabular}{llcccccccccc}', '\\toprule', HEADER, '\\midrule']
    for m in models:
        for t in ('extended', 'compact'):
            r = row(m, t)
            if r:
                lines.append(r)
    lines += ['\\bottomrule', '\\end{tabular}']
    (FIGDIR / fname).write_text('\n'.join(lines) + '\n')
    print('wrote', FIGDIR / fname)

floored = [m for m in em.REGISTRY if m.endswith('_WN') or m in ('T0_RED_REAL', 'T2_CONFUSION_ATM')]
floorless = [m for m in em.REGISTRY if not m.endswith('_WN') and m not in ('T0_RED_REAL', 'T2_CONFUSION_ATM')]
write_table(floored, 'tabD_master_floored.tex')
write_table(floorless, 'tabD_master_floorless.tex')

# code-name lookup
lines = ['\\begin{tabular}{ll}', '\\toprule', 'descriptive name & registry code names \\\\', '\\midrule']
for base, name in NAME.items():
    codes = [k for k in em.REGISTRY if k == base or k == base + '_WN']
    if codes:
        lines.append(f'{name} & ' + ', '.join(f'\\code{{{cn(c)}}}' for c in codes) + ' \\\\')
lines += ['\\bottomrule', '\\end{tabular}']
(FIGDIR / 'tabD_codenames.tex').write_text('\n'.join(lines) + '\n')
print('wrote', FIGDIR / 'tabD_codenames.tex')
