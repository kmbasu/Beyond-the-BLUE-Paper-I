"""
figcommon.py -- shared setup for the Paper-I figure scripts.

Purpose
-------
Locates the repository root relative to this file (the figure scripts live in
paper/), imports eta_pipeline, and fixes
the matplotlib style, the model display names and the output directory so that
every figure of the paper is produced with one consistent look.

Usage
-----
    from figcommon import *          # in every fig*.py script
    ...
    savefig(fig, 'fig01_noise_gallery')   # writes PDF + PNG into paper/figures/

Everything runs on numpy/scipy/matplotlib only (no torch): the scripts read the
campaign's registry (eta_pipeline.models.REGISTRY) and the result JSONs in
results/eta_results/ (override with the BTB_RESULTS environment variable).
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paths
import os
HERE = Path(__file__).resolve().parent                 # <repo>/paper
NCC = HERE.parent                                      # <repo> (holds noise_lib/, eta_pipeline/, results/)
RESULTS = Path(os.environ.get('BTB_RESULTS', NCC / 'results' / 'eta_results'))
FIGDIR = Path(os.environ.get('BTB_FIGDIR', HERE / 'figures'))
FIGDIR.mkdir(exist_ok=True, parents=True)

sys.path.insert(0, str(NCC))
import eta_pipeline as ep                               # noqa: E402
from eta_pipeline import models as em                   # noqa: E402
from eta_pipeline import whitening as wh                # noqa: E402
import noise_lib as nl                                  # noqa: E402

# ---------------------------------------------------------------- style
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# one consistent palette (colour-blind safe)
C = {
    'anchor': '#7f7f7f',      # grey     : linear anchor / MF
    'rung':   '#1f5fa8',      # blue     : ladder rungs (attained, lower bracket)
    'exact':  '#2a9d4f',      # green    : exact ceiling (quadrature / theorem)
    'bound':  '#c8401f',      # red      : complete-data upper bound
    'ext':    '#1f5fa8',      # blue     : extended template
    'cmp':    '#e08a1e',      # orange   : compact template
    'floor':  '#7f7f7f',
    'pending': '#8e44ad',     # purple   : pending / placeholder
}

# ---------------------------------------------------------------- names
# short descriptive names used in the paper (macros.tex \mXXX); code names in App. D
NAME = {
    'T0_WHITE':        'white noise',
    'T0_RED':          r'red ($1/f^{3}$) noise',
    'T0_RED_REAL':     'red + white noise',
    'T0_ANISO':        'fixed-direction anisotropy',
    'T1_PSRAND':       'spectral-tilt mixture',
    'T1_PSRAND_SLOPE': 'slope-only tilt mixture',
    'T1_PSRAND_AMP':   'amplitude-only mixture',
    'T1_RANDOR':       'random-orientation anisotropy',
    'T2_MEDIAN':       'row-median residual',
    'T2_CROSS_SYM':    'scan crossings (sym.)',
    'T2_CROSS_POS':    'scan crossings (pos.)',
    'T2_GLITCH':       'sub-threshold glitches',
    'T2_PCA':          'PCA leakage',
    'T2_CONFUSION':    'source confusion',
    'T2_CONFUSION_ATM': 'beam-sharing companion',
}
def display_name(model):
    base = model[:-3] if model.endswith('_WN') else model
    return NAME.get(base, model)

RUNG_LABEL = {'linear': 'linear', 'reweight': 'reweight', 'quadratic': 'quadratic',
              'cubic': 'cubic', 'cnn': 'conv'}          # cnn -> conv rung in the paper
RUNG_ORDER = ['linear', 'reweight', 'quadratic', 'cubic', 'cnn']

# ---------------------------------------------------------------- helpers
def load_cell(model, template):
    p = RESULTS / f'{model}__{template}.json'
    return json.load(open(p)) if p.exists() else None

def radial_profile(P2d, nbins=48):
    """Azimuthal average of a 2-D FFT-grid array; returns (k_centres, mean)."""
    ny, nx = P2d.shape
    fy = np.fft.fftfreq(ny).reshape(-1, 1)
    fx = np.fft.fftfreq(nx).reshape(1, -1)
    k = np.sqrt(fx**2 + fy**2)
    edges = np.logspace(np.log10(1.0 / nx), np.log10(0.5 * np.sqrt(2)), nbins + 1)
    idx = np.digitize(k.ravel(), edges) - 1
    out = np.full(nbins, np.nan)
    v = P2d.ravel()
    for b in range(nbins):
        m = idx == b
        if m.any():
            out[b] = v[m].mean()
    kc = np.sqrt(edges[:-1] * edges[1:])
    ok = np.isfinite(out)
    return kc[ok], out[ok]

def savefig(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIGDIR / f'{name}.{ext}')
    print('wrote', FIGDIR / f'{name}.pdf')
