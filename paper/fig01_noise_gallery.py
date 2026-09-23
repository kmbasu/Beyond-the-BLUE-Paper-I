"""
fig01_noise_gallery.py -- FIG 1: one realisation of every noise model in the paper.

Purpose
-------
The noise taxonomy of Sec. 5 as pictures: one 128x128 realisation per model in the
realistic (white-floored) configuration, drawn from the campaign registry with the
registry's own seed, so that every panel is reproducible from (model, seed, image 0).
The two source templates are shown as insets in the first panel.

Layout
------
2 rows x 5 columns:
  row 1  Tier 0 / Tier 1 : white | red+white | fixed anisotropy | spectral-tilt mixture | random-orientation
  row 2  Tier 2          : row-median residual | scan crossings (sym.) | glitches | PCA leakage | confusion
Each panel uses a symmetric colour scale at +-3 sigma of its own map, so that the
*texture* of each model is visible; the colour bar is therefore per panel (values
in the paper's arbitrary units, confusion in mJy/beam).

Usage
-----
    python fig01_noise_gallery.py            # writes ../Figures/fig01_noise_gallery.{pdf,png}
"""
from figcommon import *

MODELS = [('T0_WHITE', 'white'), ('T0_RED_REAL', 'red + white'),
          ('T0_ANISO', 'fixed-direction anisotropy'),
          ('T1_PSRAND_WN', 'spectral-tilt mixture'), ('T1_RANDOR_WN', 'random-orientation anisotropy'),
          ('T2_MEDIAN_WN', 'row-median residual'), ('T2_CROSS_SYM_WN', 'scan crossings'),
          ('T2_GLITCH_WN', 'sub-threshold glitches'), ('T2_PCA_WN', 'PCA leakage'),
          ('T2_CONFUSION_WN', 'source confusion')]
TIER = {'T0': 'Tier 0', 'T1': 'Tier 1', 'T2': 'Tier 2'}

def first_map(model, n=4):
    entry = em.REGISTRY[model]
    maps, _ = entry['gen'](n, em.model_seed(model))
    return np.asarray(maps[0], dtype=float)

fig, axes = plt.subplots(2, 5, figsize=(11.5, 5.0))
for ax, (model, label) in zip(axes.ravel(), MODELS):
    m = first_map(model)
    m = m - np.median(m)
    s = np.std(m)
    im = ax.imshow(m, cmap='RdBu_r', vmin=-3 * s, vmax=3 * s, origin='lower',
                   interpolation='nearest')
    tier = TIER[em.REGISTRY[model]['tier']]
    if model == 'T2_PCA_WN':
        tier = 'Tier 1 (reclassified)'
    ax.set_title(f'{label}\n{tier}', fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    unit = 'mJy/beam' if 'CONFUSION' in model else 'arb.'
    ax.text(0.02, 0.02, rf'$\pm3\sigma$, $\sigma$={s:.2g} {unit}', transform=ax.transAxes,
            fontsize=6.5, color='k', va='bottom',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))

# templates inset in the first panel
tpl = em.make_templates()
ax0 = axes[0, 0]
for j, (name, key) in enumerate((('compact', 'compact'), ('extended', 'extended'))):
    ins = ax0.inset_axes([0.62 - 0.36 * j, 0.62, 0.34, 0.34])
    t = tpl[key]
    c = t.shape[0] // 2
    ins.imshow(t[c - 16:c + 16, c - 16:c + 16], cmap='Greys', origin='lower')
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_title(name, fontsize=6.5, pad=1.5)
fig.subplots_adjust(wspace=0.08, hspace=0.35)
savefig(fig, 'fig01_noise_gallery')
