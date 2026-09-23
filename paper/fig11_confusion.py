"""
fig11_confusion.py -- FIG 11: source confusion.

Purpose
-------
(a) A realisation of the production row (confusion + white floor, mJy/beam) and the
    one-point distribution of the pure-confusion field with the analytic (Campbell)
    mean, rms, skewness and excess kurtosis annotated against the measured ones.
(b) The per-mode signal-to-noise |tau_k|^2 / P_hat(k), azimuthally averaged, for the
    compact and extended templates: floorless (flat for the compact template -- the
    matched filter has no spectral leverage) and floored (rolled off past the knee).

Data: the registry generators (T2_CONFUSION, T2_CONFUSION_WN), noise_lib.confusion stats.
Usage: python fig11_confusion.py
"""
from figcommon import *
from scipy import stats as sst

N = 400
maps0, _ = em.REGISTRY['T2_CONFUSION']['gen'](N, em.model_seed('T2_CONFUSION'))
mapsw, _ = em.REGISTRY['T2_CONFUSION_WN']['gen'](N, em.model_seed('T2_CONFUSION_WN'))
maps0 = np.asarray(maps0, dtype=float); mapsw = np.asarray(mapsw, dtype=float)
cs = em.CONFUSION_STATS
tpl = em.make_templates()

fig = plt.figure(figsize=(11.5, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.0, 1.1], wspace=0.45)

# (a) map
ax = fig.add_subplot(gs[0])
m = mapsw[0]
im = ax.imshow(m, cmap='inferno', origin='lower', vmin=np.percentile(m, 1), vmax=np.percentile(m, 99.7))
ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label(r'mJy beam$^{-1}$', fontsize=7); cb.ax.tick_params(labelsize=6)
ax.set_title(r'(a) confusion + white floor ($f_N = 0.5$)', loc='left')

# (b) P(D)
ax = fig.add_subplot(gs[1])
v = maps0.ravel()
ax.hist(v, bins=200, density=True, histtype='step', color=C['ext'], lw=1.0, label='pure confusion, 400 maps')
mu, sd = v.mean(), v.std(); sk, ku = sst.skew(v), sst.kurtosis(v)
ax.set_yscale('log'); ax.set_xlim(np.percentile(v, 0.02), np.percentile(v, 99.98))
ax.set_xlabel(r'pixel value [mJy beam$^{-1}$]'); ax.set_ylabel(r'$P(D)$')
txt = (f"measured / Campbell\nmean  {mu:.2f} / {cs['mean']:.2f}\n"
       f"rms   {sd:.2f} / {cs['sigma_c']:.2f}\nskew  {sk:.2f} / {cs['skew']:.2f}\n"
       f"ex.kurt {ku:.2f} / {cs['exkurt']:.2f}")
ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=7, ha='right', va='top', family='monospace',
        bbox=dict(boxstyle='round', fc='white', ec='0.7'))
ax.set_title('(b) one-point distribution', loc='left')

# (c) per-mode SNR
ax = fig.add_subplot(gs[2])
for maps, lab, ls in ((maps0, 'floorless', '--'), (mapsw, 'floored', '-')):
    Ph = wh.estimate_psd2d(maps)
    for key, col in (('compact', C['cmp']), ('extended', C['ext'])):
        snr = np.abs(np.fft.fft2(tpl[key]))**2 / Ph
        snr[0, 0] = np.nan
        k, s = radial_profile(np.nan_to_num(snr, nan=np.nanmedian(snr)))
        ax.loglog(k, s / s[np.argmin(np.abs(k - 0.03))], color=col, ls=ls, label=f'{key}, {lab}')
ax.set_xlabel(r'$|\mathbf{k}|$ [cycles px$^{-1}$]'); ax.set_ylabel(r'$|\tilde\tau_{\mathbf{k}}|^2/\hat P(\mathbf{k})$ (norm. at $k = 0.03$)')
ax.legend(frameon=False, fontsize=7, loc='lower left')
ax.set_title('(c) per-mode signal-to-noise', loc='left')
ax.set_xlim(k[0], 0.7); ax.set_ylim(1e-8, 10)
savefig(fig, 'fig11_confusion')
print('stats keys:', list(cs.keys()))
