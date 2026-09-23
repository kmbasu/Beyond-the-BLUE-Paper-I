"""
fig02_fisher_band.py -- FIG 2: the two templates and their Fisher bands.

Purpose
-------
The figure that carries most of the paper's template-dependence argument (Sec. 2.1):
where in k-space each template draws its amplitude information, on the ensemble
power spectrum of the red+white model (T0_RED_REAL), and where the white floor
terminates the compact template's band.

Panels
------
(a) the two templates in the map (central 48x48 px) and their radial profiles;
(b) the azimuthally averaged ensemble spectrum P(k) of red+white noise (1500-map
    PSD split, as the pipeline estimates it), with the beam-smoothed red part and
    the white floor drawn separately, and the knee marked;
(c) the normalised Fisher weights f_k = |tau_k|^2 / P(k) (per mode, azimuthally
    averaged and multiplied by the mode density 2 pi k so that the curve integrates
    to the information fraction per log k), for both templates, with the band
    centroid exp<log k>_f marked -- the two templates look at different parts of
    the same noise.

Numbers printed to stdout: <k>_f, exp<log k>_f, Var_f[log k], sigma_MF for both
templates -- quote these in Sec. 2.1 / Sec. 5.

Usage
-----
    python fig02_fisher_band.py
"""
from figcommon import *

MODEL = 'T0_RED_REAL'
N_PSD = 1500

entry = em.REGISTRY[MODEL]
maps, _ = entry['gen'](N_PSD, em.model_seed(MODEL))
P_hat = wh.estimate_psd2d(np.asarray(maps, dtype=float))
tpl = em.make_templates()

# analytic pieces for the spectrum panel
P_red = nl.psd_isotropic_powerlaw(em.SHAPE, *em.P_ISO_ARGS) * em.beam2()
P_w = em.p_white(em.WHITE_REAL)

ny, nx = em.SHAPE
fy = np.fft.fftfreq(ny).reshape(-1, 1); fx = np.fft.fftfreq(nx).reshape(1, -1)
kgrid = np.sqrt(fx**2 + fy**2)

stats = {}
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), gridspec_kw={'width_ratios': [0.9, 1.1, 1.1]})

# (a) templates
ax = axes[0]
c = ny // 2
r = np.arange(0, 24)
for key, col in (('compact', C['cmp']), ('extended', C['ext'])):
    t = tpl[key]
    prof = t[c, c:c + 24] / t.max()
    ax.plot(r, prof, color=col, label=key)
ax.set_xlabel('radius [px]'); ax.set_ylabel(r'$\tau(r)/\tau(0)$')
ax.legend(frameon=False, loc='upper right')
ax.set_title('(a) templates', loc='left')
ins = ax.inset_axes([0.42, 0.28, 0.45, 0.45])
t = tpl['extended']; ins.imshow(t[c - 24:c + 24, c - 24:c + 24], cmap='Greys', origin='lower')
ins.set_xticks([]); ins.set_yticks([]); ins.set_title('extended', fontsize=6.5, pad=1)

# (b) spectrum
ax = axes[1]
k, p = radial_profile(P_hat)
_, pr = radial_profile(P_red)
_, pw = radial_profile(P_w)
ax.loglog(k, p, color='k', label=r'ensemble $\hat P(k)$')
ax.loglog(k, pr, color=C['ext'], ls='--', lw=0.9, label=r'beam-smoothed $1/f^{3}$')
ax.loglog(k, pw, color=C['floor'], ls=':', lw=0.9, label='white floor')
# knee: where the two analytic parts cross
kk = k[np.nanargmin(np.abs(np.log(pr) - np.log(pw)))]
ax.axvline(kk, color=C['floor'], lw=0.6)
ax.text(kk * 1.08, np.nanmin(pw) * 0.05, 'knee', fontsize=7, color=C['floor'])
ax.set_xlabel(r'$|\mathbf{k}|$ [cycles px$^{-1}$]'); ax.set_ylabel(r'$P(k)$ [arb.]')
ax.legend(frameon=False, loc='upper right')
ax.set_title('(b) red + white noise spectrum', loc='left')
ax.set_xlim(k[0], 0.5)
ax.set_ylim(np.nanmin(pw) * 1e-3, np.nanmax(p) * 5)

# (c) Fisher weights
ax = axes[2]
for key, col in (('compact', C['cmp']), ('extended', C['ext'])):
    tau = tpl[key]
    tau_k = np.fft.fft2(tau)
    f = np.abs(tau_k)**2 / P_hat
    f /= f.sum()
    kb, fb = radial_profile(f)          # mean weight per mode in each ring
    # information per log k: weight per mode x number of modes per log k  (∝ k^2 for log bins)
    dens = fb * kb**2
    dens /= np.nansum(dens)
    ax.plot(kb, dens, color=col, label=key)
    logk = np.log(kgrid[kgrid > 0]); w = f[kgrid > 0]
    kmean = (kgrid * f).sum()
    lk = (logk * w).sum() / w.sum(); vlk = ((logk - lk)**2 * w).sum() / w.sum()
    smf = wh.sigma_mf_from_psd(tau, P_hat)
    stats[key] = dict(k_mean=kmean, k_geo=np.exp(lk), var_logk=vlk, sigma_mf=float(smf))
    ax.axvline(np.exp(lk), color=col, lw=0.6, ls='--')
ax.set_xscale('log')
ax.set_xlabel(r'$|\mathbf{k}|$ [cycles px$^{-1}$]')
ax.set_ylabel(r'information fraction per $\log k$')
ax.legend(frameon=False)
ax.set_title('(c) Fisher weights and band centroids', loc='left')
ax.set_xlim(k[0], 0.5)

fig.subplots_adjust(wspace=0.32)
savefig(fig, 'fig02_fisher_band')
print('knee k =', kk)
for key, s in stats.items():
    print(key, {kk_: round(v, 4) for kk_, v in s.items()})
json.dump(dict(knee=float(kk), **stats), open(FIGDIR / 'fig02_fisher_band_stats.json', 'w'), indent=1)
