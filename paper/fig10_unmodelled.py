"""
fig10_unmodelled.py -- FIG 10: three unmodelled channels in k-space.

Purpose
-------
(a) The DC mode: per-mode signal-to-noise |tau_k|^2 / P(k) of the extended template on
    red noise, with the DC bin as the pipeline treats it (regularised to the fundamental
    power) and as mean subtraction leaves it (noise power -> 0, SNR -> infinity): the
    zeroed DC channel of the first campaign.
(b) The band edge: the compact template's Fisher weight per mode on the floorless
    spectral-tilt mixture (beam cancels, weight ~ k^3 up to the mask) and on the
    floored one (terminated by the white floor).
(c) Mantissa rounding: the radial power spectrum of the floorless PCA-leakage model
    stored in float32 and in float64, and their difference (the rounding power), with
    the crossover at |k| ~ 0.42 where rounding exceeds the beam-suppressed physics.
    Inset: the float32 / float64 VAL trajectories of the conv rung on that model, if
    the A/B results are present (eta_results_hpc_f64/).

Usage
-----
    python fig10_unmodelled.py
"""
import os
from figcommon import *

tpl = em.make_templates()
ny, nx = em.SHAPE
fy = np.fft.fftfreq(ny).reshape(-1, 1); fx = np.fft.fftfreq(nx).reshape(1, -1)
kgrid = np.sqrt(fx**2 + fy**2)

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

# ---------------- (a) DC
ax = axes[0]
P = nl.psd_isotropic_powerlaw(em.SHAPE, *em.P_ISO_ARGS) * em.beam2()
tk2 = np.abs(np.fft.fft2(tpl['extended']))**2
snr = tk2 / P
snr_nodc = snr.copy(); snr_nodc[0, 0] = np.nan
k, s = radial_profile(np.nan_to_num(snr_nodc, nan=0.0))
norm = np.nanmax(s)
ax.loglog(k, s / norm, color=C['ext'], label='extended template, per-mode SNR')
ax.plot([k[0] * 0.5], [snr[0, 0] / norm], marker='s', color=C['exact'], ls='none',
        label='DC bin, regularised (pipeline)')
ax.annotate('DC bin after mean subtraction:\nnoise power = 0, SNR $\\to\\infty$', xy=(k[0] * 0.5, 3e3),
            xytext=(k[0] * 1.6, 3e3), fontsize=7, color=C['bound'], va='center',
            arrowprops=dict(arrowstyle='-|>', color=C['bound'], lw=0.8))
ax.plot([k[0] * 0.5], [3e3], marker='^', color=C['bound'], ls='none')
ax.set_xlim(k[0] * 0.3, 0.5); ax.set_ylim(1e-6, 1e4)
ax.set_xlabel(r'$|\mathbf{k}|$ [cycles px$^{-1}$]'); ax.set_ylabel(r'$|\tilde\tau_{\mathbf{k}}|^2/P(\mathbf{k})$ (normalised)')
ax.set_title('(a) the DC mode', loc='left'); ax.legend(frameon=False, fontsize=7, loc='lower left')

# ---------------- (b) band edge
ax = axes[1]
for model, lab, col, ls in (('T1_PSRAND', 'floorless', C['bound'], '-'), ('T1_PSRAND_WN', 'floored', C['exact'], '-')):
    maps, _ = em.REGISTRY[model]['gen'](600, em.model_seed(model))
    Ph = wh.estimate_psd2d(np.asarray(maps, dtype=float))
    f = np.abs(np.fft.fft2(tpl['compact']))**2 / Ph
    mask = em.beam2() > 1e-8
    f = np.where(mask, f, np.nan); f /= np.nansum(f)
    kb, fb = radial_profile(np.nan_to_num(f))
    ax.loglog(kb, fb * kb**2 / np.nansum(fb * kb**2), color=col, ls=ls, label=f'compact template, {lab}')
ax.axvline(0.42, color=C['anchor'], lw=0.6, ls=':')
ax.text(0.04, 3e-4, r'floorless: $\propto k^{3}$ up to the mask', fontsize=7, color=C['bound'])
ax.set_xlim(kb[0], 0.7); ax.set_ylim(1e-5, 1)
ax.set_xlabel(r'$|\mathbf{k}|$ [cycles px$^{-1}$]'); ax.set_ylabel(r'information fraction per $\log k$')
ax.set_title('(b) the band edge without a floor', loc='left'); ax.legend(frameon=False, fontsize=7, loc='lower left')

# ---------------- (c) rounding
ax = axes[2]
specs = {}
for dt, flag in (('float32', '0'), ('float64', '1')):
    os.environ['ETA_MAPS_FLOAT64'] = flag
    maps, _ = em.REGISTRY['T2_PCA']['gen'](300, em.model_seed('T2_PCA'))
    specs[dt] = wh.estimate_psd2d(np.asarray(maps, dtype=np.float64))
os.environ.pop('ETA_MAPS_FLOAT64', None)
k32, p32 = radial_profile(specs['float32']); k64, p64 = radial_profile(specs['float64'])
ax.loglog(k64, p64, color=C['ext'], label='physical spectrum (float64 storage)')
ax.loglog(k32, p32, color=C['bound'], ls='--', label='as stored in float32')
diff = np.clip(p32 - p64, 1e-30, None)
sel = k32 > 0.25
ax.loglog(k32[sel], diff[sel], color=C['anchor'], ls=':', label='difference = rounding power')
ax.axvline(0.42, color=C['anchor'], lw=0.6)
ax.text(0.43, p64.max() * 0.1, r'$|\mathbf{k}|\approx0.42$', fontsize=7, color=C['anchor'])
ax.set_xlim(k64[0], 0.7); ax.set_ylim(max(p64.min() * 0.1, 1e-32), p64.max() * 5)
ax.set_xlabel(r'$|\mathbf{k}|$ [cycles px$^{-1}$]'); ax.set_ylabel(r'$\hat P(k)$ [arb.]')
ax.set_title('(c) mantissa rounding (PCA leakage, floorless)', loc='left'); ax.legend(frameon=False, fontsize=7, loc='lower left')

# inset: VAL trajectories float32 vs float64 conv rung, if available
f64 = NCC / 'results' / 'eta_results_hpc_f64' / 'T2_PCA__extended.json'
f32 = RESULTS / 'T2_PCA__extended.json'
try:
    ins = ax.inset_axes([0.58, 0.55, 0.4, 0.4])
    for path, col, lab in ((f32, C['bound'], 'float32'), (f64, C['ext'], 'float64')):
        d = json.load(open(path))
        v2 = d.get('variational2') or {}
        rung = (v2.get('rungs') or {}).get('cnn') or {}
        tel = (rung.get('telemetry') or {}).get('restarts') or []
        for i, t in enumerate(tel):
            ins.plot(t['val_hist'], color=col, lw=0.6, alpha=0.8, label=lab if i == 0 else None)
    ins.axhline(1.0, color=C['anchor'], lw=0.5)
    ins.set_xlabel('epoch', fontsize=6); ins.set_ylabel('VAL bound', fontsize=6)
    ins.tick_params(labelsize=5.5); ins.legend(frameon=False, fontsize=5.5)
    ins.set_title('conv rung, 6 restarts', fontsize=6)
except Exception as e:  # noqa: BLE001
    print('inset skipped:', e)

fig.subplots_adjust(wspace=0.3)
savefig(fig, 'fig10_unmodelled')
