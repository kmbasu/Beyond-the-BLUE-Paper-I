"""
diagnostics.py — quick statistical diagnostics for simulated ensembles
======================================================================

Consolidates the diagnostic helpers scattered across the legacy modules
(radial_psd, psd_1d_cut, moment summaries, postage-stamp panels) into one
place, used by every driver's DIAGNOSTICS cell and by the regression suite.

Conventions
-----------
* ``radial_psd`` and ``psd_1d_cut`` are bit-compatible ports of the legacy
  helpers (unnormalized-FFT periodogram, azimuthal average over integer
  radial bins / axis cuts, DC bin excluded).
* ``estimate_psd2d`` is the ensemble mean 2-D periodogram — the object the
  eta pipeline will use for whitening (M4 Sec. 4.4: full 2-D PSD, never a
  radial average, for anything anisotropic).
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------
# Power spectra
# ----------------------------------------------------------------------------------

def periodogram2d(image):
    """|fft2(image)|^2 (unnormalized-FFT convention), FFT-ordered."""
    F = np.fft.fft2(image)
    return np.abs(F)**2


def estimate_psd2d(images, n_sample=None, method='mean', rng=None):
    """Ensemble 2-D PSD estimate: mean (or median) periodogram.

    Parameters
    ----------
    images   : (N, H, W) array
    n_sample : number of images to use (None = all)
    method   : 'mean' | 'median'
    rng      : Generator for the subsampling draw

    Returns
    -------
    P_hat : (H, W) ndarray, FFT-ordered, unnormalized-FFT convention
            (E[|fft2|^2]); comparable directly to spectra.psd_* outputs.
    """
    if n_sample is not None and n_sample < len(images):
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(images), size=n_sample, replace=False)
        images = images[idx]
    pgrams = np.abs(np.fft.fft2(images, axes=(-2, -1)))**2
    return np.median(pgrams, axis=0) if method == 'median' else pgrams.mean(axis=0)


def radial_psd(image):
    """Azimuthally averaged PSD of one image (legacy helper).

    Returns (r, psd_1d): integer radial bins (DC excluded) and the mean
    periodogram power in each bin, unnormalized-FFT convention.
    """
    ny, nx = image.shape
    p2d = np.fft.fftshift(periodogram2d(image))
    cy, cx = ny // 2, nx // 2
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    r_max = min(cy, cx)
    psd_1d = np.array([p2d[r == ri].mean() for ri in range(1, r_max + 1)])
    return np.arange(1, r_max + 1), psd_1d


def psd_1d_cut(image, axis='kx'):
    """1-D PSD cut along the kx (ky) axis through the origin, DC excluded.

    Used to expose anisotropy: for scan-direction noise the kx cut is
    steep/red while the ky cut is flat(ter).
    Returns (k, power) for the positive-frequency half.
    """
    p2d = periodogram2d(image)
    ny, nx = image.shape
    if axis == 'kx':
        cut = p2d[0, :]                     # ky = 0 row
        k = np.fft.fftfreq(nx)
    elif axis == 'ky':
        cut = p2d[:, 0]                     # kx = 0 column
        k = np.fft.fftfreq(ny)
    else:
        raise ValueError("axis must be 'kx' or 'ky'")
    sel = k > 0
    return k[sel], cut[sel]


def fit_psd_slope(r, psd, r_range=(2, 20)):
    """Log-log linear fit of a radial PSD; returns (slope, intercept).

    The fitted slope of P(r) ~ r^s is NEGATIVE for red noise; compare
    -slope to the generation parameter alpha.
    """
    sel = (r >= r_range[0]) & (r <= r_range[1]) & (psd > 0)
    coeff = np.polyfit(np.log(r[sel]), np.log(psd[sel]), 1)
    return coeff[0], coeff[1]


# ----------------------------------------------------------------------------------
# Moments
# ----------------------------------------------------------------------------------

def moment_summary(images):
    """Per-ensemble pixel-moment summary.

    Returns dict with mean, std, skewness and excess kurtosis of the pooled
    pixel distribution — the quick one-point non-Gaussianity indicators
    (remember: one-point Gaussianity is NOT field Gaussianity; these are
    coarse diagnostics only).
    """
    x = np.asarray(images, dtype=np.float64).ravel()
    mu = x.mean()
    xc = x - mu
    var = np.mean(xc**2)
    skew = np.mean(xc**3) / var**1.5
    kurt = np.mean(xc**4) / var**2 - 3.0
    return {'mean': mu, 'std': np.sqrt(var),
            'skewness': skew, 'excess_kurtosis': kurt}


# ----------------------------------------------------------------------------------
# Quick-look figure
# ----------------------------------------------------------------------------------

def quicklook(images, I0=None, title='', n_show=9, seed=0, show=True):
    """Postage-stamp grid + radial PSD + pixel histogram for one ensemble.

    A compact version of the legacy diagnostics sections, intended for the
    drivers' final cell.  Returns the figure object.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=min(n_show, len(images)), replace=False)

    fig = plt.figure(figsize=(12, 8))
    # postage stamps
    for p, i in enumerate(idx):
        ax = fig.add_subplot(3, 6, p + 1 + (p // 3) * 3)
        ax.imshow(images[i], cmap='RdBu_r')
        lab = f'#{i}' + (f'  I0={I0[i]:.2f}' if I0 is not None else '')
        ax.set_title(lab, fontsize=8)
        ax.axis('off')
    # radial PSD (single representative image + ensemble mean of a few)
    ax_psd = fig.add_subplot(3, 6, (4, 12))
    r, p1 = radial_psd(images[idx[0]])
    sub = images[rng.choice(len(images), size=min(20, len(images)), replace=False)]
    p_mean = np.mean([radial_psd(im)[1] for im in sub], axis=0)
    ax_psd.loglog(r, p1, alpha=0.5, label='single image')
    ax_psd.loglog(r, p_mean, lw=2, label='mean of 20')
    ax_psd.set_xlabel('r (pixels in k-space)')
    ax_psd.set_ylabel('P(r)')
    ax_psd.legend(fontsize=8)
    # histogram
    ax_h = fig.add_subplot(3, 6, (16, 18))
    x = np.asarray(sub).ravel()
    ax_h.hist(x, bins=100, density=True, alpha=0.7)
    m = moment_summary(sub)
    ax_h.set_title(f"skew={m['skewness']:.3f}  ex.kurt={m['excess_kurtosis']:.3f}",
                   fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
