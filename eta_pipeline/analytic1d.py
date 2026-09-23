"""
analytic1d.py — exactly solvable 1-D references (memo §15.2)
============================================================

Known-η validation family: the two-component Gaussian scale mixture

    p(x) = (1−ε) N(0, σ₁²) + ε N(0, σ₂²),

standardized to unit variance.  Its Fisher information I = ∫ p′²/p dx is a
1-D numerical quadrature, and for a unit-variance density η_pix = I·Var =
I.  The weak-non-Gaussianity formula η − 1 ≈ γ₃²/2 + γ₄²/6 (verified to
≲1% in the memo's weak limit) provides the perturbative cross-check;
symmetric mixtures have γ₃ = 0 and excess kurtosis
γ₄ = 3(E[s⁴]/E[s²]² − 1) for the mixing variance s².
"""

import numpy as np


def scale_mixture_params(eps, R):
    """Unit-variance two-component mixture with variance ratio R = σ₂²/σ₁².

    Returns (sigma1, sigma2) with (1−ε)σ₁² + εσ₂² = 1.
    """
    s1sq = 1.0 / (1.0 - eps + eps * R)
    return np.sqrt(s1sq), np.sqrt(R * s1sq)


def scale_mixture_pdf(x, eps, R):
    s1, s2 = scale_mixture_params(eps, R)
    g1 = np.exp(-0.5 * (x / s1)**2) / (np.sqrt(2 * np.pi) * s1)
    g2 = np.exp(-0.5 * (x / s2)**2) / (np.sqrt(2 * np.pi) * s2)
    return (1.0 - eps) * g1 + eps * g2


def scale_mixture_eta_1d(eps, R, x_max=None, n=200_001):
    """Exact 1-pixel η = I·Var = ∫ p′²/p dx (unit variance) by quadrature."""
    s1, s2 = scale_mixture_params(eps, R)
    if x_max is None:
        x_max = 12.0 * max(s1, s2)
    x = np.linspace(-x_max, x_max, n)
    g1 = np.exp(-0.5 * (x / s1)**2) / (np.sqrt(2 * np.pi) * s1)
    g2 = np.exp(-0.5 * (x / s2)**2) / (np.sqrt(2 * np.pi) * s2)
    p = (1.0 - eps) * g1 + eps * g2
    dp = (1.0 - eps) * g1 * (-x / s1**2) + eps * g2 * (-x / s2**2)
    trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(trapz(dp**2 / np.maximum(p, 1e-300), x))


def scale_mixture_moments(eps, R):
    """(skewness, excess kurtosis) of the unit-variance mixture (γ₃ = 0)."""
    s1, s2 = scale_mixture_params(eps, R)
    m4 = 3.0 * ((1.0 - eps) * s1**4 + eps * s2**4)
    return 0.0, float(m4 - 3.0)


def eta_weak_1d(gamma3, gamma4):
    """Weak-limit formula η ≈ 1 + γ₃²/2 + γ₄²/6 (memo §6, 1-D reduction)."""
    return 1.0 + gamma3**2 / 2.0 + gamma4**2 / 6.0


def sample_scale_mixture(shape, eps, R, rng):
    """i.i.d.-pixel unit-variance mixture maps (B, H, W) for validation."""
    s1, s2 = scale_mixture_params(eps, R)
    z = rng.standard_normal(shape)
    s = np.where(rng.random(shape) < eps, s2, s1)
    return s * z


# ----------------------------------------------------------------------------------
# Compound Poisson + Gaussian: the exactly-solvable CONFUSION reference (2026-09-03)
# ----------------------------------------------------------------------------------
# Added for T2_CONFUSION_ATM.  Whitening a confusion field whose Gaussian companion
# shares the beam returns  Y = D + W  per pixel, i.i.d. across pixels, with
#
#     D = sum_{j=1..N} a_j ,  N ~ Poisson(lambda),  a_j ~ f_a  (the truncated dN/dS)
#     W ~ N(0, rho * Var[D])                                   (the companion)
#
# so eta = Var[Y] * I(Y) exactly, with no template dependence whatsoever
# (J = I_1 * Id for i.i.d. pixels; null test N4).  See README_T2_Confusion_v2.md Sec. 6.3.
#
# The Fisher information is evaluated through Tweedie's identity rather than by
# differentiating the density.  For Y = D + W with W Gaussian and independent,
#
#     p_Y'(y) = [ m(y) - y p_Y(y) ] / sigma_W^2 ,   m(y) = int d p_D(d) g_W(y-d) dd,
#
# so the score is (m/p_Y - y)/sigma_W^2 and I = E_Y[score^2].  Both p_Y and m come
# from the SAME characteristic-function FFT -- p_Y from phi_D phi_W and m from
# phi_D (lambda psi_a) phi_W, with psi_a(w) = E[a e^{i w a}] -- which avoids
# numerical differentiation of a density that has a near-atom at the origin
# (mass e^-lambda: the pixels containing no source at all).


def _cf_grid(y_lo, y_hi, n):
    """Uniform y grid and its conjugate angular-frequency grid."""
    y = np.linspace(y_lo, y_hi, n, endpoint=False)
    dy = y[1] - y[0]
    w = 2.0 * np.pi * np.fft.fftfreq(n, d=dy)
    return y, dy, w


def _invert(phi, y, dy, w):
    """p(y) = (1/2pi) int phi(w) e^{-i w y} dw on the grid of ``_cf_grid``."""
    n = len(y)
    return np.real(np.fft.fft(phi * np.exp(-1j * w * y[0]))) / (n * dy)


def _forward(f, y, dy, w):
    """phi(w) = int f(y) e^{+i w y} dy — the exact inverse of ``_invert``."""
    n = len(y)
    return np.fft.ifft(f) * n * dy * np.exp(1j * w * y[0])


def compound_poisson_pdf(marks, weights, lam, rho, n=1 << 20, n_sd=60.0):
    """Density of Y = D + W on a grid, plus the pieces the score needs.

    Parameters
    ----------
    marks, weights : mark support and its (unnormalised) weights, e.g. the flux
        grid and dN/dS weights of ``noise_lib.confusion``
    lam  : mean number of marks per pixel
    rho  : Var[W] / Var[D].  rho = 0 is the floorless limit, where the density
           has a genuine atom of mass e^-lambda at the origin and the Fisher
           information diverges; it is rejected rather than silently truncated.
    n    : FFT length.  n_sd : half-width in units of sd(Y); the grid is also
           forced to cover several times the largest mark, since a single bright
           source puts real probability there and FFT wraparound would alias it
           back onto the core.

    The mark density is binned onto the SAME grid and transformed by FFT, so the
    cost is O(n log n) rather than O(n * n_marks) — the direct outer product is
    a 34 TB allocation at these grid sizes.

    Returns
    -------
    dict with y, dy, p (density), score, mean, var, sigma_w, var_d
    """
    a = np.asarray(marks, float)
    wt = np.asarray(weights, float)
    wt = wt / wt.sum()
    m1 = float((wt * a).sum())
    m2 = float((wt * a ** 2).sum())
    mean_d, var_d = lam * m1, lam * m2
    if rho <= 0:
        raise ValueError("rho must be > 0: at rho = 0 the density has an atom "
                         "at the origin and the Fisher information is infinite "
                         "(that is the floorless T2_CONFUSION case)")
    sig_w = np.sqrt(rho * var_d)
    var_y = var_d * (1.0 + rho)
    sd_y = np.sqrt(var_y)
    y_lo = -max(10.0 * sd_y, 8.0 * sig_w)
    y_hi = mean_d + max(n_sd * sd_y, 3.0 * float(a.max()))
    y, dy, w = _cf_grid(y_lo, y_hi, n)

    # bin the mark density onto the grid (mass-preserving), then transform
    idx = np.clip(np.searchsorted(y, a) - 1, 0, n - 1)
    f_a = np.zeros(n)
    np.add.at(f_a, idx, wt)
    g_a = np.zeros(n)
    np.add.at(g_a, idx, wt * a)
    phi_a = _forward(f_a / dy, y, dy, w)
    psi_a = _forward(g_a / dy, y, dy, w)          # E[a e^{i w a}]

    phi_d = np.exp(lam * (phi_a - 1.0))
    phi_w = np.exp(-0.5 * sig_w ** 2 * w ** 2)
    p = np.maximum(_invert(phi_d * phi_w, y, dy, w), 0.0)
    m = _invert(phi_d * (lam * psi_a) * phi_w, y, dy, w)
    with np.errstate(divide='ignore', invalid='ignore'):
        score = np.where(p > 0, (m / np.where(p > 0, p, 1.0) - y) / sig_w ** 2,
                         0.0)
    return dict(y=y, dy=dy, p=p, score=score, mean=float(mean_d),
                var=float(var_y), sigma_w=float(sig_w), var_d=float(var_d))


def compound_poisson_eta_1d(marks, weights, lam, rho, n=1 << 19, n_sd=40.0,
                            p_floor=1e-14):
    """Exact 1-pixel eta = Var[Y] * I(Y) for Y = D + W (see the module note).

    ``p_floor`` drops grid points where the density has fallen below that
    fraction of its peak, which is where the FFT inversion is dominated by its
    own ringing; the returned value should be checked for stability in
    (n, n_sd, p_floor) -- ``tools/eta_exact_confusion.py`` does exactly that.

    The result satisfies eta <= (1 + rho)/rho, the complete-data bound for this
    model (perfect removal of D leaves W), which is an independent check.
    """
    d = compound_poisson_pdf(marks, weights, lam, rho, n=n, n_sd=n_sd)
    p, s, dy = d['p'], d['score'], d['dy']
    keep = p > p_floor * p.max()
    fisher = float((s[keep] ** 2 * p[keep]).sum() * dy)
    norm = float(p[keep].sum() * dy)                # ~1; guards a truncated grid
    return float(d['var'] * fisher / norm)
