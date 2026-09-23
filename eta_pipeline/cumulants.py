"""
cumulants.py — Method B: pair-trick contracted-cumulant estimators (memo §17)
=============================================================================

Direct Monte-Carlo estimates of the weak-non-Gaussianity norms

    ‖κ̃₃(t̂)‖²_F = E_pairs[ y y' (x†x')² ],
    ‖κ̃₄(t̂)‖²_F = E_pairs[ y y' (x†x')³ ] − 6 E[y²‖x‖²] + (3M + 6),

with y = t̂†x, from two DISJOINT whitened batches (both whitened with the
SPLIT_PSD spectrum), and the perturbative ceiling

    η_pert − 1 ≈ ½‖κ̃₃(t̂)‖² + ⅙‖κ̃₄(t̂)‖².

Error estimation (important): every pair statistic shares both batches, so
a bootstrap over batch-1 rows alone MASSIVELY underestimates the error
(the common batch-2 fluctuation is a shared systematic — found the hard
way in the Phase-B1 null tests).  All errors here come from a TWO-AXIS
bootstrap: batch-1 and batch-2 image indices are resampled jointly and the
full statistic is recomputed on the resampled pair matrix.

Caveats (memo §17): the κ₄ expression subtracts O(M) disconnected pieces
from an O(M) raw term — Monte-Carlo hungry (errors grow with pixel count);
heavy-tailed marks with infinite fourth moment (Student-t df ≤ 4) make the
estimators formally divergent.  Production numbers for such models come
from the variational rungs; these estimators serve as perturbative
cross-checks on weakly non-Gaussian, light-tailed ensembles.
"""

import numpy as np


def _flatten(x):
    x = np.asarray(x)
    return x.reshape(len(x), -1)


def _pair_matrices(x1, x2, t_hat):
    """Shared pieces: y1, y2, Gram G = X1 X2ᵀ, disconnected row term d1."""
    X1, X2 = _flatten(x1), _flatten(x2)
    th = np.asarray(t_hat).ravel()
    y1, y2 = X1 @ th, X2 @ th
    G = X1 @ X2.T
    d1 = 6.0 * (y1**2) * (X1**2).sum(axis=1)         # 6 y² ||x||², batch 1
    M = X1.shape[1]
    return y1, y2, G, d1, M


def _two_axis_boot(stat_fn, n1, n2, n_boot, seed):
    """Bootstrap a pair statistic by resampling both batches jointly."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i1 = rng.integers(0, n1, size=n1)
        i2 = rng.integers(0, n2, size=n2)
        vals[b] = stat_fn(i1, i2)
    return float(vals.std())


def k3_norm(x1, x2, t_hat, n_boot=400, seed=0):
    """‖κ̃₃(t̂)‖²_F estimate: (value, two-axis bootstrap error)."""
    y1, y2, G, d1, M = _pair_matrices(x1, x2, t_hat)
    P3 = (y1[:, None] * y2[None, :]) * G**2

    def stat(i1, i2):
        return P3[np.ix_(i1, i2)].mean()

    return float(P3.mean()), _two_axis_boot(stat, len(y1), len(y2), n_boot, seed)


def k4_norm(x1, x2, t_hat, n_boot=400, seed=0):
    """‖κ̃₄(t̂)‖²_F estimate (disconnected parts removed): (value, error)."""
    y1, y2, G, d1, M = _pair_matrices(x1, x2, t_hat)
    P4 = (y1[:, None] * y2[None, :]) * G**3
    const = 3.0 * M + 6.0

    def stat(i1, i2):
        return P4[np.ix_(i1, i2)].mean() - d1[i1].mean() + const

    return (float(P4.mean() - d1.mean() + const),
            _two_axis_boot(stat, len(y1), len(y2), n_boot, seed))


def eta_perturbative(x1, x2, t_hat, n_boot=400, seed=0):
    """Perturbative η with two-axis bootstrap errors.

    Returns (eta, eta_err, (k3, k3_err), (k4, k4_err)),
    η − 1 ≈ ½‖κ̃₃‖² + ⅙‖κ̃₄‖² (the combined error preserves the κ₃/κ₄
    correlation by bootstrapping the combined statistic).
    """
    y1, y2, G, d1, M = _pair_matrices(x1, x2, t_hat)
    yy = y1[:, None] * y2[None, :]
    P3, P4 = yy * G**2, yy * G**3
    const = 3.0 * M + 6.0

    def s3(i1, i2):
        return P3[np.ix_(i1, i2)].mean()

    def s4(i1, i2):
        return P4[np.ix_(i1, i2)].mean() - d1[i1].mean() + const

    def s_eta(i1, i2):
        return 1.0 + 0.5 * s3(i1, i2) + s4(i1, i2) / 6.0

    n1, n2 = len(y1), len(y2)
    k3 = float(P3.mean())
    k4 = float(P4.mean() - d1.mean() + const)
    eta = 1.0 + 0.5 * k3 + k4 / 6.0
    return (eta, _two_axis_boot(s_eta, n1, n2, n_boot, seed),
            (k3, _two_axis_boot(s3, n1, n2, n_boot, seed + 1)),
            (k4, _two_axis_boot(s4, n1, n2, n_boot, seed + 2)))
