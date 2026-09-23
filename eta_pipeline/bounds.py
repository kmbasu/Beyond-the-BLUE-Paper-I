"""
bounds.py — Method E: complete-data (perfect-removal) upper bound (memo §20)
============================================================================

For structured-plus-Gaussian noise n = c(z) + g, g ~ N(0, N_g), the exact
score is the Gaussian score after MMSE subtraction of the structure
(Tweedie, memo §8), which yields the one-line envelope from the generative
spectra (stationary case):

    η ≤ [ Σ_k |τ̃|²/P_g ] / [ Σ_k |τ̃|²/P_tot ]      (perfect removal).

Every reported η curve must lie inside this envelope (null test 3 /
ladder consistency).
"""

import numpy as np


def complete_data_bound(tau, P_background, P_total, k_mask=None):
    """Perfect-removal η envelope from generative spectra.

    P_background : (H, W) spectrum of the irreducible Gaussian part g
                   (background only — the structure removed)
    P_total      : (H, W) total noise spectrum (background + structure);
                   use the analytic total or the SPLIT_PSD estimate.
    k_mask       : optional boolean (H, W) mask of modes to INCLUDE.

    WARNING (v2).  Both sums are Fisher-weighted and are only meaningful on a
    band where the per-mode SNR is regularized.  For a COMPACT template on
    beam-smoothed red noise with no white floor the beam cancels in the
    numerator (|tau~|^2 = B^2, P_g = B^2 P_red) but not in the denominator
    (P_tot -> P_struct at high k, because the Phase-A pipeline injects artifacts
    AFTER beam smoothing), and the ratio diverges: the glitch/compact bound runs
    1.42, 2.69, 6.12, 12.4, 34.0 as the cutoff is relaxed, and 152 unmasked.
    Always pair this with bound_stability(); an unstable bound means the band is
    unregularized, not that the value is uncertain.
    """
    tau_F2 = np.abs(np.fft.fft2(tau))**2
    if k_mask is not None:
        m = np.asarray(k_mask, dtype=bool)
        tau_F2 = np.where(m, tau_F2, 0.0)
    num = np.sum(tau_F2 / np.maximum(P_background, 1e-20))
    den = np.sum(tau_F2 / np.maximum(P_total, 1e-20))
    return float(num / den)


def bound_stability(tau, P_background, P_total, beam_sq, cuts=(1e-4, 1e-8, 1e-14),
                    tol=0.02):
    """Is the complete-data bound a property of the noise or of the mode cutoff?

    Evaluates the bound on nested masks B^2 > cut and reports the maximum
    relative spread.  ``stable`` False means the Fisher band is unregularized
    (no white floor) and the number must not be quoted — this is null test N8.

    Returns {'values': {cut: eta}, 'max_rel_drift': float, 'stable': bool}.
    """
    vals = {float(c): complete_data_bound(tau, P_background, P_total,
                                          k_mask=(beam_sq > c)) for c in cuts}
    v = np.array(list(vals.values()), dtype=float)
    drift = float((v.max() - v.min()) / max(v.min(), 1e-30))
    return {'values': vals, 'max_rel_drift': drift, 'stable': bool(drift < tol)}
