"""
Locally-Unbiased-Band Diagnostics for CNN Image Regression
===========================================================
Companion module to ``shrinkage_diagnostics.py``.

This module locates the amplitude ranges in which a CNN amplitude estimator can
be fairly compared against the matched-filter (MF) Cramer-Rao limit, and reports
the variance-reduction metrics there.

Two distinct ranges are reported, because they answer different questions:

  * the BIAS-CONSISTENT REGION  -- the (wide) contiguous range where the
    conditional bias g(A) = E[A_hat|A] - A is consistent with zero
    (|g| < max(2 SE, floor)).  This is where the *point predictor* is
    trustworthy.

  * the STRICT g'~0 BAND        -- the (narrow) contiguous range where, in
    addition, the bias gradient g'(A) is within eps_slope of zero.  This is the
    conservative range where a *bare* V(A) vs sigma_MF**2 comparison needs no
    correction.

Why both.  The biased Cramer-Rao bound is

        Var[A_hat | A]  >=  [1 + g'(A)]**2 * sigma_MF**2          (Gaussian MF)

so a bare variance comparison is only fair where g'(A) ~ 0.  A bias-correcting
loss flattens the *magnitude* of g (widening the bias-consistent region) but does
NOT necessarily flatten the *gradient* g' (which controls the bound).  Insisting
on g'~0 is therefore over-conservative.  The clean way to use the whole
bias-consistent region is the shrinkage-cleaned ratio

        R(A) = V(A) / ( [1 + g'(A)]**2 * sigma_MF**2 )

which divides the shrinkage credit out explicitly: R >= 1 for any Gaussian
estimator (R = 1 = efficient), and R < 1 only with a genuine non-Gaussian Fisher
gain.  R(A) averaged over the bias-consistent region is the PRIMARY
variance-vs-CRLB output; the strict g'~0 band is a cross-check.

sigma_MF is OPTIONAL.  Without it, the regions are still located (g' needs no
external input), the within-region bias-corrected RMSE / MACB / bias-fraction /
regression are reported, and the residual gradient dg/dA is printed.  The
genuine-advantage verdict and R(A) require sigma_MF and are otherwise declined.

Notes on robustness
-------------------
* g'(A) is obtained by splining g(A) DIRECTLY (weighted by 1/SE[g]); this makes
  the smoothing parameter effective.  (Splining mu(A) ~ A is dominated by the
  identity trend, so its smoothing is inert -- a subtle trap.)
* The band is the longest contiguous run of qualifying bins, allowing up to
  ``max_gap`` isolated dropouts so a single threshold-crossing bin does not
  shatter it.

References
----------
* S. M. Kay, *Fundamentals of Statistical Signal Processing: Estimation Theory*
  (Prentice Hall, 1993), Sec. 3.4 -- biased Cramer-Rao bound.
* Y. C. Eldar, "Minimum Variance in Biased Estimation," IEEE Trans. Signal
  Process. 52(7), 2004 -- bias-gradient-constrained CRB / locally-unbiased
  formalism.

Part of the "Beyond the BLUE" code release (K. Basu).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm, pearsonr
from typing import Tuple, Dict, Optional, List


# =====================================================================
#  1.  Conditional statistics  (binned in true amplitude A)
# =====================================================================
def compute_conditional_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 25,
    min_count: int = 30,
) -> Dict:
    """
    Conditional bias g(A), variance V(A), MSE(A) and their MC errors, binned in A.

    Returns a dict with keys: 'bin_centers', 'bin_edges', 'cond_mean_pred',
    'cond_bias' (g), 'cond_var' (V), 'cond_bias_sq', 'cond_mse',
    'cond_bias_err' (SE[g]=std(resid)/sqrt(N)), 'cond_var_err'
    (SE[V]~V*sqrt(2/(N-1))), 'counts', 'valid'.
    """
    bin_edges = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.clip(np.digitize(y_true, bin_edges) - 1, 0, n_bins - 1)

    cond_mean_pred = np.full(n_bins, np.nan)
    cond_bias = np.full(n_bins, np.nan)
    cond_var = np.full(n_bins, np.nan)
    cond_bias_sq = np.full(n_bins, np.nan)
    cond_mse = np.full(n_bins, np.nan)
    cond_bias_err = np.full(n_bins, np.nan)
    cond_var_err = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = bin_idx == i
        counts[i] = mask.sum()
        if counts[i] < min_count:
            continue
        preds = y_pred[mask]
        resid = preds - y_true[mask]
        cond_mean_pred[i] = np.mean(preds)
        cond_bias[i] = np.mean(resid)
        cond_var[i] = np.var(preds, ddof=1)
        cond_bias_sq[i] = cond_bias[i] ** 2
        cond_mse[i] = np.mean(resid ** 2)
        cond_bias_err[i] = np.std(resid, ddof=1) / np.sqrt(counts[i])
        cond_var_err[i] = cond_var[i] * np.sqrt(2.0 / max(counts[i] - 1, 1))

    valid = ~np.isnan(cond_bias)
    return {
        'bin_centers': bin_centers, 'bin_edges': bin_edges,
        'cond_mean_pred': cond_mean_pred, 'cond_bias': cond_bias,
        'cond_var': cond_var, 'cond_bias_sq': cond_bias_sq, 'cond_mse': cond_mse,
        'cond_bias_err': cond_bias_err, 'cond_var_err': cond_var_err,
        'counts': counts, 'valid': valid,
    }


# =====================================================================
#  2.  Local slope:  spline g(A) DIRECTLY, then g'(A) = d g / dA
# =====================================================================
def compute_local_slope(
    bin_centers: np.ndarray,
    cond_bias: np.ndarray,
    cond_bias_err: np.ndarray,
    valid: np.ndarray,
    smoothing_factor: Optional[float] = None,
    weighted: bool = True,
    n_fine: int = 400,
) -> Tuple[np.ndarray, np.ndarray, object, np.ndarray]:
    """
    Fit a smoothing spline to g(A) = cond_bias (NOT to mu(A)) and differentiate.

    Splining g(A) directly is what makes ``smoothing_factor`` effective: a spline
    of mu(A) ~ A is dominated by the identity slope and its smoothing barely
    touches the small residual where g lives.

    Parameters
    ----------
    bin_centers, cond_bias, cond_bias_err, valid : arrays from
        compute_conditional_statistics.
    smoothing_factor : float or None
        UnivariateSpline ``s``.  None -> s = n_valid (weighted) so the fit is
        statistically consistent with the error bars (chi^2 ~ dof): fluctuations
        within ~1 SE are smoothed, real trends are kept.
    weighted : bool
        If True, weight points by 1/SE[g] (recommended; heteroscedastic bins).
    n_fine : int
        Points on the fine plotting grid.

    Returns
    -------
    A_fine          : fine grid of A.
    mu_prime        : mu'(A) = 1 + g'(A) on the fine grid (for plotting panel c).
    gspline         : the fitted spline of g(A) (callers differentiate it).
    g_prime_centers : g'(A) at the (all) bin centres; index with `valid`.
    """
    x = bin_centers[valid]
    g = cond_bias[valid]
    if weighted:
        w = 1.0 / np.clip(cond_bias_err[valid], 1e-12, None)
        s = smoothing_factor if smoothing_factor is not None else len(x)
        gspline = UnivariateSpline(x, g, w=w, s=s, k=3)
    else:
        s = smoothing_factor if smoothing_factor is not None else len(x) * np.var(g)
        gspline = UnivariateSpline(x, g, s=s, k=3)

    dg = gspline.derivative()
    A_fine = np.linspace(x.min(), x.max(), n_fine)
    mu_prime = 1.0 + dg(A_fine)
    g_prime_centers = dg(bin_centers)
    return A_fine, mu_prime, gspline, g_prime_centers


# =====================================================================
#  3.  Longest contiguous run with gap tolerance
# =====================================================================
def _longest_run_gap(mask: np.ndarray, max_gap: int = 0) -> Tuple[int, int, int]:
    """
    Longest run of True allowing up to ``max_gap`` consecutive False inside it.
    Returns (span_length, i0, i1) with i0,i1 the first/last True of the best run
    (both endpoints are True). span_length = i1 - i0 + 1.  (0,-1,-1) if none.
    """
    n = len(mask)
    best = (0, -1, -1)
    for i in range(n):
        if not mask[i]:
            continue
        j, last, gap = i, i, 0
        while j + 1 < n:
            if mask[j + 1]:
                j += 1; last = j; gap = 0
            elif gap < max_gap:
                j += 1; gap += 1
            else:
                break
        if last - i + 1 > best[0]:
            best = (last - i + 1, i, last)
    return best


# =====================================================================
#  4.  The two ranges:  bias-consistent region + strict g'~0 band
# =====================================================================
def find_locally_unbiased_band(
    stats: Dict,
    gspline,
    eps_slope: float = 0.05,
    sigma_mf: Optional[float] = None,
    bias_floor_frac: float = 0.10,
    k_se: float = 2.0,
    max_gap: int = 1,
    min_band_bins: int = 3,
) -> Dict:
    """
    Locate (i) the bias-consistent region and (ii) the strict g'~0 band.

    Gates (per valid bin):
        bias gate :  |g(A)|  < max( k_se * SE[g] , floor )       floor uses
                     sigma_MF if supplied else internal sigma_int
        slope gate:  |g'(A)| < eps_slope                          g' from gspline

    The bias-consistent REGION is the longest gap-tolerant run of bias-gate
    passes; the strict BAND is the longest gap-tolerant run of (bias & slope).

    Returns a dict with (selected keys):
      strict band : 'found','band_mask','band_bins','A_lo','A_hi'
      region      : 'region_found','region_mask','region_bins','region_A_lo','region_A_hi'
      gates       : 'g_prime','slope_gate','bias_gate','combined'
      scale       : 'sigma_scale','sigma_source','sigma_int','floor'
      echoes      : 'eps_slope','bias_floor_frac','k_se','max_gap'
    """
    v = stats['valid']
    n_bins = len(v)
    bc = stats['bin_centers']
    g = stats['cond_bias']
    g_err = stats['cond_bias_err']
    V = stats['cond_var']

    # g'(A) at every bin centre (NaN out invalid bins)
    g_prime = np.where(v, gspline.derivative()(bc), np.nan)
    slope_gate = v & (np.abs(g_prime) < eps_slope)

    # internal noise scale (used iff sigma_mf is None) -- high-SNR sqrt(V) plateau
    sqrtV = np.sqrt(np.where(v, V, np.nan))
    A_med = np.nanmedian(bc[v]) if v.any() else np.nan
    hi = slope_gate & (bc >= A_med)
    if np.count_nonzero(hi) >= 2:
        sigma_int = float(np.nanmedian(sqrtV[hi]))
    elif np.count_nonzero(slope_gate) >= 1:
        sigma_int = float(np.nanmedian(sqrtV[slope_gate]))
    else:
        hiA = v & (bc >= np.nanpercentile(bc[v], 75))
        sigma_int = float(np.nanmedian(sqrtV[hiA])) if hiA.any() else float(np.nanmedian(sqrtV))

    if sigma_mf is not None:
        sigma_scale, sigma_source = float(sigma_mf), 'sigma_MF'
    else:
        sigma_scale, sigma_source = sigma_int, 'sigma_int'
    floor = bias_floor_frac * sigma_scale

    bias_tol = np.maximum(k_se * g_err, floor)
    bias_gate = v & (np.abs(g) < bias_tol)
    combined = slope_gate & bias_gate

    def _run(mask):
        L, i0, i1 = _longest_run_gap(mask, max_gap)
        if i0 < 0:
            return False, np.zeros(n_bins, bool), np.array([], int), np.nan, np.nan
        m = np.zeros(n_bins, bool)
        idx = [k for k in range(i0, i1 + 1) if mask[k]]
        m[idx] = True
        found = len(idx) >= min_band_bins
        return found, m, np.array(idx), float(stats['bin_edges'][i0]), float(stats['bin_edges'][i1 + 1])

    b_found, b_mask, b_bins, b_lo, b_hi = _run(combined)
    r_found, r_mask, r_bins, r_lo, r_hi = _run(bias_gate)

    return {
        'found': b_found, 'band_mask': b_mask, 'band_bins': b_bins,
        'A_lo': b_lo, 'A_hi': b_hi,
        'region_found': r_found, 'region_mask': r_mask, 'region_bins': r_bins,
        'region_A_lo': r_lo, 'region_A_hi': r_hi,
        'g_prime': g_prime, 'slope_gate': slope_gate, 'bias_gate': bias_gate,
        'combined': combined,
        'sigma_scale': sigma_scale, 'sigma_source': sigma_source,
        'sigma_int': sigma_int, 'floor': floor,
        'eps_slope': eps_slope, 'bias_floor_frac': bias_floor_frac,
        'k_se': k_se, 'max_gap': max_gap,
    }


# =====================================================================
#  5.  Aggregation over a set of bins (strict band OR bias region)
# =====================================================================
def _ols_with_se(x: np.ndarray, y: np.ndarray):
    """OLS y = b0 + b1 x; return (b1, b0, se_b1, se_b0)."""
    n = len(x); xbar = x.mean()
    Sxx = np.sum((x - xbar) ** 2)
    b1 = np.sum((x - xbar) * (y - y.mean())) / Sxx
    b0 = y.mean() - b1 * xbar
    resid = y - (b0 + b1 * x)
    s2 = np.sum(resid ** 2) / (n - 2)
    se_b1 = np.sqrt(s2 / Sxx)
    se_b0 = np.sqrt(s2 * (1.0 / n + xbar ** 2 / Sxx))
    return float(b1), float(b0), float(se_b1), float(se_b0)


def _aggregate(stats, mask, g_prime, y_true, y_pred, A_lo, A_hi,
               sigma_mf: Optional[float]) -> Dict:
    """Aggregate bias/variance/regression/R over the bins in `mask` and the
    pooled pairs in [A_lo, A_hi]. Returns {} if the selection is empty."""
    v = stats['valid']
    sel = mask & v
    if sel.sum() == 0:
        return {}
    g = stats['cond_bias'][sel]
    V = stats['cond_var'][sel]
    counts = stats['counts'][sel]
    gp = g_prime[sel]
    w = counts / counts.sum()

    mean_V = float(np.sum(w * V))
    mean_g2 = float(np.sum(w * g ** 2))
    mean_mse = mean_g2 + mean_V
    out = {
        'n_bins': int(sel.sum()), 'n_eff': int(counts.sum()),
        'A_lo': float(A_lo), 'A_hi': float(A_hi),
        'RMSE_bc': float(np.sqrt(mean_V)),
        'RMSE_raw_binned': float(np.sqrt(mean_mse)),
        'MACB': float(np.mean(np.abs(g))),
        'bias_fraction': mean_g2 / mean_mse if mean_mse > 0 else 0.0,
        'mean_cond_var': mean_V,
        'mean_gprime': float(np.sum(w * gp)),
        'max_abs_gprime': float(np.nanmax(np.abs(gp))),
    }

    # pooled (pixel-level) residuals, regression, residual gradient dg/dA = b1-1
    in_band = (y_true >= A_lo) & (y_true < A_hi)
    yt, yp = y_true[in_band], y_pred[in_band]
    resid = yp - yt
    out['n_pairs'] = int(in_band.sum())
    out['RMSE_pooled'] = float(np.sqrt(np.mean(resid ** 2)))
    b1, b0, se1, se0 = _ols_with_se(yt, yp)
    r_pear, p_pear = pearsonr(yt, yp)
    ss_res = np.sum(resid ** 2); ss_tot = np.sum((yt - yt.mean()) ** 2)
    out.update({
        'reg_slope': b1, 'reg_intercept': b0,
        'reg_slope_se': se1, 'reg_intercept_se': se0,
        'dg_dA': b1 - 1.0, 'dg_dA_se': se1,
        'dg_dA_z': (b1 - 1.0) / se1 if se1 > 0 else np.nan,
        'reg_r2': float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        'reg_pearson_r': float(r_pear), 'reg_pearson_p': float(p_pear),
    })

    if sigma_mf is not None:
        sig2 = float(sigma_mf) ** 2
        Rc = V / ((1.0 + gp) ** 2 * sig2)          # pointwise shrinkage-cleaned R
        R = float(np.sum(w * Rc))
        var_VoverMF = mean_V / sig2 if sig2 > 0 else np.nan   # <V>/sigma_MF^2 (<1 = below MF)
        # significance of the genuine advantage = significance of R < 1.
        # The dominant error is on <V> (SE[V]/V = sqrt(2/N)); g' uncertainty is
        # subdominant, so SE[R]/R ~ sqrt(2/N_eff).
        se_R = R * np.sqrt(2.0 / max(out['n_eff'] - 1, 1))
        z_R = (1.0 - R) / se_R if se_R > 0 else np.nan       # >0 means R below 1
        # (kept for reference) significance of bare <V> < sigma_MF^2
        se_meanV = mean_V * np.sqrt(2.0 / max(out['n_eff'] - 1, 1))
        z_var = (sig2 - mean_V) / se_meanV if se_meanV > 0 else np.nan
        out.update({
            'sigma_mf': float(sigma_mf), 'sigma_mf_sq': sig2,
            'R': R, 'R_se': float(se_R),
            'z_R': float(z_R), 'p_R': float(norm.sf(z_R)) if np.isfinite(z_R) else np.nan,
            'var_VoverMF': float(var_VoverMF),
            'z_var_below_MF': float(z_var),
            'genuine_advantage': bool((R < 1.0) and np.isfinite(z_R) and (z_R > 2.0)),
        })
    return out


def compute_band_summaries(stats, band, y_true, y_pred,
                           sigma_mf: Optional[float] = None) -> Dict:
    """
    Aggregate over BOTH the strict g'~0 band and the (wide) bias-consistent
    region.  Returns {'strict': {...}|None, 'region': {...}|None}.

    For each range: bias-corrected RMSE sqrt(<V>), raw pooled RMSE, MACB,
    bias-fraction, within-range OLS regression and the residual gradient
    dg/dA (= slope-1) with its significance, and -- if sigma_mf is given --
    the shrinkage-cleaned R(A), the bare variance ratio, and the
    significance of <V> < sigma_MF**2 with a genuine-advantage flag.
    """
    gp = band['g_prime']
    strict = (_aggregate(stats, band['band_mask'], gp, y_true, y_pred,
                         band['A_lo'], band['A_hi'], sigma_mf)
              if band['found'] else None)
    region = (_aggregate(stats, band['region_mask'], gp, y_true, y_pred,
                         band['region_A_lo'], band['region_A_hi'], sigma_mf)
              if band['region_found'] else None)
    return {'strict': strict or None, 'region': region or None}


def compute_R_ratio(stats: Dict, band: Dict, sigma_mf: float) -> np.ndarray:
    """Pointwise R(A) = V(A) / ([1+g'(A)]^2 sigma_MF^2) at bin centres (NaN where
    invalid).  Requires sigma_mf."""
    v = stats['valid']
    sig2 = float(sigma_mf) ** 2
    R = stats['cond_var'] / ((1.0 + band['g_prime']) ** 2 * sig2)
    return np.where(v, R, np.nan)


# =====================================================================
#  6.  Robustness scan over spline smoothing / eps_slope
# =====================================================================
def band_robustness_scan(
    y_true, y_pred, n_bins: int = 25,
    smoothing_factors: Optional[list] = None,
    eps_slopes: Optional[list] = None,
    sigma_mf: Optional[float] = None,
    max_gap: int = 1,
) -> list:
    """
    Re-derive the strict band and the bias region over a grid of spline-smoothing
    and eps_slope values.  Returns a list of dicts with the spans and the
    bias-corrected within-range RMSE for each combination.
    """
    if smoothing_factors is None:
        smoothing_factors = [None]
    if eps_slopes is None:
        eps_slopes = [0.03, 0.05, 0.08]

    stats = compute_conditional_statistics(y_true, y_pred, n_bins=n_bins)
    out = []
    for s in smoothing_factors:
        _, _, gsp, _ = compute_local_slope(
            stats['bin_centers'], stats['cond_bias'], stats['cond_bias_err'],
            stats['valid'], smoothing_factor=s)
        for eps in eps_slopes:
            band = find_locally_unbiased_band(stats, gsp, eps_slope=eps,
                                              sigma_mf=sigma_mf, max_gap=max_gap)
            summ = compute_band_summaries(stats, band, y_true, y_pred,
                                          sigma_mf=sigma_mf)
            st = summ['strict']; rg = summ['region']
            out.append({
                's': s, 'eps_slope': eps,
                'band_found': band['found'],
                'band_A': (band['A_lo'], band['A_hi']),
                'band_nbins': int(band['band_mask'].sum()),
                'band_RMSE_bc': st['RMSE_bc'] if st else np.nan,
                'region_A': (band['region_A_lo'], band['region_A_hi']),
                'region_nbins': int(band['region_mask'].sum()),
                'region_RMSE_bc': rg['RMSE_bc'] if rg else np.nan,
                'region_R': (rg.get('R', np.nan) if rg else np.nan),
            })
    return out


# =====================================================================
#  7.  Plot: standalone conditional bias g(A) with both ranges marked
# =====================================================================
def plot_conditional_bias_with_band(
    stats: Dict, band: Dict, summaries: Optional[Dict] = None,
    scalar_summaries: Optional[Dict] = None, title_prefix: str = "CNN",
    ax: Optional[plt.Axes] = None, save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Standalone g(A) plot with the wide bias-consistent region (light) and the
    strict g'~0 band (dark) both shaded, the +/-2 SE null ribbon, the floor, and
    an annotation block.
    """
    v = stats['valid']
    bc = stats['bin_centers'][v]
    g = stats['cond_bias'][v]
    g_err = stats['cond_bias_err'][v]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
    else:
        fig = ax.figure

    if band['region_found']:
        ax.axvspan(band['region_A_lo'], band['region_A_hi'], color='steelblue',
                   alpha=0.10, label=f"bias-consistent region [{band['region_A_lo']:.2f}, {band['region_A_hi']:.2f}]")
    if band['found']:
        ax.axvspan(band['A_lo'], band['A_hi'], color='mediumseagreen', alpha=0.22,
                   label=f"strict $g'\\!\\approx\\!0$ band [{band['A_lo']:.2f}, {band['A_hi']:.2f}]")

    ax.fill_between(bc, -2 * g_err, 2 * g_err, alpha=0.15, color='gray',
                    label=r'$\pm 2\,\mathrm{SE}[g]$ (unbiased null)')
    ax.axhline(0, color='black', ls='--', lw=0.8)
    floor = band['floor']
    sname = 'MF' if band['sigma_source'] == 'sigma_MF' else 'int'
    ax.axhline(+floor, color='darkorange', ls=':', lw=1.1,
               label=f"$\\pm$floor = {band['bias_floor_frac']:.2f}$\\sigma_{{\\rm {sname}}}$ = {floor:.3f}")
    ax.axhline(-floor, color='darkorange', ls=':', lw=1.1)

    ax.errorbar(bc, g, yerr=2 * g_err, fmt='o-', markersize=4, capsize=2,
                color='steelblue', ecolor='lightblue', zorder=5,
                label=r'$g(A)=\langle\hat A\rangle - A$')
    if band['found']:
        bm = band['band_mask'][v]
        ax.plot(bc[bm], g[bm], 'o', markersize=6, color='darkgreen', zorder=6,
                label='strict-band bins')

    ax.set_xlabel('True Amplitude $A$')
    ax.set_ylabel('Conditional Bias $g(A)$')
    ax.set_title(f'{title_prefix}: Conditional Bias with Unbiased Ranges')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', framealpha=0.9)

    lines = [f"$\\varepsilon_s$={band['eps_slope']:.3f}  "
             f"($\\sigma_{{\\rm {sname}}}$={band['sigma_scale']:.3f}, gap={band['max_gap']})"]
    if scalar_summaries is not None:
        lines.append(f"global MACB = {scalar_summaries['MACB']:.4f}")
    if summaries is not None:
        rg = summaries.get('region'); st = summaries.get('strict')
        if rg is not None:
            extra = f", $dg/dA$={rg['dg_dA']:+.3f}±{rg['dg_dA_se']:.3f}"
            lines.append(f"region: RMSE_bc={rg['RMSE_bc']:.4f}{extra}")
            if 'R' in rg:
                lines.append(f"region: $\\langle R\\rangle$={rg['R']:.3f}")
        if st is not None:
            lines.append(f"strict band: RMSE_bc={st['RMSE_bc']:.4f}")
    ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes, fontsize=8.5,
            va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# =====================================================================
#  8.  Full multi-panel diagnostics (band-aware; sigma_mf optional)
# =====================================================================
def plot_biasfixed_diagnostics(
    stats, band, A_fine, mu_prime, summaries, scalar_summaries,
    sigma_mf: Optional[float] = None, title_prefix: str = "CNN",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Band-aware figure.  2x2 if sigma_mf supplied (adds R(A)), else 1x3.
      (a) g(A) with both ranges     (b) bias-variance (+ biased-CRLB if sigma_mf)
      (c) bias gradient g'(A)        (d) R(A)            [only if sigma_mf]
    """
    v = stats['valid']
    bc = stats['bin_centers'][v]
    cv = stats['cond_var'][v]
    cb2 = stats['cond_bias_sq'][v]
    cmse = stats['cond_mse'][v]
    gp_c = band['g_prime']
    have_mf = sigma_mf is not None
    rg = summaries.get('region'); stb = summaries.get('strict')

    if have_mf:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
        ax_a, ax_b, ax_c = axes; ax_d = None
    fig.suptitle(f"{title_prefix}: Bias-Fixed Diagnostics"
                 f"{'  (sigma_MF supplied)' if have_mf else '  (checkpoint-only)'}",
                 fontsize=14)

    # (a)
    plot_conditional_bias_with_band(stats, band, summaries=summaries,
                                    scalar_summaries=scalar_summaries,
                                    title_prefix=title_prefix, ax=ax_a)
    ax_a.set_title('(a) Conditional Bias $g(A)$')

    def _shade(ax):
        if band['region_found']:
            ax.axvspan(band['region_A_lo'], band['region_A_hi'], color='steelblue', alpha=0.10)
        if band['found']:
            ax.axvspan(band['A_lo'], band['A_hi'], color='mediumseagreen', alpha=0.20)

    # (b)
    ax_b.plot(bc, cv, 's-', ms=4, color='forestgreen', label=r'Var$[\hat A|A]$')
    ax_b.plot(bc, cb2, '^-', ms=4, color='tomato', label=r'$g(A)^2$')
    ax_b.plot(bc, cmse, 'D-', ms=4, color='steelblue', label=r'MSE$(A)$')
    if have_mf:
        sig2 = float(sigma_mf) ** 2
        ax_b.axhline(sig2, color='black', ls=':', lw=1.5,
                     label=f'$\\sigma^2_{{\\rm MF}}$={sig2:.3f}')
        ax_b.plot(bc, (1.0 + gp_c[v]) ** 2 * sig2, '-', color='purple', lw=1.3,
                  alpha=0.8, label=r'$[1+g^\prime]^2\sigma^2_{\rm MF}$')
    _shade(ax_b)
    ax_b.set_xlabel('True Amplitude $A$'); ax_b.set_ylabel('Variance / MSE')
    ax_b.set_title('(b) Bias-Variance Decomposition')
    ax_b.set_ylim(bottom=0); ax_b.grid(True, alpha=0.3); ax_b.legend(fontsize=8)
    if rg is not None:
        txt = f"region RMSE_bc={rg['RMSE_bc']:.4f}\n$dg/dA$={rg['dg_dA']:+.3f}±{rg['dg_dA_se']:.3f}"
        if have_mf:
            txt += f"\nregion $\\langle R\\rangle$={rg['R']:.3f}"
        ax_b.text(0.02, 0.98, txt, transform=ax_b.transAxes, fontsize=8.5,
                  va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # (c) bias gradient g'(A)
    ax_c.plot(A_fine, mu_prime - 1.0, '-', color='steelblue', lw=1.5,
              label=r"$g'(A)=\mu'(A)-1$")
    ax_c.plot(bc, gp_c[v], 'o', ms=3, color='gray', alpha=0.6, label='at bin centres')
    ax_c.axhline(0.0, color='black', ls='--', lw=0.8)
    ax_c.axhline(+band['eps_slope'], color='darkorange', ls=':', lw=1.0)
    ax_c.axhline(-band['eps_slope'], color='darkorange', ls=':', lw=1.0,
                 label=r"$\pm\varepsilon_s$")
    _shade(ax_c)
    ax_c.set_xlabel('True Amplitude $A$'); ax_c.set_ylabel(r"Bias gradient $g'(A)$")
    ax_c.set_title("(c) Bias Gradient $g'(A)$  [gate: $|g'|<\\varepsilon_s$]")
    ax_c.grid(True, alpha=0.3); ax_c.legend(fontsize=8)

    # (d) R(A)
    if have_mf and ax_d is not None:
        R = compute_R_ratio(stats, band, sigma_mf)[v]
        ax_d.plot(bc, R, 'o-', ms=4, color='crimson',
                  label=r'$R(A)=V/[(1+g^\prime)^2\sigma^2_{\rm MF}]$')
        ax_d.axhline(1.0, color='black', ls='--', lw=0.9, label=r'$R=1$ (Gaussian floor)')
        _shade(ax_d)
        ax_d.fill_between(bc, 0, 1, where=(R < 1.0), color='gold', alpha=0.25)
        ax_d.set_xlabel('True Amplitude $A$'); ax_d.set_ylabel('$R(A)$')
        ax_d.set_title('(d) Shrinkage-Cleaned Ratio $R(A)$  [$<1$ = genuine gain]')
        ax_d.grid(True, alpha=0.3); ax_d.legend(fontsize=8)
        if rg is not None:
            adv = "YES" if rg['genuine_advantage'] else "no"
            ax_d.text(0.02, 0.02,
                      f"region $\\langle R\\rangle$={rg['R']:.3f}±{rg['R_se']:.3f}\n"
                      f"$R<1$ at z={rg['z_R']:+.1f}\n"
                      f"genuine advantage: {adv}",
                      transform=ax_d.transAxes, fontsize=8.5, va='bottom',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# =====================================================================
#  9.  Scalar summaries (global; kept for the pair)
# =====================================================================
def compute_scalar_summaries(stats: Dict, sigma_mf: Optional[float] = None) -> Dict:
    """Global population-weighted scalar diagnostics, independent of the band."""
    v = stats['valid']
    g = stats['cond_bias'][v]
    g_err = stats['cond_bias_err'][v]
    cond_var = stats['cond_var'][v]
    counts = stats['counts'][v]
    weights = counts / counts.sum()

    macb = float(np.mean(np.abs(g)))
    mean_bias_sq = float(np.sum(weights * stats['cond_bias_sq'][v]))
    mean_cond_var = float(np.sum(weights * cond_var))
    mean_mse = mean_bias_sq + mean_cond_var
    chi2_val = float(np.sum((g / g_err) ** 2))
    n_dof = int(len(g))
    out = {
        'MACB': macb,
        'MACB_weighted': float(np.sum(weights * np.abs(g))),
        'RMS_bias': float(np.sqrt(np.mean(g ** 2))),
        'mean_cond_var': mean_cond_var, 'mean_bias_sq': mean_bias_sq,
        'bias_fraction': mean_bias_sq / mean_mse if mean_mse > 0 else 0.0,
        'chi2_unbiased': chi2_val, 'chi2_dof': n_dof,
        'chi2_pval': float(1.0 - chi2_dist.cdf(chi2_val, df=n_dof)),
    }
    if sigma_mf is not None:
        out['variance_ratio_vs_MF'] = float(sigma_mf) ** 2 / mean_cond_var
    return out


# =====================================================================
#  10.  Top-level driver
# =====================================================================
def run_biasfixed_analysis(
    y_true, y_pred, sigma_mf: Optional[float] = None, n_bins: int = 25,
    eps_slope: float = 0.05, bias_floor_frac: float = 0.10, max_gap: int = 1,
    smoothing_factor: Optional[float] = None, title_prefix: str = "CNN",
    make_plot: bool = True, save_path: Optional[str] = None,
):
    """
    End-to-end: conditional stats -> g(A) spline -> ranges -> summaries -> report
    (+plot).  Reports the wide bias-consistent region (primary, with R(A) if
    sigma_mf given) and the strict g'~0 band (cross-check), plus the residual
    gradient dg/dA.  Returns (stats, band, summaries, scalar_summaries, fig).
    """
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]

    stats = compute_conditional_statistics(y_true, y_pred, n_bins=n_bins)
    A_fine, mu_prime, gsp, _ = compute_local_slope(
        stats['bin_centers'], stats['cond_bias'], stats['cond_bias_err'],
        stats['valid'], smoothing_factor=smoothing_factor)
    band = find_locally_unbiased_band(stats, gsp, eps_slope=eps_slope,
                                      sigma_mf=sigma_mf,
                                      bias_floor_frac=bias_floor_frac,
                                      max_gap=max_gap)
    scalar = compute_scalar_summaries(stats, sigma_mf=sigma_mf)
    summ = compute_band_summaries(stats, band, y_true, y_pred, sigma_mf=sigma_mf)
    rg, stb = summ['region'], summ['strict']

    bar = "=" * 70
    print(bar)
    print(f"BIAS-FIXED ANALYSIS  -  {title_prefix}")
    print(f"  mode: {'sigma_MF supplied' if sigma_mf is not None else 'CHECKPOINT-ONLY (no sigma_MF)'}")
    print(bar)
    print(f"  global MACB={scalar['MACB']:.5f}   global bias-fraction={scalar['bias_fraction']:.2%}")
    print(f"  chi^2/nu={scalar['chi2_unbiased']:.1f}/{scalar['chi2_dof']} (p={scalar['chi2_pval']:.1e})"
          f"   g'(A) range=[{(mu_prime-1).min():+.3f}, {(mu_prime-1).max():+.3f}]")

    # ---- PRIMARY: bias-consistent region ----
    print("-" * 70)
    if rg is None:
        print("  BIAS-CONSISTENT REGION : NOT FOUND (try larger n_bins / floor).")
    else:
        print(f"  BIAS-CONSISTENT REGION (primary) : A in [{rg['A_lo']:.3f}, {rg['A_hi']:.3f}]"
              f"  ({rg['n_bins']} bins, N={rg['n_eff']})")
        print(f"    RMSE (bias-corrected) = {rg['RMSE_bc']:.5f}   MACB = {rg['MACB']:.5f}"
              f"   bias-frac = {rg['bias_fraction']:.2%}")
        print(f"    net residual gradient dg/dA = {rg['dg_dA']:+.4f} +/- {rg['dg_dA_se']:.4f}"
              f"  ({rg['dg_dA_z']:+.1f}sigma over the region)")
        print(f"    local max|g'(A)| in region = {rg['max_abs_gprime']:.3f}  "
              f"(vs eps_s={band['eps_slope']:.3f}; this local excursion limits the strict band)")
        if sigma_mf is not None:
            print(f"    <V>/sigma_MF^2 = {rg['var_VoverMF']:.3f}   "
                  f"shrinkage-cleaned <R(A)> = {rg['R']:.3f} +/- {rg['R_se']:.3f}")
            print(f"    advantage (R<1): "
                  f"{'YES' if rg['genuine_advantage'] else 'no'}"
                  f"  (R below 1 at {rg['z_R']:+.1f} sigma)")
        else:
            print("    sigma_MF not supplied -> R(A) / advantage verdict DECLINED.")

    # ---- CROSS-CHECK: strict band ----
    print("-" * 70)
    if stb is None:
        print("  STRICT g'~0 BAND : NOT FOUND (residual gradient exceeds eps_slope).")
    else:
        print(f"  STRICT g'~0 BAND (cross-check)   : A in [{stb['A_lo']:.3f}, {stb['A_hi']:.3f}]"
              f"  ({stb['n_bins']} bins)")
        print(f"    RMSE (bias-corrected) = {stb['RMSE_bc']:.5f}   "
              f"reg. slope b1 = {stb['reg_slope']:.4f} +/- {stb['reg_slope_se']:.4f}")
        if sigma_mf is not None:
            print(f"    <R(A)> = {stb['R']:.3f}")
    print(bar)

    fig = None
    if make_plot:
        fig = plot_biasfixed_diagnostics(stats, band, A_fine, mu_prime, summ,
                                         scalar, sigma_mf=sigma_mf,
                                         title_prefix=title_prefix,
                                         save_path=save_path)
    return stats, band, summ, scalar, fig


# =====================================================================
#  11.  Multi-run comparison table (e.g. std-MSE vs bias-corrected MSE)
# =====================================================================
def compare_runs(
    runs: List[Tuple[str, np.ndarray, np.ndarray]],
    sigma_mf: Optional[float] = None, n_bins: int = 25, eps_slope: float = 0.05,
    bias_floor_frac: float = 0.10, smoothing_factor: Optional[float] = None,
    max_gap: int = 1,
) -> List[Dict]:
    """
    Print a side-by-side verdict table for several runs and return the rows.

    ``runs`` : list of (label, y_true, y_pred).

    Columns: net residual gradient dg/dA over the bias-consistent region
    (+/- SE, sigma); the local max|g'| there (what limits the strict band);
    the bias-consistent region (bins, span); the strict g'~0 band (bins, span);
    and -- if sigma_mf given -- <V>/sigma_MF^2 and the shrinkage-cleaned <R(A)>
    with its significance below 1.
    """
    rows = []
    for label, yt, yp in runs:
        m = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[m], yp[m]
        stats = compute_conditional_statistics(yt, yp, n_bins=n_bins)
        _, _, gsp, _ = compute_local_slope(
            stats['bin_centers'], stats['cond_bias'], stats['cond_bias_err'],
            stats['valid'], smoothing_factor=smoothing_factor)
        band = find_locally_unbiased_band(stats, gsp, eps_slope=eps_slope,
                                          sigma_mf=sigma_mf,
                                          bias_floor_frac=bias_floor_frac,
                                          max_gap=max_gap)
        summ = compute_band_summaries(stats, band, yt, yp, sigma_mf=sigma_mf)
        rg = summ['region']
        rows.append({
            'label': label,
            'dg_dA': rg['dg_dA'] if rg else np.nan,
            'dg_dA_se': rg['dg_dA_se'] if rg else np.nan,
            'dg_dA_sigma': rg['dg_dA_z'] if rg else np.nan,
            'max_abs_gprime': rg['max_abs_gprime'] if rg else np.nan,
            'region_A': (band['region_A_lo'], band['region_A_hi']),
            'region_nbins': int(band['region_mask'].sum()),
            'band_A': (band['A_lo'], band['A_hi']),
            'band_nbins': int(band['band_mask'].sum()),
            'var_VoverMF': (rg.get('var_VoverMF', np.nan) if rg else np.nan),
            'R': (rg.get('R', np.nan) if rg else np.nan),
            'R_se': (rg.get('R_se', np.nan) if rg else np.nan),
            'z_R': (rg.get('z_R', np.nan) if rg else np.nan),
            'RMSE_bc': (rg['RMSE_bc'] if rg else np.nan),
        })

    # ---- pretty print ----
    h = (f"{'run':<14}{'dg/dA region (sig)':>20}{'max|g′|':>9}"
         f"{'bias-region (bins)':>21}{'g′~0 band (bins)':>20}"
         f"{'<V>/σ²':>9}{'<R> (sig<1)':>16}")
    print("=" * len(h)); print(h); print("-" * len(h))
    for r in rows:
        ra = f"[{r['region_A'][0]:.2f},{r['region_A'][1]:.2f}]({r['region_nbins']})"
        ba = f"[{r['band_A'][0]:.2f},{r['band_A'][1]:.2f}]({r['band_nbins']})"
        dg = f"{r['dg_dA']:+.3f}±{r['dg_dA_se']:.3f}({r['dg_dA_sigma']:+.1f})" if np.isfinite(r['dg_dA']) else "  -  "
        mg = f"{r['max_abs_gprime']:.3f}" if np.isfinite(r['max_abs_gprime']) else "  -  "
        vr = f"{r['var_VoverMF']:.3f}" if np.isfinite(r['var_VoverMF']) else "  -  "
        R = f"{r['R']:.3f}({r['z_R']:+.1f}σ)" if np.isfinite(r['R']) else "  -  "
        print(f"{r['label']:<14}{dg:>20}{mg:>9}{ra:>21}{ba:>20}{vr:>9}{R:>16}")
    print("=" * len(h))
    if sigma_mf is not None:
        print("Reading: <V>/σ²<1 looks like a variance win, but only R<1 (sig.) is a"
              " genuine non-Gaussian gain;\nR~1 = at the CRLB; R>1 = the apparent"
              " win is shrinkage. dg/dA is the net tilt over the region;\nmax|g′| is"
              " the local excursion that fragments the strict g′~0 band.")
    return rows


# =====================================================================
#  Example usage (synthetic; replace with checkpoint arrays)
# =====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 8000
    sigma_mf_true = 0.54

    A = rng.uniform(0.0, 4.0, N)
    # std-MSE-like: smooth shrinkage S-curve (steep ends, flat middle)
    yp_std = A - 0.45 * ((A - 2) / 2) ** 3 + rng.normal(0, sigma_mf_true, N)
    # bias-corrected-like: flat |g| but a mild residual positive gradient
    g_ends = 0.20 * np.exp(-A / 0.35) - 0.16 * np.exp(-(4 - A) / 0.35)
    yp_bc = A + g_ends + 0.04 * (A - 2) + rng.normal(0, sigma_mf_true, N)

    print("\n### single run (bias-corr-like, sigma_MF supplied) ###")
    run_biasfixed_analysis(A, yp_bc, sigma_mf=sigma_mf_true,
                           title_prefix="Synthetic bias-corr",
                           save_path="/tmp/biasfixed_demo.png")

    print("\n### comparison table ###")
    compare_runs([("std-MSE", A, yp_std), ("bias-corr", A, yp_bc)],
                 sigma_mf=sigma_mf_true)
