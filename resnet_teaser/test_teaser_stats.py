"""Self-test of the teaser_eval statistics path with a synthetic estimator.

A_hat = c (A + e_MF) + (1 - c) A0  is a shrunk matched filter: g'(A) = c - 1,
V(A) = c^2 sigma^2, so R(A) = V / ([1 + g']^2 sigma^2) must be 1 at every A,
and V/sigma^2 = c^2 -- "below the floor" only because of shrinkage credit.
"""
import numpy as np, sys
sys.path.insert(0, '.')
import teaser_eval as te, biasfixed_diagnostics as bd
rng = np.random.default_rng(1)
sigma, L, c, A0, n = 0.863, 5.0, 0.7, 2.5, 4000
A = np.linspace(0, L, 41)
e = rng.normal(0, sigma, n)
Ahat = c * (A[:, None] + e[None, :]) + (1 - c) * A0
st = te.grid_stats(A, Ahat)
A_fine, mu_p, gsp, _ = bd.compute_local_slope(st['bin_centers'], st['cond_bias'], st['cond_bias_err'], st['valid'])
band = bd.find_locally_unbiased_band(st, gsp, eps_slope=0.05, sigma_mf=sigma)
gp = np.asarray(band['g_prime'])
R = st['cond_var'] / ((1 + gp)**2 * sigma**2)
print("g' (expect %.3f): %.4f .. %.4f" % (c - 1, gp.min(), gp.max()))
print("V/sigma^2 (expect %.3f): %.4f .. %.4f" % (c**2, (st['cond_var']/sigma**2).min(), (st['cond_var']/sigma**2).max()))
print("R (expect 1): %.4f .. %.4f" % (R.min(), R.max()))
print("bias-consistent region found:", band.get('region_found'), " strict band found:", band.get('found'))
assert abs(gp.mean() - (c - 1)) < 0.01 and abs(R.mean() - 1) < 0.03
# unbiased case: g = 0 everywhere, region should cover the grid, R = 1
Ahat2 = A[:, None] + e[None, :]
st2 = te.grid_stats(A, Ahat2)
_, _, gsp2, _ = bd.compute_local_slope(st2['bin_centers'], st2['cond_bias'], st2['cond_bias_err'], st2['valid'])
b2 = bd.find_locally_unbiased_band(st2, gsp2, eps_slope=0.05, sigma_mf=sigma)
m = np.asarray(b2['region_mask'], bool)
print("unbiased MF: region covers %d/41 points, strict band %d/41; <V>/sigma^2 = %.4f" % (m.sum(), np.asarray(b2['band_mask'],bool).sum(), (st2['cond_var']/sigma**2)[m].mean()))
boot = te.bootstrap_band(A, Ahat2, np.asarray(b2['g_prime']), sigma, m, np.asarray(b2['band_mask'],bool), 200)
print("bootstrap <V>/sigma^2 = %.4f +- %.4f, <R> = %.4f +- %.4f" % (*boot['V_region'], *boot['R_region']))
print("crossover test:", te.crossover(A, np.linspace(0.5, 1.5, 41)))
print("SELF-TEST PASSED")
