"""
eta_pipeline — advantage-ceiling (η) computation for the Beyond-the-BLUE program
================================================================================

Implements the eta framework of "Beyond the BLUE I" (arXiv:2609.10475,
Secs. 2-4 and App. B): given a NOISE-ONLY image ensemble (generated in memory
from the registry in models.py, or read from HDF5) and an analytic template,
compute the advantage ceiling

    η = 1 + σ_MF² · τ†Δτ,      Δ = 𝒥 − N⁻¹ ⪰ 0,

the maximum variance-reduction factor over the matched filter available to
ANY estimator.  η is prior-free: no source amplitudes, no I0_range, no
training set enters — only the noise ensemble and the template.

Submodules
----------
whitening   splits (PSD/FIT/VAL/EVAL), ensemble 2-D PSD, whitening to
            E[xx†] = I, whitened unit templates t̂, σ_MF  (memo §14)
variational Method A — score-matching variational bound with the capacity
            ladder (linear / quadratic / cubic / CNN), PyTorch  (memo §16)
cumulants   Method B — pair-trick contracted-cumulant estimators
            ‖κ̃₃(t̂)‖²_F, ‖κ̃₄(t̂)‖²_F and the perturbative η  (memo §17)
quadrature  Method D — exact marginal-score η for conditionally-Gaussian
            latent mixtures (PSRAND, random orientation)  (memo §19)
analytic1d  exactly solvable 1-D references: scale-mixture η by quadrature,
            the weak-limit formula γ₃²/2 + γ₄²/6  (memo §15.2)
bounds      complete-data (perfect-removal) upper bound  (memo §20)
channels    coherent / incoherent channel decomposition (Secs. 3 and 7)
models      the noise-model registry (one row per model of the paper)
registry_tools  generate_ensemble / reconstruct_map / campaign_splits / export
variational2    production variational ladder (round 2, App. B)

("memo §N" references are to the internal development notes; the paper is
the public reference.)

Conventions (identical to noise_lib)
------------------------------------
* Spectra in the unnormalized-FFT convention, E[|fft2(n)|²] = P(k).
* Whitened maps: x = ifft2(fft2(n)·√(npix/P̂)).real ⇒ E[|fft2(x)|²] = npix
  (unit white).  Whitened template t: same filter on τ; ‖t‖² = Σ|τ̃|²/P̂ =
  1/σ_MF², and t̂ = t/‖t‖.
* Split discipline is non-negotiable (memo §14): P̂ from SPLIT_PSD only;
  fitting on SPLIT_FIT; early stopping on SPLIT_VAL; every REPORTED number
  and its bootstrap error from SPLIT_EVAL.  (SPLIT_VAL is a refinement over
  the memo, which early-stopped on the reporting split — that mildly
  selects upward MC fluctuations; a separate VAL split removes the bias.)
* DFT-normalization correctness is certified by the §15 null tests
  (run_null_tests.py); trust no η produced before they pass.
"""

__version__ = "0.1.0"

from . import whitening, cumulants, quadrature, analytic1d, bounds


def __getattr__(name):
    """Import ``variational`` / ``variational2`` on first use (PEP 562).

    Both need PyTorch.  Nothing else in the package does, and the cheap
    campaign mode advertises that it "runs anywhere"
    (``run_eta_campaign.py`` docstring) while using neither -- but an eager
    import here made ``import eta_pipeline`` fail outright on any machine
    without torch (a bare container, a login node, a numpy-only validation
    environment), so "runs anywhere" was not true.  The orchestrator already
    defers both imports into ``run_variational`` / ``run_variational2``, so
    deferring them here as well changes nothing on a torch-bearing machine and
    makes the numpy-only path work as documented.  (2026-09-03)
    """
    if name in ('variational', 'variational2'):
        import importlib
        mod = importlib.import_module('.' + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .whitening import (make_splits, estimate_psd2d, whiten_maps,
                        whitened_template, sigma_mf_from_psd)
from .cumulants import k3_norm, k4_norm, eta_perturbative
from .quadrature import eta_marg_quadrature
from .analytic1d import scale_mixture_eta_1d, eta_weak_1d
from .bounds import complete_data_bound, bound_stability
