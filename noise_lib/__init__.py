"""
noise_lib — shared library for the "Beyond the BLUE" noise-simulation framework
===============================================================================

Unified, script-based (no-notebook) implementation of every noise model used in
the CNN-vs-MF paper, consolidating the legacy modules (not part of this release):

    map_realnoise.ipynb / modified_noisemap.ipynb   (Tier-0 Gaussian cases)
    scanning_residuals.py / algorithm_scanning_artifact.py
    anisotropic_median.py / anisotropic_glitches.py / PCA_leakage.py

Design contract
---------------
* All random draws flow through an explicit ``numpy.random.Generator``; every
  driver spawns one child generator per image from a single master seed
  (``numpy.random.SeedSequence``), so any single image is exactly
  reproducible from (master_seed, image_index).
* All Gaussian random fields are produced by the transfer-function method
  (real white field -> FFT -> multiply by sqrt(P) -> IFFT), which is exactly
  Hermitian and delivers the target power spectrum *without* the factor-1/2
  suppression of the legacy Hermitian-projection generators (see grf.py).
* Power spectra are always specified in the UNNORMALIZED-FFT convention:
      E[ |numpy.fft.fft2(noise)|^2 ] = P(kx, ky),
  with frequencies in cycles/pixel (numpy.fft.fftfreq convention).
* Per-image latent variables (spectral parameters, artifact positions and
  amplitudes, orientation angles, ...) are returned by every generator and
  are stored in the output HDF5 files — they are required downstream by the
  eta-ceiling pipeline (oracle MF, Tier-1 quadrature, Gaussianization
  surrogates).

Submodules
----------
spectra        2-D power-spectrum builders (isotropic, two-component
               anisotropic, rotated-anisotropic)
grf            Gaussian random field generator + white noise
sources        beta-model / point sources, flux priors, map-domain templates
artifacts      scan-crossing residuals (sign-convention flag), sub-threshold
               glitches, PCA leaked modes
psrand         per-image and patchwise randomized-PSD 1/f noise (Tier-1)
processing     row/column median removal, k-space high-pass, PCA transfer
               function, Fourier beam smoothing
confusion      extragalactic point-source confusion: truncated dN/dS marks on a
               Poisson field, beam-convolved (the Campbell / P(D) Tier-2 rung)
io_h5          HDF5 dataset writer/reader with latents, parameters and seeds
diagnostics    radial/1-D PSD estimators, moment summaries, quick-look plots
"""

__version__ = "1.0.0"

from . import (spectra, grf, sources, artifacts, psrand, processing,
               confusion, io_h5, diagnostics)

from .grf import white_noise, grf_from_psd
from .spectra import (freq_grids, psd_isotropic_powerlaw, psd_two_component,
                      psd_scan_powerlaw)
from .sources import draw_source, beta_model, make_template
from .artifacts import (scan_crossing_artifacts, add_glitch_residuals,
                        glitch_profile_1d, pca_leaked_modes)
from .psrand import psrand_1overf_noise, patchwise_1overf_noise
from .processing import (row_median_removal, kspace_highpass, pca_transfer,
                         beam_smooth)
from .confusion import (schechter_dnds, flux_moment, confusion_stats,
                        mark_sampler, n_sources_mean, confusion_map,
                        confusion_psd)
from .io_h5 import save_dataset, load_dataset
