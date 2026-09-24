# Beyond the BLUE I — code release

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22924969.svg)](https://doi.org/10.5281/zenodo.22924969)

Code, noise models, and result files for

> **K. Basu (2026), *Beyond the BLUE I: the advantage ceiling — how much can any estimator beat the matched filter in mm/submm survey data?*** [arXiv:2609.10475](https://arxiv.org/abs/2609.10475)

The paper defines the **advantage ceiling** η: the largest factor by which *any* estimator of a source amplitude can reduce the variance of the matched filter (MF), for a given noise ensemble and source template. For Gaussian noise η = 1, which is the Gauss–Markov/BLUE statement. For non-Gaussian or conditionally Gaussian noise, η > 1 measures how much room a nonlinear method, for example a convolutional neural network, can have. This repository contains:

1. **the η framework** (`eta_pipeline/`): exact quadratures, complete-data bounds, cumulant estimators and the variational capacity ladder;
2. **the Tier-0/1/2 noise models** (`noise_lib/`, with the model registry in `eta_pipeline/models.py`);
3. **the ResNet "teaser" runs** of the paper (`resnet_teaser/`).

It also includes the scripts that produced every figure and the master tables of the paper, the paper's null-test battery, and the final result files, so that all tables and figures can be regenerated without re-running the computations.

---

## Contents

```
.
├── README.md             this file
├── LICENSE               MIT
├── CITATION.cff          citation metadata (GitHub "Cite this repository", Zenodo)
├── pyproject.toml        installs noise_lib and eta_pipeline (pip install -e .)
│
├── noise_lib/            (2) noise-model building blocks: Gaussian random fields,
│                             power spectra, beams, scan crossings, glitches, PCA leakage,
│                             per-image randomized spectra, point-source confusion, HDF5 I/O
├── eta_pipeline/         (1) the η framework
│   ├── models.py             THE MODEL REGISTRY: one row per noise model of the paper
│   ├── registry_tools.py     generate_ensemble / reconstruct_map / campaign_splits / export
│   ├── whitening.py          splits, ensemble PSD, whitening, sigma_MF
│   ├── quadrature.py         exact eta for conditionally Gaussian (Tier-1) mixtures
│   ├── bounds.py             complete-data (perfect-removal) upper bound
│   ├── cumulants.py          perturbative eta from contracted 3rd/4th cumulants
│   ├── analytic1d.py         exactly solvable 1-D references (scale mixtures, compound Poisson)
│   ├── channels.py           coherent / incoherent channel decomposition
│   └── variational.py, variational2.py   the variational capacity ladder (PyTorch)
│
├── scripts/
│   ├── run_eta_campaign.py   campaign driver: all models x both templates -> results/eta_results/
│   ├── run_null_tests.py     the null battery of Sec. 4.4 / App. C (nine tests, 100 checks)
│   ├── run_b2b_cheap.py      channel decomposition, closure tests, high-precision quadratures
│   ├── run_r3_cheap.py       inferability, Method-E envelopes, matched-seed comparisons
│   ├── run_pca_mixture_quad.py   exact eta of PCA leakage as a Gaussian scale mixture
│   ├── export_registry.py    writes configs/registry.json and configs/REGISTRY.md
│   ├── generate/             stand-alone drivers that write Tier-0/1/2 ensembles to HDF5
│   ├── tools/                calibration and diagnostic tools (white floor, DC check,
│   │                         rounding channel, confusion validation, re-run comparison, ...)
│   └── hpc/                  Slurm array jobs and the exact cell lists of the production runs
│
├── configs/
│   ├── registry.json         the registry as data: conventions, seeds, floors, parameters
│   └── REGISTRY.md           the same as readable tables
│
├── results/              the final numbers of the paper (see results/README.md)
│   ├── eta_results/          one JSON per (model, template) cell + auxiliary files
│   ├── eta_results_hpc_f64/  the float64 A/B run used in Fig. 8
│   └── tests_and_results/    confusion reference calculations; the final null-battery log
│
├── paper/                figure and table scripts of the paper (write to paper/figures/)
├── resnet_teaser/        (3) ResNet data generation, training, evaluation, Fig. 3
│                             (see resnet_teaser/README.md)
└── tests/                quick consistency tests of the release (pytest)
```

---

## Installation

Python ≥ 3.10. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate      # or a conda environment
pip install -e .                  # numpy, scipy, h5py, matplotlib: noise models, exact eta, figures
pip install -e ".[torch,test]"    # + PyTorch and pandas (variational ladder, ResNet) and pytest
```

Everything except the variational ladder (`eta_pipeline/variational*.py`) and the ResNet code runs without PyTorch. The scripts find the two packages by walking up to the repository root, so they also work without installation when run from inside the repository. The variational ladder and the ResNet code select `cuda`, Apple-silicon `mps`, or `cpu` automatically.

The release was checked in a clean Linux environment with Python 3.11, NumPy 2.4, SciPy 1.17, h5py 3.16, Matplotlib 3.11 and PyTorch 2.14 (see *Reproducing the results* below).

---

## Quick start

```bash
python -m pytest tests -q                       # release-integrity tests, well under a minute
python paper/fig09_ceiling_chart.py             # a paper figure from the shipped results
```

```python
from eta_pipeline import registry_tools as rt

maps, latents = rt.generate_ensemble('T1_PSRAND_WN', 100)   # first 100 maps of a campaign ensemble
m = rt.reconstruct_map('T2_PCA_WN', 4321)                    # map 4321 of that ensemble, on its own
splits = rt.campaign_splits(6000)                            # PSD / FIT / VAL / EVAL indices
```

---

## The model registry, the seeds and the configuration

The paper states that "every number in this paper is reconstructable from (model, image index)". The release makes that concrete as follows.

**The registry is code.** `eta_pipeline/models.py` defines the dictionary `REGISTRY`, with one row per noise model (24 rows: the Tier-0 Gaussian controls, the Tier-1 conditionally Gaussian mixtures, the Tier-2 non-Gaussian models, their `*_WN` white-floor variants, and the three confusion rows). Each row holds the generator function, an explicit master seed, the white-noise floor σ_w, and the per-method settings (variational rungs, quadrature builder, complete-data-bound background, cumulant validity). The severity parameters are module constants at the top of the file (`P_ISO_ARGS`, `PSRAND_ARGS`, `CROSS_ARGS`, `GLITCH_ARGS`, `PCA_ARGS`, `WN_FLOORS`, …). This file is the single source of truth. It is not duplicated anywhere else.

**The registry is also exported as data.** Some rows contain executable objects, so a human-readable copy is generated from the code:

* [`configs/REGISTRY.md`](configs/REGISTRY.md) is a table of all models: code name, paper name, tier, master seed, white floor, rungs, and methods. It also lists every seed stream and the severity parameters of each model.
* [`configs/registry.json`](configs/registry.json) contains the same information, machine-readable.

`python scripts/export_registry.py` regenerates both files. `tests/test_registry.py` fails if they no longer match the code.

**Seeds.** Image *i* of model *M* is drawn from child *i* of `numpy.random.SeedSequence(master_seed[M])`. A child depends only on the master seed and its index, so any single map can be regenerated without the rest of the ensemble (`reconstruct_map`), and appending maps never changes existing ones. The other random streams are fixed offsets and are listed in `REGISTRY.md`:

* the `*_WN` floors: master seed + 606000;
* the ResNet training and evaluation sets: + 103000 and + 213000;
* the PSD/FIT/VAL/EVAL split: seed 12345.

No map is stored anywhere. Every ensemble is regenerated in memory in seconds to minutes.

**Configuration of each computed number.** Every result file in `results/eta_results/` records, next to the numbers, the model, template, tier, ensemble size, master seed, split seed and floor. For the variational ladder it also records the full run configuration (`variational2.config`: epochs, restarts, patience, learning rate, pooling basis, FIT/VAL/EVAL sizes, device) and per-restart training telemetry. The Slurm cell lists in `scripts/hpc/cells*.txt` are the exact command-line settings of the production passes. The files are commented, including the reasons for the choices. `cells_final.txt` holds the final confusion pass.

---

## Paper figures and tables → scripts

Figure numbers refer to arXiv:2609.10475v1. All figure scripts read `results/` and write PDF + PNG to `paper/figures/`. Set `BTB_FIGDIR` to write elsewhere, or `BTB_RESULTS` to plot a re-run instead of the shipped results.

| paper | content | script |
|---|---|---|
| Fig. 1 | Fisher bands of the two templates | `paper/fig02_fisher_band.py` |
| Fig. 2 | gallery of the noise models | `paper/fig01_noise_gallery.py` |
| Fig. 3 | ResNet teaser: variance ratio vs amplitude | `resnet_teaser/make_fig12_teaser.py` (see below) |
| Fig. 4 | coherent / incoherent channels of the Tier-1 mixtures | `paper/fig07_channels.py` |
| Fig. 5 | Method-E envelope | `paper/fig06_methodE.py` |
| Fig. 6 | the variational ladders of the Tier-2 models | `paper/fig08_ladders.py` |
| Fig. 7 | source confusion | `paper/fig11_confusion.py` |
| Fig. 8 | unmodeled channels (float32 rounding side channel) | `paper/fig10_unmodelled.py` |
| Fig. 9 | summary of all ceilings | `paper/fig09_ceiling_chart.py` |
| App. E tables | master tables (floored, floorless) and code-name table | `paper/tabD_master.py` (writes LaTeX) |
| Sec. 4.4, App. C | null battery | `scripts/run_null_tests.py`; final log in `results/tests_and_results/` |

The file names come from an earlier draft order and do not match the final figure numbers. The table above gives the correspondence.

Fig. 3 from the shipped evaluation files:

```bash
python resnet_teaser/make_fig12_teaser.py \
  resnet_teaser/results/TEASER_T0_RED_REAL_L5_bc15.pth.best_eval.json \
  resnet_teaser/results/TEASER_T0_RED_REAL_L5_mse.pth.best_eval.json \
  resnet_teaser/results/TEASER_T1_PSRAND_WN_L7_bc15.pth.best_eval.json \
  resnet_teaser/results/TEASER_T1_PSRAND_WN_L7_mse.pth.best_eval.json
```

---

## Reproducing the results: three levels

**(a) Tables and figures from the shipped numbers (minutes, laptop).** Run the scripts in `paper/` as above. `python scripts/run_eta_campaign.py --report` rebuilds `results/eta_results/master_table.csv` from the JSON files.

**(b) Re-computing the deterministic parts (minutes to hours, laptop).** This covers σ_MF, the exact Tier-1 quadratures, the complete-data bounds and the cumulant estimators (`--mode cheap`), for any subset of models. Write into a separate directory and compare with the shipped files:

```bash
python scripts/run_null_tests.py                        # the validation battery comes first
python scripts/run_eta_campaign.py --mode cheap --models T0_RED_REAL,T1_PSRAND_WN --out rerun_cheap
python scripts/tools/compare_with_shipped.py rerun_cheap
```

These numbers are deterministic. In the release check they agreed with the shipped files to a relative difference of 3e-12 (floating-point summation order) on a different platform and NumPy version from the ones that produced them.

**(c) The variational ladder (GPU hours).** `--mode variational2` trains the capacity ladder (linear, reweight, quadratic, cubic, conv rungs) with restarts. The production ladders ran as one-GPU-per-cell Slurm arrays on a GPU cluster (`scripts/hpc/eta_ladder_array.slurm`, with the cell lists next to it; the Slurm files are templates in which the account, partition and GPU request must be set). A full five-rung cell (200 epochs, four restarts, 10 800 FIT maps) took about 3.5 h on one GH200. These rungs are trained networks, so a re-run reproduces them within their quoted bootstrap errors, not bit for bit. `compare_with_shipped.py` reports a z-score per rung. Before re-running a stored cell, note that `--force` overwrites that cell's stored rungs. Use `--out` to write elsewhere.

`run_null_tests.py` must pass on the machine that produces any new η number. The battery is the paper's guard against normalization, whitening and split-discipline errors, and test N6 audits the shipped result files against their exact ceilings.

---

## The noise models (`noise_lib/`, `scripts/generate/`)

`noise_lib` contains the building blocks. The registry composes them into the paper's models: code names in `configs/REGISTRY.md`, physical descriptions in Sec. 5 of the paper. All power spectra use the unnormalized-FFT convention E|fft2(n)|² = P(k), with k in cycles per pixel, and every generator returns its per-image latent variables (spectral slope and amplitude, scan angle, event positions and amplitudes, source counts), which the exact Tier-1 quadratures use.

`scripts/generate/gen_t*.py` are stand-alone drivers. They write labeled ensembles (maps with injected sources, their latents, parameters and seeds) to HDF5 via `noise_lib.io_h5`, for training networks on the same noise models. They can be run as scripts or cell by cell in Spyder (`#%%` cells).

---

## The ResNet teaser runs (`resnet_teaser/`)

These are the scripts behind Sec. 6.1 and Fig. 3: the campaign-exact training and evaluation sets, the ResNet amplitude regressor (plain and bias-corrected MSE losses), the amplitude-grid evaluation against the η reference lines, and the figure. The four best-validation checkpoints behind Fig. 3 (5.8 MB each) and their evaluation summaries are included. With them, the paper's networks can be evaluated without a GPU. The full evaluation of one checkpoint (41 amplitudes × 4000 maps) takes of order an hour on a laptop CPU, whereas re-training takes about 4 GPU-hours per run. See [`resnet_teaser/README.md`](resnet_teaser/README.md).

---

## Notes

* **Code names vs paper names.** The code uses registry names such as `T1_PSRAND_WN`. The paper uses descriptive names such as "spectral-tilt mixture + white floor". The correspondence is in `configs/REGISTRY.md` and in the code-name table of Appendix E.
* **Comments in the code** occasionally refer to internal development notes (for example "memo §16", `README_PhaseB1`, `WN_Floor_Implementation_Plan.md`). Those working documents are not part of the release. The paper, especially Sec. 4 and Appendices B and C, is the public description of every method and test. Dated comments ("2026-09-03: …") record why a piece of code has its present form and are kept on purpose.
* **Precision.** Maps are generated in float32, except the confusion rows, which always use float64. `ETA_MAPS_FLOAT64=1` forces float64 everywhere. Sec. 10 and Fig. 8 of the paper explain why this matters for the floorless models.

---

## Citation

If you use this code, please cite the paper and, to pin the exact code version, the archived release on Zenodo. The version used for the paper is v1.0, [doi:10.5281/zenodo.22924970](https://doi.org/10.5281/zenodo.22924970). The concept DOI [10.5281/zenodo.22924969](https://doi.org/10.5281/zenodo.22924969) always resolves to the latest release.

```bibtex
@article{Basu2026BeyondTheBLUE1,
  author        = {Basu, Kaustuv},
  title         = {Beyond the {BLUE} {I}: the advantage ceiling -- how much can any estimator
                   beat the matched filter in mm/submm survey data?},
  year          = {2026},
  eprint        = {2609.10475},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.IM},
  doi           = {10.48550/arXiv.2609.10475}
}

@software{Basu2026BeyondTheBLUE1code,
  author    = {Basu, Kaustuv},
  title     = {Beyond the {BLUE} {I}: code for the advantage-ceiling (eta) framework,
               the Tier-0/1/2 noise models and the {ResNet} teaser runs},
  year      = {2026},
  version   = {v1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22924970},
  url       = {https://doi.org/10.5281/zenodo.22924970}
}
```

GitHub's "Cite this repository" button (from `CITATION.cff`) gives the software citation.

## License

MIT; see [`LICENSE`](LICENSE).

## Acknowledgments

The entire codebase was written with the help of Claude (Anthropic), as stated in the paper; the scientific content, the methods and their validation are the author's responsibility. Computing time for the variational ladders and the ResNet runs was provided by the Jülich Supercomputing Centre.

## Contact

Kaustuv Basu ([ORCID 0000-0001-5276-8730](https://orcid.org/0000-0001-5276-8730)), Argelander-Institut für Astronomie, Universität Bonn — kbasu@uni-bonn.de. Questions and bug reports are welcome as GitHub issues.
