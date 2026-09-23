# Result files

These are the final numbers of Paper I, exactly as the figure and table scripts read them. Two kinds of metadata in them were rewritten for the release: host names became generic labels ("laptop (Apple M3)", "HPC node (NVIDIA GH200)", "cloud container (CPU)"), and absolute paths became `<TEASER_DATA>/…`. In addition, the `master_seed` field of the four `T1_PSRAND_SLOPE` and `T1_PSRAND_AMP` files was corrected from 127 and 144 to the registry seeds 777085 and 777102. The old values were a stale label; regenerating both ensembles showed that the stored numbers come from the registry seeds. No number was changed.

## `eta_results/`

**One JSON per cell**, named `<MODEL>__<template>.json`, for 24 registry models × {`extended`, `compact`}. The sections are:

| key | content |
|---|---|
| `model`, `template`, `tier`, `n_ens`, `master_seed`, `split_seed`, `floor`, `note` | provenance: the cell can be regenerated from these (see `configs/REGISTRY.md`) |
| `sigma_mf` | predicted (from the PSD split) and empirical (EVAL split) matched-filter error |
| `moments` | pixel mean, standard deviation, skewness, excess kurtosis of the EVAL maps |
| `quadrature` | exact η of the Tier-1 mixtures (Method D) with bootstrap error |
| `bound_complete_data`, `bound_complete_data_dc_excluded` | complete-data upper bound (the DC-excluded value is the quotable one for the confusion rows) |
| `cumulants` | perturbative η from the contracted third and fourth cumulants, with a `reliable` flag |
| `bound_stability`, `bound_note` | stability of the bound against the mode mask |
| `diagnostics` | event diagnostics of the Tier-2 models (structured power fraction, per-event significance) |
| `variational2` | the production capacity ladder: per rung (`linear`, `reweight`, `quadratic`, `cubic`, `cnn` = the paper's conv rung) η ± err, per-restart telemetry, and the full run configuration in `config` |
| `variational2_superseded`, `variational` | earlier passes (round 2 before the final re-runs; round 1), kept for the record and not quoted in the paper |
| `variational2_float64` | the float64 A/B rungs attached to the two floorless cells where the rounding channel was tested |

Rungs carrying `physical: false` (seven rungs in the floorless cells, stamped by `scripts/tools/stamp_unphysical.py`) are float32 rounding-channel artifacts of the floorless models (paper Sec. 10). They are kept in the files but are not physical lower bounds.

**Auxiliary files**:

| file | written by | content |
|---|---|---|
| `master_table.csv` | `run_eta_campaign.py --report` | one row per cell (the basis of the App. E tables) |
| `master_table_v2.csv`, `b2b_*.json` | `run_b2b_cheap.py` | channel decomposition, closure tests, high-precision quadratures, cross-checks |
| `r3_channels_wn.json`, `r3_inferability.json`, `r3_matched_seed.json` | `run_r3_cheap.py` | Method-E envelopes (Figs. 4, 5), latent inferability, matched-seed comparisons |
| `r3_rounding_band.json` | `tools/diag_rounding_band.py` | the float32 rounding band |
| `pca_mixture_quadrature.json` | `run_pca_mixture_quad.py` | exact η of PCA leakage as a Gaussian scale mixture |
| `confusion_bound_dc_check.json` | `tools/check_bound_dc.py` | DC-consistent complete-data bounds of the confusion rows |

## `eta_results_hpc_f64/`

`T2_PCA__extended.json` is the float64 A/B re-run of the floorless PCA cell. It is the evidence for the rounding side channel (Fig. 8, inset of panel c) and for the `physical: false` stamps.

## `tests_and_results/`

| file | content |
|---|---|
| `null_tests_final_20260907.txt` | the final null-battery log quoted in the paper (nine tests, 100 of 100 checks passed) |
| `confusion_exact_*.json` | exact η of the beam-sharing reference `T2_CONFUSION_ATM` (`tools/eta_exact_confusion.py`); the last one is read by null test N7 |
| `confusion_method_c_*.json` | the Method-C analysis of the template ordering in confusion (Sec. 9) |
| `confusion_sweeps_*.json` | analytic severity sweeps of the confusion model |
| `confusion_validation_*.json` | the acceptance battery of the confusion model |
