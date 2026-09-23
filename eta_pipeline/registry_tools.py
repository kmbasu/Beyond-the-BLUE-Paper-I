"""
registry_tools.py -- read-only helpers around the noise-model registry
=======================================================================

The single source of truth for every noise model of the paper is the
``REGISTRY`` dictionary in :mod:`eta_pipeline.models`: one row per model,
carrying its generator, master seed, white-noise floor and per-method
configuration.  This module adds nothing to that definition; it only makes
it easy to *use* and to *inspect*:

``generate_ensemble(name, n)``
    The noise-only ensemble of ``n`` maps exactly as the eta campaign drew it
    (same generator, same master seed).

``reconstruct_map(name, index)``
    Map number ``index`` of a model's campaign ensemble, on its own.  Every
    generator draws image *i* from the *i*-th child of
    ``numpy.random.SeedSequence(master_seed)``, and a SeedSequence child
    depends only on (master seed, child index) -- not on how many children
    are spawned -- so image *i* is the same whether the ensemble had
    ``i + 1`` maps or 30 000.  This is what the paper means by "every number
    is reconstructable from (model, image index)".

``campaign_splits(n_ens)``
    The PSD / FIT / VAL / EVAL index sets used for an ensemble of ``n_ens``
    maps (fixed split seed 12345; fractions 0.25 / 0.40 / 0.10 / 0.25).

``registry_record()``
    A JSON-serialisable description of the whole registry (global
    conventions, seed streams, and one entry per model with its tier,
    generator, severity parameters, floor, seed and ladder configuration).
    ``scripts/export_registry.py`` writes it to ``configs/registry.json`` and
    renders ``configs/REGISTRY.md``; ``tests/test_registry.py`` checks that
    the shipped files still agree with the code.

Nothing here needs PyTorch.
"""

import numpy as np

from . import models as em
from .whitening import make_splits

# Seeds and seed offsets used by the campaign drivers (all documented in the
# modules that use them; collected here so that the exported record is complete).
SPLIT_SEED = 12345                       # run_eta_campaign.SPLIT_SEED
SPLIT_FRACTIONS = dict(psd=0.25, fit=0.40, val=0.10, eval=0.25)
DEFAULT_N_ENS = 6000                     # run_eta_campaign.DEFAULT_N_ENS
SEED_OFFSETS = {
    'white_floor': em.WN_SEED_OFFSET,    # model seed + 606000: independent stream of the *_WN floors
    'components': 991,                   # model seed + 991: event-diagnostic component ensembles
    'teaser_train': 103000,              # model seed + 103000: resnet_teaser TRAIN files
    'teaser_eval': 213000,               # model seed + 213000: resnet_teaser noise-only EVAL files
}
# Two auxiliary streams are keyed to the global base 777000 rather than to a model seed:
OTHER_SEED_STREAMS = {
    'high_precision_quadrature': '777000 + 555000 + (row index in run_b2b_cheap.py)',
    'closure_tests': '777000 + 900001 + int(1000 * sigma_alpha)   (run_b2b_cheap.py)',
}

# Descriptive names used in the paper (Appendix D maps them to the code names).
DISPLAY_NAME = {
    'T0_WHITE': 'white noise',
    'T0_RED': 'red (1/f^3) noise',
    'T0_RED_REAL': 'red + white noise',
    'T0_ANISO': 'fixed-direction anisotropy',
    'T1_PSRAND': 'spectral-tilt mixture',
    'T1_PSRAND_SLOPE': 'slope-only tilt mixture',
    'T1_PSRAND_AMP': 'amplitude-only mixture',
    'T1_RANDOR': 'random-orientation anisotropy',
    'T2_MEDIAN': 'row-median residual',
    'T2_CROSS_SYM': 'scan crossings (sym.)',
    'T2_CROSS_POS': 'scan crossings (pos.)',
    'T2_GLITCH': 'sub-threshold glitches',
    'T2_PCA': 'PCA leakage',
    'T2_CONFUSION': 'source confusion',
    'T2_CONFUSION_ATM': 'beam-sharing companion',
}


def display_name(name):
    """Paper name of a registry row (``*_WN`` rows add '+ white floor')."""
    if name.endswith('_WN'):
        return DISPLAY_NAME.get(name[:-3], name[:-3]) + ' + white floor'
    return DISPLAY_NAME.get(name, name)


# ----------------------------------------------------------------------------------
# Using the registry
# ----------------------------------------------------------------------------------

def generate_ensemble(name, n, seed=None):
    """Noise-only ensemble (maps, latents) of registry model ``name``.

    ``seed`` defaults to the model's campaign master seed, so the result is
    the campaign ensemble itself (its first ``n`` maps).
    """
    entry = em.REGISTRY[name]
    return entry['gen'](int(n), em.model_seed(name) if seed is None else int(seed))


def reconstruct_map(name, index):
    """Map ``index`` (0-based) of model ``name``'s campaign ensemble.

    Generates ``index + 1`` maps and returns the last one: generators are
    per-image (child ``i`` of the master SeedSequence), so this equals element
    ``index`` of any larger campaign ensemble exactly.
    """
    maps, _ = generate_ensemble(name, int(index) + 1)
    return maps[int(index)]


def campaign_splits(n_ens=DEFAULT_N_ENS):
    """PSD / FIT / VAL / EVAL index sets of an ``n_ens``-map campaign ensemble."""
    return make_splits(int(n_ens), f_psd=SPLIT_FRACTIONS['psd'],
                       f_fit=SPLIT_FRACTIONS['fit'], f_val=SPLIT_FRACTIONS['val'],
                       seed=SPLIT_SEED)


# ----------------------------------------------------------------------------------
# Describing the registry
# ----------------------------------------------------------------------------------

def _severity(name):
    """The generator parameters of one model, read from the live module constants.

    Mirrors the generator bodies in models.py; tests/test_registry.py guards
    the parts that can drift (seeds, floors, tiers, rung sets).
    """
    base = name[:-3] if name.endswith('_WN') else name
    iso = dict(zip(('slope', 'amplitude'), em.P_ISO_ARGS))
    aniso = dict(zip(('slope_scan', 'A_scan', 'slope_iso', 'A_iso'), em.P_ANISO_ARGS))
    ps = dict(zip(('slope_mean', 'slope_sigma', 'amp_mean', 'amp_sigma'), em.PSRAND_ARGS))
    beam = {'beam_fwhm_pix': em.BEAM_FWHM}
    if base == 'T0_WHITE':
        return {'white_sigma': 1.0}
    if base == 'T0_RED':
        return {'psd_isotropic': iso, **beam}
    if base == 'T0_RED_REAL':
        return {'psd_isotropic': iso, **beam, 'white_sigma_unsmoothed': em.WHITE_REAL}
    if base == 'T0_ANISO':
        return {'psd_two_component': aniso, **beam}
    if base == 'T1_PSRAND':
        return {'psrand': ps, **beam}
    if base == 'T1_PSRAND_SLOPE':
        return {'psrand': {**ps, 'amp_sigma': 0.0}, **beam}
    if base == 'T1_PSRAND_AMP':
        return {'psrand': {**ps, 'slope_sigma': 0.0}, **beam}
    if base == 'T1_RANDOR':
        return {'psd_two_component': aniso, 'scan_angle': 'uniform on [0, pi)', **beam}
    if base == 'T2_MEDIAN':
        return {'psd_two_component': aniso, **beam, 'processing': 'row-median removal (axis=1)'}
    if base in ('T2_CROSS_SYM', 'T2_CROSS_POS'):
        sign = 'symmetric' if base.endswith('SYM') else 'positive'
        return {'psd_isotropic': iso, **beam, 'scan_crossings': dict(em.CROSS_ARGS),
                'sign_convention': sign}
    if base == 'T2_GLITCH':
        return {'psd_two_component': aniso, **beam, 'glitches': dict(em.GLITCH_ARGS)}
    if base == 'T2_PCA':
        return {'psd_isotropic': iso, **beam, 'pca_leakage': dict(em.PCA_ARGS)}
    if base in ('T2_CONFUSION', 'T2_CONFUSION_ATM'):
        conf = {'counts': 'Schechter fit to 350 um counts (noise_lib.confusion)',
                's_min_mJy': em.CONFUSION_ARGS['s_lo'], 's_cut_mJy': em.CONFUSION_ARGS['s_cut'],
                'sigma_c_mJy_per_beam': float(em.CONFUSION_SIGMA_C),
                'pixel_arcsec': 4.0, 'beam_fwhm_arcsec': 20.0}
        if base == 'T2_CONFUSION_ATM':
            conf['beam_correlated_companion_rho'] = em.CONFUSION_RHO_ATM
        if name == 'T2_CONFUSION_WN':
            conf['f_N'] = em.CONFUSION_F_N
        return {'confusion': conf}
    return {}


def _jsonable(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def registry_record():
    """JSON-serialisable description of the registry and the campaign conventions."""
    models = {}
    for name, e in em.REGISTRY.items():
        gen = e['gen']
        models[name] = {
            'display_name': display_name(name),
            'tier': e['tier'],
            'master_seed': em.model_seed(name),
            'white_floor_sigma': float(e.get('floor') or 0.0),
            'generator': getattr(gen, '__name__', repr(gen)),
            'severity': _severity(name),
            'variational_rungs': list(e['rungs']) if e.get('rungs') else None,
            'exact_quadrature': e.get('quad') is not None,
            'complete_data_bound': e.get('bound_bg') is not None,
            'cumulant_estimator_reliable': bool(e.get('cumulant_ok')),
            'note': e.get('note', ''),
        }
    return _jsonable({
        'description': ('Noise-model registry of "Beyond the BLUE I" '
                        '(eta_pipeline/models.py), exported by scripts/export_registry.py. '
                        'Image i of model M is child i of numpy.random.SeedSequence('
                        'master_seed); see eta_pipeline.registry_tools.reconstruct_map.'),
        'conventions': {
            'map_size_pix': em.SIZE,
            'beam_fwhm_pix': em.BEAM_FWHM,
            'templates': {'extended': f'beta model, r_c = {em.TEMPLATE_RC} px, convolved with the beam',
                          'compact': 'the beam (point source)'},
            'power_spectrum_convention': 'unnormalized FFT: E|fft2(n)|^2 = P(k), k in cycles/pixel',
            'map_dtype': 'float32 (float64 for the confusion rows; ETA_MAPS_FLOAT64=1 forces float64)',
            'split_seed': SPLIT_SEED,
            'split_fractions': SPLIT_FRACTIONS,
            'default_n_ens': DEFAULT_N_ENS,
            'seed_offsets': SEED_OFFSETS,
            'other_seed_streams': OTHER_SEED_STREAMS,
            'white_floor_knee_factor': em.WN_KNEE_FACTOR,
        },
        'models': models,
    })


def registry_markdown(record=None):
    """Render ``registry_record()`` as a Markdown table plus parameter blocks."""
    import json
    r = record or registry_record()
    c = r['conventions']
    out = ['# Noise-model registry', '',
           '*Generated by `scripts/export_registry.py` from `eta_pipeline/models.py`. '
           'Do not edit by hand; `tests/test_registry.py` fails if this file and the code disagree.*', '',
           f"Maps are {c['map_size_pix']}x{c['map_size_pix']} pixels with a Gaussian beam of FWHM "
           f"{c['beam_fwhm_pix']} px. Spectra use the {c['power_spectrum_convention']} convention. "
           f"Ensembles are split into PSD / FIT / VAL / EVAL with fractions "
           f"{' / '.join(str(v) for v in c['split_fractions'].values())} and split seed {c['split_seed']} "
           f"(default ensemble size {c['default_n_ens']}).", '',
           'Image *i* of a model is drawn from child *i* of `numpy.random.SeedSequence(master_seed)`, '
           'so any single map can be regenerated with `eta_pipeline.registry_tools.reconstruct_map(model, i)`. '
           'White floors of the `*_WN` rows come from an independent stream at master seed + '
           f"{c['seed_offsets']['white_floor']}.", '',
           'Notes on the columns. *tier* is the code\'s label (the prefix of the code name); the paper '
           'shows that PCA leakage is a Gaussian scale mixture and discusses it with the Tier-1 models '
           '(Sec. 7), while the code name `T2_PCA` is kept so that result files stay addressable. '
           '*default rungs* is the ladder a bare `run_eta_campaign.py` run uses (`cnn` is the paper\'s '
           '"conv rung"); the production cells added the `reweight` rung and other settings per cell -- '
           'those are recorded in `scripts/hpc/cells*.txt` and, authoritatively, in the '
           '`variational2.config` block of every result file in `results/eta_results/`.', '',
           '| code name | paper name | tier | master seed | white floor sigma_w | default rungs | exact quadrature | complete-data bound |',
           '|---|---|---|---|---|---|---|---|']
    for name, m in r['models'].items():
        rungs = ', '.join(m['variational_rungs']) if m['variational_rungs'] else '(cheap only)'
        out.append(f"| `{name}` | {m['display_name']} | {m['tier']} | {m['master_seed']} | "
                   f"{m['white_floor_sigma']:g} | {rungs} | {'yes' if m['exact_quadrature'] else '-'} | "
                   f"{'yes' if m['complete_data_bound'] else '-'} |")
    out += ['', '## Seed streams', '', '| stream | seed |', '|---|---|']
    out += [f'| {k} | model seed + {v} |' for k, v in c['seed_offsets'].items()]
    out += [f'| {k} | {v} |' for k, v in c['other_seed_streams'].items()]
    out += ['', '## Severity parameters per model', '']
    for name, m in r['models'].items():
        out += [f"### `{name}`", '', f"Generator `{m['generator']}`. {m['note']}", '',
                '```json', json.dumps(m['severity'], indent=2), '```', '']
    return '\n'.join(out) + '\n'
