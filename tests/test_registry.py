"""
Release-integrity tests (fast; numpy/scipy only).

Run from the repository root:   python -m pytest tests -q

These do NOT replace the paper's null battery (scripts/run_null_tests.py),
which validates the eta pipeline itself and takes much longer.  They check
that the repository is internally consistent:

* the exported registry files (configs/) agree with eta_pipeline/models.py;
* every master seed is explicit and equal to the historical positional rule;
* a single map regenerated with reconstruct_map() equals the same map inside a
  larger ensemble (the "reconstructable from (model, image index)" contract);
* every shipped result file parses and refers to a registry model.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import torch  # noqa: F401
except Exception:                       # import eta_pipeline submodules without torch
    _pkg = types.ModuleType('eta_pipeline')
    _pkg.__path__ = [str(ROOT / 'eta_pipeline')]
    sys.modules['eta_pipeline'] = _pkg

from eta_pipeline import models as em               # noqa: E402
from eta_pipeline import registry_tools as rt       # noqa: E402


def test_exported_registry_is_current():
    rec = rt.registry_record()
    shipped = json.loads((ROOT / 'configs' / 'registry.json').read_text())
    assert shipped == json.loads(json.dumps(rec)), \
        'configs/registry.json is out of date: run python scripts/export_registry.py'
    assert (ROOT / 'configs' / 'REGISTRY.md').read_text() == rt.registry_markdown(rec)


def test_seeds_explicit_and_positional():
    for i, (name, entry) in enumerate(em.REGISTRY.items()):
        assert entry.get('seed') is not None, name
        assert entry['seed'] == em.SEED_BASE + 17 * i, name


@pytest.mark.parametrize('name', ['T0_RED_REAL', 'T1_PSRAND_WN', 'T1_RANDOR',
                                  'T2_MEDIAN_WN', 'T2_PCA_WN', 'T2_CONFUSION_WN'])
def test_reconstruct_single_map(name):
    maps, _ = rt.generate_ensemble(name, 6)
    for i in (0, 4):
        np.testing.assert_array_equal(rt.reconstruct_map(name, i), maps[i])


def test_splits_disjoint_and_complete():
    sp = rt.campaign_splits(6000)
    allidx = np.concatenate([sp[k] for k in ('psd', 'fit', 'val', 'eval')])
    assert len(allidx) == 6000 and len(np.unique(allidx)) == 6000
    assert [len(sp[k]) for k in ('psd', 'fit', 'val', 'eval')] == [1500, 2400, 600, 1500]


def test_result_files_parse():
    files = sorted((ROOT / 'results' / 'eta_results').glob('T*__*.json'))
    assert len(files) == 48
    for f in files:
        d = json.loads(f.read_text())
        assert d['model'] in em.REGISTRY, f.name
        assert d['master_seed'] == em.model_seed(d['model']), f.name
