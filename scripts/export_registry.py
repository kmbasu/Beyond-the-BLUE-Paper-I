#!/usr/bin/env python
"""
export_registry.py -- write the noise-model registry to human-readable files
============================================================================

The registry of the paper's noise models is Python code
(``eta_pipeline/models.py``, dictionary ``REGISTRY``), because each row
carries executable objects (its generator, its quadrature builder).  This
script exports everything a reader needs to *see* -- conventions, master
seeds, seed streams, white-noise floors, severity parameters, ladder
configuration -- to two files:

    configs/registry.json   machine-readable record
    configs/REGISTRY.md     the same as tables, for reading on GitHub

Both files are committed to the repository.  ``tests/test_registry.py``
regenerates the record and fails if it no longer matches the shipped JSON,
so the files cannot drift silently from the code.

Usage (from the repository root; numpy/scipy only, no torch):
    python scripts/export_registry.py            # write both files
    python scripts/export_registry.py --check    # exit 1 if they are out of date
"""
import argparse
import json
import sys
import types
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
try:
    import torch  # noqa: F401
except Exception:                       # import eta_pipeline submodules without torch
    _pkg = types.ModuleType('eta_pipeline')
    _pkg.__path__ = [str(_root / 'eta_pipeline')]
    sys.modules['eta_pipeline'] = _pkg

from eta_pipeline import registry_tools as rt   # noqa: E402

CONFIG_DIR = _root / 'configs'


def main(argv=None):
    ap = argparse.ArgumentParser(description='Export the noise-model registry.')
    ap.add_argument('--check', action='store_true',
                    help='compare with the shipped files instead of writing them')
    args = ap.parse_args(argv)

    rec = rt.registry_record()
    js = json.dumps(rec, indent=2, ensure_ascii=False) + '\n'
    md = rt.registry_markdown(rec)
    fj, fm = CONFIG_DIR / 'registry.json', CONFIG_DIR / 'REGISTRY.md'
    if args.check:
        ok = fj.exists() and fj.read_text() == js and fm.exists() and fm.read_text() == md
        print('registry files are up to date' if ok else 'registry files are OUT OF DATE -- '
              'run python scripts/export_registry.py')
        return 0 if ok else 1
    CONFIG_DIR.mkdir(exist_ok=True)
    fj.write_text(js)
    fm.write_text(md)
    print(f'wrote {fj.relative_to(_root)} and {fm.relative_to(_root)} '
          f'({len(rec["models"])} models)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
