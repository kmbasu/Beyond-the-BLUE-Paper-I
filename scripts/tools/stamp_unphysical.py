#!/usr/bin/env python
"""
stamp_unphysical.py — mark rounding-channel rungs as non-physical and attach the float64 A/B
==========================================================================================

Why (v4 §5, §12 item 4).  The floorless T2_PCA / T2_MEDIAN variational sections in
eta_results/ hold the fair-protocol GPU-cluster re-run on FLOAT32 ensembles.  Those numbers
are correct measurements of that configuration, but the float64 A/B
(eta_results_hpc_f64/) showed that every value above the `reweight` rung there is
the float32 mantissa-rounding side channel, not the noise.  Rather than delete
them (they are the evidence), this tool

  1. stamps each listed rung with  physical: False  and a reason, so that
     run_eta_campaign.report() excludes it from eta_var2_best;
  2. copies the float64 section into the same JSON under
     'variational2_float64' (rungs + config + provenance), so the A/B lives
     with the record it corrects.

Idempotent; backs up eta_results/ first.   Usage: python scripts/tools/stamp_unphysical.py
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / 'results'
LOCAL, F64 = HERE / 'eta_results', HERE / 'eta_results_hpc_f64'
REASON = ('float32 rounding side channel (v4 §5): float64 A/B on the same seeds gives 0/6 '
          'restarts off the anchor; see variational2_float64')

# (file, rungs to stamp)
STAMP = {
    'T2_PCA__extended.json':      ['cnn', 'cubic', 'quadratic'],
    'T2_MEDIAN__extended.json':   ['cnn', 'cubic', 'quadratic'],
    'T2_CROSS_SYM__extended.json': ['cnn'],          # the retired 1.142 "violation" (v4 §4.2)
}

stamp_time = datetime.now().isoformat(timespec='seconds')
bak = HERE / f"eta_results_backup_{datetime.now().strftime('%Y-%m-%d_%H%M')}"
shutil.copytree(LOCAL, bak)
print(f'backed up eta_results -> {bak.name}')
for fname, rungs in STAMP.items():
    p = LOCAL / fname
    d = json.loads(p.read_text())
    v2 = d.get('variational2', {}).get('rungs', {})
    for r in rungs:
        if r in v2:
            v2[r]['physical'] = False
            v2[r]['reason'] = REASON
            v2[r]['stamped'] = stamp_time
            print(f'  {fname}: {r} eta={v2[r]["eta"]:.3f} -> physical: False')
    f64 = F64 / fname
    if f64.exists():
        r64 = json.loads(f64.read_text())
        d['variational2_float64'] = {
            'rungs': r64['variational2']['rungs'], 'config': r64['variational2']['config'],
            'host': r64.get('host'), 'updated': r64.get('updated'),
            'note': 'ETA_MAPS_FLOAT64=1 A/B run (maps stored in float64 before whitening); '
                    'same seeds/splits as variational2; NOT merged into rungs by design'}
        print(f'  {fname}: attached variational2_float64 '
              f'({", ".join(f"{k}={v[chr(101)+chr(116)+chr(97)]:.3f}" for k, v in r64["variational2"]["rungs"].items())})')
    d['updated'] = stamp_time
    p.write_text(json.dumps(d, indent=1, default=float))
print('done — now run:  python scripts/run_eta_campaign.py --report')
