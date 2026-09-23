#!/usr/bin/env python
"""
compare_with_shipped.py -- check a re-run against the released result files
===========================================================================

After re-running part of the campaign into a separate directory, e.g.

    python scripts/run_eta_campaign.py --mode cheap --models T0_RED_REAL,T1_PSRAND_WN \
           --out rerun_cheap

this script compares every numeric entry of the cheap sections (sigma_MF,
moments, Tier-1 quadrature, complete-data bounds) of the re-run JSONs with
the shipped ones in results/eta_results/ and prints the relative
differences.  Deterministic sections should agree to floating-point
summation noise (~1e-12); variational rungs are stochastic (training) and
are therefore compared only as eta +- err, with a z-score.

Usage (from the repository root):
    python scripts/tools/compare_with_shipped.py rerun_cheap
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / 'results' / 'eta_results'
DETERMINISTIC = ('sigma_mf', 'moments', 'quadrature', 'bound_complete_data')


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main(rerun_dir):
    rerun_dir = Path(rerun_dir)
    worst = 0.0
    for f in sorted(rerun_dir.glob('T*__*.json')):
        ref = SHIPPED / f.name
        if not ref.exists():
            print(f'{f.name}: no shipped counterpart'); continue
        a, b = json.loads(f.read_text()), json.loads(ref.read_text())
        print(f'== {f.stem}')
        for sec in DETERMINISTIC:
            if sec not in a or sec not in b:
                continue
            va, vb = a[sec], b[sec]
            items = va.items() if isinstance(va, dict) else [('value', va)]
            for k, x in items:
                y = vb.get(k) if isinstance(vb, dict) else vb
                if _num(x) and _num(y) and k != 'runtime_s':
                    rel = abs(x - y) / max(abs(y), 1e-300)
                    worst = max(worst, rel)
                    print(f'   {sec}.{k:<18} shipped {y:>14.8g}  re-run {x:>14.8g}  rel.diff {rel:.1e}')
        ra = (a.get('variational2') or {}).get('rungs') or {}
        rb = (b.get('variational2') or {}).get('rungs') or {}
        for rung in sorted(set(ra) & set(rb)):
            ea, sa = ra[rung].get('eta'), ra[rung].get('err')
            eb, sb = rb[rung].get('eta'), rb[rung].get('err')
            if all(_num(v) for v in (ea, sa, eb, sb)):
                z = (ea - eb) / max((sa ** 2 + sb ** 2) ** 0.5, 1e-300)
                print(f'   rung {rung:<10} shipped {eb:.4f} +- {sb:.4f}   re-run {ea:.4f} +- {sa:.4f}   z = {z:+.2f}')
    print(f'\nlargest relative difference in the deterministic sections: {worst:.1e}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
