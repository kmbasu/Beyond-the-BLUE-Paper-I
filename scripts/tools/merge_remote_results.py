#!/usr/bin/env python
"""
merge_remote_results.py — fold remote (GPU-cluster) variational2 rungs into eta_results/
=======================================================================================

Purpose
-------
The cluster array job (scripts/hpc/eta_ladder_array.slurm) writes one JSON per
(model, template) cell into a *separate* directory (eta_results_hpc/),
containing only the 'variational2' section (plus the model/template header
fields).  This tool merges those rungs into the local documentation-of-record
JSONs in eta_results/ at RUNG level, while keeping every value it displaces
auditable:

  * the local 'variational2' section, as it stood before the merge, is
    appended to a list  res['variational2_superseded']  with a timestamp and
    the reason 'merge_remote', so the Pass-B (patience-15) numbers remain
    inspectable and the memo erratum can quote them;
  * remote rungs overwrite local rungs of the same name; local rungs the
    remote run did not compute are kept;
  * res['variational2']['config'] becomes the remote config (epochs,
    patience, min_epochs, device, host …) and gains 'merged_from' and
    'merged_at'.

Nothing in the cheap layer (sigma_mf, quadrature, bounds, cumulants, …) is
touched, because the remote run never computed it.

Usage (from the repository root)
---------------------------------------
    python scripts/tools/merge_remote_results.py results/eta_results_hpc            # dry run: prints what would change
    python scripts/tools/merge_remote_results.py results/eta_results_hpc --apply    # writes, after backing up eta_results/
    python scripts/run_eta_campaign.py --report                              # rebuild master_table.csv

A backup copy of eta_results/ is made as eta_results_backup_<timestamp>/ before
anything is written.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / 'results'   # remote dirs are resolved relative to cwd
LOCAL = HERE / 'eta_results'


def summarize(rungs):
    """One line per rung: eta ± err, restarts, and how many were patience-killed."""
    out = []
    for name, v in rungs.items():
        tele = v.get('telemetry', {})
        rs = tele.get('restarts', [])
        killed = sum(1 for t in rs if t.get('stopped_by_patience', t.get('epochs_run', 0) < 200))
        moved = sum(1 for t in rs if t.get('moved_off_init'))
        best_ep = [t.get('best_epoch') for t in rs]
        out.append(f"      {name:9s} eta={v['eta']:.3f}±{v['err']:.3f}"
                   + (f"  moved {moved}/{len(rs)}  best_epoch {best_ep}" if rs else '  (anchor)'))
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('remote', help='directory holding the remote T*__*.json files')
    ap.add_argument('--local', default=str(LOCAL))
    ap.add_argument('--apply', action='store_true', help='write the merge (default: dry run)')
    args = ap.parse_args(argv)
    remote, local = Path(args.remote), Path(args.local)
    files = sorted(remote.glob('T*__*.json'))
    if not files:
        sys.exit(f'no T*__*.json in {remote}')

    if args.apply:
        stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        bak = local.parent / f'{local.name}_backup_{stamp}'
        shutil.copytree(local, bak)
        print(f'backed up {local} -> {bak}')

    for f in files:
        rem = json.loads(f.read_text())
        v2 = rem.get('variational2')
        if not v2 or not v2.get('rungs'):
            print(f'{f.name}: no variational2 rungs — skipped'); continue
        lp = local / f.name
        loc = json.loads(lp.read_text()) if lp.exists() else {}
        old = loc.get('variational2')
        print(f'\n=== {f.name}  (remote host {rem.get("host")}, device {v2.get("config", {}).get("device")})')
        if old:
            print('   local rungs before merge:'); print(summarize(old.get('rungs', {})))
        print('   remote rungs:'); print(summarize(v2['rungs']))
        if not args.apply:
            continue
        merged = dict(old.get('rungs', {})) if old else {}
        merged.update(v2['rungs'])
        cfg = dict(v2.get('config', {}))
        cfg.update({'merged_from': str(f), 'merged_at': datetime.now().isoformat(timespec='seconds'),
                    'remote_host': rem.get('host'), 'remote_updated': rem.get('updated')})
        if old:
            loc.setdefault('variational2_superseded', []).append(
                {'superseded_at': cfg['merged_at'], 'reason': 'merge_remote',
                 'replaced_rungs': sorted(v2['rungs']), 'section': old})
        for k in ('model', 'template', 'tier', 'n_ens', 'master_seed', 'floor', 'split_seed', 'note'):
            loc.setdefault(k, rem.get(k))
        loc['variational2'] = {'rungs': merged, 'config': cfg}
        loc['updated'] = cfg['merged_at']
        lp.write_text(json.dumps(loc, indent=1, default=float))
        print(f'   -> merged into {lp}')
    if not args.apply:
        print('\n(dry run — add --apply to write)')
    else:
        print('\nnow run:  python scripts/run_eta_campaign.py --report')


if __name__ == '__main__':
    main()
