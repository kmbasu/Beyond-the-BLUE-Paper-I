#!/usr/bin/env python
"""
run_eta_campaign.py — the Phase-B2 η campaign orchestrator
==========================================================

Runs the η-ceiling computation for the registered noise models
(eta_pipeline/models.py) at the paper conventions (128², beam FWHM 5 px)
for both templates (extended rc=7 β⊛beam and compact beam), and stores
resumable per-(model, template) JSON results in ``eta_results/``.

The campaign is split by METHOD COST, matching the agreed hybrid workflow:

  --mode cheap        σ_MF (predicted + empirical) and design ratio r,
                      pooled moments, exact Tier-1 latent quadratures,
                      complete-data bounds, cumulant cross-checks (flagged
                      by validity), event diagnostics (f_s, ξ_rms).
                      Minutes per model — runs anywhere.
  --mode variational  the production Method-A capacity ladder (PyTorch,
                      restarts, B1-validated protocol).  Run this on the
                      M3 (mps) — hours for the full set; resumable.
  --mode all          both.

Ensembles are generated IN MEMORY from fixed per-model seeds: a cheap run
in the cloud container and a variational run on the M3 use IDENTICAL maps
and splits, so their sections merge into the same JSON consistently.  No
HDF5 datasets are needed for η (those are for the later CNN campaign).

Usage (from the repository root):
  python scripts/run_eta_campaign.py --mode cheap                       # all models
  python scripts/run_eta_campaign.py --mode variational                 # M3 campaign
  python scripts/run_eta_campaign.py --mode variational --models T2_PCA,T2_GLITCH
  python scripts/run_eta_campaign.py --report                           # master table

Already-computed sections are skipped unless --force; interrupting and
re-running resumes where it left off.  The §15 null-test battery
(run_null_tests.py) must be passing on the executing machine first.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
for _p in (_here, *_here.parents):
    if (_p / 'eta_pipeline').is_dir():
        sys.path.insert(0, str(_p))
        break

import noise_lib as nl
import eta_pipeline as ep
from eta_pipeline import models as em

SPLIT_SEED = 12345
SEED_BASE = 777000          # per-model ensemble seed = SEED_BASE + 17 * index
DEFAULT_N_ENS = 6000
N_COMPONENTS = 300          # ensemble size for the f_s / xi diagnostics


# ----------------------------------------------------------------------------------
# Results I/O (resumable, merging)
# ----------------------------------------------------------------------------------

def result_path(out_dir, model, template):
    return Path(out_dir) / f'{model}__{template}.json'


def load_result(out_dir, model, template):
    p = result_path(out_dir, model, template)
    return json.loads(p.read_text()) if p.exists() else {}


def save_result_merged(out_dir, model, template, res):
    """Save, merging variational2 rungs with whatever is ALREADY on disk.

    Why (2026-09-05): heavy cells are now split across array tasks (e.g. the
    cubic rung in its own task) that write the same <model>__<template>.json.
    A plain save would overwrite the other task's finished rungs with this
    task's in-memory copy, which was loaded before the other task saved.
    Re-reading at save time and merging at rung level makes concurrent tasks
    on one cell safe up to a race window of milliseconds.
    """
    p = result_path(out_dir, model, template)
    if p.exists() and 'variational2' in res:
        try:
            cur = json.loads(p.read_text())
        except ValueError:
            cur = {}
        cur_r = cur.get('variational2', {}).get('rungs', {})
        merged = dict(cur_r)
        # write back ONLY the rungs THIS process computed (config['rungs_this_run']);
        # rungs merely loaded at start-up are stale copies and must not overwrite
        # what a concurrent task saved in the meantime (bug found 2026-09-05: the
        # cubic-only task overwrote the (quadratic, cnn) task's fresh values)
        mine = res['variational2'].get('config', {}).get('rungs_this_run')
        src = res['variational2']['rungs']
        merged.update({k: src[k] for k in (mine if mine is not None else src) if k in src})
        res['variational2']['rungs'] = merged
        res['variational2']['config']['rungs_completed'] = sorted(merged)
        for k in ('variational2_superseded', 'variational2_float64'):
            if k in cur and k not in res:
                res[k] = cur[k]
    save_result(out_dir, model, template, res)


def save_result(out_dir, model, template, res):
    Path(out_dir).mkdir(exist_ok=True)
    res['updated'] = datetime.now().isoformat(timespec='seconds')
    res['host'] = platform.node() or 'unknown'
    result_path(out_dir, model, template).write_text(
        json.dumps(res, indent=1, default=float))


# ----------------------------------------------------------------------------------
# Shared per-model preparation
# ----------------------------------------------------------------------------------

def model_seed(name):
    """Delegate to the registry's explicit seeds (models.model_seed)."""
    return em.model_seed(name)


def prepare_model(name, n_ens, n_gen=None):
    """Generate the ensemble, splits, and the SPLIT_PSD spectrum.

    n_gen > n_ens (variational2) appends extra maps AFTER the frozen
    campaign ensemble: SeedSequence children are index-stable, so the
    first n_ens maps — and hence the PSD/FIT/VAL/EVAL splits, the
    whitening, and every cheap result — are bit-identical to the B2a run.
    """
    entry = em.REGISTRY[name]
    t0 = time.time()
    maps, latents = entry['gen'](n_gen or n_ens, model_seed(name))
    sp = ep.make_splits(n_ens, seed=SPLIT_SEED)
    P_hat = ep.estimate_psd2d(maps[sp['psd']])
    print(f'  [{name}] generated {len(maps)} maps + PSD in {time.time()-t0:.0f} s')
    return entry, maps, latents, sp, P_hat


# ----------------------------------------------------------------------------------
# Cheap methods
# ----------------------------------------------------------------------------------

def run_cheap(name, template_name, tau, entry, maps, latents, sp, P_hat,
              res, force=False):
    if 'sigma_mf' in res and not force:
        print(f'  [{name}|{template_name}] cheap section exists — skipped')
        return res

    t_hat, sigma_mf = ep.whitened_template(tau, P_hat)
    A_hat = ep.whitening.mf_amplitudes(maps[sp['eval']], tau, P_hat)
    res['sigma_mf'] = {'pred': sigma_mf, 'emp': float(A_hat.std()),
                       'r_L5': float(np.sqrt(12) * A_hat.std() / 5.0)}
    res['moments'] = nl.diagnostics.moment_summary(maps[sp['eval']])

    # exact Tier-1 quadrature
    if entry.get('quad') is not None:
        t0 = time.time()
        runner = entry['quad']()
        qres = runner(maps[sp['eval']], tau)
        qres['n_eval'] = int(len(sp['eval']))
        qres['runtime_s'] = round(time.time() - t0, 1)
        res['quadrature'] = qres
        eta_q, err_q = qres['eta'], qres['err']
        # closure prediction for the amplitude-only variant (erratum 1)
        if name == 'T1_PSRAND_AMP':
            from eta_pipeline.quadrature import truncnorm_grid
            g, w = truncnorm_grid(em.PSRAND_ARGS[2], em.PSRAND_ARGS[3], 1e-6)
            res['quadrature']['analytic_E2Einv2'] = float(
                np.sum(w * g**2) * np.sum(w / g**2))
        print(f"  [{name}|{template_name}] quadrature eta = {eta_q:.4f} ± "
              f"{err_q:.4f}  ({res['quadrature']['runtime_s']} s)")

    # complete-data (perfect-removal) envelope
    if entry.get('bound_bg') is not None:
        res['bound_complete_data'] = ep.complete_data_bound(
            tau, entry['bound_bg'](), P_hat)
        # N8: is that number a property of the noise or of the mode cutoff?
        res['bound_stability'] = ep.bound_stability(
            tau, entry['bound_bg'](), P_hat, em.beam2())
        if not res['bound_stability']['stable']:
            print(f"  [{name}|{template_name}] ** BOUND UNSTABLE vs k-mask "
                  f"(drift {res['bound_stability']['max_rel_drift']:.1%}) — "
                  f"unregularized band, do not quote **")

    # cumulant cross-check (validity-flagged; memo §17 caveats)
    xe = ep.whiten_maps(maps[sp['eval']], P_hat)
    half = len(xe) // 2
    e_p, e_e, (k3, k3e), (k4, k4e) = ep.cumulants.eta_perturbative(
        xe[:half], xe[half:], t_hat, n_boot=200)
    res['cumulants'] = {'eta': e_p, 'err': e_e, 'k3': [k3, k3e],
                        'k4': [k4, k4e],
                        'reliable': bool(entry['cumulant_ok']),
                        'note': ('' if entry['cumulant_ok'] else
                                 'heavy-tailed marks: diagnostic only')}

    # event diagnostics: severity coordinates (f_s, xi_rms) of memo §11
    if entry.get('components') is not None and 'diagnostics' not in res:
        bg, st, counts = entry['components'](N_COMPONENTS,
                                             model_seed(name) + 991)
        P_s = ep.estimate_psd2d(st)
        P_t = P_hat
        tau_F2 = np.abs(np.fft.fft2(tau))**2
        fw = tau_F2 / P_t
        fw /= fw.sum()
        f_s = float((fw * np.minimum(P_s / P_t, 1.0)).sum())
        # per-event MF significance: q_i = sum_k |s_i|^2 / P_tot ~ sum_e xi_e^2
        # (convention check: white noise sigma, delta event a -> q = a^2/sigma^2)
        S = np.fft.fft2(st, axes=(-2, -1))
        q = (np.abs(S)**2 / P_t).sum(axis=(-2, -1))
        mean_count = float(np.mean(counts))
        xi_rms = float(np.sqrt(q.mean() / max(mean_count, 1e-9)))
        res['diagnostics'] = {'f_s': f_s, 'xi_rms': xi_rms,
                              'mean_count': mean_count}
        print(f'  [{name}|{template_name}] f_s = {f_s:.3f}  xi_rms = {xi_rms:.2f}')

    return res


# ----------------------------------------------------------------------------------
# Variational method (Method A production ladder)
# ----------------------------------------------------------------------------------

def run_variational(name, template_name, tau, entry, maps, sp, P_hat, res,
                    epochs, restarts, batch, force=False):
    if entry.get('rungs') is None:
        return res
    if 'variational' in res and not force:
        print(f'  [{name}|{template_name}] variational section exists — skipped')
        return res

    from eta_pipeline import variational as ev
    t0 = time.time()
    t_hat, _ = ep.whitened_template(tau, P_hat)
    W = {k: ep.whiten_maps(maps[sp[k]], P_hat) for k in ('fit', 'val', 'eval')}
    ladder = ev.run_ladder(W['fit'], W['val'], W['eval'], t_hat,
                           rungs=entry['rungs'], epochs=epochs,
                           patience=max(8, epochs // 5), batch=batch,
                           lr=1e-2, reg_to_init=1e-3, restarts=restarts,
                           verbose=True)
    res['variational'] = {
        'rungs': {k: {'eta': v[0], 'err': v[1],
                      'restart_etas': (v[2] if len(v) > 2 else [v[0]])}
                  for k, v in ladder.items()},
        'config': {'epochs': epochs, 'restarts': restarts, 'batch': batch,
                   'device': ev.default_device(),
                   'runtime_s': round(time.time() - t0, 1)},
    }
    best = max(ladder, key=lambda k: ladder[k][0])
    print(f"  [{name}|{template_name}] variational best rung '{best}': "
          f"eta = {ladder[best][0]:.4f} ± {ladder[best][1]:.4f} "
          f"({res['variational']['config']['runtime_s']} s)")
    return res


# ----------------------------------------------------------------------------------
# Variational method, round 2 (parameter-efficient ladder; see variational2.py)
# ----------------------------------------------------------------------------------

def run_variational2(name, template_name, tau, entry, maps, sp, P_hat, res,
                     n_ens, epochs, restarts, batch, force=False, rungs=None,
                     patience=None, min_epochs=None, lr=3e-3, kmax=3, pool='fourier',
                     save_cb=None):
    """Round-2 ladder on the extended ensemble.

    `maps` must hold n_ens + n_extra images: the first n_ens are the
    frozen B2a campaign ensemble (same SeedSequence children — the PSD
    and EVAL splits are bit-identical to the cheap runs), the remainder
    is round-2 FIT/VAL data (90/10).
    """
    if entry.get('rungs') is None:
        return res
    from eta_pipeline import variational2 as ev2
    want = rungs or (ev2.RUNGS2_T2 if entry['tier'] == 'T2' else ev2.RUNGS2_T1)
    have = set(res.get('variational2', {}).get('rungs', {}))
    if set(want) <= have and not force:
        print(f'  [{name}|{template_name}] variational2 rungs exist — skipped')
        return res
    rungs = [r for r in want if r not in have or force]
    if 'linear' in want and 'linear' not in rungs:
        rungs = ['linear'] + rungs            # anchor is cheap; keep it fresh
    from eta_pipeline.variational import default_device
    t0 = time.time()
    t_hat, _ = ep.whitened_template(tau, P_hat)
    extra = maps[n_ens:]
    n_val = max(1, len(extra) // 10)
    x_fit = ep.whiten_maps(extra[:-n_val], P_hat)
    x_val = ep.whiten_maps(extra[-n_val:], P_hat)
    x_eval = ep.whiten_maps(maps[sp['eval']], P_hat)
    # Early-stopping protocol (2026-09-02): patience scales with --epochs and
    # a warm-up protects the slow-starting CNN/cubic rungs.  Until this
    # revision run_ladder2's fixed default patience=15 applied regardless of
    # --epochs, so no Pass-B CNN restart ran past epoch 19.
    if patience is None:
        patience = max(15, epochs // 3)
    if min_epochs is None:
        min_epochs = min(30, epochs)
    # One rung at a time, CHECKPOINTING after each (2026-09-05): a cell killed at
    # the wallclock limit used to lose every rung it had finished, because the
    # JSON was written only after the whole ladder.  Signal-bearing cubic/CNN
    # rungs now run to 200 epochs (confusion: 8.7-8.9 h per cell against a 12 h
    # limit), so each finished rung is merged and saved immediately via save_cb.
    ladder = {}
    for rung in rungs:
        part = ev2.run_ladder2(x_fit, x_val, x_eval, t_hat, rungs=(rung,),
                               epochs=epochs, batch=batch, restarts=restarts,
                               lr=lr, patience=patience, min_epochs=min_epochs,
                               kmax=kmax, pool=pool, verbose=True)
        ladder.update(part)
        prev = res.get('variational2', {}).get('rungs', {})
        prev.update(part)                     # rung-level merge (resumable)
        res['variational2'] = {
            'rungs': prev,
            'config': {'epochs': epochs, 'restarts': restarts, 'batch': batch,
                       'patience': int(patience), 'min_epochs': int(min_epochs),
                       'lr': float(lr), 'kmax': int(kmax), 'pool': pool,
                       'n_fit': int(len(x_fit)), 'n_val': int(len(x_val)),
                       'n_eval': int(len(x_eval)),
                       'device': default_device(),
                       'runtime_s': round(time.time() - t0, 1),
                       'rungs_completed': sorted(prev),
                       'rungs_this_run': sorted(ladder)},
        }
        if save_cb is not None:
            save_cb(res)
            print(f"  [{name}|{template_name}] checkpoint: rung '{rung}' saved "
                  f"({res['variational2']['config']['runtime_s']} s elapsed)")
    best = max(ladder, key=lambda k: ladder[k]['eta'])
    print(f"  [{name}|{template_name}] variational2 best rung '{best}': "
          f"eta = {ladder[best]['eta']:.4f} ± {ladder[best]['err']:.4f} "
          f"({res['variational2']['config']['runtime_s']} s)")
    return res


# ----------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------

def report(out_dir):
    rows = []
    # campaign results only: the b2b_* analysis JSONs in the same directory
    # have no model/template fields (their presence crashed the printer after
    # the first round-2 pass — the campaign JSONs were already saved)
    for p in sorted(Path(out_dir).glob('T*__*.json')):
        try:
            r = json.loads(p.read_text())
        except ValueError:                 # concurrent array task mid-write
            print(f'  report: skipping unreadable {p.name}')
            continue
        var = r.get('variational', {}).get('rungs', {})
        best = max(var.values(), key=lambda v: v['eta']) if var else None
        var2 = r.get('variational2', {}).get('rungs', {})
        # rungs stamped physical: False (tools/stamp_unphysical.py — the float32
        # rounding channel, v4 §5) are kept in the JSON as evidence but excluded here
        best2 = max((v['eta'] for v in var2.values()
                     if v.get('physical', True)), default=None) if var2 else None
        rows.append({
            'model': r.get('model'), 'template': r.get('template'),
            'floor': r.get('floor', em.REGISTRY.get(r.get('model'), {})
                            .get('floor')),
            'sigma_MF': r.get('sigma_mf', {}).get('emp'),
            'eta_quad': r.get('quadrature', {}).get('eta'),
            'eta_var2_best': best2,
            'eta_var_best': best['eta'] if best else None,
            'eta_lin': var.get('linear', {}).get('eta') if var else None,
            'bound': r.get('bound_complete_data_dc_excluded', r.get('bound_complete_data')),
            'f_s': r.get('diagnostics', {}).get('f_s'),
            'xi': r.get('diagnostics', {}).get('xi_rms'),
        })
    if not rows:
        print('no results yet'); return

    def fmt(v, n=3):
        return '—' if v is None else f'{v:.{n}f}'
    hdr = (f"{'model':16s} {'template':9s} {'sig_MF':>7s} {'eta_quad':>9s} "
           f"{'eta_var2':>8s} {'eta_var':>8s} {'eta_lin':>8s} {'bound':>7s} "
           f"{'f_s':>6s} {'xi':>5s}")
    print(hdr); print('-' * len(hdr))
    for r in rows:
        print(f"{r['model']:16s} {r['template']:9s} {fmt(r['sigma_MF']):>7s} "
              f"{fmt(r['eta_quad'], 4):>9s} {fmt(r['eta_var2_best'], 4):>8s} "
              f"{fmt(r['eta_var_best'], 4):>8s} "
              f"{fmt(r['eta_lin'], 4):>8s} {fmt(r['bound'], 2):>7s} "
              f"{fmt(r['f_s']):>6s} {fmt(r['xi'], 2):>5s}")
    # CSV alongside
    import csv
    with open(Path(out_dir) / 'master_table.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print(f"\nwritten {Path(out_dir) / 'master_table.csv'}")


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--models', default='all',
                    help="comma list of registry names, or one of "
                         "'all' / 'base' (floorless) / 'wn' (white-floor set)")
    ap.add_argument('--templates', default='both',
                    choices=['extended', 'compact', 'both'])
    ap.add_argument('--mode', default='cheap',
                    choices=['cheap', 'variational', 'variational2', 'all'])
    ap.add_argument('--n-ens', type=int, default=DEFAULT_N_ENS)
    ap.add_argument('--n-extra', type=int, default=12000,
                    help='extra FIT/VAL maps for --mode variational2 '
                         '(appended to the frozen n-ens campaign ensemble)')
    ap.add_argument('--rungs', default=None,
                    help="variational2 only: comma list of rungs (default: "
                         "per-tier ladder; e.g. 'linear,reweight' for a fast "
                         "first pass — later runs merge missing rungs in)")
    ap.add_argument('--epochs', type=int, default=None,
                    help='default: 60 (variational), 80 (variational2)')
    ap.add_argument('--restarts', type=int, default=2)
    ap.add_argument('--patience', type=int, default=None,
                    help='variational2 only: early-stop patience in epochs '
                         '(default max(15, epochs//3); was a fixed 15 '
                         'before 2026-09-02)')
    ap.add_argument('--min-epochs', type=int, default=None,
                    help='variational2 only: warm-up epochs during which '
                         'early stopping cannot fire (default min(30, epochs))')
    ap.add_argument('--lr', type=float, default=3e-3,
                    help='variational2 only: Adam learning rate (default 3e-3)')
    ap.add_argument('--kmax', type=int, default=3,
                    help='variational2 only: max |f| (cycles/map) of the Fourier '
                         'pooling basis (default 3 = 49 maps; 8 = ~200; 16 = ~800)')
    ap.add_argument('--pool', default='fourier',
                    choices=['fourier', 'template', 'both'],
                    help="variational2 only: pooling basis for quadratic/cubic/cnn "
                         "rungs. 'fourier' = campaign default; 'template' = maps "
                         "derived from the whitened template (makes the i.i.d. "
                         "optimum representable for compact templates); 'both' = union")
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--out', default=str(_p / 'results' / 'eta_results'))
    ap.add_argument('--force', action='store_true',
                    help='recompute sections that already exist')
    ap.add_argument('--report', action='store_true',
                    help='print/write the master table from stored results')
    args = ap.parse_args(argv)

    if args.report:
        report(args.out); return
    if args.epochs is None:
        args.epochs = 80 if args.mode == 'variational2' else 60

    if args.models == 'all':
        names = list(em.REGISTRY)
    elif args.models == 'base':                 # the floorless set
        names = list(em.BASE_MODELS)
    elif args.models == 'wn':                   # the white-floor set
        names = list(em.WN_MODELS)
    else:
        names = [m.strip() for m in args.models.split(',')]
    for m in names:
        if m not in em.REGISTRY:
            sys.exit(f"unknown model '{m}'; choose from {list(em.REGISTRY)}")
    template_names = (['extended', 'compact'] if args.templates == 'both'
                      else [args.templates])
    templates = em.make_templates()

    for name in names:
        print(f'===== {name} =====')
        if args.mode == 'variational2' and em.REGISTRY[name]['rungs'] is None:
            continue
        n_gen = (args.n_ens + args.n_extra if args.mode == 'variational2'
                 else args.n_ens)
        entry, maps, latents, sp, P_hat = prepare_model(name, args.n_ens,
                                                        n_gen=n_gen)
        for tname in template_names:
            tau = templates[tname]
            res = load_result(args.out, name, tname)
            res.update({'model': name, 'template': tname,
                        'tier': entry['tier'], 'n_ens': args.n_ens,
                        'master_seed': model_seed(name),
                        'floor': entry.get('floor', 0.0),
                        'split_seed': SPLIT_SEED, 'note': entry['note']})
            if args.mode in ('cheap', 'all'):
                res = run_cheap(name, tname, tau, entry, maps, latents, sp,
                                P_hat, res, force=args.force)
                save_result(args.out, name, tname, res)
            if args.mode in ('variational', 'all'):
                res = run_variational(name, tname, tau, entry, maps, sp, P_hat,
                                      res, epochs=args.epochs,
                                      restarts=args.restarts, batch=args.batch,
                                      force=args.force)
                save_result(args.out, name, tname, res)
            if args.mode == 'variational2':
                rungs = (tuple(r.strip() for r in args.rungs.split(','))
                         if args.rungs else None)
                res = run_variational2(name, tname, tau, entry, maps, sp,
                                       P_hat, res, n_ens=args.n_ens,
                                       epochs=args.epochs,
                                       restarts=args.restarts,
                                       batch=args.batch, force=args.force,
                                       rungs=rungs, patience=args.patience,
                                       min_epochs=args.min_epochs, lr=args.lr,
                                       kmax=args.kmax, pool=args.pool,
                                       save_cb=lambda r, _n=name, _t=tname:
                                           save_result_merged(args.out, _n, _t, r))
                save_result_merged(args.out, name, tname, res)
        del maps
    print('\ncampaign pass complete.')
    report(args.out)


if __name__ == '__main__':
    main()
