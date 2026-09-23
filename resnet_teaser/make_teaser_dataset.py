#!/usr/bin/env python
"""
make_teaser_dataset.py -- campaign-exact training / evaluation sets for the
ResNet teaser runs of Paper I (Sec. 6.1)
==========================================================================

Purpose
-------
Build the HDF5 datasets on which the ResNet regressors of Sec. 6.1 of Paper I
are trained and evaluated, such that the eta-campaign numbers (sigma_MF, the
ceiling, the reweight rung, the Method-E envelope) apply to them WITHOUT any
re-derivation.  The maps are produced by the very generators the campaign
measured,

    eta_pipeline.models.REGISTRY[model]['gen'](n, seed)

(128^2, beam FWHM 5 px, P_ISO_ARGS, PSRAND_ARGS, WHITE_REAL, WN_FLOORS, pipeline
order correlated noise -> beam -> UNSMOOTHED white floor), and the source is
injected as

    d = A * tau_ext + n ,     tau_ext = noise_lib.make_template(128, 'beta', 7, 5,
                                                                 normalize='none').

Why this is identical to the Phase-A driver pipeline: the drivers compute
beam(img + n_1f) + w with img the PRE-beam beta profile of peak amplitude A.  The
beam is linear, so beam(img + n_1f) + w = A * beam(beta) + [beam(n_1f) + w] =
A * tau_ext + n.  A is therefore the pre-beam peak amplitude (the paper's unit),
and tau_ext is the same template object that defines sigma_MF in the campaign
JSONs -- comparability is obtained by construction rather than by copying
parameters into a legacy driver.

Two files per model
-------------------
1. TRAIN file  (default 30k maps):  legacy layout, read-compatible with the
   DDP training scripts (they read 'noisy' and 'I0' only):

       /noisy      (N,128,128) float32   A*tau_ext + noise
       /clean      (N,128,128) float32   A*beta_pre  (pre-beam profile, as legacy)
       /I0         (N,)        float32   A ~ U[0, L]
       /has_source (N,)        int8      all 1 (uniform prior touches A=0 anyway)
       /latents/*                        per-map latents (PSRAND slope/amplitude)
       /template_ext (128,128) float64   tau_ext (peak ~0.9, pre-beam peak 1)
       attrs: model, L, r_design, seeds, registry parameters, n_samples, ...

2. EVAL file (default 6000 NOISE-ONLY maps from an independent seed):

       /noise      (M,128,128) float32
       /latents/*
       /template_ext, /P_hat (128,128) float64  -- ensemble 2-D PSD from the
                                                    first n_psd maps (SPLIT_PSD)
       attrs: n_psd, n_eval, sigma_mf_pred (from P_hat), sigma_mf_emp (ensemble
              MF applied to the EVAL maps), campaign value for comparison, ...

   The evaluation script forms A_j*tau_ext + n_i for an A-grid on the M - n_psd
   EVAL maps (common random numbers across A) and applies the ensemble MF to the
   same maps for a paired V_CNN / V_MF comparison.

Seeds
-----
Master seeds must differ from the campaign's (777000 + 17 i, the _WN rows
777221...).  Default: train = registry seed + 103000, eval = registry seed +
213000 (T0_RED_REAL: 880034 / 990034; T1_PSRAND_WN: 880221 / 990221).  The maps
are generated in chunks; chunk c uses the integer seed drawn as the c-th 64-bit
state of numpy.random.SeedSequence(master) -- recorded in attrs['chunk_seeds'],
so any chunk is reproducible on its own.  The amplitudes A are drawn from a
separate stream seeded with master + 1.

Design ratio
------------
r = sqrt(12) sigma_MF / L  (Phase-A r-audit): L = 5 for T0_RED_REAL (sigma_MF =
0.86 -> r = 0.60), L = 7 for T1_PSRAND_WN (sigma_MF = 1.21 -> r = 0.60).  The
script prints r from the measured sigma_MF and warns if it leaves [0.5, 0.85].

Usage
-----
    python make_teaser_dataset.py --model T0_RED_REAL  --L 5 --outdir $TEASER_DATA
    python make_teaser_dataset.py --model T1_PSRAND_WN --L 7 --outdir $TEASER_DATA
    python make_teaser_dataset.py --model T0_RED_REAL --L 5 --n-train 400 --n-eval 300 \
           --outdir /tmp/smoke                                   # smoke test

Needs numpy, scipy, h5py and the packages noise_lib and eta_pipeline of this
repository (the script finds them by walking up to the repository root, or via
``pip install -e .``).  No torch.
"""

#%% ================================================================================
# === IMPORTS AND PATH SETUP ===
# ==================================================================================

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py

_here = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
for _p in (_here, *_here.parents):                       # finds the repository root
    if (_p / 'noise_lib').is_dir() and (_p / 'eta_pipeline').is_dir():
        sys.path.insert(0, str(_p))
        break

import noise_lib as nl                                   # noqa: E402
from eta_pipeline import models as em                    # noqa: E402
from eta_pipeline import whitening as wh                 # noqa: E402

# campaign reference values (results/eta_results JSONs), extended
# template, for the sigma_MF consistency check
CAMPAIGN_SIGMA_MF = {
    'T0_WHITE':     dict(pred=0.1211, emp=0.1211),   # sigma_w = 1 (registry); scales with sigma_w
    'T0_RED_REAL':  dict(pred=0.8636, emp=0.8537),
    'T1_PSRAND_WN': dict(pred=1.2999, emp=1.2053),
}
TRAIN_SEED_OFFSET = 103000
EVAL_SEED_OFFSET = 213000


#%% ================================================================================
# === CONFIG (used when the script is run WITHOUT command-line arguments, e.g. in Spyder) ===
# ==================================================================================
# From a terminal the same settings are command-line flags, e.g.
#   python make_teaser_dataset.py --model T0_RED_REAL  --L 5 --outdir $TEASER_DATA
#   python make_teaser_dataset.py --model T1_PSRAND_WN --L 7 --outdir $TEASER_DATA
#   python make_teaser_dataset.py --model T0_WHITE --white-sigma 9 --L 5 --outdir $TEASER_DATA
# Command-line flags always win; this cell only supplies defaults for an argument-less run.

CONFIG = dict(
    model       = 'T1_PSRAND_WN',   # registry name: T0_RED_REAL | T1_PSRAND_WN | T0_WHITE | ...
    L           = 7.0,             # upper edge of the uniform prior on A  (r = sqrt(12) sigma_MF / L)
    white_sigma = None,            # T0_WHITE only: white rms per pixel; 9.0 -> sigma_MF 1.09, r 0.75
    n_train     = 30000,           # maps in the TRAIN file
    n_eval      = 6000,            # noise-only maps in the EVAL file ...
    n_psd       = 2000,            # ... of which the first n_psd form the PSD split
    outdir      = os.environ.get('TEASER_DATA', str(Path.cwd() / 'teaser_data')),   # WHERE THE HDF5 FILES ARE WRITTEN
    chunk       = 2000,
)


def config_argv(cfg):
    """Turn the CONFIG dict into an argv list for main()."""
    argv = ['--model', cfg['model'], '--L', str(cfg['L']), '--n-train', str(cfg['n_train']),
            '--n-eval', str(cfg['n_eval']), '--n-psd', str(cfg['n_psd']),
            '--outdir', cfg['outdir'], '--chunk', str(cfg['chunk'])]
    if cfg.get('white_sigma') is not None:
        argv += ['--white-sigma', str(cfg['white_sigma'])]
    return argv


#%% ================================================================================
# === HELPERS ===
# ==================================================================================

def chunk_seeds(master, n_chunks):
    """Integer seeds for the chunks: 63-bit states of SeedSequence(master).

    Independent of numpy version for a given master (SeedSequence is a fixed
    hash), and far away from every 777xxx / 8xxxxx campaign seed.
    """
    ss = np.random.SeedSequence(int(master))
    return [int(s) for s in ss.generate_state(n_chunks, dtype=np.uint64) >> np.uint64(1)]


WHITE_SIGMA_OVERRIDE = None     # set by --white-sigma (T0_WHITE only)


def _gen_white(sigma_w):
    """T0_WHITE at a chosen rms: the registry generator with sigma_w instead of 1.0
    (same per-image child RNG structure as em.gen_t0_white)."""
    def gen(n, seed):
        return em._ensemble(n, seed, lambda rng: nl.white_noise(em.SHAPE, sigma_w, rng)), {}
    gen.__name__ = f'gen_t0_white_sigma{sigma_w:g}'
    return gen


def get_generator(model):
    if model == 'T0_WHITE' and WHITE_SIGMA_OVERRIDE is not None:
        return _gen_white(WHITE_SIGMA_OVERRIDE)
    return em.REGISTRY[model]['gen']


def generate_noise(model, n, master, chunk=2000, verbose=True):
    """Noise-only maps from the registry generator, in chunks.

    Yields (start, maps, latents) with maps float32 (n_chunk, 128, 128) and
    latents a dict of per-map arrays (may be empty).
    """
    gen = get_generator(model)
    n_chunks = int(np.ceil(n / chunk))
    seeds = chunk_seeds(master, n_chunks)
    t0 = time.time()
    for c in range(n_chunks):
        m = min(chunk, n - c * chunk)
        maps, lat = gen(m, seeds[c])
        maps = np.asarray(maps, dtype=np.float32)
        if verbose:
            print(f"  chunk {c + 1}/{n_chunks}: {m} maps  seed {seeds[c]}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
        yield c * chunk, maps, {k: np.asarray(v) for k, v in lat.items()}


def exact_mean_psd(model):
    """Exact ensemble-mean spectrum P_bar for the models whose mixture is analytic.

    T0_RED_REAL : P = A^2/|k|^3 B^2 + n_pix sigma_w^2   (sigma_w = WHITE_REAL)
    T1_PSRAND_WN: P_bar = E[a^2] E_s[|k|^-s] B^2 + n_pix sigma_w^2  with the clipped
                  normal priors of noise_lib.psrand, integrated on the campaign's
                  truncnorm_grid (201 slope nodes; converged to < 1e-3).
    Returns None for other models (the eval script then falls back to P_hat).
    sigma_MF(P_bar) is the exact matched-filter error the eta ratios refer to;
    the EVAL-split empirical value of a covariance mixture scatters by ~10 %
    (campaign: 1.205 empirical vs 1.300 from its PSD split vs 1.3285 exact).
    """
    from eta_pipeline.quadrature import truncnorm_grid
    B2 = em.beam2()
    if model == 'T0_RED_REAL':
        return nl.psd_isotropic_powerlaw(em.SHAPE, *em.P_ISO_ARGS) * B2 + em.p_white(em.WHITE_REAL)
    if model == 'T0_RED':
        return nl.psd_isotropic_powerlaw(em.SHAPE, *em.P_ISO_ARGS) * B2
    if model == 'T0_WHITE':
        return em.p_white(1.0 if WHITE_SIGMA_OVERRIDE is None else WHITE_SIGMA_OVERRIDE)
    if model in ('T1_PSRAND', 'T1_PSRAND_WN'):
        sm, ss, am, asg = em.PSRAND_ARGS
        s_grid, s_w = truncnorm_grid(sm, ss, 1.0, n=201)
        a_grid, a_w = truncnorm_grid(am, asg, 1e-6, n=201)
        Ea2 = float(np.sum(a_w * a_grid**2))
        Pbar = sum(w * nl.psd_isotropic_powerlaw(em.SHAPE, s, 1.0)
                   for s, w in zip(s_grid, s_w)) * Ea2 * B2
        if model.endswith('_WN'):
            Pbar = Pbar + em.p_white(em.WN_FLOORS[model])
        return Pbar
    return None


def registry_params(model):
    """Every registry-level parameter that fixes the model, for the attrs."""
    entry = em.REGISTRY[model]
    p = dict(
        model=model, tier=entry['tier'], registry_seed=int(entry['seed']),
        floor_sigma_w=float(entry.get('floor', 0.0)),
        size=em.SIZE, beam_fwhm=em.BEAM_FWHM, template_rc=em.TEMPLATE_RC,
        P_ISO_ARGS=list(em.P_ISO_ARGS), PSRAND_ARGS=list(em.PSRAND_ARGS),
        WHITE_REAL=em.WHITE_REAL, WN_SEED_OFFSET=em.WN_SEED_OFFSET,
        registry_note=entry.get('note', ''),
        pipeline='correlated noise -> beam -> unsmoothed white floor; source = A*tau_ext '
                 '(= beam(A*beta_pre)), A = pre-beam peak amplitude; no DC zeroing',
    )
    if model in em.WN_FLOORS:
        p['WN_FLOOR'] = em.WN_FLOORS[model]
    if model == 'T0_WHITE':
        p['white_sigma'] = 1.0 if WHITE_SIGMA_OVERRIDE is None else float(WHITE_SIGMA_OVERRIDE)
        p['generator'] = get_generator(model).__name__
    return p


def write_attrs(f, params):
    f.attrs['params_json'] = json.dumps(params, default=str)
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, np.integer, np.floating)):
            f.attrs[k] = v
    f.attrs['creation_date'] = datetime.now().isoformat()
    f.attrs['noise_lib_version'] = nl.__version__
    f.attrs['numpy_version'] = np.__version__


#%% ================================================================================
# === TRAIN FILE ===
# ==================================================================================

def make_train_file(model, L, n, master, outpath, chunk=2000, has_source_all=True):
    """A*tau_ext + noise, legacy HDF5 layout, written chunk by chunk."""
    S = em.SIZE
    tau = em.make_templates(S)['extended']                  # beam(beta_pre), peak ~0.9
    beta_pre = nl.beta_model(S, em.TEMPLATE_RC, 1.0)         # pre-beam profile, peak 1

    rngA = np.random.default_rng(int(master) + 1)
    A = rngA.uniform(0.0, L, size=n).astype(np.float64)

    params = registry_params(model)
    params.update(dict(kind='train', n_samples=n, L=float(L), flux_prior='uniform[0,L]',
                       master_seed=int(master), amplitude_seed=int(master) + 1,
                       chunk=chunk, template='extended', template_peak=float(tau.max()),
                       source_injection='A*tau_ext added to the noise map (post-beam, '
                                        'identical to beam(A*beta_pre + n_1f) + w)'))
    n_chunks = int(np.ceil(n / chunk))
    params['chunk_seeds'] = chunk_seeds(master, n_chunks)

    print(f"[train] {model}: n={n}  L={L}  master={master} -> {outpath}", flush=True)
    latent_store = {}
    with h5py.File(outpath, 'w') as f:
        kw = dict(compression='lzf', shuffle=True, dtype=np.float32, chunks=(1, S, S))
        d_noisy = f.create_dataset('noisy', shape=(n, S, S), **kw)
        d_clean = f.create_dataset('clean', shape=(n, S, S), **kw)
        f.create_dataset('I0', data=A.astype(np.float32), chunks=(min(1000, n),),
                         compression='gzip', compression_opts=1)
        f.create_dataset('has_source', data=np.ones(n, dtype=np.int8) if has_source_all
                         else (A > 0).astype(np.int8))
        f.create_dataset('template_ext', data=tau)
        f.create_dataset('beta_pre', data=beta_pre)

        gmin, gmax = np.inf, -np.inf
        for start, maps, lat in generate_noise(model, n, master, chunk):
            m = len(maps)
            a = A[start:start + m]
            noisy = maps + (a[:, None, None] * tau[None]).astype(np.float32)
            d_noisy[start:start + m] = noisy
            d_clean[start:start + m] = (a[:, None, None] * beta_pre[None]).astype(np.float32)
            gmin, gmax = min(gmin, float(noisy.min())), max(gmax, float(noisy.max()))
            for k, v in lat.items():
                latent_store.setdefault(k, []).append(v)

        if latent_store:
            g = f.create_group('latents')
            for k, parts in latent_store.items():
                g.create_dataset(k, data=np.concatenate(parts))
        # global range over ALL maps (train+val): the training script's normalisation
        # constants if it reads them from the attrs, else it recomputes on the train split
        d_noisy.attrs['min'] = gmin
        d_noisy.attrs['max'] = gmax
        params.update(dict(noisy_min=gmin, noisy_max=gmax))
        write_attrs(f, params)
        f.attrs['n_samples'] = n
        f.attrs['model_name'] = model

    mb = os.path.getsize(outpath) / 1024**2
    print(f"[train] written {n} maps -> {outpath} ({mb:.0f} MB); noisy range "
          f"[{gmin:.3f}, {gmax:.3f}]", flush=True)
    return params


#%% ================================================================================
# === EVAL FILE (noise only) + sigma_MF CHECK ===
# ==================================================================================

def make_eval_file(model, L, n, n_psd, master, outpath, chunk=2000):
    """Noise-only maps; P_hat from the first n_psd (SPLIT_PSD), sigma_MF checks."""
    S = em.SIZE
    tau = em.make_templates(S)['extended']
    params = registry_params(model)
    n_chunks = int(np.ceil(n / chunk))
    params.update(dict(kind='eval_noise_only', n_samples=n, n_psd=n_psd, n_eval=n - n_psd,
                       L=float(L), master_seed=int(master), chunk=chunk,
                       chunk_seeds=chunk_seeds(master, n_chunks), template='extended',
                       template_peak=float(tau.max())))

    print(f"[eval] {model}: n={n} (psd {n_psd} / eval {n - n_psd})  master={master} "
          f"-> {outpath}", flush=True)
    latent_store = {}
    with h5py.File(outpath, 'w') as f:
        d = f.create_dataset('noise', shape=(n, S, S), compression='lzf', shuffle=True,
                             dtype=np.float32, chunks=(1, S, S))
        for start, maps, lat in generate_noise(model, n, master, chunk):
            d[start:start + len(maps)] = maps
            for k, v in lat.items():
                latent_store.setdefault(k, []).append(v)
        if latent_store:
            g = f.create_group('latents')
            for k, parts in latent_store.items():
                g.create_dataset(k, data=np.concatenate(parts))

        # ---- sigma_MF from the PSD split; paired ensemble MF on the EVAL split ----
        psd_maps = d[:n_psd].astype(np.float64)
        P_hat = wh.estimate_psd2d(psd_maps, method='mean')
        del psd_maps
        sig_pred = float(wh.sigma_mf_from_psd(tau, P_hat))
        ev = d[n_psd:].astype(np.float64)
        a_mf = wh.mf_amplitudes(ev, tau, P_hat)
        del ev
        sig_emp = float(a_mf.std(ddof=1))
        sig_emp_err = sig_emp / np.sqrt(2.0 * (len(a_mf) - 1))
        mf_mean = float(a_mf.mean())

        f.create_dataset('template_ext', data=tau)
        f.create_dataset('P_hat', data=P_hat)
        f.create_dataset('mf_amplitudes_eval', data=a_mf)
        P_exact = exact_mean_psd(model)
        if P_exact is not None:
            f.create_dataset('P_exact', data=P_exact)
            sig_exact = float(wh.sigma_mf_from_psd(tau, P_exact))
            # paired MF built from the EXACT spectrum on the same EVAL maps
            ev = d[n_psd:].astype(np.float64)
            a_mf_ex = wh.mf_amplitudes(ev, tau, P_exact)
            del ev
            f.create_dataset('mf_amplitudes_eval_exact', data=a_mf_ex)
            sig_emp_exact = float(a_mf_ex.std(ddof=1))
        else:
            sig_exact, sig_emp_exact = float('nan'), float('nan')
        ref = CAMPAIGN_SIGMA_MF.get(model, {})
        sig_ref = sig_exact if np.isfinite(sig_exact) else sig_pred
        r_design = float(np.sqrt(12.0) * sig_ref / L)
        params.update(dict(sigma_mf_exact=sig_exact, sigma_mf_emp_exactfilter=sig_emp_exact,
                           sigma_mf_pred=sig_pred, sigma_mf_emp=sig_emp,
                           sigma_mf_emp_err=float(sig_emp_err), mf_mean_eval=mf_mean,
                           campaign_sigma_mf_pred=ref.get('pred', np.nan),
                           campaign_sigma_mf_emp=ref.get('emp', np.nan),
                           r_design=r_design))
        write_attrs(f, params)
        f.attrs['n_samples'] = n
        f.attrs['model_name'] = model

    print(f"[eval] sigma_MF exact (analytic P_bar) = {sig_exact:.4f}   <- primary reference")
    print(f"[eval] sigma_MF(P_hat, {n_psd} maps) = {sig_pred:.4f}  (campaign PSD-split value {ref.get('pred', float('nan')):.4f})")
    print(f"[eval] sigma_MF empirical on {len(a_mf)} EVAL maps = {sig_emp:.4f} +- {sig_emp_err:.4f}"
          f"  (campaign emp {ref.get('emp', float('nan')):.4f});  MF mean at A=0: {mf_mean:+.4f}")
    print(f"[eval] design ratio r = sqrt(12) sigma_MF / L = {r_design:.3f}  (target ~0.6)")
    if np.isfinite(sig_exact) and abs(sig_pred / sig_exact - 1) > 0.05:
        print("[eval] WARNING: sigma_MF(P_hat) differs from the exact value by > 5 %.")
    if ref and WHITE_SIGMA_OVERRIDE is None and abs(sig_ref / ref['pred'] - 1) > 0.05:
        print("[eval] WARNING: sigma_MF(P_hat) differs from the campaign value by > 5 % -- "
              "the dataset is NOT the campaign's model; investigate before training.")
    if not 0.5 <= r_design <= 0.85:
        print(f"[eval] WARNING: r = {r_design:.2f} outside [0.5, 0.7]; reconsider L (target 0.6-0.8, must be < 1).")
    return params


#%% ================================================================================
# === MAIN ===
# ==================================================================================

def main(argv=None):
    if argv is None and len(sys.argv) <= 1:      # no flags given (Spyder / bare run): use CONFIG
        argv = config_argv(CONFIG)
        print('no command-line arguments -> using the CONFIG cell:', ' '.join(argv))
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=list(em.REGISTRY))
    ap.add_argument('--L', type=float, required=True, help='upper edge of the uniform prior on A')
    ap.add_argument('--n-train', type=int, default=30000)
    ap.add_argument('--n-eval', type=int, default=6000, help='noise-only maps in the EVAL file')
    ap.add_argument('--n-psd', type=int, default=2000, help='of which used for P_hat / sigma_MF')
    ap.add_argument('--seed-train', type=int, default=None,
                    help='default: registry seed + %d' % TRAIN_SEED_OFFSET)
    ap.add_argument('--seed-eval', type=int, default=None,
                    help='default: registry seed + %d' % EVAL_SEED_OFFSET)
    ap.add_argument('--chunk', type=int, default=2000)
    ap.add_argument('--outdir', default='.',
                    help='directory for the HDF5 files (default: the current working directory)')
    ap.add_argument('--tag', default='', help='extra string in the file names')
    ap.add_argument('--white-sigma', type=float, default=None,
                    help='T0_WHITE only: white-noise rms per pixel (registry value 1.0 gives '
                         'r = 0.08 at L = 5 -- signal dominated; 9.0 gives sigma_MF = 1.09, r = 0.75)')
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--skip-eval', action='store_true')
    args = ap.parse_args(argv)

    global WHITE_SIGMA_OVERRIDE
    if args.white_sigma is not None:
        if args.model != 'T0_WHITE':
            raise SystemExit('--white-sigma applies to T0_WHITE only')
        WHITE_SIGMA_OVERRIDE = float(args.white_sigma)
    reg_seed = em.model_seed(args.model)
    seed_train = args.seed_train if args.seed_train is not None else reg_seed + TRAIN_SEED_OFFSET
    seed_eval = args.seed_eval if args.seed_eval is not None else reg_seed + EVAL_SEED_OFFSET
    for s in (seed_train, seed_eval):
        if any(s == em.model_seed(m) for m in em.REGISTRY):
            raise SystemExit(f"seed {s} collides with a campaign registry seed")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Ltag = ('%g' % args.L).replace('.', 'p')
    tag = f"_{args.tag}" if args.tag else ''
    if WHITE_SIGMA_OVERRIDE is not None:
        tag = f"_sw{WHITE_SIGMA_OVERRIDE:g}" + tag
    ntag = f"{args.n_train // 1000}k" if args.n_train % 1000 == 0 else str(args.n_train)
    f_train = outdir / f"TEASER_{args.model}_train{ntag}_L{Ltag}_s{seed_train}{tag}.h5"
    f_eval = outdir / f"TEASER_{args.model}_evalnoise{args.n_eval}_s{seed_eval}{tag}.h5"

    summary = {}
    t0 = time.time()
    if not args.skip_eval:
        summary['eval'] = make_eval_file(args.model, args.L, args.n_eval, args.n_psd,
                                         seed_eval, f_eval, args.chunk)
    if not args.skip_train:
        summary['train'] = make_train_file(args.model, args.L, args.n_train, seed_train,
                                           f_train, args.chunk)
    with open(outdir / f"TEASER_{args.model}_L{Ltag}{tag}_manifest.json", 'w') as fh:
        json.dump(dict(train_file=str(f_train), eval_file=str(f_eval), **summary), fh,
                  indent=1, default=str)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
