"""
variational2.py — Method A, round 2: parameter-efficient capacity ladder
========================================================================

Why a round 2 exists.  The round-1 ladder (variational.py) parameterized
its spatial weight maps freely — ~n_pixels = 16384 parameters per map
against n_fit = 2400 images.  In that p ≈ n regime every optimizer step
off the initialization degrades the VAL bound (Adam's scale-free
per-coordinate steps inject an O(lr) random perturbation into the maps),
so the best-VAL selection returned the EXACT initialization for every
rung of every model: the M3 campaign silently produced only the linear
anchor, at full precision, everywhere.  The guards worked; the
parameterization was wrong for the data volume.

Round-2 design rules:

1. NO free per-pixel parameter anywhere.  Spatial weight maps live in a
   fixed low-k Fourier basis (physics: the relevant weighting is smooth —
   the noise structure is stationary or large-scale), so a "map" costs
   ~50 coefficients, not 16384.  Convolution kernels (weight-shared,
   effective sample n_fit × n_pixels) remain free.
2. A physics-informed rung for the mixture (Tier-1) channel: band-power
   reweighting  f = t̂†x · (1 + Σ_j a_j q̃_j) + Σ_j d_j q̃_j  with q_j the
   whitened band powers (radial + sectored) — exactly the b(ẑ)·Â_MF
   estimator family of v5 §5.2.5, so its η should approach the quadrature
   ceiling for well-inferable scalar latents (and the count-fluctuation
   channel for event models).
3. Mandatory telemetry: every fit stores its VAL trajectory, the init
   anchor value, and a moved_off_init flag; a run in which nothing moves
   can never again masquerade as a result.
4. Bigger data: the campaign driver generates extra FIT/VAL maps
   (n_extra ≈ 12000) beyond the frozen 6000-map B2a ensemble, whose PSD
   split still defines the whitening and whose EVAL split still defines
   the reported value (paired with the cheap methods by construction).

Everything else (Rayleigh-form bound, detached-EMA centering, split
discipline, best-VAL restart selection, EVAL bootstrap) is inherited
unchanged from round 1.
"""

import numpy as np
import torch
import torch.nn as nn

from .variational import (LinearRung, bound_parts, rayleigh_bound,
                          eval_rung, default_device)


# ----------------------------------------------------------------------------------
# Fixed low-k spatial basis
# ----------------------------------------------------------------------------------

def fourier_basis(shape, kmax=3):
    """Real 2-D Fourier maps with integer frequencies |f| ≤ kmax (cyc/map).

    Returns (n_b, H, W) float32, each map unit-RMS.  One map per
    independent (fy, fx) pair on a Hermitian half-plane: cos and sin
    components (sin dropped for the self-conjugate DC pair).  kmax = 3
    gives 29 maps (kmax = 8: 197) — smooth weighting fields down to
    1/3-map scales.  (Earlier versions of this docstring said 49; the
    logs print the true count.)
    """
    ny, nx = shape
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    maps = []
    for fy in range(0, kmax + 1):
        for fx in range(-kmax, kmax + 1):
            if fy == 0 and fx < 0:
                continue                       # Hermitian partner
            if fy**2 + fx**2 > kmax**2:
                continue
            ph = 2 * np.pi * (fy * y / ny + fx * x / nx)
            maps.append(np.cos(ph))
            if not (fy == 0 and fx == 0):
                maps.append(np.sin(ph))
    B = np.stack(maps).astype(np.float32)
    B /= np.sqrt((B**2).mean(axis=(-2, -1), keepdims=True))
    return B


def template_pool_maps(t_hat, blur_px=(1.0, 2.0, 4.0)):
    """Pooling maps derived from the whitened template (added 2026-09-05).

    Why.  The optimal test function is f* = t_hat^T s(x) (H2, v3 §2.5).  For an
    i.i.d. whitened field -- T2_CONFUSION_ATM, where the answer is known exactly --
    s acts pixelwise, so f* = sum_p t_hat_p g(x_p): a template-WEIGHTED sum of a
    pointwise nonlinearity.  The fixed low-k Fourier pooling basis (kmax=3, 49 maps,
    ~20-px resolution) cannot represent that weighting when t_hat is compact: for
    the beam template the whitened t_hat of _ATM is a single pixel, and the closest
    smooth-pooled statistic is a ~400-pixel sum whose variance buries the signal.
    Measured: the ladder recovered 63 % of eta = 3.5688 on the extended template
    and 0 % on the compact one.  Pooling the conv features with the template itself
    (and a few smoothed/squared variants, for a little flexibility) makes the
    i.i.d. optimum exactly representable for ANY template, at a cost of six pooled
    features per channel and no free per-pixel parameter -- the round-1 design
    rule is untouched.

    Returns (n_maps, H, W) float32, each unit-RMS: t, |t|, t^2, and t blurred with
    Gaussian sigma = blur_px.
    """
    t = np.asarray(t_hat, dtype=np.float64)
    maps = [t, np.abs(t), t**2]
    ny, nx = t.shape
    fy = np.fft.fftfreq(ny).reshape(-1, 1); fx = np.fft.fftfreq(nx).reshape(1, -1)
    k2 = fx**2 + fy**2
    T = np.fft.fft2(t)
    for sig in blur_px:
        maps.append(np.fft.ifft2(T * np.exp(-2 * np.pi**2 * sig**2 * k2)).real)
    B = np.stack(maps).astype(np.float32)
    B /= np.sqrt((B**2).mean(axis=(-2, -1), keepdims=True)).clip(1e-12)
    return B


def make_pool_basis(shape, t_hat, kmax=3, pool='fourier'):
    """Pooling basis for the quadratic/cubic/CNN rungs.

    pool = 'fourier'  : the campaign default, fourier_basis(kmax) only;
           'template' : template_pool_maps(t_hat) only;
           'both'     : the union (recommended for production from 2026-09-05).
    """
    parts = []
    if pool in ('fourier', 'both'):
        parts.append(fourier_basis(shape, kmax=kmax))
    if pool in ('template', 'both'):
        parts.append(template_pool_maps(t_hat))
    if not parts:
        raise ValueError(f"unknown pool '{pool}'")
    return np.concatenate(parts, axis=0)


# ----------------------------------------------------------------------------------
# Band-power features (the Tier-1 physics rung)
# ----------------------------------------------------------------------------------

def band_masks(shape, n_rad=10, n_sect=6, n_sect_rad=3, f_min=None):
    """Fourier-plane masks: log-radial annuli + (sector × radial) wedges.

    Sectors are in the angle mod π (real-field symmetry), resolving the
    anisotropy direction for the random-orientation model.  Returns
    (n_bands, H, W) float32 masks (not normalized).
    """
    ny, nx = shape
    fy = np.fft.fftfreq(ny).reshape(-1, 1)
    fx = np.fft.fftfreq(nx).reshape(1, -1)
    k = np.sqrt(fy**2 + fx**2)
    if f_min is None:
        f_min = 1.0 / max(ny, nx)
    edges = np.geomspace(f_min, 0.5, n_rad + 1)
    masks = []
    for i in range(n_rad):
        masks.append(((k >= edges[i]) & (k < edges[i + 1])).astype(np.float32))
    ang = np.mod(np.arctan2(fy, fx), np.pi)
    sedges = np.linspace(0, np.pi, n_sect + 1)
    redges = np.geomspace(f_min, 0.5, n_sect_rad + 1)
    for i in range(n_sect):
        for j in range(n_sect_rad):
            m = ((ang >= sedges[i]) & (ang < sedges[i + 1])
                 & (k >= redges[j]) & (k < redges[j + 1]))
            if m.sum() > 8:
                masks.append(m.astype(np.float32))
    return np.stack(masks)


class _FeatureNorm(nn.Module):
    """Detached EMA standardization of a feature vector: φ̂ = (φ − μ)/σ.

    Why it exists: the raw nonlinear features have wildly heterogeneous
    scales (an 8-mode band power fluctuates ~35% while a 5000-mode one
    fluctuates ~1%; product features on glitch maps carry outliers
    squared), so Adam's per-coordinate steps on the head coefficients were
    enormous relative to the useful values and the first epoch destroyed
    the bound (B2b container smoke test).  After standardization every
    head coefficient is an O(1) regression weight on a unit-variance
    feature and lr = 3e-3 means what it says.

    Like _EMACenter, the statistics are DETACHED (no gradient through μ,
    σ — the Stein identity stays intact) and freeze at eval time: the
    evaluated f is a fixed function.  Buffers initialize from the first
    training batch; before any training μ = 0, σ = 1 and the zero-init
    head keeps f exactly at the linear anchor.
    """

    def __init__(self, n_feat, momentum=0.05):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('mu', torch.zeros(n_feat))
        self.register_buffer('sd', torch.ones(n_feat))
        self.register_buffer('warm', torch.zeros((), dtype=torch.bool))

    def forward(self, feat):                     # (B, n_feat)
        if self.training:
            with torch.no_grad():
                m = feat.mean(dim=0)
                s = feat.std(dim=0).clamp_min(1e-8)
                if not bool(self.warm):
                    self.mu.copy_(m); self.sd.copy_(s)
                    self.warm.fill_(True)
                else:
                    self.mu += self.momentum * (m - self.mu)
                    self.sd += self.momentum * (s - self.sd)
        return (feat - self.mu) / self.sd


class ReweightRung2(nn.Module):
    """The mixture-MF rung: per-band reweighting of band-restricted MFs.

        f = t̂†x + Σ_{jm} C_{jm}·[ (t̂_j†x) · ĝ_m(x) ] + Σ_m d_m·ĝ_m(x),

    t̂_j = band-filtered copies of the whitened template (the full t̂ plus
    the radial and sectored Fourier bands with non-negligible template
    power), g_m = three functional families of the whitened band powers —
    q, log q, 1/q — standardized (ĝ) by the detached _FeatureNorm.

    Why this family: the optimal Tier-1 estimator is τ†N̂(ẑ)⁻¹x — a
    per-image REBUILT filter.  Restricting N̂ to be band-diagonal makes
    that exactly Σ_j β_j(ẑ)·(t̂_j†x) with β_j rational in the measured
    band powers, i.e. inside this head's span:
      * a scalar β (j = full template) is b(ẑ)·Â_MF — the coherent / IVW
        channel of v5 §5.2.5 (PSRAND);
      * band/sector-dependent β_j is filter reshaping — the incoherent
        channel (random-orientation anisotropy), which no scalar rescale
        can reach.
    The d_m terms supply the pure-quadratic (log-det-type) score part.
    Zero-init head ⇒ exact anchor start.
    """

    def __init__(self, shape, t_hat, masks, min_frac=1e-4):
        super().__init__()
        self.lin = LinearRung(shape, init=t_hat)
        self.lin.w.requires_grad_(False)
        m = torch.as_tensor(masks)
        self.register_buffer('masks', m)
        self.register_buffer('norms', m.sum(dim=(-2, -1)) * float(np.prod(shape)))
        # band-restricted whitened templates (keep bands holding template power)
        tF = np.fft.fft2(np.asarray(t_hat))
        t_tot = float(np.sum(np.abs(tF)**2))
        t_bands = [np.asarray(t_hat, dtype=np.float32)]
        for mk in np.asarray(masks):
            tb = np.fft.ifft2(tF * mk).real.astype(np.float32)
            if np.sum(np.abs(tF)**2 * mk) > min_frac * t_tot:
                t_bands.append(tb)
        self.register_buffer('t_bands', torch.as_tensor(np.stack(t_bands)))
        n_lin, n_pow = len(t_bands), 3 * len(masks)
        self.n_lin, self.n_pow = n_lin, n_pow
        n_feat = n_lin * n_pow + n_pow
        self.norm = _FeatureNorm(n_feat)
        self.head = nn.Parameter(torch.zeros(n_feat))

    def band_powers(self, x):                   # (B,1,H,W) -> (B, n_bands)
        F = torch.fft.fft2(x[:, 0])
        p2 = (F.real**2 + F.imag**2)
        return torch.einsum('bhw,jhw->bj', p2, self.masks) / self.norms

    def forward(self, x):
        lin = self.lin(x)
        lin_j = torch.einsum('jhw,bhw->bj', self.t_bands, x[:, 0])
        q = self.band_powers(x)
        g = torch.cat([q, torch.log(q.clamp_min(1e-12)),
                       1.0 / q.clamp_min(1e-12)], dim=1)
        feat = torch.cat([(lin_j[:, :, None] * g[:, None, :]).flatten(1), g],
                         dim=1)
        return lin + self.norm(feat) @ self.head


# ----------------------------------------------------------------------------------
# Basis-weighted product and CNN rungs
# ----------------------------------------------------------------------------------

class _ProductFeatures2(nn.Module):
    """Basis-pooled product features with a standardized linear head:

        φ_{cj}(x) = Σ_p B_j(p) · Π_o conv_o(x)_{c,p},
        block(x)  = Σ_{cj} C_{cj} · φ̂_{cj}(x)          (C zero-init).

    Round-2 version of round 1's free-map _ProductFeatures: the spatial
    weighting lives in the fixed low-k basis (J maps) instead of 16384
    free pixels, and every pooled feature is EMA-standardized before the
    head (see _FeatureNorm).
    """

    def __init__(self, basis, order=2, channels=2, ksize=7):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(1, channels, ksize, padding=ksize // 2, bias=False)
            for _ in range(order))
        for c in self.convs:
            nn.init.normal_(c.weight, std=0.05)
        self.register_buffer('B', torch.as_tensor(basis))
        n_feat = channels * len(basis)
        self.norm = _FeatureNorm(n_feat)
        self.head = nn.Parameter(torch.zeros(n_feat))

    def forward(self, x):
        prod = self.convs[0](x)
        for c in self.convs[1:]:
            prod = prod * c(x)
        feat = torch.einsum('bchw,jhw->bcj', prod, self.B).flatten(1)
        return self.norm(feat) @ self.head


class PolyRung2(nn.Module):
    """linear (frozen t̂) + quadratic [+ cubic] basis-weighted products."""

    def __init__(self, shape, t_hat, basis, order=2, channels=2, ksize=7):
        super().__init__()
        self.lin = LinearRung(shape, init=t_hat)
        self.lin.w.requires_grad_(False)
        self.blocks = nn.ModuleList(
            _ProductFeatures2(basis, order=o, channels=channels, ksize=ksize)
            for o in range(2, order + 1))

    def forward(self, x):
        f = self.lin(x)
        for b in self.blocks:
            f = f + b(x)
        return f


class CNNRung2(nn.Module):
    """linear (frozen t̂) + standardized basis-pooled CNN features:

        φ_{cj}(x) = Σ_p B_j(p) · body(x)_{c,p},
        f = t̂†x + Σ_{cj} C_{cj} φ̂_{cj}(x)              (C zero-init).

    The smooth (SiLU) conv body supplies generic nonlinear local features
    (event-profile responses included); the low-k basis pools them into
    channels·J numbers per image, standardized before the head.
    """

    def __init__(self, shape, t_hat, basis, channels=6, ksize=7, depth=2):
        super().__init__()
        self.lin = LinearRung(shape, init=t_hat)
        self.lin.w.requires_grad_(False)
        layers, c_in = [], 1
        for _ in range(depth):
            layers += [nn.Conv2d(c_in, channels, ksize, padding=ksize // 2),
                       nn.SiLU()]
            c_in = channels
        self.body = nn.Sequential(*layers)
        self.register_buffer('B', torch.as_tensor(basis))
        n_feat = channels * len(basis)
        self.norm = _FeatureNorm(n_feat)
        self.head = nn.Parameter(torch.zeros(n_feat))

    def forward(self, x):
        feat = torch.einsum('bchw,jhw->bcj', self.body(x), self.B).flatten(1)
        return self.lin(x) + self.norm(feat) @ self.head


# ----------------------------------------------------------------------------------
# Trainer with telemetry
# ----------------------------------------------------------------------------------

class _Scaled(nn.Module):
    """f = e^γ · inner(x): learnable global output scale for RAW-bound training.

    Why: the Rayleigh ratio's batch estimate carries an upward bias
    Var(proj)/B, so maximizing it batchwise rewards VARIANCE-INFLATING
    directions — on heavy-tailed models (glitches) training chased them
    and the VAL bound collapsed (caught in the B2b container smoke test).
    The raw bound 2E[proj] − E[f²] is unbiased per batch and penalizes
    useless output variance; the learnable log-scale γ supplies the output
    normalization the Rayleigh form provided analytically.  VAL selection
    and EVAL reporting stay in the scale-invariant Rayleigh form.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return torch.exp(self.gamma) * self.inner(x)


def fit_rung2(model, x_fit, x_val, t_hat, epochs=80, batch=512, lr=3e-3,
              patience=15, min_epochs=0, device=None, verbose=False, seed=0):
    """Round-2 trainer: RAW bound on FIT, Rayleigh early-stop on VAL, telemetry.

    Returns (model at best-VAL state, telemetry dict).  moved_off_init is
    the honesty flag: False means the reported value IS the init anchor.

    Early stopping (2026-09-02 revision).  `patience` counts epochs since
    the best VAL bound; `min_epochs` is a warm-up during which the counter
    can never fire.  Why both: the CNN and cubic rungs have zero-initialised
    heads, so the conv body receives no gradient until the head has moved,
    and the VAL bound sits at (or dips below) the linear anchor for ~30-70
    epochs before it takes off (floorless T2_PCA / T2_MEDIAN winners: best
    epoch 77 and 75 of 80).  With the old fixed patience=15 every restart
    that did not nudge above init_val inside 15 epochs was killed at
    epochs_run=15 -- which is what happened to ALL Pass-B CNN restarts on
    the *_WN models.  best_epoch/epochs_run in the telemetry show whether a
    restart was stopped by patience (epochs_run < epochs) or ran out.
    """
    device = device or default_device()
    model = model.to(device).train()
    th = torch.as_tensor(t_hat, dtype=torch.float32, device=device)
    xf = torch.as_tensor(x_fit, dtype=torch.float32, device=device).unsqueeze(1)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device).unsqueeze(1)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                           lr=lr)
    gen = torch.Generator().manual_seed(seed)

    def val_bound():
        vals = []
        for i in range(0, len(xv), 2048):
            vals.append(float(rayleigh_bound(model, xv[i:i + 2048], th).detach())
                        * len(xv[i:i + 2048]))
        return sum(vals) / len(xv)

    init_val = val_bound()
    best_val, best_epoch = init_val, -1
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    since, hist = 0, [init_val]
    import time as _time
    t_start = _time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(xf), generator=gen)
        for i in range(0, len(xf), batch):
            idx = perm[i:i + batch]
            proj, y = bound_parts(model, xf[idx], th)
            loss = -(2.0 * proj.mean() - y.pow(2).mean())   # raw bound: unbiased
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val = val_bound()
        hist.append(val)
        if verbose:
            print(f"    epoch {ep:3d}  val = {val:.4f}   "
                  f"({(_time.time() - t_start) / (ep + 1):.1f} s/epoch)", flush=True)
        if val > best_val + 1e-5:
            best_val, best_epoch, since = val, ep, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience and ep + 1 >= min_epochs:
                break
    model.load_state_dict(best_state)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tele = {'init_val': init_val, 'best_val': best_val,
            'best_epoch': best_epoch, 'epochs_run': len(hist) - 1,
            'moved_off_init': bool(best_epoch >= 0),
            'stopped_by_patience': bool(len(hist) - 1 < epochs),
            'patience': int(patience), 'min_epochs': int(min_epochs),
            'lr': float(lr),
            'sec_per_epoch': round((_time.time() - t_start) / max(1, len(hist) - 1), 2),
            'n_params_trainable': int(n_par),
            'val_hist': [round(v, 5) for v in hist]}
    return model, tele


RUNGS2_T1 = ('linear', 'reweight', 'quadratic', 'cnn')
RUNGS2_T2 = ('linear', 'reweight', 'quadratic', 'cubic', 'cnn')


def _make_rung2(name, shape, t_hat, basis, masks):
    if name == 'linear':
        return LinearRung(shape, init=t_hat)
    if name == 'reweight':
        return _Scaled(ReweightRung2(shape, t_hat, masks))
    if name == 'quadratic':
        return _Scaled(PolyRung2(shape, t_hat, basis, order=2))
    if name == 'cubic':
        return _Scaled(PolyRung2(shape, t_hat, basis, order=3))
    if name == 'cnn':
        return _Scaled(CNNRung2(shape, t_hat, basis))
    raise ValueError(f"unknown rung '{name}'")


def run_ladder2(x_fit, x_val, x_eval, t_hat, rungs=RUNGS2_T2, epochs=80,
                batch=512, lr=3e-3, patience=15, min_epochs=0, restarts=2,
                device=None, verbose=False, seed=0, kmax=3, pool='fourier'):
    """Round-2 ladder.  Returns {rung: {'eta', 'err', 'restart_etas',
    'telemetry'}}; 'linear' is the untrained anchor (eval only).

    Anything with moved_off_init = False across ALL restarts is reported
    but stamped — the value is the anchor, not a training result.
    """
    shape = x_fit.shape[-2:]
    basis = make_pool_basis(shape, t_hat, kmax=kmax, pool=pool)
    if verbose:
        print(f"  pooling basis: {len(basis)} maps (kmax={kmax}, pool='{pool}')")
    masks = band_masks(shape)
    x_ref = np.concatenate([x_fit, x_val])
    out = {}
    for name in rungs:
        if name == 'linear':
            model = LinearRung(shape, init=t_hat)
            eta, err = eval_rung(model, x_ref, x_eval, t_hat, device=device,
                                 seed=seed)
            out[name] = {'eta': float(eta), 'err': float(err),
                         'restart_etas': [float(eta)],
                         'telemetry': {'anchor': True}}
            if verbose:
                print(f"  [linear   ] eta = {eta:.4f} ± {err:.4f}  (anchor)")
            continue
        best_model, best_val, teles, restart_etas = None, -np.inf, [], []
        for r in range(max(1, restarts)):
            torch.manual_seed(seed + 1000 * r)
            m_r = _make_rung2(name, shape, t_hat, basis, masks)
            m_r, tele = fit_rung2(m_r, x_fit, x_val, t_hat, epochs=epochs,
                                  batch=batch, lr=lr, patience=patience,
                                  min_epochs=min_epochs, device=device,
                                  verbose=verbose, seed=seed + r)
            e_r, _ = eval_rung(m_r, x_ref, x_eval, t_hat, device=device,
                               seed=seed)
            restart_etas.append(float(e_r))
            teles.append(tele)
            if tele['best_val'] > best_val:
                best_val, best_model = tele['best_val'], m_r
        eta, err = eval_rung(best_model, x_ref, x_eval, t_hat, device=device,
                             seed=seed)
        moved = any(t['moved_off_init'] for t in teles)
        out[name] = {'eta': float(eta), 'err': float(err),
                     'restart_etas': restart_etas,
                     'telemetry': {'restarts': teles,
                                   'moved_off_init_any': bool(moved)}}
        if verbose:
            tag = '' if moved else '  ** NEVER MOVED OFF INIT **'
            print(f"  [{name:9s}] eta = {eta:.4f} ± {err:.4f}  "
                  f"restarts {['%.3f' % v for v in restart_etas]}{tag}")
    trained = [n for n in out if n != 'linear']
    if trained and not any(out[n]['telemetry'].get('moved_off_init_any')
                           for n in trained):
        print('  WARNING: no rung moved off its initialization — the ladder '
              'produced only anchor values (see README_PhaseB2b).')
    return out
