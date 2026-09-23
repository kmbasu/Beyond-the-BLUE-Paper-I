"""
variational.py — Method A: score-matching variational bound (memo §16)
======================================================================

Estimates η as a CERTIFIED LOWER BOUND from whitened noise-only maps via
the exact variational identity (memo §9)

    η = sup_f { 2 E[t̂†∇_x f(x)] − E[f(x)²] },

whose optimum is f* = t̂†s_x(x).  Every f — undertrained, wrong capacity —
gives a lower bound up to Monte-Carlo error, so suboptimality is
conservative, never misleading.

Capacity ladder (each rung is a deliverable; values must be monotone):

  linear     f = (w·x)                      → 1 exactly (stationary case);
             η_lin > 1 flags non-stationarity (better-linear-filter forecast)
  quadratic  + Σ u·(conv_a x)(conv_b x)     → 1 + ½‖κ̃₃(t̂)‖²   (bispectrum)
  cubic      + Σ v·(conv_c x)(conv_d x)(conv_e x) → + ⅙‖κ̃₄(t̂)‖² (trispectrum)
  pixelwise  + Σ m·φ(x)  (shared 1×1 MLP)  — captures i.i.d.-pixel scores
  cnn        + small generic CNN head       → tops the ladder

Architectural notes (memo §16): activations must be smooth (SiLU) for the
double backprop; do NOT hard-code an odd nonlinearity (skewed models);
quadratic/cubic feature terms are batch-mean-subtracted to keep E[f] ≈ 0.
Nonlinear rungs embed the linear term initialized at t̂, so each rung
starts from the previous rung's optimum and can only add value.

Training protocol: maximize the bound on SPLIT_FIT (Adam), early-stop on
the SPLIT_VAL bound, REPORT the frozen network's SPLIT_EVAL value with a
bootstrap-over-images error (whitening.bootstrap_mean).

Device: picks 'mps' (Apple Silicon) > 'cuda' > 'cpu' automatically; no DDP.
"""

import numpy as np
import torch
import torch.nn as nn

from .whitening import bootstrap_mean


def default_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# ----------------------------------------------------------------------------------
# Rung architectures
# ----------------------------------------------------------------------------------

class LinearRung(nn.Module):
    """f(x) = Σ w·x.  Optimum (stationary whitened case): w = t̂, value 1."""

    def __init__(self, shape, init=None):
        super().__init__()
        if init is None:
            # small random init: from-scratch convergence to t̂ is itself a
            # null test (zeros would be a stationary point of the Rayleigh
            # objective)
            w0 = 0.01 * torch.randn(shape, generator=torch.Generator().manual_seed(0))
        else:
            w0 = torch.as_tensor(init, dtype=torch.float32).clone()
        self.w = nn.Parameter(w0)

    def forward(self, x):                    # x: (B, 1, H, W)
        return (self.w * x[:, 0]).sum(dim=(-2, -1))


class _EMACenter(nn.Module):
    """Centering by a DETACHED running mean.

    The variational identity is per-image; subtracting an in-graph batch
    mean couples the images and breaks the Stein integration by parts —
    a network can then cancel its own output variance while the gradient
    projection survives, inflating the "bound" arbitrarily (caught by null
    test N4).  A detached EMA constant is a legitimate test-function shift
    (f → f − c), keeps E[f] ≈ 0, and freezes at evaluation time.
    """

    def __init__(self, momentum=0.05):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('mu', torch.zeros(()))

    def forward(self, feat):
        if self.training:
            with torch.no_grad():
                self.mu += self.momentum * (feat.mean() - self.mu)
        return feat - self.mu


class _ProductFeatures(nn.Module):
    """Translation-covariant n-fold product features, spatially weighted:
    Σ_pix u · Π_j conv_j(x), EMA-centered (keeps E[f] ≈ 0)."""

    def __init__(self, shape, order=2, channels=2, ksize=7):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(1, channels, ksize, padding=ksize // 2, bias=False)
            for _ in range(order))
        for c in self.convs:
            nn.init.normal_(c.weight, std=0.05)
        self.u = nn.Parameter(torch.zeros(channels, *shape))
        self.center = _EMACenter()

    def forward(self, x):                    # (B, 1, H, W) -> (B,)
        prod = self.convs[0](x)
        for c in self.convs[1:]:
            prod = prod * c(x)
        feat = (self.u * prod).sum(dim=(-3, -2, -1))
        return self.center(feat)


class PolyRung(nn.Module):
    """linear (init t̂) + quadratic [+ cubic] product features.

    order=2 estimates the bispectrum rung, order=3 adds the trispectrum
    rung (both product blocks active).
    """

    def __init__(self, shape, t_hat, order=2, channels=2, ksize=7):
        super().__init__()
        self.lin = LinearRung(shape, init=t_hat)
        self.blocks = nn.ModuleList(
            _ProductFeatures(shape, order=o, channels=channels, ksize=ksize)
            for o in range(2, order + 1))

    def forward(self, x):
        f = self.lin(x)
        for b in self.blocks:
            f = f + b(x)
        return f


class PixelwiseRung(nn.Module):
    """linear (init t̂) + Σ_pix m·φ(x_pix) with a shared 1×1-conv MLP φ.

    Captures arbitrary i.i.d.-pixel scores exactly (optimal for the
    scale-mixture validation ensemble: m → t̂ shape, φ → nonlinear part of
    the 1-D score); batch-mean-subtracted.
    """

    def __init__(self, shape, t_hat, hidden=8, freeze_maps=True):
        super().__init__()
        self.lin = LinearRung(shape, init=t_hat)
        self.phi = nn.Sequential(
            nn.Conv2d(1, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, 1, 1))
        # zero-init the output layer: the rung STARTS exactly at the linear
        # optimum and the nonlinear correction grows only when it helps
        nn.init.zeros_(self.phi[-1].weight); nn.init.zeros_(self.phi[-1].bias)
        # weight map initialized at t̂: the nonlinear correction engages
        # first where the template carries Fisher weight
        self.m = nn.Parameter(torch.as_tensor(t_hat, dtype=torch.float32).clone())
        self.center = _EMACenter()
        if freeze_maps:
            # spatial maps pinned at the linear-rung optimum; only the shared
            # nonlinearity phi trains.  With learnable maps the optimizer has
            # ~n_pixels parameters and finds p ~ n overfitting directions
            # (B-shrinking along near-null sample-covariance modes) before the
            # low-dimensional physical solution — found in null test N3.  For
            # i.i.d. noise m = t_hat is exactly optimal, so nothing is lost.
            self.lin.w.requires_grad_(False)
            self.m.requires_grad_(False)

    def forward(self, x):
        feat = (self.m * self.phi(x)[:, 0]).sum(dim=(-2, -1))
        return self.lin(x) + self.center(feat)


class CNNRung(nn.Module):
    """linear (init t̂) + small generic CNN scalar head (memo §16: smooth
    activations, no enforced parity), batch-mean-subtracted."""

    def __init__(self, shape, t_hat, channels=8, ksize=5, depth=2):
        super().__init__()
        self.lin = LinearRung(shape, init=t_hat)
        layers, c_in = [], 1
        for _ in range(depth):
            layers += [nn.Conv2d(c_in, channels, ksize, padding=ksize // 2),
                       nn.SiLU()]
            c_in = channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(c_in, 1, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.m = nn.Parameter(torch.as_tensor(t_hat, dtype=torch.float32).clone())
        self.center = _EMACenter()

    def forward(self, x):
        feat = (self.m * self.head(self.body(x))[:, 0]).sum(dim=(-2, -1))
        return self.lin(x) + self.center(feat)


# ----------------------------------------------------------------------------------
# The bound and the trainer
# ----------------------------------------------------------------------------------

def bound_parts(f, x, t_hat):
    """Per-image (proj_i, y_i): proj = t̂†∇f(x_i), y = f(x_i).

    The raw bound is mean(2·proj − y²); the SCALE-OPTIMIZED (Rayleigh) form
    maximizes over an output rescaling c·f analytically,

        max_c E[2c·proj − c²y²] = (E proj)² / E[y²]   at c* = E proj / E y²,

    which is still a certified lower bound (it is the bound of c*·f) but is
    invariant under the output scale of f — the raw form is violently
    sensitive to optimizer step sizes on pixel-map parameters (Adam's
    per-parameter steps are scale-free), which destabilized the first
    implementation; the Rayleigh form removes that failure mode.
    """
    x = x.requires_grad_(True)
    y = f(x)
    g = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    proj = (g[:, 0] * t_hat).sum(dim=(-2, -1))
    return proj, y


def rayleigh_bound(f, x, t_hat, eps=1e-12):
    """Scale-optimized bound (E proj)²/E[y²] on one batch (torch scalar)."""
    proj, y = bound_parts(f, x, t_hat)
    return proj.mean()**2 / (y.pow(2).mean() + eps)


def fit_rung(model, x_fit, x_val, t_hat, epochs=60, batch=256, lr=3e-3,
             patience=8, reg_to_init=0.0, device=None, verbose=False, seed=0):
    """Maximize the Rayleigh bound on FIT; early-stop on the VAL bound.

    reg_to_init > 0 adds a decay-to-initialization penalty
    reg · Σ‖θ − θ_init‖² to the training loss (NOT to the reported bound):
    with pixel-map parameters the fit problem is in the p ~ n regime and
    unregularized training finds sample-covariance null directions before
    the physical solution; anchoring at the init (= previous rung's
    optimum) suppresses that channel.  Returns the model restored to its
    best-VAL state and the VAL history.
    """
    device = device or default_device()
    model = model.to(device).train()
    th = torch.as_tensor(t_hat, dtype=torch.float32, device=device)
    xf = torch.as_tensor(x_fit, dtype=torch.float32, device=device).unsqueeze(1)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device).unsqueeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    init_state = ({k: v.detach().clone() for k, v in model.state_dict().items()}
                  if reg_to_init > 0 else None)

    # the INIT state is a selection candidate: nonlinear rungs start at the
    # previous rung's optimum (linear part = t̂), and training must never be
    # able to end below it
    best_val = float(rayleigh_bound(model, xv, th).detach())
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    since, hist = 0, [best_val]
    for ep in range(epochs):
        perm = torch.randperm(len(xf), generator=gen)
        for i in range(0, len(xf), batch):
            idx = perm[i:i + batch]
            loss = -rayleigh_bound(model, xf[idx], th)
            if reg_to_init > 0:
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        loss = loss + reg_to_init * (p - init_state[name]).pow(2).sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val = float(rayleigh_bound(model, xv, th).detach())
        hist.append(val)
        if verbose:
            print(f"  epoch {ep:3d}  val bound = {val:.4f}")
        if val > best_val + 1e-5:
            best_val, since = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def eval_rung(model, x_ref, x_eval, t_hat, device=None, batch=512, seed=0,
              return_terms=False):
    """Frozen-model SPLIT_EVAL bound: (eta, boot_err[, terms]).

    The output rescaling c* is fixed on x_ref — any EVAL-disjoint data;
    FIT∪VAL is the stable choice (c* is one number, so reusing fit data
    costs nothing) — then the per-image raw-bound terms of c*·f are
    averaged over SPLIT_EVAL with a bootstrap-over-images error.
    return_terms=True also returns the per-image terms (index-aligned with
    x_eval), enabling PAIRED comparisons between runs that share the same
    evaluation images — the correlated part of the fluctuations cancels in
    a shared-index bootstrap of the difference.
    """
    x_val = x_ref
    device = device or default_device()
    model = model.to(device).eval()          # freeze EMA centering buffers
    th = torch.as_tensor(t_hat, dtype=torch.float32, device=device)

    def collect(xs):
        P, Y = [], []
        xs = torch.as_tensor(xs, dtype=torch.float32, device=device).unsqueeze(1)
        for i in range(0, len(xs), batch):
            proj, y = bound_parts(model, xs[i:i + batch], th)
            P.append(proj.detach().cpu().numpy())
            Y.append(y.detach().cpu().numpy())
        return np.concatenate(P), np.concatenate(Y)

    pv, yv = collect(x_val)
    c_star = pv.mean() / max(np.mean(yv**2), 1e-12)
    pe, ye = collect(x_eval)
    terms = 2.0 * c_star * pe - c_star**2 * ye**2
    m, e = bootstrap_mean(terms, seed=seed)
    return (m, e, terms) if return_terms else (m, e)


def _make_rung(name, shape, t_hat):
    """Fresh rung model.  'linear' inits at t̂: it is the SANITY ANCHOR —
    its value must sit at 1 (stationary case) and training must not move
    it; any deviation flags a whitening/DFT-convention error.  (A random
    init cannot converge when n_fit < n_pixels — the B<M overfit regime —
    so from-scratch recovery is not used as a test.)"""
    if name == 'linear':
        return LinearRung(shape, init=t_hat)
    if name == 'quadratic':
        return PolyRung(shape, t_hat, order=2)
    if name == 'cubic':
        return PolyRung(shape, t_hat, order=3)
    if name == 'pixelwise':
        return PixelwiseRung(shape, t_hat)
    if name == 'cnn':
        return CNNRung(shape, t_hat)
    raise ValueError(f"unknown rung '{name}'")


def run_ladder(x_fit, x_val, x_eval, t_hat, rungs=('linear', 'quadratic',
                                                   'cubic', 'cnn'),
               epochs=60, device=None, verbose=False, seed=0, restarts=1,
               return_terms=False, **fit_kw):
    """Train and evaluate the requested rungs.

    Returns {rung: (eta, err, restart_etas)} — a THREE-tuple.  (Before the
    Phase-B1 restarts update this was (eta, err); run_eta_campaign.py was
    updated with it and run_null_tests.py was not, which is why the battery
    raised "too many values to unpack" on the first Gaussian null.  Unpack as
    (eta, err, *_) or slice [:2].)

    Rung names: 'linear', 'quadratic', 'cubic', 'pixelwise', 'cnn'.

    restarts > 1 trains each nonlinear rung several times from different
    torch seeds and keeps the best-VAL model.  Every restart's value is a
    certified lower bound, so taking the best is legitimate — and it
    absorbs the training stochasticity (initialization + non-deterministic
    device kernels) that per-run bootstrap errors do NOT cover.
    return_terms=True returns (results, {rung: per-image EVAL terms}).
    """
    shape = x_fit.shape[-2:]
    out, terms_out = {}, {}
    for name in rungs:
        rung_epochs = epochs
        n_try = 1 if name == 'linear' else max(1, restarts)
        x_ref = np.concatenate([x_fit, x_val])
        best_model, best_val = None, -np.inf
        restart_etas = []
        for r in range(n_try):
            torch.manual_seed(seed + 1000 * r)     # deterministic conv inits
            m_r = _make_rung(name, shape, t_hat)
            m_r, hist = fit_rung(m_r, x_fit, x_val, t_hat, epochs=rung_epochs,
                                 device=device, verbose=verbose,
                                 seed=seed + r, **fit_kw)
            # every restart's EVAL value is recorded (the restart SPREAD is
            # the training systematic); the REPORTED value is the best-VAL
            # restart's — selection never touches EVAL
            e_r, _ = eval_rung(m_r, x_ref, x_eval, t_hat, device=device,
                               seed=seed)
            restart_etas.append(float(e_r))
            if max(hist) > best_val:
                best_val, best_model = max(hist), m_r
        model = best_model
        res = eval_rung(model, x_ref, x_eval, t_hat, device=device,
                        seed=seed, return_terms=True)
        out[name] = res[:2] + (restart_etas,)
        terms_out[name] = res[2]
        if verbose:
            print(f"[{name:9s}] eta = {out[name][0]:.4f} ± {out[name][1]:.4f}"
                  f"  restarts {['%.3f' % v for v in restart_etas]}")
    return (out, terms_out) if return_terms else out
