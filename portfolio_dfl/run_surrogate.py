"""Test 3 - LP -> QP surrogate.

A. How much does 0.5*eta*|w|^2 change the decisions (semantics) and how much gradient does it give?
B. End-to-end training of theta (loss = -realised PnL net of kappa|w|, i.e. the original semantics),
   through the LP layers and through the QP surrogate, deploying the original LP on the test dates.
"""
import time
import warnings

import numpy as np
import torch

from problem import make_data, ic, lp_exact, qp_exact, qp_exact_vjp, net_pnl, cvx_layer, ols, SCS_TIGHT
from altdiff import altdiff_layer

warnings.filterwarnings("ignore")
bps = lambda v: 1e4 * v

n, T_TRAIN, T_TEST, BATCH = 3000, 250, 250, 32
X, theta_star, r, kappa, lo, up = make_data(T_TRAIN + T_TEST, n)
Xtr, rtr, Xte, rte = X[:T_TRAIN], r[:T_TRAIN], X[T_TRAIN:], r[T_TRAIN:]
kt, lot, upt, rtr_t, Xtr_t = map(torch.tensor, (kappa, lo, up, rtr, Xtr))
print(f"{n} stocks, {T_TRAIN} train + {T_TEST} test dates.  single-factor IC "
      f"{', '.join(f'{ic(X[..., j], r):.3f}' for j in range(X.shape[-1]))};  IC of X theta*: "
      f"{ic(X @ theta_star, r):.3f} (rank {ic(X @ theta_star, r, rank=True):.3f})")


def lp_pnl(theta, Xs, rs):
    return bps(net_pnl(lp_exact(Xs @ theta, kappa, lo, up), rs, kappa).mean())


def cos(a, b):
    return a @ b / np.linalg.norm(a) / np.linalg.norm(b)


# ---------------------------------------------------------------- A. semantics of the quadratic term
print("\nA. decisions of the QP vs the LP on the test dates (signal c = X theta*)")
C = Xte @ theta_star
W_lp = lp_exact(C, kappa, lo, up)
f_lp = lambda W: (C * W).sum(1) - (kappa * abs(W)).sum(1)      # the LP objective with predicted c
print(f"{'eta':>7s} {'dates = LP':>10s} {'assets changed':>14s} {'free/date':>9s} {'dates>=2 free':>13s} "
      f"{'LP-obj loss':>11s} {'bound':>8s} {'realised QP':>11s} {'|grad|':>8s}")
for eta in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
    W, F = qp_exact(C, kappa, eta, lo, up)
    diff = np.abs(W - W_lp) > 1e-12
    grad = np.einsum("tnk,tn->k", Xte, qp_exact_vjp(F, eta, rte - kappa * np.sign(W)))
    print(f"{eta:7.2g} {np.mean(~diff.any(1)):10.0%} {diff.mean():14.1%} {F.sum(1).mean():9.1f} "
          f"{np.mean(F.sum(1) >= 2):13.0%} {bps((f_lp(W_lp) - f_lp(W)).mean()):9.3f}bp "
          f"{bps(0.5 * eta * np.maximum(lo ** 2, up ** 2).sum()):6.2f}bp {bps(net_pnl(W, rte, kappa).mean()):9.2f}bp "
          f"{np.linalg.norm(grad):8.3f}")
print(f"(LP decisions: realised net PnL {bps(net_pnl(W_lp, rte, kappa).mean()):.2f}bp/day, "
      f"predicted objective {bps(f_lp(W_lp).mean()):.2f}bp/day, gross trade {abs(W_lp).sum(1).mean():.2f} x NAV)")


# ---------------------------------------------------------------- B. training
class ClosedFormQP(torch.autograd.Function):
    """Exact QP layer for this particular problem (closed form + analytic Jacobian)."""

    @staticmethod
    def forward(ctx, theta, idx, eta):
        W, F = qp_exact(Xtr[idx] @ theta.detach().numpy(), kappa, eta, lo, up)
        ctx.idx, ctx.F, ctx.eta = idx, F, eta
        return torch.tensor(W)

    @staticmethod
    def backward(ctx, grad_W):
        g = np.einsum("tnk,tn->k", Xtr[ctx.idx], qp_exact_vjp(ctx.F, ctx.eta, grad_W.numpy()))
        return torch.tensor(g), None, None


batches = np.random.default_rng(2).integers(0, T_TRAIN, size=(300, BATCH))   # same dates for every method
lp_cvx, qp_cvx = cvx_layer(n, kappa, lo, up), cvx_layer(n, kappa, lo, up, 0.3)
eta_t = torch.full((n,), 0.3)
LAYERS = {   # name: (layer(theta, date indices), training steps)
    "LP  cvxpylayers SCS eps=1e-8": (lambda th, i: lp_cvx(Xtr_t[i] @ th, solver_args=dict(SCS_TIGHT))[0], 10),
    "LP  Alt-Diff stopped at 100 it": (lambda th, i: altdiff_layer(th, Xtr_t[i], kt, lot, upt, rho=1.0, max_iter=100, tol=0), 60),
    "QP  eta=0.3 closed form": (lambda th, i: ClosedFormQP.apply(th, i, 0.3), 60),
    "QP  eta=0.3 Alt-Diff rho=1": (lambda th, i: altdiff_layer(th, Xtr_t[i], kt, lot, upt, eta_t, rho=1.0), 60),
    "QP  eta=0.3 cvxpylayers eps=1e-8": (lambda th, i: qp_cvx(Xtr_t[i] @ th, solver_args=dict(SCS_TIGHT))[0], 5),
}


def train(layer, theta0, steps, lr=0.05):
    phi = torch.tensor(theta0 / 1e-3, requires_grad=True)     # theta in units of 1e-3
    opt = torch.optim.Adam([phi], lr=lr)
    path = {}
    t0 = time.time()
    for step in range(steps):
        idx = batches[step]
        opt.zero_grad()
        loss = -net_pnl(layer(phi * 1e-3, idx), rtr_t[idx], kt).mean()   # original semantics: return - kappa|w|
        loss.backward()
        opt.step()
        path[step + 1] = phi.detach().numpy() * 1e-3
    return path, (time.time() - t0) / steps


def report(name, theta, sec=None):
    extra = f"{sec:6.2f}s/step" if sec is not None else " " * 11
    print(f"{name:38s} {extra}  test LP {lp_pnl(theta, Xte, rte):6.2f}bp  test IC {ic(Xte @ theta, rte):6.3f}  "
          f"cos {cos(theta, theta_star):6.3f}  |theta|/|theta*| {np.linalg.norm(theta) / np.linalg.norm(theta_star):5.2f}")


g = np.random.default_rng(1)
theta0 = g.standard_normal(len(theta_star))
theta0 *= np.linalg.norm(theta_star) / np.linalg.norm(theta0)
theta_ols = ols(Xtr, rtr)
print(f"\nB. training theta end-to-end (mini-batches of {BATCH} dates; test = deploy the original LP)")
report("oracle theta*", theta_star)
report("two-stage OLS", theta_ols)
report("random init (all runs start here)", theta0)
paths = {}
for name, (layer, steps) in LAYERS.items():
    paths[name], sec = train(layer, theta0, steps)
    report(f"{name} [{steps} steps]", paths[name][steps], sec)
ref = paths["QP  eta=0.3 closed form"]
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
print(f"  |theta - theta(closed form)| / |theta| after 5 steps: Alt-Diff {rel(paths['QP  eta=0.3 Alt-Diff rho=1'][5], ref[5]):.1e}, "
      f"cvxpylayers {rel(paths['QP  eta=0.3 cvxpylayers eps=1e-8'][5], ref[5]):.1e};  after 60 steps: Alt-Diff "
      f"{rel(paths['QP  eta=0.3 Alt-Diff rho=1'][60], ref[60]):.1e}")
print(f"  LP cvxpylayers: |theta_10 - theta_0| / |theta_0| = {rel(paths['LP  cvxpylayers SCS eps=1e-8'][10], theta0):.1e}")

print("\n  QP surrogate, closed form, 300 steps from the random init")
for eta in (0.1, 0.3, 1.0):
    path, sec = train(lambda th, i: ClosedFormQP.apply(th, i, eta), theta0, steps=300)
    report(f"QP  eta={eta}", path[300], sec)

print("\n  QP surrogate, closed form, fine-tuning the two-stage OLS theta (lr=0.01)")
for eta in (0.1, 0.3, 1.0):
    path, sec = train(lambda th, i: ClosedFormQP.apply(th, i, eta), theta_ols, steps=200, lr=0.01)
    for steps in (50, 200):
        report(f"OLS -> QP eta={eta}, {steps} steps", path[steps], sec)
