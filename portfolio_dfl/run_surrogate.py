"""Test 3 - LP -> QP surrogate.

A. How much does 0.5*eta*|w|^2 change the decisions (semantics) and how much gradient does it give?
B. End-to-end training of theta (loss = -realised PnL net of kappa|w|, i.e. the original semantics),
   through the LP layers and through the QP surrogate, deploying the original LP on the test dates.
"""
import time
import warnings

import numpy as np
import torch

from problem import make_data, lp_exact, qp_exact, qp_exact_vjp, net_pnl, cvx_layer, ols, SCS_TIGHT
from altdiff import altdiff_layer

warnings.filterwarnings("ignore")
bps = lambda v: 1e4 * v

X, theta_star, r, kappa, lo, up = make_data(500)
Xtr, rtr, Xte, rte = X[:250], r[:250], X[250:], r[250:]
n = X.shape[1]
kt, lot, upt, rtr_t, Xtr_t = map(torch.tensor, (kappa, lo, up, rtr, Xtr))


def lp_pnl(theta, Xs, rs):
    return bps(net_pnl(lp_exact(Xs @ theta, kappa, lo, up), rs, kappa).mean())


def cos(a, b):
    return a @ b / np.linalg.norm(a) / np.linalg.norm(b)


# ---------------------------------------------------------------- A. semantics of the quadratic term
print("A. decisions of the QP vs the LP on the test dates (signal c = X theta*)")
C = Xte @ theta_star
W_lp = lp_exact(C, kappa, lo, up)
f_lp = lambda W: (C * W).sum(1) - (kappa * abs(W)).sum(1)      # the LP objective with predicted c
print(f"{'eta':>7s} {'dates = LP':>10s} {'assets changed':>14s} {'free/date':>9s} {'dates>=2 free':>13s} "
      f"{'LP-obj loss':>11s} {'bound':>7s} {'realised QP':>11s} {'|grad|':>8s}")
for eta in (1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0):
    W, F = qp_exact(C, kappa, eta, lo, up)
    diff = np.abs(W - W_lp) > 1e-12
    grad = np.einsum("tnk,tn->k", Xte, qp_exact_vjp(F, eta, rte - kappa * np.sign(W)))
    print(f"{eta:7.0e} {np.mean(~diff.any(1)):10.0%} {diff.mean():14.1%} {F.sum(1).mean():9.1f} "
          f"{np.mean(F.sum(1) >= 2):13.0%} {bps((f_lp(W_lp) - f_lp(W)).mean()):9.3f}bp "
          f"{bps(0.5 * eta * np.maximum(lo ** 2, up ** 2).sum()):5.2f}bp {bps(net_pnl(W, rte, kappa).mean()):9.2f}bp "
          f"{np.linalg.norm(grad):8.2f}")
print(f"(realised net PnL of the LP decisions: {bps(net_pnl(W_lp, rte, kappa).mean()):.2f}bp/day)")


# ---------------------------------------------------------------- B. training
class ClosedFormQP(torch.autograd.Function):
    """Exact QP layer for this particular problem (closed form + analytic Jacobian)."""

    @staticmethod
    def forward(ctx, theta, eta):
        W, F = qp_exact(Xtr @ theta.detach().numpy(), kappa, eta, lo, up)
        ctx.F, ctx.eta = F, eta
        return torch.tensor(W)

    @staticmethod
    def backward(ctx, grad_W):
        return torch.tensor(np.einsum("tnk,tn->k", Xtr, qp_exact_vjp(ctx.F, ctx.eta, grad_W.numpy()))), None


lp_cvx, qp_cvx = cvx_layer(n, kappa, lo, up), cvx_layer(n, kappa, lo, up, 0.1)
LAYERS = {
    "LP  cvxpylayers SCS default": lambda th: lp_cvx(Xtr_t @ th)[0],
    "LP  cvxpylayers SCS eps=1e-8": lambda th: lp_cvx(Xtr_t @ th, solver_args=dict(SCS_TIGHT))[0],
    "LP  Alt-Diff stopped at 100 it": lambda th: altdiff_layer(th, Xtr_t, kt, lot, upt, rho=0.1, max_iter=100, tol=0),
    "QP  eta=0.1 closed form": lambda th: ClosedFormQP.apply(th, 0.1),
    "QP  eta=0.1 Alt-Diff rho=0.1": lambda th: altdiff_layer(th, Xtr_t, kt, lot, upt, torch.full((n,), 0.1), rho=0.1),
    "QP  eta=0.1 cvxpylayers eps=1e-8": lambda th: qp_cvx(Xtr_t @ th, solver_args=dict(SCS_TIGHT))[0],
}


def train(layer, theta0, steps, lr=0.05):
    phi = torch.tensor(theta0 / 1e-3, requires_grad=True)     # theta in units of 1e-3
    opt = torch.optim.Adam([phi], lr=lr)
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad()
        W = layer(phi * 1e-3)
        loss = -net_pnl(W, rtr_t, kt).mean()                    # original semantics: return - kappa|w|
        loss.backward()
        opt.step()
    return phi.detach().numpy() * 1e-3, (time.time() - t0) / steps


def calibrated(theta):
    """Pick the scale of c for the LP on the training dates (the QP surrogate inflates |theta|)."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(2.0), 13))
    alpha = grid[np.argmax([lp_pnl(a * theta, Xtr, rtr) for a in grid])]
    return alpha * theta


def report(name, theta, sec=None):
    extra = f"{sec:6.2f}s/step" if sec is not None else " " * 11
    print(f"{name:34s} {extra}  test LP {lp_pnl(theta, Xte, rte):6.2f}bp  cos {cos(theta, theta_star):6.3f}  "
          f"|theta|/|theta*| {np.linalg.norm(theta) / np.linalg.norm(theta_star):5.2f}  "
          f"scale-calibrated LP {lp_pnl(calibrated(theta), Xte, rte):6.2f}bp")


g = np.random.default_rng(1)
theta0 = g.standard_normal(len(theta_star))
theta0 *= np.linalg.norm(theta_star) / np.linalg.norm(theta0)
print("\nB. training theta end-to-end (250 train dates, 250 test dates; test = deploy the original LP)")
report("oracle theta*", theta_star)
report("two-stage OLS", ols(Xtr, rtr))
report("random init (all runs start here)", theta0)
thetas = {}
for name, layer in LAYERS.items():
    thetas[name], sec = train(layer, theta0, steps=60)
    report(name + " [60 steps]", thetas[name], sec)
ref = thetas["QP  eta=0.1 closed form"]
for name in ("QP  eta=0.1 Alt-Diff rho=0.1", "QP  eta=0.1 cvxpylayers eps=1e-8"):
    print(f"  after 60 steps, |theta - theta(closed form)| / |theta| for {name.split('  ')[1]}: "
          f"{np.linalg.norm(thetas[name] - ref) / np.linalg.norm(ref):.1e}")

print("\n  QP surrogate, closed form, 300 steps")
for eta in (0.01, 0.03, 0.1, 0.3):
    theta, sec = train(lambda th: ClosedFormQP.apply(th, eta), theta0, steps=300)
    report(f"QP  eta={eta}", theta, sec)

print("\n  QP surrogate, closed form, fine-tuning the two-stage OLS theta (lr=0.01)")
for eta in (0.01, 0.03, 0.1):
    for steps in (50, 200):
        theta, sec = train(lambda th: ClosedFormQP.apply(th, eta), ols(Xtr, rtr), steps=steps, lr=0.01)
        report(f"OLS -> QP eta={eta}, {steps} steps", theta, sec)
