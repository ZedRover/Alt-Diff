"""Test 2 - the QP:  max c'w - kappa'|w| - 0.5*eta*|w|^2  s.t. sum(w) = 0, lo <= w <= up,  c = X theta.

Reference: closed-form solution and Jacobian (problem.qp_exact / qp_exact_vjp), cross-checked against
Clarabel and finite differences.  Then every layer's decisions and d(sum_t r_t'w_t)/dtheta.
"""
import sys
import time
import warnings

import cvxpy as cp
import numpy as np
import torch
from qpth.qp import QPFunction

from problem import make_data, qp_exact, qp_exact_vjp, cvx_layer, SCS_TIGHT
from altdiff import altdiff_layer, split_problem

warnings.filterwarnings("ignore")
sys.path.insert(0, "../classification")
from newlayer import diff as repo_altdiff  # noqa: E402  (the repo's own layer, unchanged)

ETA = 0.1


def exact_grad(X, theta, r, kappa, lo, up, eta=ETA):
    W, F = qp_exact(X @ theta, kappa, eta, lo, up)
    return W, np.einsum("tnk,tn->k", X, qp_exact_vjp(F, eta, r)), F


def layers(Xt, kt, lot, upt, n, T, with_repo=True):
    kappa, lo, up = kt.numpy(), lot.numpy(), upt.numpy()
    P, A, b, G, h = split_problem(kt, lot, upt, torch.full((n,), ETA))
    cvx = cvx_layer(n, kappa, lo, up, ETA)

    def qpth(th):
        c = Xt @ th
        z = QPFunction(verbose=False)(P, torch.cat([kt - c, kt + c], 1), G, h, A, b)
        return z[:, :n] - z[:, n:]

    def repo(th):
        ex = lambda M: M.unsqueeze(0).expand(T, *M.shape)
        c = Xt @ th
        z = repo_altdiff()(ex(P), torch.cat([kt - c, kt + c], 1), ex(G), ex(h), ex(A), ex(b))
        return z[:, :n] - z[:, n:]

    out = [
        ("cvxpylayers, SCS default", lambda th: cvx(Xt @ th)[0]),
        ("cvxpylayers, SCS eps=1e-8", lambda th: cvx(Xt @ th, solver_args=dict(SCS_TIGHT))[0]),
        ("qpth (OptNet)", qpth),
        ("Alt-Diff rho=0.1", lambda th: altdiff_layer(th, Xt, kt, lot, upt, torch.full((n,), ETA), rho=0.1)),
        ("Alt-Diff rho=1 (repo's value)", lambda th: altdiff_layer(th, Xt, kt, lot, upt, torch.full((n,), ETA), rho=1.0)),
    ]
    if with_repo:
        out.append(("repo classification/newlayer.py as is", repo))
    return out


def compare(T, n, with_repo=True, skip=()):
    X, theta_star, r, kappa, lo, up = make_data(T, n)
    Xt, rt, kt, lot, upt = map(torch.tensor, (X, r, kappa, lo, up))
    t0 = time.time()
    W_ref, g_ref, F = exact_grad(X, theta_star, r, kappa, lo, up)
    print(f"\n[T={T} dates, n={n} stocks, eta={ETA}] closed form {time.time() - t0:.3f}s, "
          f"{F.sum(1).mean():.1f} free assets per date, |grad| = {np.linalg.norm(g_ref):.1f}")
    print(f"{'layer':40s} {'time':>7s} {'max|w-w*|':>10s} {'grad relerr':>12s} {'cos':>9s}")
    for name, fn in layers(Xt, kt, lot, upt, n, T, with_repo):
        if name in skip:
            continue
        th = torch.tensor(theta_star, requires_grad=True)
        t0 = time.time()
        W = fn(th)
        (W * rt).sum().backward()
        dt = time.time() - t0
        g = th.grad.numpy()
        print(f"{name:40s} {dt:6.2f}s {np.abs(W.detach().numpy() - W_ref).max():10.1e} "
              f"{np.linalg.norm(g - g_ref) / np.linalg.norm(g_ref):12.1e} {g @ g_ref / np.linalg.norm(g) / np.linalg.norm(g_ref):9.6f}")


if __name__ == "__main__":
    # the reference itself: Clarabel on a few dates, finite differences of the realised PnL
    X, theta_star, r, kappa, lo, up = make_data(64, 50)
    W_ref, g_ref, _ = exact_grad(X, theta_star, r, kappa, lo, up)
    w = cp.Variable(50)
    err = 0
    for t in range(5):
        cp.Problem(cp.Maximize(X[t] @ theta_star @ w - kappa @ cp.abs(w) - 0.5 * ETA * cp.sum_squares(w)),
                   [cp.sum(w) == 0, w >= lo, w <= up]).solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
        err = max(err, np.abs(w.value - W_ref[t]).max())
    f = lambda th: (r * qp_exact(X @ th, kappa, ETA, lo, up)[0]).sum()
    fd = np.array([(f(theta_star + 1e-8 * e) - f(theta_star - 1e-8 * e)) / 2e-8 for e in np.eye(5)])
    print(f"closed form vs Clarabel: max |dw| = {err:.1e};  analytic gradient vs finite differences: "
          f"relerr = {np.linalg.norm(fd - g_ref) / np.linalg.norm(g_ref):.1e}")

    compare(64, 50)
    compare(32, 200, with_repo=False)
    compare(16, 500, with_repo=False, skip=("cvxpylayers, SCS default", "Alt-Diff rho=1 (repo's value)"))
