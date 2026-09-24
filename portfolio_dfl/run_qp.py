"""Test 2 - the QP:  max c'w - kappa'|w| - 0.5*eta*|w|^2  s.t. sum(w) = 0, lo <= w <= up,  c = X theta.

Reference: closed-form solution and Jacobian (problem.qp_exact / qp_exact_vjp), cross-checked against
Clarabel and finite differences.  Then every layer's decisions and d(sum_t r_t'w_t)/dtheta.
"""
import time
import warnings

import cvxpy as cp
import numpy as np
import torch

from problem import make_data, qp_exact, qp_exact_vjp, cvx_layer, SCS_TIGHT
from altdiff import altdiff_layer

warnings.filterwarnings("ignore")

T, n, ETA = 16, 3000, 0.3


def exact_grad(X, theta, r, kappa, lo, up, eta=ETA):
    W, F = qp_exact(X @ theta, kappa, eta, lo, up)
    return W, np.einsum("tnk,tn->k", X, qp_exact_vjp(F, eta, r)), F


X, theta_star, r, kappa, lo, up = make_data(T, n)
Xt, rt, kt, lot, upt = map(torch.tensor, (X, r, kappa, lo, up))

# the reference itself: Clarabel on two dates, finite differences of the realised PnL
t0 = time.time()
W_ref, g_ref, F = exact_grad(X, theta_star, r, kappa, lo, up)
t_closed = time.time() - t0
w = cp.Variable(n)
err = 0
for t in range(2):
    cp.Problem(cp.Maximize(X[t] @ theta_star @ w - kappa @ cp.abs(w) - 0.5 * ETA * cp.sum_squares(w)),
               [cp.sum(w) == 0, w >= lo, w <= up]).solve(solver=cp.CLARABEL, tol_gap_abs=1e-14, tol_gap_rel=1e-12, tol_feas=1e-12)
    err = max(err, np.abs(w.value - W_ref[t]).max())
f = lambda th: (r * qp_exact(X @ th, kappa, ETA, lo, up)[0]).sum()
fd = np.array([(f(theta_star + 1e-9 * e) - f(theta_star - 1e-9 * e)) / 2e-9 for e in np.eye(len(theta_star))])
print(f"closed form vs Clarabel: max |dw| = {err:.1e} (|w| <= 1.5e-3);  analytic gradient vs finite "
      f"differences: relerr = {np.linalg.norm(fd - g_ref) / np.linalg.norm(g_ref):.1e}")

cvx = cvx_layer(n, kappa, lo, up, ETA)
eta_t = torch.full((n,), ETA)
cases = [
    ("cvxpylayers, SCS default", lambda th: cvx(Xt @ th)[0]),
    ("cvxpylayers, SCS eps=1e-8", lambda th: cvx(Xt @ th, solver_args=dict(SCS_TIGHT))[0]),
    ("Alt-Diff rho=1", lambda th: altdiff_layer(th, Xt, kt, lot, upt, eta_t, rho=1.0)),
    ("Alt-Diff rho=0.1", lambda th: altdiff_layer(th, Xt, kt, lot, upt, eta_t, rho=0.1)),
]
print(f"\n[T={T} dates, n={n} stocks, eta={ETA}] closed form {t_closed:.2f}s, "
      f"{F.sum(1).mean():.0f} free assets per date, |grad| = {np.linalg.norm(g_ref):.2f}")
print(f"{'layer':32s} {'time':>7s} {'max|w-w*|':>10s} {'grad relerr':>12s} {'cos':>9s}")
for name, fn in cases:
    th = torch.tensor(theta_star, requires_grad=True)
    t0 = time.time()
    W = fn(th)
    (W * rt).sum().backward()
    dt = time.time() - t0
    g = th.grad.numpy()
    print(f"{name:32s} {dt:6.1f}s {np.abs(W.detach().numpy() - W_ref).max():10.1e} "
          f"{np.linalg.norm(g - g_ref) / np.linalg.norm(g_ref):12.1e} {g @ g_ref / np.linalg.norm(g) / np.linalg.norm(g_ref):9.6f}")
print("(qpth and classification/newlayer.py are not run at n=3000: dense 6000-variable / 12000-constraint\n"
      " KKT systems per date; qpth already needed 35s for 16 dates at n=500)")
