"""Test 1 - the LP:  max c'w - kappa'|w|  s.t. sum(w) = 0, lo <= w <= up,  c = X theta.

Checks the forward solution and d(sum_t r_t'w_t)/dtheta of every layer against the exact LP.
"""
import time
import warnings

import numpy as np
import torch

from problem import make_data, ic, lp_exact, lp_highs, qp_exact, qp_exact_vjp, cvx_layer, SCS_TIGHT
from altdiff import altdiff_layer

warnings.filterwarnings("ignore")

T, n, k = 16, 3000, 5
X, theta_star, r, kappa, lo, up = make_data(T, n, k)
Xt, rt, kt, lot, upt = map(torch.tensor, (X, r, kappa, lo, up))
C = X @ theta_star
print(f"{T} dates x {n} stocks; signal IC {ic(C, r):.3f}, single-factor IC "
      f"{', '.join(f'{ic(X[..., j], r):.3f}' for j in range(k))}")

W_lp = lp_exact(C, kappa, lo, up)
print(f"exact LP (breakpoint walk) vs HiGHS: max |dw| = {np.abs(W_lp[:3] - lp_highs(C[:3], kappa, lo, up)).max():.1e}")
at_bound = np.isclose(W_lp, up) | np.isclose(W_lp, lo)
zero = W_lp == 0
print(f"exact LP decisions: {at_bound.mean():.0%} at a bound, {zero.mean():.0%} zero, "
      f"{(~at_bound & ~zero).sum(1).mean():.1f} interior asset per date (the one closing sum(w)=0)")

# the exact gradient is 0: realised PnL of the exact LP decisions is piecewise constant in theta
f = lambda th: (r * lp_exact(X @ th, kappa, lo, up)).sum()
fd = [(f(theta_star + 1e-9 * e) - f(theta_star - 1e-9 * e)) / 2e-9 for e in np.eye(k)]
print(f"finite-difference gradient of the exact LP decisions: max |.| = {np.abs(fd).max():.1e}")
Wq, F = qp_exact(C, kappa, 0.3, lo, up)
g_qp = np.einsum("tnk,tn->k", X, qp_exact_vjp(F, 0.3, r))
print(f"(for scale: the same gradient through the QP with eta=0.3 has norm {np.linalg.norm(g_qp):.2f})\n")

layer = cvx_layer(n, kappa, lo, up)
cases = [
    ("cvxpylayers, SCS default", lambda th: layer(Xt @ th)[0]),
    ("cvxpylayers, SCS eps=1e-8", lambda th: layer(Xt @ th, solver_args=dict(SCS_TIGHT))[0]),
    ("Alt-Diff rho=1, 20000 it", lambda th: altdiff_layer(th, Xt, kt, lot, upt, rho=1.0, max_iter=20000, tol=1e-12)),
    ("Alt-Diff rho=1, stopped at 100 it", lambda th: altdiff_layer(th, Xt, kt, lot, upt, rho=1.0, max_iter=100, tol=0)),
    ("Alt-Diff rho=1, stopped at 1000 it", lambda th: altdiff_layer(th, Xt, kt, lot, upt, rho=1.0, max_iter=1000, tol=0)),
]
print(f"{'layer':40s} {'time':>7s} {'max|w-w*|':>10s} {'|grad| (exact 0)':>17s}")
for name, fn in cases:
    th = torch.tensor(theta_star, requires_grad=True)
    t0 = time.time()
    W = fn(th)
    (W * rt).sum().backward()
    print(f"{name:40s} {time.time() - t0:6.1f}s {np.abs(W.detach().numpy() - W_lp).max():10.1e} {th.grad.norm().item():17.1e}")
print("(bounds are 5-15bp = 5e-4..1.5e-3; classification/newlayer.py is not run at n=3000: it forms the full\n"
      " 6000x6000 dz/dq per date and loops over 12000 slack entries in Python every iteration)")
