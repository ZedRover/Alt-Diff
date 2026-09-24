"""Rebalancing problem shared by the portfolio experiments.

    max_w  c'w - kappa'|w|  [- 0.5 * sum_i eta_i * w_i^2]    s.t.  sum(w) = 0,  lo <= w <= up
    c = X theta   (X: stock factors of one date, theta: weights of the linear return model)

Without the bracketed term this is an LP; with it (eta > 0) it is a QP.
"""
import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

torch.set_default_dtype(torch.float64)

# SCS defaults are too loose for these problems (see run_lp.py / run_qp.py)
SCS_TIGHT = {"eps": 1e-8, "max_iters": 100000}


def make_data(T, n=50, k=5, seed=0, alpha_std=0.003, noise_std=0.02):
    """Synthetic panel: factors X (T, n, k), true theta, realised returns r (T, n),
    proportional costs kappa (5-20 bp) and per-stock trade limits lo < 0 < up."""
    g = np.random.default_rng(seed)
    X = g.standard_normal((T, n, k))
    theta_star = g.standard_normal(k)
    theta_star *= alpha_std / np.linalg.norm(theta_star)
    r = X @ theta_star + noise_std * g.standard_normal((T, n))
    kappa = g.uniform(5e-4, 2e-3, n)
    up = g.uniform(0.01, 0.03, n)
    lo = -g.uniform(0.01, 0.03, n)
    return X, theta_star, r, kappa, lo, up


def lp_highs(C, kappa, lo, up):
    """LP decisions from HiGHS (feasibility tolerance ~1e-7), one problem per row of C."""
    n = C.shape[1]
    w, c = cp.Variable(n), cp.Parameter(n)
    prob = cp.Problem(cp.Maximize(c @ w - kappa @ cp.abs(w)), [cp.sum(w) == 0, w >= lo, w <= up])
    out = []
    for ci in C:
        c.value = ci
        prob.solve(solver=cp.HIGHS)
        out.append(w.value)
    return np.array(out)


def lp_exact(C, kappa, lo, up):
    """Exact LP decisions without solver tolerance.  For a multiplier mu of sum(w) = 0 each asset
    sits at up (c_i - mu > kappa_i), 0 (|c_i - mu| < kappa_i) or lo (c_i - mu < -kappa_i); lowering mu
    from +inf moves asset i from lo to 0 at c_i + kappa_i and from 0 to up at c_i - kappa_i.  Walk the
    breakpoints until sum(w) reaches 0; the asset at that breakpoint takes the fractional amount."""
    C = np.atleast_2d(C)
    T, n = C.shape
    W = np.empty((T, n))
    for t, c in enumerate(C):
        bp = np.r_[c + kappa, c - kappa]            # breakpoints, lo->0 then 0->up
        jump = np.r_[-lo, up]                         # increase of sum(w) when mu passes below
        order = np.argsort(-bp, kind="stable")
        total = lo.sum() + np.cumsum(jump[order])     # sum(w) just below each breakpoint
        j = np.searchsorted(total, 0.0)               # first breakpoint where sum(w) >= 0
        w = lo.copy()
        for idx in order[:j]:                         # fully passed breakpoints
            i = idx % n
            w[i] = 0.0 if idx < n else up[i]
        i = order[j] % n                              # marginal asset closes the budget exactly
        w[i] -= (total[j - 1] if j > 0 else lo.sum())
        W[t] = w
    return W


def qp_exact(C, kappa, eta, lo, up, iters=200):
    """Closed-form QP decisions: w_i = clip(soft(c_i - mu, kappa_i) / eta_i, lo_i, up_i),
    with the multiplier mu of sum(w) = 0 found by bisection and then solved exactly on the free set.
    Returns W (T, n) and the free-set mask (assets that are neither 0 nor at a bound)."""
    C = np.atleast_2d(C)
    eta = np.broadcast_to(eta, C.shape[1])

    def w_of(mu):
        a = C - mu[:, None]
        return np.clip(np.sign(a) * np.maximum(np.abs(a) - kappa, 0) / eta, lo, up)

    span = np.abs(C).max() + kappa.max() + (eta * np.maximum(up, -lo)).max() + 1
    mlo, mhi = np.full(len(C), -span), np.full(len(C), span)
    for _ in range(iters):
        mid = 0.5 * (mlo + mhi)
        pos = w_of(mid).sum(1) > 0
        mlo, mhi = np.where(pos, mid, mlo), np.where(pos, mhi, mid)
    mu = 0.5 * (mlo + mhi)
    W = w_of(mu)
    free = (W != 0) & (W > lo) & (W < up)
    # exact mu on the free set:  sum_F (c_i - mu - kappa_i sgn_i) / eta_i + sum_notF w_i = 0
    d = free / eta
    has = d.sum(1) > 0
    num = (d * (C - kappa * np.sign(W))).sum(1) + np.where(free, 0, W).sum(1)
    mu = np.where(has, num / np.where(has, d.sum(1), 1), mu)
    W = np.where(free, (C - mu[:, None] - kappa * np.sign(W)) / eta, W)
    return W, free


def qp_exact_vjp(free, eta, G):
    """J^T g for J = dw/dc = diag(d_F) - d_F d_F^T / sum(d_F),  d = 1/eta on the free set F."""
    d = free / np.broadcast_to(eta, free.shape[1])
    S = d.sum(1, keepdims=True)
    return d * G - d * (d * G).sum(1, keepdims=True) / np.where(S > 0, S, 1)


def net_pnl(W, R, kappa):
    """Realised PnL of each date net of proportional costs (the objective of the original problem)."""
    return (R * W).sum(-1) - (kappa * abs(W)).sum(-1)


def cvx_layer(n, kappa, lo, up, eta=None):
    """cvxpylayers layer  c -> w  for the LP (eta=None) or the QP."""
    w, c = cp.Variable(n), cp.Parameter(n)
    obj = c @ w - kappa @ cp.abs(w)
    if eta is not None:
        obj = obj - 0.5 * cp.sum(cp.multiply(np.broadcast_to(eta, n), cp.square(w)))
    prob = cp.Problem(cp.Maximize(obj), [cp.sum(w) == 0, w >= lo, w <= up])
    assert prob.is_dpp()
    return CvxpyLayer(prob, parameters=[c], variables=[w])


def ols(X, r):
    """Two-stage baseline: pooled least squares for theta."""
    return np.linalg.lstsq(X.reshape(-1, X.shape[-1]), r.ravel(), rcond=None)[0]
