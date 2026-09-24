"""Batched Alt-Diff for the rebalancing problem.

Same recursion as numerical_experiment/opt_layer.py and classification/newlayer.py, with the
changes that matter here:
  * forward-mode tangents are taken w.r.t. theta directly (k columns) instead of the full dz/dq,
  * the whole batch of dates is solved at once and P + rho(A'A + G'G) is factored once,
  * stopping uses primal/dual residuals plus an iteration cap, and rho is a parameter,
  * altdiff_box exploits the box/budget structure so that a step is O(n) (n = 3000 is fine).
"""
import torch


def altdiff_qp(P, q, dq, A, b, G, h, rho=0.1, max_iter=20000, tol=1e-10, check_every=10):
    """ADMM on  min 0.5 z'Pz + q'z  s.t.  Az = b, Gz <= h  (slack s >= 0), differentiating every
    step along the tangents dq = dq/dtheta.  q: (T, d), dq: (T, d, k).
    Returns z (T, d), dz/dtheta (T, d, k) and the number of iterations used."""
    T, d = q.shape
    R = -torch.cholesky_inverse(torch.linalg.cholesky(P + rho * (A.T @ A + G.T @ G)))
    z = q.new_zeros(T, d)
    s, nu = q.new_zeros(T, G.shape[0]), q.new_zeros(T, G.shape[0])
    lam = q.new_zeros(T, A.shape[0])
    dz = torch.zeros_like(dq)
    ds, dnu = dq.new_zeros(T, G.shape[0], dq.shape[-1]), dq.new_zeros(T, G.shape[0], dq.shape[-1])
    dlam = dq.new_zeros(T, A.shape[0], dq.shape[-1])
    const = -rho * (A.T @ b) - rho * (G.T @ h)
    for it in range(1, max_iter + 1):
        z = (q + lam @ A + nu @ G + rho * s @ G + const) @ R.T
        dz = R @ (dq + A.T @ dlam + G.T @ dnu + rho * G.T @ ds)
        s_new = torch.relu(-nu / rho - (z @ G.T - h))
        ds = -(1 / rho) * (s_new > 0).to(dq.dtype)[..., None] * (dnu + rho * G @ dz)
        lam = lam + rho * (z @ A.T - b)
        dlam = dlam + rho * (A @ dz)
        nu = nu + rho * (z @ G.T + s_new - h)
        dnu = dnu + rho * (G @ dz + ds)
        if it % check_every == 0:
            r_prim = max((z @ A.T - b).abs().max().item(), (z @ G.T + s_new - h).abs().max().item())
            r_dual = rho * ((s_new - s) @ G).abs().max().item()
            if max(r_prim, r_dual) < tol:
                s = s_new
                break
        s = s_new
    return z, dz, it


def split_problem(kappa, lo, up, eta=None):
    """Exact reformulation with w = p - m, p, m >= 0:
        min (kappa - c)'p + (kappa + c)'m + 0.5 eta'(p^2 + m^2)
        s.t. 1'p - 1'm = 0,  0 <= p <= max(up, 0),  0 <= m <= max(-lo, 0)
    At the optimum p_i m_i = 0 (kappa > 0), so eta'(p^2 + m^2) == eta'w^2 and the problem is the
    original one; eta=None gives the LP (P = 0).  Returns P, A, b, G, h."""
    n = len(kappa)
    if eta is None:
        P = torch.zeros(2 * n, 2 * n)
    else:
        eta = torch.as_tensor(eta).expand(n)
        P = torch.diag(torch.cat([eta, eta]))
    A = torch.cat([torch.ones(1, n), -torch.ones(1, n)], 1)
    b = torch.zeros(1)
    G = torch.cat([torch.eye(2 * n), -torch.eye(2 * n)])
    h = torch.cat([up.clamp(min=0), (-lo).clamp(min=0), torch.zeros(2 * n)])
    return P, A, b, G, h


def altdiff_box(q, dq, eta2, ub, rho=0.1, max_iter=20000, tol=1e-12, check_every=10):
    """altdiff_qp specialised to split_problem():  min 0.5 z'diag(eta2)z + q'z  s.t.  a'z = 0,
    0 <= z <= ub  with a = [1; -1] and G = [I; -I].  G z = [z; -z], G'v = v_top - v_bottom and
    P + rho(aa' + G'G) = diag(eta2 + 2 rho) + rho aa' is inverted with Sherman-Morrison, so each step
    costs O(T d k) instead of O(T d^2 k).  The iterates are the same as altdiff_qp's."""
    T, d = q.shape
    k = dq.shape[-1]
    a = torch.cat([q.new_ones(d // 2), -q.new_ones(d // 2)])
    Dinv = 1 / (eta2 + 2 * rho)
    Da = Dinv * a
    coef = rho / (1 + rho * (a * Da).sum())
    R2 = lambda v: -(Dinv * v - Da * (coef * (v @ Da))[:, None])
    R3 = lambda V: -(Dinv[:, None] * V - Da[:, None] * (coef * torch.einsum("d,tdk->tk", Da, V))[:, None, :])
    Gt = lambda v: v[:, :d] - v[:, d:]
    G = lambda v: torch.cat([v, -v], 1)
    h = torch.cat([ub, torch.zeros_like(ub)])
    z, lam = q.new_zeros(T, d), q.new_zeros(T)
    s, nu = q.new_zeros(T, 2 * d), q.new_zeros(T, 2 * d)
    dz, dlam = torch.zeros_like(dq), dq.new_zeros(T, k)
    ds, dnu = dq.new_zeros(T, 2 * d, k), dq.new_zeros(T, 2 * d, k)
    for it in range(1, max_iter + 1):
        z = R2(q + lam[:, None] * a + Gt(nu) + rho * Gt(s) - rho * ub)
        dz = R3(dq + a[None, :, None] * dlam[:, None, :] + Gt(dnu) + rho * Gt(ds))
        Gz, Gdz = G(z), G(dz)
        s_new = torch.relu(-nu / rho - (Gz - h))
        ds = -(1 / rho) * (s_new > 0).to(dq.dtype)[..., None] * (dnu + rho * Gdz)
        lam = lam + rho * (z @ a)
        dlam = dlam + rho * torch.einsum("d,tdk->tk", a, dz)
        nu = nu + rho * (Gz + s_new - h)
        dnu = dnu + rho * (Gdz + ds)
        if it % check_every == 0:
            r_prim = max((z @ a).abs().max().item(), (Gz + s_new - h).abs().max().item())
            r_dual = rho * Gt(s_new - s).abs().max().item()
            if max(r_prim, r_dual) < tol:
                s = s_new
                break
        s = s_new
    return z, dz, it


def altdiff_portfolio(theta, X, kappa, lo, up, eta=None, dense=False, **kw):
    """Decisions w (T, n) for c = X theta and their Jacobian dw/dtheta (T, n, k).
    dense=True runs the generic altdiff_qp on the same problem (for checking; O(n^2) per step)."""
    n = X.shape[1]
    c = X @ theta
    q = torch.cat([kappa - c, kappa + c], 1)
    dq = torch.cat([-X, X], 1)
    if dense:
        z, dz, it = altdiff_qp(*_dense_args(q, dq, kappa, lo, up, eta), **kw)
    else:
        eta2 = torch.zeros(2 * n) if eta is None else torch.as_tensor(eta).expand(n).repeat(2)
        z, dz, it = altdiff_box(q, dq, eta2, torch.cat([up.clamp(min=0), (-lo).clamp(min=0)]), **kw)
    return z[:, :n] - z[:, n:], dz[:, :n] - dz[:, n:], it


def _dense_args(q, dq, kappa, lo, up, eta):
    P, A, b, G, h = split_problem(kappa, lo, up, eta)
    return P, q, dq, A, b, G, h


class AltDiffLayer(torch.autograd.Function):
    """theta -> w(theta); backward is dw/dtheta^T grad_w, using the tangents from the forward pass."""

    @staticmethod
    def forward(ctx, theta, X, kappa, lo, up, eta, kw):
        W, dW, it = altdiff_portfolio(theta.detach(), X, kappa, lo, up, eta, **kw)
        ctx.save_for_backward(dW)
        ctx.iters = it
        return W

    @staticmethod
    def backward(ctx, grad_W):
        (dW,) = ctx.saved_tensors
        return torch.einsum("tn,tnk->k", grad_W, dW), None, None, None, None, None, None


def altdiff_layer(theta, X, kappa, lo, up, eta=None, **kw):
    return AltDiffLayer.apply(theta, X, kappa, lo, up, eta, kw)
