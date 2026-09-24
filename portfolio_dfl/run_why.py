"""Test 4 - why is decision-focused learning (DFL) not better than two-stage OLS here?

Every PnL below is the *expected* daily net PnL on the test dates, computed with the true alpha
E[r | X] (noise-free evaluation), in bp of NAV.

E1  training/deployment mismatch: value of the LP and QP(eta) policies along theta = a * theta*,
    and OLS vs DFL(eta) deployed both ways.
E2  statistical efficiency: 8 independent 40-date panels, OLS vs DFL.
E3  misspecification: two ways the linear model can be wrong, OLS vs DFL on 4 independent panels.
"""
import time
import warnings

import numpy as np
import torch

from problem import make_data, lp_exact, qp_exact, qp_exact_vjp, net_pnl, ols

warnings.filterwarnings("ignore")
bps = lambda v: 1e4 * v

X, theta_star, r, kappa, lo, up = make_data(250 * 5)
noise = r - X @ theta_star
kt = torch.tensor(kappa)
Xte = X[-250:]


class ClosedFormQP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, Xb, eta):
        W, F = qp_exact(Xb @ theta.detach().numpy(), kappa, eta, lo, up)
        ctx.Xb, ctx.F, ctx.eta = Xb, F, eta
        return torch.tensor(W)

    @staticmethod
    def backward(ctx, grad_W):
        return torch.tensor(np.einsum("tnk,tn->k", ctx.Xb, qp_exact_vjp(ctx.F, ctx.eta, grad_W.numpy()))), None, None


def dfl(Xs, rs, theta0, eta=0.1, steps=300, lr=0.05, batch=32, seed=0):
    """Maximise realised net PnL of the QP(eta) decisions; returns the average of the last 100 iterates."""
    phi = torch.tensor(theta0 / 1e-3, requires_grad=True)
    opt = torch.optim.Adam([phi], lr=lr)
    g = np.random.default_rng(seed)
    tail = []
    for step in range(steps):
        idx = g.integers(0, len(Xs), batch) if batch else np.arange(len(Xs))
        opt.zero_grad()
        W = ClosedFormQP.apply(phi * 1e-3, Xs[idx], eta)
        (-net_pnl(W, torch.tensor(rs[idx]), kt).mean()).backward()
        opt.step()
        if step >= steps - 100:
            tail.append(phi.detach().numpy() * 1e-3)
    return np.mean(tail, 0)


def expected(W, S):
    return bps((S * W - kappa * abs(W)).sum(1).mean())


def policy(eta):
    if eta is None:
        return lambda C: lp_exact(C, kappa, lo, up)
    return lambda C: qp_exact(C, kappa, eta, lo, up)[0]


def angle(a):
    return np.degrees(np.arccos(np.clip(a @ theta_star / np.linalg.norm(a) / np.linalg.norm(theta_star), -1, 1)))


# ---------------------------------------------------------------- E1
S_te = Xte @ theta_star
alphas = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5]
print("E1a. expected test PnL of each policy for theta = a * theta*")
print("        " + " ".join(f"{a:6.2f}" for a in alphas) + "   best a")
for eta in (None, 0.1, 0.3, 1.0):
    v = [expected(policy(eta)(Xte @ (a * theta_star)), S_te) for a in alphas]
    print(f"{'LP' if eta is None else f'QP {eta}':7s} " + " ".join(f"{x:6.2f}" for x in v) + f"   {alphas[int(np.argmax(v))]}")

print("\nE1b. OLS vs DFL(eta) trained on 250 dates, each deployed as its training policy QP(eta) and as the LP")
t_ols = ols(X[:250], r[:250])
for eta in (0.1, 0.3, 1.0):
    t_dfl = dfl(X[:250], r[:250], t_ols, eta)
    print(f"eta={eta:<4} DFL scale {np.linalg.norm(t_dfl) / np.linalg.norm(theta_star):.2f} (OLS "
          f"{np.linalg.norm(t_ols) / np.linalg.norm(theta_star):.2f}) | deploy QP(eta): OLS "
          f"{expected(policy(eta)(Xte @ t_ols), S_te):6.2f}  DFL {expected(policy(eta)(Xte @ t_dfl), S_te):6.2f} | "
          f"deploy LP: OLS {expected(policy(None)(Xte @ t_ols), S_te):6.2f}  DFL {expected(policy(None)(Xte @ t_dfl), S_te):6.2f}")

# ---------------------------------------------------------------- E2
print("\nE2. 8 independent 40-date panels (well-specified), DFL with eta=0.1 started from OLS, full batch, lr=0.02")
t0 = time.time()
rows = []
for w in range(8):
    Xs, rs = X[40 * w:40 * (w + 1)], r[40 * w:40 * (w + 1)]
    a, b = ols(Xs, rs), None
    b = dfl(Xs, rs, a, eta=0.1, lr=0.02, batch=None)
    insample = [bps(net_pnl(qp_exact(Xs @ t, kappa, 0.1, lo, up)[0], rs, kappa).mean()) for t in (a, b)]
    rows.append([angle(a), angle(b), expected(policy(None)(Xte @ a), S_te), expected(policy(None)(Xte @ b), S_te),
                 *insample, qp_exact(Xs @ b, kappa, 0.1, lo, up)[1].mean()])
R = np.array(rows)
m, s = R.mean(0), R.std(0)
print(f"  direction error:                          OLS {m[0]:4.1f} +- {s[0]:.1f} deg   DFL {m[1]:4.1f} +- {s[1]:.1f} deg")
print(f"  expected test LP PnL:                     OLS {m[2]:5.2f} +- {s[2]:.2f}     DFL {m[3]:5.2f} +- {s[3]:.2f}"
      f"   (DFL - OLS per panel {np.round(R[:, 3] - R[:, 2], 2)})")
print(f"  in-sample realised QP objective (DFL's target): OLS {m[4]:5.2f}   DFL {m[5]:5.2f}")
print(f"  (date, stock) pairs that carry gradient:  {m[6]:.1%}      [{time.time() - t0:.0f}s]")

# ---------------------------------------------------------------- E3
print("\nE3. misspecified truth (the linear model c = X theta cannot represent E[r|X]); 4 independent 250-date panels")
j = int(np.argmax(theta_star))
cheap = kappa <= np.median(kappa)
Z = X @ theta_star
scenarios = {
    f"A: factor {j} is -1x in cheap, +3x in costly stocks": Z + (np.where(cheap, -1.0, 3.0) - 1) * theta_star[j] * X[..., j],
    "B: alpha saturates, 12bp * tanh(z / 12bp)": 12e-4 * np.tanh(Z / 12e-4),
}
for name, S in scenarios.items():
    scale = np.std(r) / np.std(S + noise)                    # keep the return dispersion at ~2%
    S, rr = S * scale, (S + noise) * scale
    t0 = time.time()
    out = []
    for w in range(4):
        sl = slice(250 * w, 250 * (w + 1))
        a = ols(X[sl], rr[sl])
        b = dfl(X[sl], rr[sl], a, eta=0.1, seed=w)
        out.append((expected(policy(None)(Xte @ a), S[-250:]), expected(policy(None)(Xte @ b), S[-250:])))
    o = np.array(out)
    print(f"  {name}: expected test LP PnL OLS {o[:, 0].mean():5.2f}  DFL {o[:, 1].mean():5.2f}  "
          f"(DFL - OLS per panel {np.round(o[:, 1] - o[:, 0], 2)})  [{time.time() - t0:.0f}s]")
