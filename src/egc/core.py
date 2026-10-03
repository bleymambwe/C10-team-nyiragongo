"""Effect-Gramian Certificates (EGC).

Implements the objects defined in docs/01_THEORY.md:

  Definition 1  adjoint covector          g_l^{(i)}
  Definition 2  effect Gramian            W_g(l)
  Definition 4  probe direction           w_probe
  Definition 5  steerability certificate  Sbar, S2
  Definition 6  effect participation rank r_eff
  Definition 7  dark direction            vstar   (Proposition 5)
  Proposition 4 probe/adjoint alignment   rho
  Proposition 6 usable steering radius    astar

Everything here is framework-agnostic numpy; torch enters only in the
model adapters (see egc.adapters).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------
# Read geometry: probes
# --------------------------------------------------------------------------

def fisher_probe(X: np.ndarray, y: np.ndarray, gamma: float = 1e-3) -> np.ndarray:
    """Definition 4: w_probe ~ (Sigma + gamma I)^{-1} delta, unit norm.

    X : (N, d) representations at one layer.
    y : (N,) binary labels.
    gamma is relative to mean eigenvalue of Sigma so it is scale-free.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    X1, X0 = X[y == 1], X[y == 0]
    if len(X1) == 0 or len(X0) == 0:
        raise ValueError("fisher_probe needs both classes present")
    mu1, mu0 = X1.mean(0), X0.mean(0)
    delta = mu1 - mu0
    Xc = np.concatenate([X1 - mu1, X0 - mu0], axis=0)
    Sigma = (Xc.T @ Xc) / max(len(Xc) - 2, 1)
    ridge = gamma * (np.trace(Sigma) / X.shape[1] + EPS)
    w = np.linalg.solve(Sigma + ridge * np.eye(X.shape[1]), delta)
    return unit(w)


def diff_of_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The ActAdd / CAA steering vector: mu_1 - mu_0, unit norm.

    This -- not the Fisher probe -- is what the activation-steering literature
    actually deploys, so it is the baseline the certificate must be compared
    against. Note it uses only the class means and ignores Sigma entirely, so
    it is a *different* read-side object again, and equally uninformed about
    the write geometry.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    return unit(X[y == 1].mean(0) - X[y == 0].mean(0))


def logistic_probe(X: np.ndarray, y: np.ndarray, l2: float = 1e-3,
                   steps: int = 400, lr: float = 0.5) -> np.ndarray:
    """Standardised logistic-regression probe direction (the most common probe).

    Plain full-batch gradient descent on standardised features; returned in the
    original coordinates, unit norm.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + EPS
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1])
    b = 0.0
    n = len(Z)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        gw = Z.T @ (p - y) / n + l2 * w
        gb = float((p - y).mean())
        w -= lr * gw
        b -= lr * gb
    return unit(w / sd)


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Exact rank-based ROC AUC with tie correction. No sklearn dependency."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(y).astype(int).ravel()
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0  # average of 1-based ranks
        ranks[order[i:j + 1]] = avg
        i = j + 1
    r1 = ranks[y == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def direction_auc(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """AUC obtained by reading out along direction w."""
    return auc(np.asarray(X, dtype=np.float64) @ unit(w), y)


# --------------------------------------------------------------------------
# Write geometry: effect Gramian and certificates
# --------------------------------------------------------------------------

def effect_gramian(G: np.ndarray) -> np.ndarray:
    """Definition 2: W_g = (1/N) sum_i g_i g_i^T from stacked covectors G (N,d)."""
    G = np.asarray(G, dtype=np.float64)
    return (G.T @ G) / G.shape[0]


def gbar(G: np.ndarray) -> np.ndarray:
    """Mean adjoint covector."""
    return np.asarray(G, dtype=np.float64).mean(0)


def S_bar(G: np.ndarray, w: np.ndarray) -> float:
    """Definition 5: coherent certificate |gbar^T w| / ||w||."""
    return float(abs(gbar(G) @ unit(w)))


def S_two(G: np.ndarray, w: np.ndarray) -> float:
    """Definition 5: RMS certificate sqrt(w^T W_g w) / ||w||."""
    u = unit(w)
    Gu = np.asarray(G, dtype=np.float64) @ u
    return float(np.sqrt(np.mean(Gu ** 2)))


def signed_S_bar(G: np.ndarray, w: np.ndarray) -> float:
    """Signed version; the sign says which way behaviour moves."""
    return float(gbar(G) @ unit(w))


def effect_spectrum(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of W_g, eigenvalues descending."""
    W = effect_gramian(G)
    vals, vecs = np.linalg.eigh(W)
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx]


def r_eff(G: np.ndarray) -> float:
    """Definition 6: effect participation rank tr(W_g)/lambda_1."""
    vals, _ = effect_spectrum(G)
    lam1 = max(float(vals[0]), EPS)
    return float(vals.sum() / lam1)


def stable_rank_ratio(G: np.ndarray) -> float:
    """Proposition 3: r_eff/d, the steering power a random direction retains."""
    return r_eff(G) / G.shape[1]


# --------------------------------------------------------------------------
# Alignment, dark directions
# --------------------------------------------------------------------------

def rho_alignment(G: np.ndarray, w_probe: np.ndarray) -> float:
    """Proposition 4: |cos angle(w_probe, gbar)|."""
    return float(abs(unit(gbar(G)) @ unit(w_probe)))


def dark_direction(G: np.ndarray, O_basis: np.ndarray) -> np.ndarray:
    """Proposition 5: vstar = (I - P_O) gbar, normalised.

    O_basis : (d, s) columns spanning the discriminative subspace O(l).
              Need not be orthonormal; it is orthonormalised here.
    """
    Q = orthonormalise(np.asarray(O_basis, dtype=np.float64))
    g = gbar(G)
    v = g - Q @ (Q.T @ g)
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("gbar lies inside the discriminative subspace: no dark direction")
    return v / n


def best_steer_direction(G: np.ndarray) -> np.ndarray:
    """Maximiser of Sbar: gbar / ||gbar||."""
    return unit(gbar(G))


def top_effect_eigvec(G: np.ndarray) -> np.ndarray:
    """Maximiser of S2: leading eigenvector of W_g."""
    _, vecs = effect_spectrum(G)
    return unit(vecs[:, 0])


# --------------------------------------------------------------------------
# Curvature and validity radius
# --------------------------------------------------------------------------

def estimate_curvature(effect_fn, x, w, alphas) -> float:
    """Estimate M_l along the ray x + a w by second differences.

    effect_fn(a) -> scalar behaviour Phi(x + a w).
    Returns max |second difference| / h^2 over the grid, i.e. an empirical
    bound on the directional second derivative for unit ||w||.
    """
    a = np.asarray(sorted(alphas), dtype=np.float64)
    vals = np.array([effect_fn(float(t)) for t in a], dtype=np.float64)
    curv = []
    for k in range(1, len(a) - 1):
        h1, h2 = a[k] - a[k - 1], a[k + 1] - a[k]
        if min(abs(h1), abs(h2)) < EPS:
            continue
        # non-uniform second difference
        d2 = 2.0 * (h2 * vals[k - 1] - (h1 + h2) * vals[k] + h1 * vals[k + 1]) / (h1 * h2 * (h1 + h2))
        curv.append(abs(d2))
    return float(max(curv)) if curv else 0.0


def usable_radius(G: np.ndarray, w: np.ndarray, M: float) -> float:
    """Proposition 6: astar = 2 |gbar^T w| / (M ||w||^2), for unit w."""
    s = S_bar(G, w)
    if M <= EPS:
        return float("inf")
    return float(2.0 * s / M)


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    return v / n if n > EPS else v


def orthonormalise(A: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the column space of A (d, s)."""
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    if A.shape[0] == 1 and A.shape[1] > 1:
        A = A.T
    Q, R = np.linalg.qr(A)
    keep = np.abs(np.diag(R)) > EPS * max(A.shape)
    return Q[:, keep] if keep.any() else Q[:, :0]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, tie-corrected, no scipy dependency."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > EPS else float("nan")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > EPS else float("nan")


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def bootstrap_ci(fn, n: int, reps: int = 400, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for a statistic computed on index subsets."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals = np.sort(np.array(vals))
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return lo, hi
