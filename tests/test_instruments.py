"""Validate the measurement instruments against reference implementations.

Every claim in this repository is a number produced by egc.core. If auc() or
fisher_probe() were subtly wrong, every downstream result would be wrong in a
way no amount of experimental care would catch. These tests pin them to
sklearn/scipy.

Run:  python tests/test_instruments.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from egc.core import (  # noqa: E402
    auc, dark_direction, fisher_probe, pearson, r_eff, spearman, unit,
)

TOL_EXACT = 1e-12


def test_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    worst = 0.0
    for t in range(200):
        n = int(rng.integers(20, 300))
        y = rng.integers(0, 2, size=n)
        if y.min() == y.max():
            continue
        s = rng.normal(size=n)
        if t % 3 == 0:
            s = np.round(s)          # heavy ties
        if t % 7 == 0:
            s = np.zeros(n)          # degenerate: all tied
        worst = max(worst, abs(auc(s, y) - roc_auc_score(y, s)))
    assert worst < TOL_EXACT, f"AUC deviates by {worst}"
    return worst


def test_rank_correlations_match_scipy():
    from scipy.stats import pearsonr, spearmanr
    rng = np.random.default_rng(1)
    ws = wp = 0.0
    for t in range(200):
        n = int(rng.integers(10, 200))
        a, b = rng.normal(size=n), rng.normal(size=n)
        if t % 3 == 0:
            a = np.round(a)
        ws = max(ws, abs(spearman(a, b) - spearmanr(a, b).statistic))
        wp = max(wp, abs(pearson(a, b) - pearsonr(a, b).statistic))
    assert ws < 1e-10 and wp < 1e-10, f"spearman {ws}, pearson {wp}"
    return ws, wp


def test_fisher_probe_matches_lda():
    """Under shared covariance the Fisher probe direction is the LDA direction."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    rng = np.random.default_rng(2)
    worst = 1.0
    for _ in range(30):
        d, n = 20, 4000
        y = rng.integers(0, 2, size=n)
        A = rng.normal(size=(d, d))
        S = A @ A.T / d + np.eye(d) * 0.5
        L = np.linalg.cholesky(S)
        mu = rng.normal(size=d) * 0.6
        X = rng.normal(size=(n, d)) @ L.T + np.outer(y, mu)
        mine = fisher_probe(X, y, gamma=1e-9)
        ref = unit(LinearDiscriminantAnalysis(solver="lsqr").fit(X, y).coef_.ravel())
        worst = min(worst, abs(mine @ ref))
    assert worst > 0.9999, f"probe direction cos {worst}"
    return worst


def test_dark_direction_is_orthogonal():
    rng = np.random.default_rng(3)
    G = rng.normal(size=(500, 30))
    O = rng.normal(size=(30, 3))
    v = dark_direction(G, O)
    Q = np.linalg.qr(O)[0]
    leak = float(np.abs(Q.T @ v).max())
    assert leak < TOL_EXACT, f"dark direction leaks {leak} into O"
    assert abs(np.linalg.norm(v) - 1) < TOL_EXACT
    return leak


def test_r_eff_bounds():
    rng = np.random.default_rng(4)
    for rank in (1, 5, 30):
        B = rng.normal(size=(30, rank))
        G = rng.normal(size=(2000, rank)) @ B.T
        re = r_eff(G)
        assert 1 - 1e-9 <= re <= 30 + 1e-9, f"r_eff {re} outside [1, d]"
    # a rank-1 effect Gramian must have r_eff exactly 1
    g = rng.normal(size=30)
    G1 = rng.normal(size=(1000, 1)) * g[None, :]
    assert abs(r_eff(G1) - 1.0) < 1e-8, r_eff(G1)
    return True


if __name__ == "__main__":
    checks = [
        ("auc vs sklearn (incl. ties)", test_auc_matches_sklearn),
        ("spearman/pearson vs scipy", test_rank_correlations_match_scipy),
        ("fisher probe vs sklearn LDA", test_fisher_probe_matches_lda),
        ("dark direction orthogonality", test_dark_direction_is_orthogonal),
        ("r_eff bounds and rank-1 case", test_r_eff_bounds),
    ]
    failed = 0
    for name, fn in checks:
        try:
            v = fn()
            print(f"  PASS  {name:<34} {v}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name:<34} {e}")
    print(f"\n{len(checks) - failed}/{len(checks)} instrument checks pass")
    sys.exit(1 if failed else 0)
