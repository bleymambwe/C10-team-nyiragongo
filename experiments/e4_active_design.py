"""E4 -- Active intervention design (prediction P5).

Finding a causally strong steering direction normally costs one intervention
sweep per candidate. The certificate is free: it needs adjoints already
computed for the probe-training data and no forward interventions at all.

Question: given a budget of B intervention measurements out of M candidates,
which ordering finds the strongest direction fastest?

Strategies
  oracle       descending TRUE measured effect            (upper bound)
  certificate  descending Sbar                            (ours, free prior)
  probe_auc    descending |AUC - 1/2|                     (what probing implies)
  random       uniform random order                       (control)

Reported: best-true-effect-found vs budget, and recall of the true top-5.
Ground truth is obtained by measuring all M candidates once, offline.

Run:  python experiments/e4_active_design.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from egc.adapters import LMAdapter  # noqa: E402
from egc.core import (  # noqa: E402
    S_bar, direction_auc, effect_spectrum, fisher_probe, gbar, spearman, unit,
)
from egc.toxdata import NEUTRAL_WORDS, TOXIC_WORDS, build  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "data")
os.makedirs(RESULTS, exist_ok=True)

MODEL = os.environ.get("EGC_MODEL", "gpt2")
M_CAND = 64
N_PROMPTS = 200
N_MEAS = 48
ALPHA_FRAC = 0.10
N_RANDOM_REPS = 400


def candidates(Xtr, ytr, Gtr, m, seed=0):
    """A candidate pool that is realistic rather than rigged: random directions,
    probe/adjoint mixtures, effect eigenvectors, and sparse coordinate writes."""
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    wp = fisher_probe(Xtr, ytr, gamma=1e-2)
    wg = unit(gbar(Gtr))
    _, vecs = effect_spectrum(Gtr)
    pool = [("probe", wp), ("gbar", wg)]
    for k in range(6):
        pool.append((f"eig{k}", unit(vecs[:, k])))
    for t in np.linspace(0.1, 0.9, 6):
        pool.append((f"mix{t:.1f}", unit(t * wp + (1 - t) * wg)))
    # sparse writes into single residual coordinates
    for j in rng.choice(d, size=12, replace=False):
        e = np.zeros(d); e[int(j)] = 1.0
        pool.append((f"coord{int(j)}", e))
    while len(pool) < m:
        pool.append((f"rand{len(pool)}", unit(rng.normal(size=d))))
    return pool[:m]


def budget_curves(order, truth, budgets):
    """Best true effect found after B measurements, following `order`."""
    best = []
    for B in budgets:
        idx = order[:B]
        best.append(float(np.max(truth[idx])) if len(idx) else 0.0)
    return best


def recall_at_k(order, truth, k, budgets):
    top = set(np.argsort(truth)[::-1][:k].tolist())
    return [len(top & set(order[:B].tolist())) / k for B in budgets]


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    lm = LMAdapter(MODEL)
    pos, neg = lm.token_ids(TOXIC_WORDS), lm.token_ids(NEUTRAL_WORDS)
    print(f"{MODEL}: layers={lm.n_layers} d={lm.d_model}")

    prompts, y, s = build(N_PROMPTS, spurious=0.85, seed=21)
    n = len(prompts)
    rng = np.random.default_rng(21)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    X, G, _ = lm.states_and_adjoints(prompts, pos, neg, batch=16)
    p_meas = [prompts[i] for i in te[:N_MEAS]]
    base = lm.baseline_phi(p_meas, pos, neg)
    print(f"adjoints done ({time.time()-t0:.0f}s)")

    target_layers = [lm.n_layers // 4, lm.n_layers // 2, 3 * lm.n_layers // 4]
    out = {"experiment": "e4_active_design", "model": MODEL,
           "M_cand": M_CAND, "N_meas": N_MEAS, "layers": []}
    path = os.path.join(RESULTS, "e4_active_design.json")

    def checkpoint():
        """Write partial results after every layer.

        The first attempt at this suite wrote JSON only at the end, so stopping
        it discarded hours of completed work. Never again.
        """
        out["partial"] = len(out["layers"]) < len(target_layers)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    for L in target_layers:
        Xl, Gl = X[L], G[L]
        Xtr, ytr, Gtr = Xl[tr], y[tr], Gl[tr]
        alpha = ALPHA_FRAC * float(np.linalg.norm(Xl, axis=1).mean())
        pool = candidates(Xtr, ytr, Gtr, M_CAND, seed=L)

        sbar = np.array([S_bar(Gtr, w) for _, w in pool])
        aucs = np.array([direction_auc(Xl[te], y[te], w) for _, w in pool])
        truth = np.array([
            float(np.abs(lm.measure_steering(p_meas, L, w, alpha, pos, neg,
                                             base=base)).mean())
            for _, w in pool
        ])
        print(f"  L{L}: measured all {M_CAND} candidates ({time.time()-t0:.0f}s)")

        budgets = list(range(1, M_CAND + 1))
        ord_cert = np.argsort(sbar)[::-1]
        ord_auc = np.argsort(np.abs(aucs - 0.5))[::-1]
        ord_oracle = np.argsort(truth)[::-1]

        rand_best = np.zeros(len(budgets))
        rand_rec = np.zeros(len(budgets))
        for r in range(N_RANDOM_REPS):
            o = np.random.default_rng(1000 + r).permutation(M_CAND)
            rand_best += np.array(budget_curves(o, truth, budgets))
            rand_rec += np.array(recall_at_k(o, truth, 5, budgets))
        rand_best /= N_RANDOM_REPS
        rand_rec /= N_RANDOM_REPS

        tmax = float(truth.max())

        def frac(c):
            return [v / tmax if tmax > 0 else float("nan") for v in c]

        rec = {
            "layer": L, "alpha": alpha,
            "spearman_Sbar_vs_truth": spearman(sbar, truth),
            "spearman_absauc_vs_truth": spearman(np.abs(aucs - 0.5), truth),
            "budgets": budgets,
            "best_found_frac": {
                "oracle": frac(budget_curves(ord_oracle, truth, budgets)),
                "certificate": frac(budget_curves(ord_cert, truth, budgets)),
                "probe_auc": frac(budget_curves(ord_auc, truth, budgets)),
                "random": frac(rand_best.tolist()),
            },
            "recall_at_5": {
                "oracle": recall_at_k(ord_oracle, truth, 5, budgets),
                "certificate": recall_at_k(ord_cert, truth, 5, budgets),
                "probe_auc": recall_at_k(ord_auc, truth, 5, budgets),
                "random": rand_rec.tolist(),
            },
            "candidate_names": [nm for nm, _ in pool],
            "Sbar": sbar.tolist(), "auc": aucs.tolist(), "truth": truth.tolist(),
        }

        # budget needed to reach 90% / 99% of the best attainable effect
        for thr in (0.90, 0.99):
            rec[f"budget_to_{int(thr*100)}pct"] = {}
            for k, curve in rec["best_found_frac"].items():
                hit = next((b for b, v in zip(budgets, curve) if v >= thr), None)
                rec[f"budget_to_{int(thr*100)}pct"][k] = hit
        out["layers"].append(rec)
        checkpoint()

        b90 = rec["budget_to_90pct"]
        print(f"    spearman(Sbar,truth)={rec['spearman_Sbar_vs_truth']:+.3f}  "
              f"spearman(|AUC-.5|,truth)={rec['spearman_absauc_vs_truth']:+.3f}")
        print(f"    budget to 90% of best effect: certificate={b90['certificate']} "
              f"probe_auc={b90['probe_auc']} random={b90['random']} oracle={b90['oracle']}")

    out["wall_time_s"] = time.time() - t0
    checkpoint()
    print(f"\nwrote {path}  ({out['wall_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
