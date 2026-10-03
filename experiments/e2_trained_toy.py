"""E2 -- The probe/steer gap in TRAINED Transformers.

E1 built the gap by hand. E2 asks whether gradient descent produces it on its
own, and whether the certificate predicts measured steering in a model with
LayerNorm, attention, GELU MLPs and a genuinely nonlinear tail map.

Dial: `spurious` = P(surface marker class == causal content class). Real
toxicity data sits near the high end (profanity, identity terms and dialect
markers co-occur with toxicity labels without being the causal route).

Measured per layer, per spurious level:
  probe AUC                       -- what the sensor sees
  rho = |cos(w_probe, gbar)|      -- Prop. 4 alignment
  r_eff / d                       -- Prop. 3 inert fraction
  Sbar for probe / gbar / dark    -- predicted effects
  measured D(alpha, .) for each   -- actual causal effects
  Spearman(Sbar, measured)        -- P3, certificate validity
  Spearman(AUC, measured)         -- P4, probe AUC is not a causal predictor
  astar and curvature M_l         -- Prop. 6 validity radius

Run:  python experiments/e2_trained_toy.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from egc.core import (  # noqa: E402
    S_bar, dark_direction, direction_auc, estimate_curvature, fisher_probe,
    gbar, r_eff, rho_alignment, spearman, stable_rank_ratio, unit,
)
from egc.toy import (  # noqa: E402
    ModelSpec, TaskSpec, accuracy, adjoint_covectors, make_dataset,
    measure_steering, train_toy,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "data")
os.makedirs(RESULTS, exist_ok=True)

SPURIOUS_LEVELS = (0.5, 0.7, 0.9, 0.98)
SEEDS = (0, 1, 2)
ALPHAS = (-8.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 8.0)


def analyse_layer(model, toks_te, y_te, spec, layer, alpha_probe=2.0):
    X, G = adjoint_covectors(model, toks_te, spec, layer)
    yn = y_te.numpy()

    w_probe = fisher_probe(X, yn, gamma=1e-2)
    w_grad = unit(gbar(G))                       # certificate-optimal
    try:
        v_dark = dark_direction(G, w_probe[:, None])
    except ValueError:
        v_dark = None

    rng = np.random.default_rng(layer)
    w_rand = unit(rng.normal(size=X.shape[1]))

    cand = {"probe": w_probe, "gbar": w_grad, "random": w_rand}
    if v_dark is not None:
        cand["dark"] = v_dark

    rec = {
        "layer": layer,
        "probe_auc": direction_auc(X, yn, w_probe),
        "gbar_auc": direction_auc(X, yn, w_grad),
        "random_auc": direction_auc(X, yn, w_rand),
        "dark_auc": direction_auc(X, yn, v_dark) if v_dark is not None else None,
        "rho": rho_alignment(G, w_probe),
        "r_eff": r_eff(G),
        "r_eff_over_d": stable_rank_ratio(G),
        "gbar_norm": float(np.linalg.norm(gbar(G))),
        "resid_norm_mean": float(np.linalg.norm(X, axis=1).mean()),
        "directions": {},
    }

    for name, w in cand.items():
        meas = measure_steering(model, toks_te, spec, layer, w, alpha_probe)
        rec["directions"][name] = {
            "Sbar": S_bar(G, w),
            "auc": direction_auc(X, yn, w),
            "measured_mean_effect": float(meas.mean()),
            "measured_abs_mean_effect": float(np.abs(meas).mean()),
            "predicted_effect": float(alpha_probe * S_bar(G, w)),
            "alpha": alpha_probe,
        }
    return rec, X, G


def direction_pool(X, y, G, n_random=24, n_mix=24, seed=0):
    """Pool of candidate directions spanning the (AUC, Sbar) plane."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    wp = fisher_probe(X, y, gamma=1e-2)
    wg = unit(gbar(G))
    pool = [wp, wg]
    for t in np.linspace(0.0, 1.0, n_mix):
        pool.append(unit(t * wp + (1 - t) * wg))
        pool.append(unit(t * wp - (1 - t) * wg))
    for _ in range(n_random):
        pool.append(unit(rng.normal(size=d)))
    # effect-eigenvector directions
    from egc.core import effect_spectrum
    _, vecs = effect_spectrum(G)
    for k in (0, 1, 2, 4, 8):
        if k < vecs.shape[1]:
            pool.append(unit(vecs[:, k]))
    return pool


def main():
    t0 = time.time()
    out = {"experiment": "e2_trained_toy", "runs": []}
    torch.set_num_threads(4)

    for spur in SPURIOUS_LEVELS:
        for seed in SEEDS:
            spec = TaskSpec(spurious=spur, seed=seed)
            mspec = ModelSpec(vocab=spec.vocab, seq_len=spec.seq_len)
            print(f"\n=== spurious={spur}  seed={seed}  (vocab={spec.vocab}) ===")
            model = train_toy(spec, mspec, steps=700, seed=seed, verbose=(seed == 0))

            toks_te, y_te, m_te = make_dataset(spec, 3000, seed=seed + 777)
            acc = accuracy(model, toks_te, y_te, spec)
            acc_marker = accuracy(model, toks_te, m_te, spec)
            print(f"    test acc (causal content) = {acc:.4f}   "
                  f"(spurious marker) = {acc_marker:.4f}")

            run = {"spurious": spur, "seed": seed, "test_acc": acc,
                   "test_acc_vs_marker": acc_marker, "layers": []}

            for layer in range(mspec.n_layers + 1):
                rec, X, G = analyse_layer(model, toks_te, y_te, spec, layer)
                yn = y_te.numpy()

                # ---- P3/P4 over a direction pool -------------------------
                pool = direction_pool(X, yn, G, seed=seed)
                sb, au, me = [], [], []
                for w in pool:
                    sb.append(S_bar(G, w))
                    au.append(direction_auc(X, yn, w))
                    me.append(float(np.abs(
                        measure_steering(model, toks_te[:600], spec, layer, w, 2.0)
                    ).mean()))
                rec["pool"] = {
                    "n": len(pool),
                    "spearman_Sbar_vs_measured": spearman(np.array(sb), np.array(me)),
                    "spearman_auc_vs_measured": spearman(np.array(au), np.array(me)),
                    "spearman_absauc_vs_measured": spearman(
                        np.abs(np.array(au) - 0.5), np.array(me)),
                    "Sbar": sb, "auc": au, "measured": me,
                }

                # ---- Prop. 6 curvature and validity radius ---------------
                wg = unit(gbar(G))
                sub = toks_te[:256]

                def eff_fn(a, _w=wg, _l=layer):
                    return float(measure_steering(model, sub, spec, _l, _w, a).mean())

                M = estimate_curvature(eff_fn, None, wg,
                                       np.linspace(-4, 4, 17))
                sbar_g = S_bar(G, wg)
                rec["curvature_M"] = M
                rec["astar_gbar"] = float(2 * sbar_g / M) if M > 1e-12 else float("inf")

                # ---- linearity check across alphas -----------------------
                lin = []
                for a in ALPHAS:
                    meas = measure_steering(model, sub, spec, layer, wg, a)
                    lin.append({"alpha": a, "measured": float(meas.mean()),
                                "predicted": float(a * sbar_g)})
                rec["alpha_sweep_gbar"] = lin

                run["layers"].append(rec)
                pl = rec["directions"]
                print(f"  L{layer}: AUC={rec['probe_auc']:.3f} rho={rec['rho']:.3f} "
                      f"r_eff/d={rec['r_eff_over_d']:.3f} | "
                      f"meas probe={pl['probe']['measured_mean_effect']:+.3f} "
                      f"gbar={pl['gbar']['measured_mean_effect']:+.3f} "
                      f"dark={pl.get('dark', {}).get('measured_mean_effect', float('nan')):+.3f} "
                      f"| sp(S,meas)={rec['pool']['spearman_Sbar_vs_measured']:.3f} "
                      f"sp(AUC,meas)={rec['pool']['spearman_auc_vs_measured']:.3f}")

            out["runs"].append(run)
            out["partial"] = True
            with open(os.path.join(RESULTS, "e2_trained_toy.json"), "w") as f:
                json.dump(out, f, indent=2)

    out["wall_time_s"] = time.time() - t0
    out["partial"] = False
    path = os.path.join(RESULTS, "e2_trained_toy.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}  ({out['wall_time_s']:.1f}s)")


if __name__ == "__main__":
    main()
