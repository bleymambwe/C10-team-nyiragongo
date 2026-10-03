"""E5 -- Real toxicity data, multiple models, all standard steering baselines.

Closes the three obvious objections to E3:

  "synthetic templates"   -> RealToxicityPrompts, balanced high/low toxicity
  "one model"             -> gpt2, distilgpt2, pythia-70m (two families)
  "you only beat a Fisher probe, nobody steers with those"
                          -> difference-of-means (ActAdd/CAA) and logistic
                             regression probes are included as first-class
                             baselines, since those ARE what practitioners
                             deploy as steering vectors.

Every read-side direction (fisher / logistic / diff-of-means) is fitted on a
train split; AUC and steering are evaluated on held-out prompts. Bootstrap CIs
are reported for the headline quantities.

Run:  python experiments/e5_real_data_multimodel.py
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
from egc.core import auc as _auc  # noqa: E402
from egc.core import (  # noqa: E402
    S_bar, dark_direction, diff_of_means, direction_auc, fisher_probe, gbar,
    logistic_probe, pearson, r_eff, rho_alignment, spearman, stable_rank_ratio,
    unit,
)
from egc.toxdata import NEUTRAL_WORDS, TOXIC_WORDS  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "data")
CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "cache")
os.makedirs(RESULTS, exist_ok=True)

MODELS = ["gpt2", "distilgpt2", "EleutherAI/pythia-70m"]
N_TOTAL = 400
N_STEER = 100
ALPHA_FRAC = 0.10
BOOT = 300


def load_rtp(n_total=N_TOTAL, seed=0):
    with open(os.path.join(CACHE, "rtp_balanced.json")) as f:
        d = json.load(f)
    p = np.array(d["prompts"], dtype=object)
    y = np.array(d["y"])
    tox = np.array(d["toxicity"])
    rng = np.random.default_rng(seed)
    hi = rng.choice(np.where(y == 1)[0], n_total // 2, replace=False)
    lo = rng.choice(np.where(y == 0)[0], n_total // 2, replace=False)
    idx = np.concatenate([hi, lo])
    rng.shuffle(idx)
    return list(p[idx]), y[idx], tox[idx]


def boot_mean_ci(v, reps=BOOT, seed=0):
    v = np.asarray(v, dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = [float(v[rng.integers(0, len(v), len(v))].mean()) for _ in range(reps)]
    return float(v.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def run_model(name, prompts, y, out):
    t0 = time.time()
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    lm = LMAdapter(name)
    pos, neg = lm.token_ids(TOXIC_WORDS), lm.token_ids(NEUTRAL_WORDS)
    print(f"  layers={lm.n_layers} d={lm.d_model} pos={len(pos)} neg={len(neg)}")

    n = len(prompts)
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]

    X, G, phi = lm.states_and_adjoints(prompts, pos, neg, batch=8)
    print(f"  adjoints done ({time.time()-t0:.0f}s); "
          f"phi toxic={phi[y==1].mean():+.3f} benign={phi[y==0].mean():+.3f}")

    p_steer = [prompts[i] for i in te[:N_STEER]]
    base = lm.baseline_phi(p_steer, pos, neg, batch=8)

    rec_model = {
        "model": name, "n_layers": lm.n_layers, "d_model": lm.d_model,
        "phi_toxic_mean": float(phi[y == 1].mean()),
        "phi_benign_mean": float(phi[y == 0].mean()),
        "phi_separation_auc": _auc(phi, y),
        "layers": [],
    }

    for L in range(lm.n_layers):
        Xl, Gl = X[L], G[L]
        Xtr, ytr, Gtr = Xl[tr], y[tr], Gl[tr]
        Xte, yte = Xl[te], y[te]
        alpha = ALPHA_FRAC * float(np.linalg.norm(Xl, axis=1).mean())

        dirs = {
            "fisher_probe": fisher_probe(Xtr, ytr, gamma=1e-2),
            "logistic_probe": logistic_probe(Xtr, ytr),
            "diff_of_means": diff_of_means(Xtr, ytr),
            "gbar": unit(gbar(Gtr)),
            "random": unit(np.random.default_rng(500 + L).normal(size=Xl.shape[1])),
        }
        try:
            dirs["dark"] = dark_direction(Gtr, np.stack([
                dirs["fisher_probe"], dirs["logistic_probe"], dirs["diff_of_means"]
            ], axis=1))
        except ValueError:
            pass

        rec = {
            "layer": L, "alpha": alpha,
            "resid_norm_mean": float(np.linalg.norm(Xl, axis=1).mean()),
            "r_eff": r_eff(Gtr), "r_eff_over_d": stable_rank_ratio(Gtr),
            "gbar_norm": float(np.linalg.norm(gbar(Gtr))),
            "directions": {},
        }
        for nm, w in dirs.items():
            meas = lm.measure_steering(p_steer, L, w, alpha, pos, neg, base=base, batch=8)
            m, lo, hi = boot_mean_ci(meas, seed=L)
            rec["directions"][nm] = {
                "auc_test": direction_auc(Xte, yte, w),
                "rho": rho_alignment(Gtr, w),
                "Sbar": S_bar(Gtr, w),
                "predicted_effect": float(alpha * S_bar(Gtr, w)),
                "measured_mean_effect": m,
                "measured_ci_lo": lo, "measured_ci_hi": hi,
                "measured_abs_mean": float(np.abs(meas).mean()),
            }
        # collateral damage: KL of the next-token distribution
        for nm in ("fisher_probe", "diff_of_means", "gbar"):
            if nm in dirs:
                rec["directions"][nm]["next_token_kl"] = lm.next_token_kl(
                    p_steer[:48], L, dirs[nm], alpha, batch=8)

        # certificate vs measured across the direction set at this layer
        nms = list(rec["directions"].keys())
        sb = np.array([rec["directions"][k]["Sbar"] for k in nms])
        me = np.array([abs(rec["directions"][k]["measured_mean_effect"]) for k in nms])
        au = np.array([abs(rec["directions"][k]["auc_test"] - 0.5) for k in nms])
        rec["spearman_Sbar_vs_measured"] = spearman(sb, me)
        rec["spearman_absauc_vs_measured"] = spearman(au, me)
        rec_model["layers"].append(rec)

        d = rec["directions"]
        print(f"  L{L:>2} a={alpha:6.2f} reff/d={rec['r_eff_over_d']:.4f} | "
              f"fisher AUC={d['fisher_probe']['auc_test']:.3f} rho={d['fisher_probe']['rho']:.3f} "
              f"m={d['fisher_probe']['measured_mean_effect']:+.3f} | "
              f"dom AUC={d['diff_of_means']['auc_test']:.3f} rho={d['diff_of_means']['rho']:.3f} "
              f"m={d['diff_of_means']['measured_mean_effect']:+.3f} | "
              f"gbar m={d['gbar']['measured_mean_effect']:+.3f} | "
              f"dark m={d.get('dark',{}).get('measured_mean_effect',float('nan')):+.3f} "
              f"AUC={d.get('dark',{}).get('auc_test',float('nan')):.3f}")

    # ---- model-level aggregates -------------------------------------------
    agg = {}
    for nm in ("fisher_probe", "logistic_probe", "diff_of_means", "gbar", "random", "dark"):
        vals = [l["directions"][nm] for l in rec_model["layers"] if nm in l["directions"]]
        if not vals:
            continue
        agg[nm] = {
            "mean_auc": float(np.mean([v["auc_test"] for v in vals])),
            "mean_rho": float(np.mean([v["rho"] for v in vals])),
            "mean_abs_measured": float(np.mean([abs(v["measured_mean_effect"]) for v in vals])),
            "max_abs_measured": float(np.max([abs(v["measured_mean_effect"]) for v in vals])),
            "mean_next_token_kl": (float(np.mean([v["next_token_kl"] for v in vals]))
                                   if all("next_token_kl" in v for v in vals) else None),
        }
    for nm in agg:
        if nm != "gbar" and agg.get("gbar", {}).get("mean_abs_measured", 0) > 0:
            agg[nm]["efficiency_vs_gbar"] = (
                agg[nm]["mean_abs_measured"] / agg["gbar"]["mean_abs_measured"])
    rec_model["aggregate"] = agg
    rec_model["mean_r_eff_over_d"] = float(np.mean([l["r_eff_over_d"] for l in rec_model["layers"]]))
    rec_model["wall_time_s"] = time.time() - t0
    out["models"].append(rec_model)
    _checkpoint(out)

    print(f"\n  AGGREGATE for {name} (mean over layers):")
    for nm, a in agg.items():
        print(f"    {nm:<16} AUC={a['mean_auc']:.3f} rho={a['mean_rho']:.3f} "
              f"|effect|={a['mean_abs_measured']:.4f} "
              f"eff_vs_gbar={a.get('efficiency_vs_gbar', 1.0):.3f}")
    print(f"    mean r_eff/d = {rec_model['mean_r_eff_over_d']:.4f}")
    del lm
    import gc; gc.collect()


def _checkpoint(out):
    """Persist after each model, so stopping mid-suite keeps what finished."""
    done = {m["model"] for m in out["models"] if "error" not in m}
    out["partial"] = any(m not in done for m in MODELS)
    out["models_not_completed"] = [m for m in MODELS if m not in done]
    with open(os.path.join(RESULTS, "e5_real_data_multimodel.json"), "w") as f:
        json.dump(out, f, indent=2)


def main():
    torch.set_num_threads(4)
    prompts, y, tox = load_rtp(N_TOTAL, seed=0)
    print(f"RealToxicityPrompts: n={len(prompts)} "
          f"mean tox (y=1)={tox[y==1].mean():.3f} (y=0)={tox[y==0].mean():.3f}")
    out = {"experiment": "e5_real_data_multimodel", "dataset": "RealToxicityPrompts",
           "n_prompts": len(prompts), "n_steer": N_STEER, "alpha_frac": ALPHA_FRAC,
           "models": []}
    for m in MODELS:
        try:
            run_model(m, prompts, y, out)
        except Exception as e:
            print(f"  !! {m} failed: {type(e).__name__}: {e}")
            out["models"].append({"model": m, "error": f"{type(e).__name__}: {e}"})
    path = os.path.join(RESULTS, "e5_real_data_multimodel.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
