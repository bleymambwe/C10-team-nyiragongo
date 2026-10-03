"""Regenerate src/facts.json for the video from results/data/*.json.

Same principle as paper/make_numbers.py: the composition reads measurements
from here, so no number is typed into a scene by hand.

Run:  python paper-video/make_facts.py
"""

from __future__ import annotations

import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "results", "data")


def load(n):
    with open(os.path.join(DATA, n)) as f:
        return json.load(f)


e1, e2 = load("e1_exact_decoupling.json"), load("e2_trained_toy.json")
e3, e4 = load("e3_gpt2_toxicity.json"), load("e4_active_design.json")
e5, e6 = load("e5_real_data_multimodel.json"), load("e6_independence.json")
A = e3["A_layerwise"]

out = {}
out["e1_rot"] = e1["checks"]["random_rotation"]
out["e1_corr"] = e1["sweep"]["auc_vs_Sbar_pearson"]
out["p7"] = e1["checks"]["P7_finite_sample_leakage"]["model_inv_sqrt_n"]

# ---- E3 ----
out["e3"] = {
    "layers": [r["layer"] for r in A],
    "auc": [r["probe_auc_frame_test"] for r in A],
    "rho": [r["rho_probe_gbar"] for r in A],
    "reff": [r["r_eff_over_d"] for r in A],
    "m_probe": [r["directions"]["probe"]["measured_mean_effect"] for r in A],
    "m_gbar": [r["directions"]["gbar"]["measured_mean_effect"] for r in A],
    "m_dark": [r["directions"]["dark"]["measured_mean_effect"] for r in A],
    "m_random": [r["directions"]["random"]["measured_mean_effect"] for r in A],
    "sp_s": [r["pool"]["spearman_Sbar_vs_measured"] for r in A],
    "sp_a": [r["pool"]["spearman_auc_vs_measured"] for r in A],
    "auc_gbar": [r["directions"]["gbar"]["auc_frame_test"] for r in A],
    "auc_dark": [r["directions"]["dark"]["auc_frame_test"] for r in A],
}
mp = sum(abs(v) for v in out["e3"]["m_probe"])
mg = sum(abs(v) for v in out["e3"]["m_gbar"])
md = sum(abs(v) for v in out["e3"]["m_dark"])
out["e3"].update({
    "eff_probe": mp / mg, "eff_dark": md / mg,
    "mean_rho": st.mean(out["e3"]["rho"]),
    "mean_reff": st.mean(out["e3"]["reff"]),
    "mean_sp_s": st.mean(out["e3"]["sp_s"]),
    "mean_sp_a": st.mean(x for x in out["e3"]["sp_a"] if x == x),
    "max_auc": max(out["e3"]["auc"]),
    "mean_auc_gbar": st.mean(out["e3"]["auc_gbar"]),
    "d": e3["d_model"],
})

# ---- E6 ----
s6 = e6["summary"]
out["e6"] = {
    "m": s6["matched_mean_over_layers"], "u": s6["unmatched_mean_over_layers"],
    "ratio": s6["ratio"], "wins": s6["layers_where_matched_exceeds_unmatched"],
    "n": s6["n_layers"],
    "matched_series": [r["matched_mean"] for r in e6["layers"]],
    "unmatched_series": [r["unmatched_mean"] for r in e6["layers"]],
}

# ---- E5 ----
out["e5"] = [{"name": m["model"].split("/")[-1], "agg": m["aggregate"],
              "reff": m["mean_r_eff_over_d"], "d": m["d_model"]}
             for m in e5["models"]]

# ---- E4 ----
L4 = e4["layers"]
out["e4"] = {
    "M": e4["M_cand"],
    "layers": [l["layer"] for l in L4],
    "budgets": L4[0]["budgets"],
    "curves": {k: [l["best_found_frac"][k] for l in L4]
               for k in ("oracle", "certificate", "probe_auc", "random")},
    "b90": {k: [l["budget_to_90pct"][k] for l in L4]
            for k in ("oracle", "certificate", "probe_auc", "random")},
    "sp_sbar": st.mean(l["spearman_Sbar_vs_truth"] for l in L4),
    "sp_auc": st.mean(l["spearman_absauc_vs_truth"] for l in L4),
}
out["e4"]["cert_mean"] = st.mean(out["e4"]["b90"]["certificate"])
out["e4"]["rand_mean"] = st.mean(out["e4"]["b90"]["random"])
out["e4"]["auc_mean"] = st.mean(out["e4"]["b90"]["probe_auc"])
out["e4"]["speedup"] = out["e4"]["rand_mean"] / out["e4"]["cert_mean"]

# ---- E2 ----
R2 = e2["runs"]
lv = sorted({r["spurious"] for r in R2})


def _rho(r):
    return st.mean(l["rho"] for l in r["layers"])


def _eff(r):
    p = sum(abs(l["directions"]["probe"]["measured_mean_effect"]) for l in r["layers"])
    g = sum(abs(l["directions"]["gbar"]["measured_mean_effect"]) for l in r["layers"])
    return 100 * p / g


def _dark(r):
    d = sum(abs(l["directions"]["dark"]["measured_mean_effect"]) for l in r["layers"])
    g = sum(abs(l["directions"]["gbar"]["measured_mean_effect"]) for l in r["layers"])
    return 100 * d / g


def at(v, fn):
    return [fn(r) for r in R2 if r["spurious"] == v]


out["e2"] = {
    "levels": lv,
    "n_runs": len(R2),
    "n_seeds": len({r["seed"] for r in R2}),
    "acc": st.mean(r["test_acc"] for r in R2),
    "rho_by_level": [st.mean(at(v, _rho)) for v in lv],
    "rho_sd_by_level": [st.pstdev(at(v, _rho)) for v in lv],
    "eff_by_level": [st.mean(at(v, _eff)) for v in lv],
    "dark_by_level": [st.mean(at(v, _dark)) for v in lv],
    "marker_by_level": [st.mean(at(v, lambda r: r["test_acc_vs_marker"])) for v in lv],
    "d": 64,
}
out["e2"]["rho_delta"] = out["e2"]["rho_by_level"][-1] - out["e2"]["rho_by_level"][0]
out["e2"]["seed_sd"] = st.mean(out["e2"]["rho_sd_by_level"])
out["e2"]["delta_in_sd"] = abs(out["e2"]["rho_delta"]) / out["e2"]["seed_sd"]

# ---- absolute effect rank across every model measured ----
pts = [("toy (E2)", out["e2"]["d"],
        st.mean(l["r_eff_over_d"] for r in R2 for l in r["layers"]))]
pts += [(m["name"], m["d"], m["reff"]) for m in out["e5"]]
out["scale"] = [{"name": n, "d": d, "reff_frac": f, "reff_abs": f * d} for n, d, f in pts]

path = os.path.join(HERE, "src", "facts.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1, allow_nan=False)
print(f"wrote {path}")
print(f"  E3 probe eff {100*out['e3']['eff_probe']:.1f}%  dark {100*out['e3']['eff_dark']:.1f}%")
print(f"  E4 budget cert {out['e4']['cert_mean']:.1f} vs random {out['e4']['rand_mean']:.1f} "
      f"({out['e4']['speedup']:.0f}x)")
print(f"  E2 rho {out['e2']['rho_by_level'][0]:.3f} -> {out['e2']['rho_by_level'][-1]:.3f} "
      f"({out['e2']['delta_in_sd']:.1f} seed SD)")
print(f"  E5 models: {', '.join(m['name'] for m in out['e5'])}")
print("  abs r_eff: " + ", ".join(f"{p['name']} {p['reff_abs']:.1f}" for p in out["scale"]))
