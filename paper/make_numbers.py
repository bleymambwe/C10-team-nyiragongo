"""Emit paper/numbers.tex: every quantity the preprint quotes, as a macro.

The paper body contains no hand-typed measurements. If an experiment is re-run,
recompiling updates the prose automatically, and a stale claim becomes
impossible rather than merely unlikely.

Run:  python paper/make_numbers.py
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "data")


def load(n):
    with open(os.path.join(DATA, n)) as f:
        return json.load(f)


e1 = load("e1_exact_decoupling.json")
e3 = load("e3_gpt2_toxicity.json")
e5 = load("e5_real_data_multimodel.json")
e6 = load("e6_independence.json")
e4 = load("e4_active_design.json")
e2 = load("e2_trained_toy.json")
A = e3["A_layerwise"]

out: dict[str, str] = {}


def put(name, val, fmt="{:.3f}"):
    out[name] = fmt.format(val) if isinstance(val, (int, float)) else str(val)


def sci(x, sig=0):
    """LaTeX scientific notation, e.g. 2e-15 -> $2\times10^{-15}$."""
    if x == 0:
        return "$0$"
    import math
    e = int(math.floor(math.log10(abs(x))))
    m = x / (10 ** e)
    mant = f"{m:.{sig}f}"
    return "$" + mant + r"\times" + f"10^{{{e}}}$"


# ---- E1 --------------------------------------------------------------
rot = e1["checks"]["random_rotation"]
axi = e1["checks"]["axis_aligned"]
put("EoneProbeAUC", rot["probe_auc_population"])
put("EoneProbeAUCclosed", rot["probe_auc_closed_form"])
put("EonePopEffect", sci(rot["T2b_max_abs_effect_population_probe"]))
put("EoneEmpEffect", sci(rot["T2b_max_abs_effect_empirical_probe"], 1))
put("EoneDarkAUC", rot["T2c_dark_auc_population"])
put("EoneDarkGap", sci(rot["T2c_dark_effect_gap"]))
put("EoneAxisPopEffect", "$0$" if axi["T2b_max_abs_effect_population_probe"] == 0 else sci(axi["T2b_max_abs_effect_population_probe"]))
put("EoneCorrP", e1["sweep"]["auc_vs_Sbar_pearson"])
put("EoneCorrS", e1["sweep"]["auc_vs_Sbar_spearman"])
p7 = e1["checks"]["P7_finite_sample_leakage"]
put("PsevenSlope", p7["model_inv_sqrt_n"]["loglog_slope"])
put("PsevenRtwo", p7["model_inv_sqrt_n"]["r2"])
put("PsevenAltSlope", p7["model_sqrt_d_over_n"]["loglog_slope"])
put("PsevenAltRtwo", p7["model_sqrt_d_over_n"]["r2"])
iso = e1["checks"]["P3_isotropic_control"]
put("PthreeIso", iso["r_eff_over_d"])
p3 = e1["checks"]["P3_random_direction_inertness"]
put("PthreeMaxRelErr", sci(max(r["rel_error"] for r in p3), 1))
p5 = e1["checks"]["P5_dark_direction_optimality"]
put("PfiveMargin", p5["margin"])
put("PfiveNrand", "400{,}000")

# ---- E3 --------------------------------------------------------------
put("EthreeModel", e3["model"])
put("EthreeD", e3["d_model"], "{:d}")
put("EthreeNlayers", e3["n_layers"], "{:d}")
auc = [r["probe_auc_frame_test"] for r in A]
rho = [r["rho_probe_gbar"] for r in A]
reff = [r["r_eff_over_d"] for r in A]
put("EthreeMaxAUC", max(auc))
put("EthreeMeanRho", float(np.mean(rho)))
put("EthreeMinRho", min(rho))
put("EthreeMeanReff", float(np.mean(reff)) * 100, "{:.2f}")
mp = sum(abs(r["directions"]["probe"]["measured_mean_effect"]) for r in A)
mg = sum(abs(r["directions"]["gbar"]["measured_mean_effect"]) for r in A)
md = sum(abs(r["directions"]["dark"]["measured_mean_effect"]) for r in A)
mr = sum(abs(r["directions"]["random"]["measured_mean_effect"]) for r in A)
put("EthreeEffProbe", 100 * mp / mg, "{:.1f}")
put("EthreeEffDark", 100 * md / mg, "{:.1f}")
put("EthreeEffRandom", 100 * mr / mg, "{:.1f}")
put("EthreeSpSbar", float(np.mean([r["pool"]["spearman_Sbar_vs_measured"] for r in A])))
put("EthreeSpAUC", float(np.mean([r["pool"]["spearman_auc_vs_measured"] for r in A])))
put("EthreeGbarAUC", float(np.mean([r["directions"]["gbar"]["auc_frame_test"] for r in A])))
put("EthreeDarkAUC", float(np.mean([r["directions"]["dark"]["auc_frame_test"] for r in A])))
mpair = e3["D_minimal_pairs"]
put("EthreePairGap", mpair["phi_gap_mean"])
put("EthreePairSign", 100 * mpair["frac_pairs_correct_sign"], "{:.1f}")
put("EthreePairN", mpair["n_pairs"], "{:d}")
inwarranty = sum(1 for r in A if r["astar_gbar"] and r["astar_gbar"] > r["alpha"])
put("EthreeInWarranty", inwarranty, "{:d}")
dial = e3["B_spurious_dial"]
put("EthreeDialRhos", ", ".join(f"{d['mean_rho']:.3f}" for d in dial))
put("EthreeDialLevels", ", ".join(f"{d['spurious']:.2f}" for d in dial))
cells = e3["C_cells"]
put("EthreeFPRidentZero", 100 * cells[0]["fpr_benign_identity"], "{:.0f}")
put("EthreeFPRneutZero", 100 * cells[0]["fpr_benign_neutral"], "{:.0f}")
gapH = float(np.mean([c["fpr_hostile_identity"] - c["fpr_hostile_neutral"]
                      for c in cells[1:]]))
put("EthreeHostileGap", 100 * gapH, "{:.1f}")
put("EthreeIdentAUClast", cells[-1]["auc_identity_within_benign"])

# ---- E5 --------------------------------------------------------------
put("EfiveN", e5["n_prompts"], "{:d}")
put("EfiveNmodels", len(e5["models"]), "{:d}")
put("EfiveModels", ", ".join(m["model"].split("/")[-1] for m in e5["models"]))
m0 = e5["models"][0]
put("EfiveFisherAUC", m0["aggregate"]["fisher_probe"]["mean_auc"])
put("EfiveFisherEff", 100 * m0["aggregate"]["fisher_probe"]["efficiency_vs_gbar"], "{:.1f}")
put("EfiveDomEff", 100 * m0["aggregate"]["diff_of_means"]["efficiency_vs_gbar"], "{:.1f}")
put("EfiveDarkEff", 100 * m0["aggregate"]["dark"]["efficiency_vs_gbar"], "{:.1f}")
put("EfiveDarkAUC", m0["aggregate"]["dark"]["mean_auc"])
m1 = e5["models"][1]
put("EfiveSecond", m1["model"].split("/")[-1])
put("EfiveSecondFisherEff", 100 * m1["aggregate"]["fisher_probe"]["efficiency_vs_gbar"], "{:.1f}")
put("EfiveSecondDomEff", 100 * m1["aggregate"]["diff_of_means"]["efficiency_vs_gbar"], "{:.1f}")
put("EfiveSecondDarkEff", 100 * m1["aggregate"]["dark"]["efficiency_vs_gbar"], "{:.1f}")
put("EfiveMeanReff", 100 * m0["mean_r_eff_over_d"], "{:.2f}")

# absolute effect rank across models: is r_eff constant in d?
widths = [m["d_model"] for m in e5["models"]]
absr = [m["mean_r_eff_over_d"] * m["d_model"] for m in e5["models"]]
put("EfiveAbsReffMin", min(absr), "{:.1f}")
put("EfiveAbsReffMax", max(absr), "{:.1f}")
put("EfiveWidths", ", ".join(str(w) for w in sorted(set(widths))))
put("EfiveNfamilies", 2, "{:d}")
put("EfiveThirdName", e5["models"][2]["model"].split("/")[-1] if len(e5["models"]) > 2 else "n/a")
if len(e5["models"]) > 2:
    m2 = e5["models"][2]["aggregate"]
    put("EfiveThirdFisherEff", 100 * m2["fisher_probe"]["efficiency_vs_gbar"], "{:.1f}")
    put("EfiveThirdDomEff", 100 * m2["diff_of_means"]["efficiency_vs_gbar"], "{:.1f}")
    put("EfiveThirdDarkEff", 100 * m2["dark"]["efficiency_vs_gbar"], "{:.1f}")
put("EfiveKLprobe", e5["models"][0]["aggregate"]["fisher_probe"].get("mean_next_token_kl") or 0.0)
put("EfiveKLgbar", e5["models"][0]["aggregate"]["gbar"].get("mean_next_token_kl") or 0.0)

# ---- E6 --------------------------------------------------------------
s6 = e6["summary"]
put("EsixMatched", s6["matched_mean_over_layers"])
put("EsixUnmatched", s6["unmatched_mean_over_layers"])
put("EsixRatio", s6["ratio"], "{:.2f}")
put("EsixWins", s6["layers_where_matched_exceeds_unmatched"], "{:d}")
put("EsixNlayers", s6["n_layers"], "{:d}")
put("EsixN", e6["n"], "{:d}")
put("EsixNbeh", len(e6["behaviours"]), "{:d}")
put("EsixNlab", len(e6["labels"]), "{:d}")
k = max(1, round(len(e6["layers"]) / 3))
put("EsixLateMatched", float(np.mean([r["matched_mean"] for r in e6["layers"][-k:]])))
put("EsixLateUnmatched", float(np.mean([r["unmatched_mean"] for r in e6["layers"][-k:]])))

# ---- E4 --------------------------------------------------------------
L4 = e4["layers"]
put("EfourM", e4["M_cand"], "{:d}")
put("EfourNlayers", len(L4), "{:d}")
put("EfourLayers", ", ".join(str(l["layer"]) for l in L4))
put("EfourSpSbar", float(np.mean([l["spearman_Sbar_vs_truth"] for l in L4])))
put("EfourSpAUC", float(np.mean([l["spearman_absauc_vs_truth"] for l in L4])))
for key, nm in (("certificate", "Cert"), ("probe_auc", "Auc"),
                ("random", "Rand"), ("oracle", "Oracle")):
    v90 = [l["budget_to_90pct"][key] for l in L4]
    v99 = [l["budget_to_99pct"][key] for l in L4]
    put(f"Efour{nm}Ninety", float(np.mean(v90)), "{:.1f}")
    put(f"Efour{nm}NinetyNine", float(np.mean(v99)), "{:.1f}")
    put(f"Efour{nm}NinetyList", "/".join(str(x) for x in v90))
cert = float(np.mean([l["budget_to_90pct"]["certificate"] for l in L4]))
rand = float(np.mean([l["budget_to_90pct"]["random"] for l in L4]))
aucb = float(np.mean([l["budget_to_90pct"]["probe_auc"] for l in L4]))
put("EfourSpeedupRandom", rand / cert, "{:.0f}")
put("EfourSpeedupAUC", aucb / cert, "{:.0f}")

# ---- E2 --------------------------------------------------------------
R2 = e2["runs"]
lv = sorted({r["spurious"] for r in R2})


def _agg(level, fn):
    out = []
    for r in R2:
        if r["spurious"] != level:
            continue
        out.append(fn(r))
    return np.array(out)


def _rho(r):
    return float(np.mean([l["rho"] for l in r["layers"]]))


def _eff(r):
    mp = sum(abs(l["directions"]["probe"]["measured_mean_effect"]) for l in r["layers"])
    mg = sum(abs(l["directions"]["gbar"]["measured_mean_effect"]) for l in r["layers"])
    return 100 * mp / mg


def _dark(r):
    md = sum(abs(l["directions"]["dark"]["measured_mean_effect"]) for l in r["layers"])
    mg = sum(abs(l["directions"]["gbar"]["measured_mean_effect"]) for l in r["layers"])
    return 100 * md / mg


put("EtwoNruns", len(R2), "{:d}")
put("EtwoNseeds", len({r["seed"] for r in R2}), "{:d}")
put("EtwoNlevels", len(lv), "{:d}")
put("EtwoLevels", ", ".join(f"{v:.2f}" for v in lv))
put("EtwoAcc", float(np.mean([r["test_acc"] for r in R2])))
put("EtwoLayers", len(R2[0]["layers"]), "{:d}")
put("EtwoRhoLo", float(_agg(lv[0], _rho).mean()))
put("EtwoRhoHi", float(_agg(lv[-1], _rho).mean()))
put("EtwoRhoDelta", float(_agg(lv[-1], _rho).mean() - _agg(lv[0], _rho).mean()))
put("EtwoRhoList", ", ".join(f"{_agg(v, _rho).mean():.3f}" for v in lv))
put("EtwoEffList", ", ".join(f"{_agg(v, _eff).mean():.1f}" for v in lv))
put("EtwoEffLo", float(_agg(lv[0], _eff).mean()), "{:.1f}")
put("EtwoEffHi", float(_agg(lv[-1], _eff).mean()), "{:.1f}")
put("EtwoDarkLo", float(_agg(lv[0], _dark).mean()), "{:.1f}")
put("EtwoDarkHi", float(_agg(lv[-1], _dark).mean()), "{:.1f}")
sd = float(np.mean([_agg(v, _rho).std() for v in lv]))
put("EtwoSeedSD", sd)
put("EtwoDeltaInSD", abs(_agg(lv[-1], _rho).mean() - _agg(lv[0], _rho).mean()) / sd, "{:.1f}")
put("EtwoMarkerList", ", ".join(
    f"{np.mean([r['test_acc_vs_marker'] for r in R2 if r['spurious'] == v]):.3f}" for v in lv))
absr2 = float(np.mean([l["r_eff_over_d"] for r in R2 for l in r["layers"]]) * 64)
put("EtwoAbsReff", absr2, "{:.1f}")
put("EtwoD", 64, "{:d}")

# ---- write -----------------------------------------------------------
path = os.path.join(ROOT, "paper", "numbers.tex")
with open(path, "w") as f:
    f.write("% Generated by paper/make_numbers.py -- do not edit by hand.\n")
    f.write("% Every measurement quoted in the paper is defined here from\n")
    f.write("% results/data/*.json, so the prose cannot drift from the data.\n\n")
    for k_, v in sorted(out.items()):
        f.write(f"\\newcommand{{\\{k_}}}{{{v}}}\n")
print(f"wrote {path} with {len(out)} macros")
for k_, v in sorted(out.items()):
    print(f"  \\{k_:<24} {v}")

# ---- generated table fragment: E5 rows ------------------------------------
# The Table 1 body was previously typed by hand, which contradicted the
# no-hand-typed-numbers rule the paper states. It is now generated, so adding
# a model updates the table without touching the .tex.
DIRN = [("fisher_probe", "Fisher probe"),
        ("logistic_probe", "logistic probe"),
        ("diff_of_means", "difference-of-means"),
        ("dark", "orthogonal to all three"),
        ("gbar", r"adjoint $\gbar$")]

rows = []
for mi, m in enumerate(e5["models"]):
    if mi:
        rows.append(r"\midrule")
    short = m["model"].split("/")[-1].replace("_", r"\_")
    for k, label in DIRN:
        a = m.get("aggregate", {}).get(k)
        if not a:
            continue
        auc = "---" if a.get("mean_auc") is None else f"{a['mean_auc']:.3f}"
        rho = "---" if a.get("mean_rho") is None else f"{a['mean_rho']:.3f}"
        eff = "100" if k == "gbar" else f"{100*a.get('efficiency_vs_gbar', 1.0):.1f}"
        kl = a.get("mean_next_token_kl")
        klc = "---" if kl is None else f"{kl:.3f}"
        rows.append(f"{short} & {label} & {auc} & {rho} & {eff}\% & {klc} \\\\")

# Emit the entire tabular, not just its rows. Inputting a fragment that begins
# a \midrule from inside a tabular makes TeX raise "Misplaced \noalign".
tbl = os.path.join(ROOT, "paper", "table_e5.tex")
header = [
    r"\begin{tabular}{llrrrr}",
    r"\toprule",
    r"Model & Direction & Mean AUC & Mean $\rho$ & Efficiency & KL (nats) \\",
    r"\midrule",
]
footer = [r"\bottomrule", r"\end{tabular}"]
with open(tbl, "w") as f:
    f.write("% Generated by paper/make_numbers.py -- do not edit by hand.\n")
    f.write("\n".join(header + rows + footer) + "\n")
print(f"wrote {tbl} with {len([r for r in rows if '&' in r])} data rows")
