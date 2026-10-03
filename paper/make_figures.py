"""Generate the preprint figures directly from results/data/*.json.

No hand-entered numbers: every figure is a function of the experiment output,
so recompiling after a re-run cannot silently disagree with the text.

Run:  python paper/make_figures.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "data")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

# Validated categorical palette (colour-blind checked; see dashboard).
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
GREY = "#6d818b"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.30,
    "lines.linewidth": 1.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


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


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p + ".pdf")
    plt.close(fig)
    print("  wrote", name + ".pdf")


# ---------------------------------------------------------------- Fig 1
def fig_decoupling():
    g = e1["sweep"]["auc_vs_certificate_grid"]
    thetas = sorted({r["theta_deg"] for r in g})
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    for th in thetas:
        pts = [r for r in g if r["theta_deg"] == th]
        ax.scatter([p["probe_auc"] for p in pts], [p["Sbar_probe"] for p in pts],
                   s=22, color=cmap(th / 90.0), edgecolor="white", linewidth=.5,
                   label=f"{th}$^\\circ$", zorder=3)
    ax.set_xlabel("probe AUC")
    ax.set_ylabel(r"certificate $\bar{S}$")
    ax.set_xlim(0.48, 1.02)
    ax.grid(True, linestyle=":")
    r = e1["sweep"]["auc_vs_Sbar_pearson"]
    ax.set_title(f"Pearson $r = {r:.3f}$", loc="left", color="#333")
    # A colourbar rather than a legend: the legend sat on top of the data.
    norm = matplotlib.colors.Normalize(vmin=0, vmax=90)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=ax, pad=.02, ticks=[0, 30, 60, 90])
    cb.set_label(r"$\theta$ (degrees)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    save(fig, "fig1_decoupling")


# ---------------------------------------------------------------- Fig 2
def fig_leakage():
    rows = e1["checks"]["P7_finite_sample_leakage"]["rows"]
    ds = sorted({r["d"] for r in rows})
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    for k, d in enumerate(ds):
        rr = sorted([r for r in rows if r["d"] == d], key=lambda z: z["n"])
        ax.plot([r["n"] for r in rr], [r["rho_hat_mean"] for r in rr],
                "o-", ms=3.5, color=[S1, S2, S3][k % 3], label=f"$d={d}$", zorder=3)
    ns = np.array(sorted({r["n"] for r in rows}), dtype=float)
    ax.plot(ns, 1 / np.sqrt(ns), "--", color=GREY, lw=1.2,
            label=r"$N^{-1/2}$", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training examples $N$")
    ax.set_ylabel(r"spurious alignment $\hat{\rho}$")
    ax.grid(True, which="both", linestyle=":")
    m = e1["checks"]["P7_finite_sample_leakage"]["model_inv_sqrt_n"]
    ax.set_title(f"slope {m['loglog_slope']:.3f}, $R^2={m['r2']:.3f}$",
                 loc="left", color="#333")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    save(fig, "fig2_leakage")


# ---------------------------------------------------------------- Fig 3
def fig_gpt2_layerwise():
    L = [r["layer"] for r in A]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.4))

    a1.plot(L, [r["probe_auc_frame_test"] for r in A], "o-", ms=3.5, color=S2,
            label="probe AUC (held out)")
    a1.plot(L, [r["rho_probe_gbar"] for r in A], "s-", ms=3.5, color=S1,
            label=r"alignment $\rho$")
    a1.set_ylim(-0.03, 1.05)
    a1.set_xlabel("layer")
    a1.set_ylabel("value")
    a1.grid(True, linestyle=":")
    a1.legend(frameon=False, loc="center right")

    a2.plot(L, [100 * r["r_eff_over_d"] for r in A], "^-", ms=3.5, color=S3)
    a2.set_xlabel("layer")
    a2.set_ylabel(r"$r_{\mathrm{eff}}/d$  (%)")
    a2.set_ylim(0, None)
    a2.grid(True, linestyle=":")
    mean_reff = np.mean([r["r_eff_over_d"] for r in A])
    a2.axhline(100 * mean_reff, color=GREY, ls="--", lw=1)
    a2.annotate(f"mean {100*mean_reff:.2f}%", xy=(L[-4], 100 * mean_reff),
                xytext=(0, 6), textcoords="offset points", color=GREY, fontsize=7.5)
    save(fig, "fig3_gpt2_layerwise")


# ---------------------------------------------------------------- Fig 4
def fig_gpt2_effects():
    L = [r["layer"] for r in A]
    keys = [("gbar", r"adjoint $\bar{g}$", S1),
            ("dark", "probe-orthogonal", S3),
            ("probe", "Fisher probe", S2),
            ("random", "random", GREY)]
    x = np.arange(len(L))
    w = 0.20
    fig, ax = plt.subplots(figsize=(6.9, 2.5))
    for i, (k, lab, c) in enumerate(keys):
        ax.bar(x + (i - 1.5) * w,
               [r["directions"][k]["measured_mean_effect"] for r in A],
               width=w, color=c, label=lab, edgecolor="white", linewidth=.4)
    ax.set_xticks(x)
    ax.set_xticklabels(L)
    ax.set_xlabel("layer")
    ax.set_ylabel(r"measured $\Delta\varphi$ (logits)")
    ax.axhline(0, color="#333", lw=.8)
    ax.grid(True, axis="y", linestyle=":")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    save(fig, "fig4_gpt2_effects")


# ---------------------------------------------------------------- Fig 5
def fig_independence():
    Ls = e6["layers"]
    L = [r["layer"] for r in Ls]
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    ax.plot(L, [r["matched_mean"] for r in Ls], "o-", ms=3.5, color=S1,
            label="matched")
    ax.plot(L, [r["unmatched_mean"] for r in Ls], "s-", ms=3.5, color=S2,
            label="unmatched")
    ax.set_xlabel("layer")
    ax.set_ylabel(r"alignment $\rho$")
    ax.set_ylim(0, None)
    ax.grid(True, linestyle=":")
    s = e6["summary"]
    ax.set_title(f"ratio {s['ratio']:.2f}", loc="left", color="#333")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig5_independence")


# ---------------------------------------------------------------- Fig 6
def fig_realdata():
    ms = e5["models"]
    keys = [("fisher_probe", "Fisher probe", S2),
            ("diff_of_means", "diff-of-means", S5),
            ("dark", "orthogonal to both", S3)]
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    x = np.arange(len(ms))
    w = 0.26
    for i, (k, lab, c) in enumerate(keys):
        ax.bar(x + (i - 1) * w,
               [100 * m["aggregate"][k]["efficiency_vs_gbar"] for m in ms],
               width=w, color=c, label=lab, edgecolor="white", linewidth=.4)
    ax.set_xticks(x)
    ax.set_xticklabels([m["model"].split("/")[-1] for m in ms])
    ax.set_ylabel(r"% of available control")
    ax.set_ylim(0, 125)
    ax.grid(True, axis="y", linestyle=":")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    save(fig, "fig6_realdata")


# ---------------------------------------------------------------- Fig 7
def fig_calibration():
    xs, ys, ls = [], [], []
    for r in A:
        a = r["alpha"]
        for sb, me in zip(r["pool"]["Sbar"], r["pool"]["measured"]):
            xs.append(abs(a * sb))
            ys.append(me)
            ls.append(r["layer"])
    xs, ys, ls = np.array(xs), np.array(ys), np.array(ls)
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    sc = ax.scatter(xs, ys, c=ls, cmap="viridis", s=9, alpha=.85,
                    edgecolor="none", zorder=3)
    lim = max(xs.max(), ys.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color=GREY, lw=1, zorder=2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"predicted $|\alpha \bar{S}|$  (no interventions)")
    ax.set_ylabel(r"measured $|\Delta\varphi|$")
    ax.grid(True, linestyle=":")
    cb = fig.colorbar(sc, ax=ax, pad=.02)
    cb.set_label("layer", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    save(fig, "fig7_calibration")


# ---------------------------------------------------------------- Fig 8
def fig_budget():
    Ls = e4["layers"]
    fig, axes = plt.subplots(1, len(Ls), figsize=(6.9, 2.2), sharey=True)
    if len(Ls) == 1:
        axes = [axes]
    styles = [("oracle", GREY, "--"), ("certificate", S1, "-"),
              ("probe_auc", S2, "-"), ("random", S4, "-")]
    for ax, l in zip(axes, Ls):
        for k, c, ls in styles:
            ax.plot(l["budgets"], l["best_found_frac"][k], ls, color=c, lw=1.4,
                    label=k.replace("_", " "))
        ax.axhline(0.9, color="#bbb", lw=.7, ls=":")
        ax.set_xlabel("budget $B$")
        ax.set_title(f"layer {l['layer']}", fontsize=9, loc="left", color="#333")
        ax.grid(True, linestyle=":")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("fraction of best found")
    axes[-1].legend(frameon=False, fontsize=6.5, loc="lower right")
    save(fig, "fig8_budget")


# ---------------------------------------------------------------- Fig 9
def fig_confound_dial():
    """Both dials side by side: pretrained (E3) and trained-from-scratch (E2).
    Neither shows the widening the pre-registered prediction expected."""
    R = e2["runs"]
    lv = sorted({r["spurious"] for r in R})

    def per_level(fn):
        mu, sd = [], []
        for v in lv:
            vals = [fn(r) for r in R if r["spurious"] == v]
            mu.append(np.mean(vals)); sd.append(np.std(vals))
        return np.array(mu), np.array(sd)

    rho = lambda r: np.mean([l["rho"] for l in r["layers"]])
    eff = lambda r: 100 * (sum(abs(l["directions"]["probe"]["measured_mean_effect"]) for l in r["layers"])
                           / sum(abs(l["directions"]["gbar"]["measured_mean_effect"]) for l in r["layers"]))
    rm, rs = per_level(rho)
    em, es = per_level(eff)

    dial3 = e3.get("B_spurious_dial", [])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.3))

    a1.errorbar(lv, rm, yerr=rs, fmt="o-", ms=4, color=S1, capsize=3,
                label="E2 (trained from scratch)")
    if dial3:
        a1.plot([d["spurious"] for d in dial3], [d["mean_rho"] for d in dial3],
                "s--", ms=4, color=S2, label="E3 (pretrained gpt2)")
    a1.set_xlabel("P(confound matches label)")
    a1.set_ylabel(r"alignment $rho$")
    a1.set_ylim(0, max(rm.max(), 0.2) * 1.6)
    a1.grid(True, linestyle=":")
    a1.legend(frameon=False, fontsize=7)

    a2.errorbar(lv, em, yerr=es, fmt="o-", ms=4, color=S3, capsize=3)
    a2.set_xlabel("P(confound matches label)")
    a2.set_ylabel("probe steering (%)")
    a2.set_ylim(0, max(em.max() * 1.6, 20))
    a2.grid(True, linestyle=":")
    a2.set_title("E2", fontsize=9, loc="left", color="#333")
    save(fig, "fig9_confound_dial")


if __name__ == "__main__":
    print("generating figures from results/data/*.json")
    fig_decoupling()
    fig_leakage()
    fig_gpt2_layerwise()
    fig_gpt2_effects()
    fig_independence()
    fig_realdata()
    fig_calibration()
    fig_budget()
    fig_confound_dial()
    print(f"done -> {FIG}")
