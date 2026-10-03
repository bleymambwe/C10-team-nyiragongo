"""Build the results dashboard from results/data/*.json.

Reads whatever experiment outputs exist, summarises them, and injects the
summary plus a render script into dashboard/template.html.

Run:  python dashboard/build.py
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("EGC_DATA_DIR") or os.path.join(ROOT, "results", "data")


def load(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and x == x]
    return sum(xs) / len(xs) if xs else None


def summarise():
    e1, e2 = load("e1_exact_decoupling.json"), load("e2_trained_toy.json")
    e3, e4 = load("e3_gpt2_toxicity.json"), load("e4_active_design.json")
    e5, e6 = load("e5_real_data_multimodel.json"), load("e6_independence.json")
    out = {"e1": None, "e2": None, "e3": None, "e4": None, "e5": None, "e6": None,
           "have": []}

    # ---------------- E1 ----------------
    if e1:
        out["have"].append("e1")
        c = e1["checks"]
        g = e1["sweep"]["auc_vs_certificate_grid"]
        p7 = c["P7_finite_sample_leakage"]
        out["e1"] = {
            "exact": {k: c[k] for k in ("axis_aligned", "random_rotation")},
            "grid": [{"auc": r["probe_auc"], "sbar": r["Sbar_probe"],
                      "theta": r["theta_deg"], "dm": r["delta_mu"], "rho": r["rho"]} for r in g],
            "corr_auc_sbar": e1["sweep"]["auc_vs_Sbar_pearson"],
            "corr_auc_sbar_sp": e1["sweep"]["auc_vs_Sbar_spearman"],
            "p3": c["P3_random_direction_inertness"],
            "p3_iso": c["P3_isotropic_control"],
            "p5": c["P5_dark_direction_optimality"],
            "p7_rows": p7["rows"],
            "p7_n": p7["model_inv_sqrt_n"], "p7_dn": p7["model_sqrt_d_over_n"],
            "p7_verdict": p7["verdict"],
        }

    # ---------------- E2 ----------------
    if e2 and e2.get("runs"):
        out["have"].append("e2")
        by_spur = {}
        for r in e2["runs"]:
            by_spur.setdefault(r["spurious"], []).append(r)
        levels = sorted(by_spur)
        n_layers = len(e2["runs"][0]["layers"])
        series = {}
        for key in ("probe_auc", "rho", "r_eff_over_d"):
            series[key] = {s: [mean([run["layers"][L][key] for run in by_spur[s]])
                               for L in range(n_layers)] for s in levels}
        eff = {}
        for s in levels:
            eff[s] = []
            for L in range(n_layers):
                p = mean([abs(run["layers"][L]["directions"]["probe"]["measured_mean_effect"])
                          for run in by_spur[s]])
                gg = mean([abs(run["layers"][L]["directions"]["gbar"]["measured_mean_effect"])
                           for run in by_spur[s]])
                eff[s].append(p / gg if gg else None)
        dirbars = {}
        for nm in ("probe", "gbar", "dark", "random"):
            dirbars[nm] = [mean([abs(run["layers"][L]["directions"].get(nm, {})
                                     .get("measured_mean_effect", float("nan")))
                                 for run in by_spur[levels[-1]]]) for L in range(n_layers)]
        sp_s = [mean([run["layers"][L]["pool"]["spearman_Sbar_vs_measured"]
                      for s in levels for run in by_spur[s]]) for L in range(n_layers)]
        sp_a = [mean([run["layers"][L]["pool"]["spearman_auc_vs_measured"]
                      for s in levels for run in by_spur[s]]) for L in range(n_layers)]
        out["e2"] = {
            "levels": levels, "n_layers": n_layers, "series": series, "eff": eff,
            "dirbars": dirbars, "sp_sbar": sp_s, "sp_auc": sp_a,
            "top_spur": levels[-1],
            "acc": mean([r["test_acc"] for r in e2["runs"]]),
            "n_runs": len(e2["runs"]), "seeds": sorted({r["seed"] for r in e2["runs"]}),
            "mean_rho": mean([l["rho"] for r in e2["runs"] for l in r["layers"]]),
            "mean_reff": mean([l["r_eff_over_d"] for r in e2["runs"] for l in r["layers"]]),
            "wall": e2.get("wall_time_s"),
            "partial": bool(e2.get("partial")),
            "n_expected": 12,
            "rho_by_level": {s: mean([mean([l["rho"] for l in run["layers"]])
                                      for run in by_spur[s]]) for s in levels},
            "rho_sd_by_level": {
                s: (lambda v: (sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** 0.5)(
                    [mean([l["rho"] for l in run["layers"]]) for run in by_spur[s]])
                for s in levels},
            "marker_by_level": {s: mean([run["test_acc_vs_marker"] for run in by_spur[s]])
                                for s in levels},
        }

    # ---------------- E3 ----------------
    if e3 and e3.get("A_layerwise"):
        out["have"].append("e3")
        A = e3["A_layerwise"]
        L = [r["layer"] for r in A]

        def dser(nm, field="measured_mean_effect"):
            return [r["directions"].get(nm, {}).get(field) for r in A]
        out["e3"] = {
            "model": e3["model"], "d": e3["d_model"], "layers": L,
            "auc_frame": [r["probe_auc_frame_test"] for r in A],
            "auc_ident": [r["probe_auc_identity_test"] for r in A],
            "rho": [r["rho_probe_gbar"] for r in A],
            "reff": [r["r_eff_over_d"] for r in A],
            "cos_probe_ident": [r["cos_probe_identityprobe"] for r in A],
            "meas": {nm: dser(nm) for nm in ("probe", "gbar", "dark", "random", "identity_probe")},
            "sbar": {nm: dser(nm, "Sbar") for nm in ("probe", "gbar", "dark", "random")},
            "auc_of": {nm: dser(nm, "auc_frame_test") for nm in ("probe", "gbar", "dark", "random")},
            "sp_sbar": [r["pool"]["spearman_Sbar_vs_measured"] for r in A],
            "sp_auc": [r["pool"]["spearman_auc_vs_measured"] for r in A],
            "alpha": [r["alpha"] for r in A],
            "pool_points": [{"sbar": s, "auc": a, "meas": m, "layer": r["layer"],
                             "pred": r["alpha"] * s}
                            for r in A for s, a, m in zip(r["pool"]["Sbar"], r["pool"]["auc"],
                                                          r["pool"]["measured"])],
            "alpha_sweep": [{"layer": r["layer"], "pts": r["alpha_sweep_gbar"],
                             "astar": r.get("astar_gbar"), "M": r.get("curvature_M"),
                             "alpha": r["alpha"]} for r in A],
            "ppl": {nm: [r.get("perplexity_delta", {}).get(nm) for r in A]
                    for nm in ("probe", "gbar")},
            "minpair": e3.get("D_minimal_pairs"),
            "dial": e3.get("B_spurious_dial"),
            "cells": e3.get("C_cells"),
            "wall": e3.get("wall_time_s"),
        }
        mg = [abs(v) for v in out["e3"]["meas"]["gbar"] if v is not None]
        mp = [abs(v) for v in out["e3"]["meas"]["probe"] if v is not None]
        md = [abs(v) for v in out["e3"]["meas"]["dark"] if v is not None]
        out["e3"]["eff_probe"] = (sum(mp) / sum(mg)) if mg else None
        out["e3"]["eff_dark"] = (sum(md) / sum(mg)) if mg else None
        out["e3"]["mean_rho"] = mean(out["e3"]["rho"])
        out["e3"]["mean_reff"] = mean(out["e3"]["reff"])
        out["e3"]["mean_sp_sbar"] = mean(out["e3"]["sp_sbar"])
        out["e3"]["mean_sp_auc"] = mean(out["e3"]["sp_auc"])
        out["e3"]["max_auc"] = max([a for a in out["e3"]["auc_frame"] if a is not None] or [0])

    # ---------------- E6 ----------------
    if e6 and e6.get("layers"):
        out["have"].append("e6")
        Ls = e6["layers"]
        out["e6"] = {
            "model": e6["model"], "n": e6["n"],
            "behaviours": e6["behaviours"], "labels": e6["labels"],
            "matched": e6["matched"],
            "layers": [r["layer"] for r in Ls],
            "matched_series": [r["matched_mean"] for r in Ls],
            "unmatched_series": [r["unmatched_mean"] for r in Ls],
            "rho_matrix_last": Ls[-1]["rho_matrix"],
            "rho_matrix_mid": Ls[len(Ls) // 2]["rho_matrix"],
            "probe_auc_mid": Ls[len(Ls) // 2]["probe_auc"],
            "summary": e6["summary"],
        }

    # ---------------- E4 ----------------
    if e4 and e4.get("layers"):
        out["have"].append("e4")
        out["e4"] = {"model": e4["model"], "M": e4["M_cand"], "layers": e4["layers"]}

    # ---------------- E5 ----------------
    if e5 and e5.get("models"):
        ok = [m for m in e5["models"] if "error" not in m]
        if ok:
            out["have"].append("e5")
            out["e5"] = {
                "dataset": e5["dataset"], "n": e5["n_prompts"],
                "partial": e5.get("partial", False),
                "recovery_note": e5.get("recovery_note"),
                "not_completed": e5.get("models_not_completed", []),
                "models": [{
                    "name": m["model"], "n_layers": m["n_layers"], "d": m["d_model"],
                    "phi_auc": m.get("phi_separation_auc"),
                    "mean_reff": m.get("mean_r_eff_over_d"),
                    "agg": m.get("aggregate", {}),
                    "layers": [r["layer"] for r in m["layers"]],
                    "series": {nm: [r["directions"].get(nm, {}).get("measured_mean_effect")
                                    for r in m["layers"]]
                               for nm in ("fisher_probe", "logistic_probe", "diff_of_means",
                                          "gbar", "dark", "random")},
                    "auc": {nm: [r["directions"].get(nm, {}).get("auc_test") for r in m["layers"]]
                            for nm in ("fisher_probe", "logistic_probe", "diff_of_means",
                                       "gbar", "dark", "random")},
                    "rho": {nm: [r["directions"].get(nm, {}).get("rho") for r in m["layers"]]
                            for nm in ("fisher_probe", "logistic_probe", "diff_of_means")},
                    "reff": [r["r_eff_over_d"] for r in m["layers"]],
                } for m in ok],
                "errors": [m for m in e5["models"] if "error" in m],
            }
    return out


def provenance():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=ROOT, text=True).strip()
    except Exception:
        sha = "n/a"
    import platform
    import numpy
    try:
        import torch
        tv = torch.__version__
    except Exception:
        tv = "n/a"
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "commit": sha, "python": platform.python_version(),
        "torch": tv, "numpy": numpy.__version__,
        "hardware": "CPU only (4 cores, 8.5 GB RAM), no CUDA",
    }


def ascii_armour(html: str) -> str:
    """Make the page charset-proof.

    The artifact host and a plain file:// open may disagree about the default
    encoding, and a mis-guessed charset turns every em-dash and Greek letter
    into mojibake. Escaping all non-ASCII removes the dependency entirely:
    numeric entities in markup, \\uXXXX escapes inside <script> (where entities
    are not decoded). <style> is verified ASCII-only, so it needs neither.
    """
    import re

    def esc_html(s):
        return "".join(c if ord(c) < 128 else f"&#{ord(c)};" for c in s)

    def esc_js(s):
        return "".join(c if ord(c) < 128 else f"\\u{ord(c):04x}" for c in s)

    parts, pos = [], 0
    for m in re.finditer(r"<script\b[^>]*>.*?</script>", html, re.S | re.I):
        parts.append(esc_html(html[pos:m.start()]))
        body = m.group(0)
        i, j = body.index(">") + 1, body.rindex("</script>")
        parts.append(body[:i] + esc_js(body[i:j]) + body[j:])
        pos = m.end()
    parts.append(esc_html(html[pos:]))
    return "".join(parts)


def main():
    summary = summarise()
    summary["prov"] = provenance()
    with open(os.path.join(HERE, "template.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(HERE, "render.js"), encoding="utf-8") as f:
        render = f.read()
    # NaN/Infinity are valid in Python's json output but NOT in JavaScript's
    # JSON.parse, so one leaking through would blank the whole page. Fail loudly.
    blob = json.dumps(summary, allow_nan=False)
    html = tpl.replace("__DATA__", blob).replace("__RENDER__", render)
    html = ascii_armour(html)
    assert all(ord(c) < 128 for c in html), "non-ASCII survived armouring"
    outp = os.environ.get("EGC_OUT_HTML") or os.path.join(HERE, "index.html")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {outp}  ({len(html)/1024:.0f} KB)  sections: {summary['have']}")


if __name__ == "__main__":
    main()
