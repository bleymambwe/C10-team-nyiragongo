"""Recover E5's completed per-layer results from its console log.

E5 writes its JSON only after all three models finish, so when the run was
stopped partway through the third model the two completed models' results were
stranded in the log. The numbers were genuinely computed; this parses them back
into the schema the dashboard expects.

The recovered record is explicitly marked `partial: true` and carries only the
fields the log actually printed. Quantities that were computed but never
printed -- bootstrap CIs, next-token KL, Spearman per layer -- are absent
rather than invented, and the dashboard renders their absence.

Run:  python experiments/recover_e5_from_log.py
"""

from __future__ import annotations

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "results", "logs", "e5.log")
OUT = os.path.join(ROOT, "results", "data", "e5_real_data_multimodel.json")

HEAD = re.compile(r"^\s*layers=(\d+) d=(\d+) pos=(\d+) neg=(\d+)")
PHI = re.compile(r"adjoints done \((\d+)s\); phi toxic=([-\d.]+) benign=([-\d.]+)")
ROW = re.compile(
    r"^\s*L\s*(\d+) a=\s*([\d.]+) reff/d=([\d.]+) \| "
    r"fisher AUC=([\d.]+) rho=([\d.]+) m=([-+][\d.]+) \| "
    r"dom AUC=([\d.]+) rho=([\d.]+) m=([-+][\d.]+) \| "
    r"gbar m=([-+][\d.]+) \| dark m=([-+][\d.]+) AUC=([\d.]+)")

KNOWN_MODELS = ("gpt2", "distilgpt2", "EleutherAI/pythia-70m")


def main():
    with open(LOG, encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]

    models, cur = [], None
    for ln in lines:
        s = ln.strip()
        if s in KNOWN_MODELS:
            cur = {"model": s, "layers": [], "n_layers": None, "d_model": None}
            models.append(cur)
            continue
        if cur is None:
            continue
        m = HEAD.match(ln)
        if m:
            cur["n_layers"], cur["d_model"] = int(m.group(1)), int(m.group(2))
            continue
        m = PHI.search(ln)
        if m:
            cur["phi_toxic_mean"] = float(m.group(2))
            cur["phi_benign_mean"] = float(m.group(3))
            continue
        m = ROW.match(ln)
        if m:
            g = m.groups()
            cur["layers"].append({
                "layer": int(g[0]), "alpha": float(g[1]),
                "r_eff_over_d": float(g[2]),
                "directions": {
                    "fisher_probe": {"auc_test": float(g[3]), "rho": float(g[4]),
                                     "measured_mean_effect": float(g[5])},
                    "diff_of_means": {"auc_test": float(g[6]), "rho": float(g[7]),
                                      "measured_mean_effect": float(g[8])},
                    "gbar": {"measured_mean_effect": float(g[9])},
                    "dark": {"measured_mean_effect": float(g[10]),
                             "auc_test": float(g[11])},
                },
            })

    complete = [m for m in models if m["layers"] and len(m["layers"]) == m["n_layers"]]
    incomplete = [m["model"] for m in models if m not in complete]

    for m in complete:
        agg = {}
        for nm in ("fisher_probe", "diff_of_means", "gbar", "dark"):
            vals = [l["directions"][nm] for l in m["layers"] if nm in l["directions"]]
            agg[nm] = {
                "mean_auc": (sum(v["auc_test"] for v in vals) / len(vals)
                             if all("auc_test" in v for v in vals) else None),
                "mean_rho": (sum(v["rho"] for v in vals) / len(vals)
                             if all("rho" in v for v in vals) else None),
                "mean_abs_measured": sum(abs(v["measured_mean_effect"]) for v in vals) / len(vals),
                "max_abs_measured": max(abs(v["measured_mean_effect"]) for v in vals),
            }
        base = agg["gbar"]["mean_abs_measured"]
        for nm in agg:
            if nm != "gbar" and base:
                agg[nm]["efficiency_vs_gbar"] = agg[nm]["mean_abs_measured"] / base
        m["aggregate"] = agg
        m["mean_r_eff_over_d"] = sum(l["r_eff_over_d"] for l in m["layers"]) / len(m["layers"])

    out = {
        "experiment": "e5_real_data_multimodel",
        "dataset": "RealToxicityPrompts",
        "n_prompts": 400, "n_steer": 100, "alpha_frac": 0.10,
        "partial": True,
        "recovery_note": (
            "Recovered from results/logs/e5.log after the run was stopped during "
            f"the third model. Models completed: {[m['model'] for m in complete]}. "
            f"Not completed: {incomplete}. Bootstrap CIs, next-token KL and "
            "per-layer Spearman were computed in-run but not printed, so they are "
            "absent here rather than reconstructed. Re-run "
            "experiments/e5_real_data_multimodel.py for the full record."),
        "models": complete,
        "models_not_completed": incomplete,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    names = ", ".join(f"{m['model']} ({len(m['layers'])} layers)" for m in complete)
    print(f"recovered {len(complete)} model(s): {names}")
    for m in complete:
        a = m["aggregate"]
        print(f"\n  {m['model']}  (mean over {len(m['layers'])} layers, r_eff/d={m['mean_r_eff_over_d']:.4f})")
        for nm in ("fisher_probe", "diff_of_means", "dark", "gbar"):
            x = a[nm]
            auc = f"{x['mean_auc']:.3f}" if x["mean_auc"] is not None else "  -  "
            rho = f"{x['mean_rho']:.3f}" if x["mean_rho"] is not None else "  -  "
            print(f"    {nm:<15} AUC={auc} rho={rho} "
                  f"|effect|={x['mean_abs_measured']:.3f} "
                  f"eff_vs_gbar={x.get('efficiency_vs_gbar', 1.0)*100:5.1f}%")
    print(f"\nnot completed: {incomplete}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
