"""E3 -- The probe/steer gap for toxicity in GPT-2.

The application experiment. A toxicity probe is trained on hostile-vs-friendly
framing, exactly as practitioners do, and we ask what it is worth causally.

Parts
  A  layerwise certificate vs measured steering (main result)
  B  the spurious-correlation dial: does the gap widen with confounding?
  C  what the probe actually responds to -- 2x2 cell decomposition and the
     benign-identity false-positive audit
  D  minimal-pair ground truth for the behaviour functional
  E  collateral damage (perplexity) of probe-steering vs certificate-steering

All probes are fitted on a TRAIN split and every AUC is reported on a held-out
TEST split. Steering is measured on the test split.

Run:  python experiments/e3_gpt2_toxicity.py
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
    S_bar, dark_direction, direction_auc, effect_spectrum, estimate_curvature,
    fisher_probe, gbar, pearson, r_eff, rho_alignment, spearman,
    stable_rank_ratio, unit,
)
from egc.toxdata import (  # noqa: E402
    NEUTRAL_WORDS, TOXIC_WORDS, build, build_balanced_cells, minimal_pairs,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "data")
os.makedirs(RESULTS, exist_ok=True)

MODEL = os.environ.get("EGC_MODEL", "gpt2")
N_MAIN = 256           # prompts for probe fitting + adjoints
N_STEER = 96           # prompts for measured steering (cost control)
N_POOL = 64            # prompts for the direction-pool sweep
POOL_SIZE = 18
ALPHA_FRAC = 0.10      # steering magnitude as a fraction of mean ||x_l||


def split_idx(n, frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    k = int(n * frac)
    return p[:k], p[k:]


def direction_pool(Xtr, ytr, Gtr, size=POOL_SIZE, seed=0):
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    wp = fisher_probe(Xtr, ytr, gamma=1e-2)
    wg = unit(gbar(Gtr))
    pool = [("probe", wp), ("gbar", wg)]
    for t in np.linspace(0.15, 0.85, 5):
        pool.append((f"mix{t:.2f}", unit(t * wp + (1 - t) * wg)))
        pool.append((f"mixneg{t:.2f}", unit(t * wp - (1 - t) * wg)))
    _, vecs = effect_spectrum(Gtr)
    for k in (0, 1, 3):
        pool.append((f"eig{k}", unit(vecs[:, k])))
    while len(pool) < size:
        pool.append((f"rand{len(pool)}", unit(rng.normal(size=d))))
    return pool[:size]


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    print(f"loading {MODEL} ...")
    lm = LMAdapter(MODEL)
    pos = lm.token_ids(TOXIC_WORDS)
    neg = lm.token_ids(NEUTRAL_WORDS)
    print(f"  layers={lm.n_layers} d={lm.d_model} pos_tokens={len(pos)} neg_tokens={len(neg)}")

    out = {"experiment": "e3_gpt2_toxicity", "model": MODEL,
           "n_layers": lm.n_layers, "d_model": lm.d_model,
           "n_pos_tokens": len(pos), "n_neg_tokens": len(neg),
           "config": {"N_MAIN": N_MAIN, "N_STEER": N_STEER, "N_POOL": N_POOL,
                      "POOL_SIZE": POOL_SIZE, "ALPHA_FRAC": ALPHA_FRAC}}

    # ==================================================================
    # D. minimal-pair ground truth for the behaviour functional
    # ==================================================================
    print("\n[D] minimal-pair ground truth")
    hos, fri, sid = minimal_pairs(96, seed=3)
    ph = lm.baseline_phi(hos, pos, neg)
    pf = lm.baseline_phi(fri, pos, neg)
    out["D_minimal_pairs"] = {
        "n_pairs": len(hos),
        "phi_hostile_mean": float(ph.mean()), "phi_friendly_mean": float(pf.mean()),
        "phi_gap_mean": float((ph - pf).mean()),
        "phi_gap_std": float((ph - pf).std()),
        "frac_pairs_correct_sign": float((ph > pf).mean()),
        "phi_gap_identity_subject": float((ph - pf)[sid == 1].mean()),
        "phi_gap_neutral_subject": float((ph - pf)[sid == 0].mean()),
    }
    d0 = out["D_minimal_pairs"]
    print(f"  phi(hostile)-phi(friendly) = {d0['phi_gap_mean']:+.3f} "
          f"(sd {d0['phi_gap_std']:.3f}); correct sign on {d0['frac_pairs_correct_sign']*100:.1f}% of pairs")

    # ==================================================================
    # A. layerwise certificate vs measured steering
    # ==================================================================
    print("\n[A] layerwise analysis (spurious=0.85)")
    prompts, y, s = build(N_MAIN, spurious=0.85, seed=0)
    tr, te = split_idx(len(prompts), 0.5, seed=0)
    X, G, phi = lm.states_and_adjoints(prompts, pos, neg, batch=16)
    print(f"  states+adjoints done ({time.time()-t0:.0f}s)")

    p_te = [prompts[i] for i in te]
    steer_idx = te[:N_STEER]
    p_steer = [prompts[i] for i in steer_idx]
    base_steer = lm.baseline_phi(p_steer, pos, neg)
    pool_idx = te[:N_POOL]
    p_pool = [prompts[i] for i in pool_idx]
    base_pool = lm.baseline_phi(p_pool, pos, neg)

    layers = []
    for L in range(lm.n_layers):
        Xl, Gl = X[L], G[L]
        Xtr, ytr, Gtr = Xl[tr], y[tr], Gl[tr]
        Xte, yte, ste = Xl[te], y[te], s[te]

        w_probe = fisher_probe(Xtr, ytr, gamma=1e-2)
        w_ident = fisher_probe(Xtr, s[tr], gamma=1e-2)      # confound probe
        w_gbar = unit(gbar(Gtr))
        try:
            v_dark = dark_direction(Gtr, w_probe[:, None])
        except ValueError:
            v_dark = None
        rng = np.random.default_rng(100 + L)
        w_rand = unit(rng.normal(size=Xl.shape[1]))

        resid_norm = float(np.linalg.norm(Xl, axis=1).mean())
        alpha = ALPHA_FRAC * resid_norm

        rec = {
            "layer": L,
            "resid_norm_mean": resid_norm,
            "alpha": alpha,
            "probe_auc_frame_test": direction_auc(Xte, yte, w_probe),
            "probe_auc_identity_test": direction_auc(Xte, ste, w_probe),
            "identity_probe_auc_identity_test": direction_auc(Xte, ste, w_ident),
            "rho_probe_gbar": rho_alignment(Gtr, w_probe),
            "rho_identityprobe_gbar": rho_alignment(Gtr, w_ident),
            "cos_probe_identityprobe": float(abs(unit(w_probe) @ unit(w_ident))),
            "r_eff": r_eff(Gtr),
            "r_eff_over_d": stable_rank_ratio(Gtr),
            "gbar_norm": float(np.linalg.norm(gbar(Gtr))),
            "directions": {},
        }

        cand = {"probe": w_probe, "gbar": w_gbar, "random": w_rand,
                "identity_probe": w_ident}
        if v_dark is not None:
            cand["dark"] = v_dark

        for nm, w in cand.items():
            meas = lm.measure_steering(p_steer, L, w, alpha, pos, neg, base=base_steer)
            rec["directions"][nm] = {
                "Sbar": S_bar(Gtr, w),
                "predicted_effect": float(alpha * S_bar(Gtr, w)),
                "measured_mean_effect": float(meas.mean()),
                "measured_std": float(meas.std()),
                "measured_abs_mean": float(np.abs(meas).mean()),
                "auc_frame_test": direction_auc(Xte, yte, w),
                "auc_identity_test": direction_auc(Xte, ste, w),
            }

        # --- pool: does Sbar rank-predict measured effect, does AUC? -------
        pool = direction_pool(Xtr, ytr, Gtr, seed=L)
        sb, au, me = [], [], []
        for nm, w in pool:
            sb.append(S_bar(Gtr, w))
            au.append(direction_auc(Xte, yte, w))
            me.append(float(np.abs(
                lm.measure_steering(p_pool, L, w, alpha, pos, neg, base=base_pool)
            ).mean()))
        rec["pool"] = {
            "names": [n for n, _ in pool],
            "Sbar": sb, "auc": au, "measured": me,
            "spearman_Sbar_vs_measured": spearman(np.array(sb), np.array(me)),
            "pearson_Sbar_vs_measured": pearson(np.array(sb), np.array(me)),
            "spearman_auc_vs_measured": spearman(np.array(au), np.array(me)),
            "spearman_absauc_vs_measured": spearman(np.abs(np.array(au) - 0.5),
                                                    np.array(me)),
        }

        # --- Prop 6: curvature and validity radius -------------------------
        sub = p_steer[:48]
        base_sub = base_steer[:48]

        def eff_fn(a, _L=L, _w=w_gbar):
            return float(lm.measure_steering(sub, _L, _w, a, pos, neg,
                                             base=base_sub).mean())

        grid = np.linspace(-2 * alpha, 2 * alpha, 9)
        M = estimate_curvature(eff_fn, None, w_gbar, grid)
        sb_g = S_bar(Gtr, w_gbar)
        rec["curvature_M"] = M
        rec["astar_gbar"] = float(2 * sb_g / M) if M > 1e-12 else float("inf")
        rec["astar_over_alpha"] = (rec["astar_gbar"] / alpha) if alpha > 0 else float("inf")
        rec["alpha_sweep_gbar"] = [
            {"alpha": float(a), "measured": eff_fn(float(a)),
             "predicted": float(a * sb_g)} for a in grid
        ]

        # --- E: collateral damage -----------------------------------------
        rec["perplexity_delta"] = {
            nm: lm.perplexity_delta(sub, L, cand[nm], alpha)
            for nm in ("probe", "gbar")
        }

        layers.append(rec)
        dd = rec["directions"]
        print(f"  L{L:>2}: AUC={rec['probe_auc_frame_test']:.3f} "
              f"AUCid={rec['probe_auc_identity_test']:.3f} "
              f"rho={rec['rho_probe_gbar']:.3f} reff/d={rec['r_eff_over_d']:.4f} | "
              f"meas probe={dd['probe']['measured_mean_effect']:+.3f} "
              f"gbar={dd['gbar']['measured_mean_effect']:+.3f} "
              f"dark={dd.get('dark',{}).get('measured_mean_effect',float('nan')):+.3f} "
              f"rand={dd['random']['measured_mean_effect']:+.3f} | "
              f"sp(S,m)={rec['pool']['spearman_Sbar_vs_measured']:+.2f} "
              f"sp(AUC,m)={rec['pool']['spearman_auc_vs_measured']:+.2f}")

    out["A_layerwise"] = layers

    # ==================================================================
    # B. spurious dial (certificates + 3 directions, cheaper)
    # ==================================================================
    print("\n[B] spurious-correlation dial")
    dial = []
    for spur in (0.50, 0.70, 0.85, 0.95):
        pr, yy, ss = build(N_MAIN, spurious=spur, seed=11)
        tr2, te2 = split_idx(len(pr), 0.5, seed=11)
        X2, G2, _ = lm.states_and_adjoints(pr, pos, neg, batch=16)
        sidx = te2[:N_STEER]
        ps = [pr[i] for i in sidx]
        bs = lm.baseline_phi(ps, pos, neg)
        rows = []
        for L in range(lm.n_layers):
            Xl, Gl = X2[L], G2[L]
            wp = fisher_probe(Xl[tr2], yy[tr2], gamma=1e-2)
            wg = unit(gbar(Gl[tr2]))
            alpha = ALPHA_FRAC * float(np.linalg.norm(Xl, axis=1).mean())
            mp = float(lm.measure_steering(ps, L, wp, alpha, pos, neg, base=bs).mean())
            mg = float(lm.measure_steering(ps, L, wg, alpha, pos, neg, base=bs).mean())
            rows.append({
                "layer": L,
                "probe_auc": direction_auc(Xl[te2], yy[te2], wp),
                "probe_auc_identity": direction_auc(Xl[te2], ss[te2], wp),
                "rho": rho_alignment(Gl[tr2], wp),
                "r_eff_over_d": stable_rank_ratio(Gl[tr2]),
                "measured_probe": mp, "measured_gbar": mg,
                "efficiency": float(mp / mg) if abs(mg) > 1e-9 else float("nan"),
            })
        best = max(rows, key=lambda r: r["probe_auc"])
        dial.append({"spurious": spur, "rows": rows,
                     "mean_rho": float(np.mean([r["rho"] for r in rows])),
                     "mean_efficiency": float(np.nanmean([r["efficiency"] for r in rows])),
                     "best_auc_layer": best["layer"], "best_auc": best["probe_auc"]})
        print(f"  spurious={spur:.2f}: mean rho={dial[-1]['mean_rho']:.3f} "
              f"mean steering efficiency={dial[-1]['mean_efficiency']:.3f} "
              f"best probe AUC={best['probe_auc']:.3f} @L{best['layer']}")
    out["B_spurious_dial"] = dial

    # ==================================================================
    # C. what does the probe actually respond to?
    # ==================================================================
    print("\n[C] cell decomposition and benign-identity false positives")
    pc, yc, sc, cell = build_balanced_cells(64, seed=5)
    Xc, Gc, _ = lm.states_and_adjoints(pc, pos, neg, batch=16)
    cells = []
    for L in range(lm.n_layers):
        Xl = Xc[L]
        trc, tec = split_idx(len(pc), 0.5, seed=5)
        wp = fisher_probe(Xl[trc], yc[trc], gamma=1e-2)
        sc_scores = Xl @ unit(wp)
        thr = float(np.median(sc_scores[trc]))
        row = {"layer": L,
               "auc_frame": direction_auc(Xl[tec], yc[tec], wp),
               "auc_identity_within_benign": direction_auc(
                   Xl[(cell == "benign_identity") | (cell == "benign_neutral")],
                   (cell[(cell == "benign_identity") | (cell == "benign_neutral")]
                    == "benign_identity").astype(int), wp)}
        for cname in ("hostile_identity", "hostile_neutral",
                      "benign_identity", "benign_neutral"):
            m = cell == cname
            row[f"mean_score_{cname}"] = float(sc_scores[m].mean())
            row[f"fpr_{cname}"] = float((sc_scores[m] > thr).mean())
        cells.append(row)
        if L in (0, lm.n_layers // 2, lm.n_layers - 1):
            print(f"  L{L:>2}: AUC(frame)={row['auc_frame']:.3f}  "
                  f"flag-rate benign_identity={row['fpr_benign_identity']:.3f} vs "
                  f"benign_neutral={row['fpr_benign_neutral']:.3f}  "
                  f"AUC(identity|benign)={row['auc_identity_within_benign']:.3f}")
    out["C_cells"] = cells

    out["wall_time_s"] = time.time() - t0
    path = os.path.join(RESULTS, "e3_gpt2_toxicity.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}  ({out['wall_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
