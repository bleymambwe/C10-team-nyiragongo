"""E1 -- Exact decoupling of probe quality and steering efficacy.

Numerically verifies Theorem 2, Propositions 3-5 and Proposition 7 of
docs/01_THEORY.md in a setting where the theory is exact (linear Phi, so the
second-order remainder vanishes identically).

Claims tested
  T2(a) probe AUC matches the closed form Ncdf(Dm / (s sqrt 2))
  T2(b) steering along the POPULATION probe direction is exactly zero
  T2(c) a probe-invisible direction (AUC = 1/2) attains the MAXIMAL effect
  P3    E_w[S2^2] = tr(W_g)/d over random unit directions
  P4    Sbar(w_probe) = ||gbar|| * rho
  P5    the closed-form dark direction is the constrained optimum
  P7    a FINITE-SAMPLE probe acquires spurious causal effect of order
        sqrt(d/N) even when the population effect is exactly zero

Also sweeps the (AUC, Sbar) plane to show the two axes are independently
settable -- Corollary 2.1.

Run:  python experiments/e1_exact_decoupling.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from egc.core import (  # noqa: E402
    S_bar, S_two, dark_direction, direction_auc, effect_gramian, fisher_probe,
    gbar, pearson, r_eff, rho_alignment, spearman, stable_rank_ratio, unit,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "data")
os.makedirs(RESULTS, exist_ok=True)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def build_system(d, n, delta_mu, sigma, rng, rotate=False):
    """Theorem 2 construction.

    Behaviour   Phi(x) = e_1^T x           -> g = e_1 for every example
    Represent.  x | y ~ N(y * delta_mu * e_2, sigma^2 I)
    Optionally conjugate everything by a random rotation Q to show the result
    is basis-free, not an axis-alignment artefact. Returns the POPULATION
    probe direction as well, which under this construction is exactly e_2
    (rotated).
    """
    y = rng.integers(0, 2, size=n)
    e1 = np.zeros(d); e1[0] = 1.0
    e2 = np.zeros(d); e2[1] = 1.0
    X = rng.normal(0.0, sigma, size=(n, d)) + np.outer(y, delta_mu * e2)
    g_dir, w_pop = e1.copy(), e2.copy()
    if rotate:
        Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
        X = X @ Q.T
        g_dir = Q @ g_dir
        w_pop = Q @ w_pop
    G = np.tile(g_dir, (n, 1))          # exact adjoints: Phi is linear
    return X, y, G, g_dir, w_pop


def phi_linear(X, g_dir):
    return X @ g_dir


def max_abs_steer_effect(X, g_dir, w, alphas=(-8.0, -1.0, -0.1, 0.1, 1.0, 8.0)):
    base = phi_linear(X, g_dir)
    m = 0.0
    for a in alphas:
        eff = phi_linear(X + a * unit(w)[None, :], g_dir) - base
        m = max(m, float(np.abs(eff).max()))
    return m


def main():
    rng = np.random.default_rng(0)
    out = {"experiment": "e1_exact_decoupling", "checks": {}, "sweep": {}}

    # ------------------------------------------------------------------
    # Theorem 2, axis-aligned and rotated.
    # Population probe -> exact statements. Empirical probe -> P7.
    # ------------------------------------------------------------------
    for tag, rotate in (("axis_aligned", False), ("random_rotation", True)):
        d, n, sigma, delta_mu = 64, 20000, 1.0, 4.0
        X, y, G, g_dir, w_pop = build_system(d, n, delta_mu, sigma, rng, rotate=rotate)

        w_emp = fisher_probe(X, y, gamma=1e-6)
        auc_pred = normal_cdf(delta_mu / (sigma * math.sqrt(2.0)))

        # --- exact claims, evaluated at the population probe -------------
        eff_pop = max_abs_steer_effect(X, g_dir, w_pop)
        v_dark_pop = dark_direction(G, w_pop[:, None])
        base = phi_linear(X, g_dir)
        eff_dark_pop = float(np.mean(
            phi_linear(X + 1.0 * v_dark_pop[None, :], g_dir) - base))
        gb = gbar(G)

        out["checks"][tag] = {
            "d": d, "n": n, "delta_mu": delta_mu, "sigma": sigma,
            # T2(a)
            "probe_auc_population": direction_auc(X, y, w_pop),
            "probe_auc_empirical": direction_auc(X, y, w_emp),
            "probe_auc_closed_form": auc_pred,
            "probe_auc_abs_error_population": abs(direction_auc(X, y, w_pop) - auc_pred),
            # T2(b) -- exact at the population probe
            "T2b_max_abs_effect_population_probe": eff_pop,
            "T2b_max_abs_effect_empirical_probe": max_abs_steer_effect(X, g_dir, w_emp),
            # T2(c) -- exact at the population dark direction
            "T2c_dark_auc_population": direction_auc(X, y, v_dark_pop),
            "T2c_dark_auc_abs_error_from_half": abs(direction_auc(X, y, v_dark_pop) - 0.5),
            "T2c_dark_effect_per_unit_alpha": eff_dark_pop,
            "T2c_max_attainable_effect_norm_gbar": float(np.linalg.norm(gb)),
            "T2c_dark_effect_gap": abs(eff_dark_pop - float(np.linalg.norm(gb))),
            # P4
            "P4_rho_population_probe": rho_alignment(G, w_pop),
            "P4_rho_empirical_probe": rho_alignment(G, w_emp),
            "P4_Sbar_population_probe": S_bar(G, w_pop),
            "P4_Sbar_predicted_population": float(np.linalg.norm(gb) * rho_alignment(G, w_pop)),
        }

    # ------------------------------------------------------------------
    # P7: spurious causal leakage of a finite-sample probe.
    # Population truth: rho = 0 exactly. Measure rho_hat(N, d) and test the
    # predicted sqrt(d/N) scaling.
    # ------------------------------------------------------------------
    leak = []
    for d in (16, 64, 256):
        for n in (200, 500, 1000, 4000, 16000, 64000):
            if n < 4 * d:
                continue
            rhos, effs = [], []
            for trial in range(12):
                r2 = np.random.default_rng(1000 * d + n + trial)
                X, y, G, g_dir, w_pop = build_system(d, n, 4.0, 1.0, r2, rotate=True)
                w_emp = fisher_probe(X, y, gamma=1e-6)
                rhos.append(rho_alignment(G, w_emp))
                effs.append(S_bar(G, w_emp))
            leak.append({
                "d": d, "n": n,
                "rho_hat_mean": float(np.mean(rhos)),
                "rho_hat_std": float(np.std(rhos)),
                "spurious_Sbar_mean": float(np.mean(effs)),
                "inv_sqrt_n": float(1.0 / math.sqrt(n)),
                "sqrt_d_over_n": float(math.sqrt(d / n)),
                "rho_times_sqrt_n": float(np.mean(rhos) * math.sqrt(n)),
                "rho_over_sqrt_d_over_n": float(np.mean(rhos) / math.sqrt(d / n)),
            })
    # Competing scalings: rho ~ N^{-1/2} (dimension-free) vs rho ~ sqrt(d/N).
    ly = np.log(np.array([r["rho_hat_mean"] for r in leak]))
    lx_n = np.log(np.array([r["inv_sqrt_n"] for r in leak]))
    lx_dn = np.log(np.array([r["sqrt_d_over_n"] for r in leak]))
    b_n, a_n = np.polyfit(lx_n, ly, 1)
    b_dn, a_dn = np.polyfit(lx_dn, ly, 1)
    out["checks"]["P7_finite_sample_leakage"] = {
        "rows": leak,
        "model_inv_sqrt_n": {
            "loglog_slope": float(b_n), "loglog_intercept": float(a_n),
            "predicted_slope": 1.0, "r2": float(pearson(lx_n, ly) ** 2),
        },
        "model_sqrt_d_over_n": {
            "loglog_slope": float(b_dn), "loglog_intercept": float(a_dn),
            "predicted_slope": 1.0, "r2": float(pearson(lx_dn, ly) ** 2),
        },
        "verdict": "dimension-free N^{-1/2}" if pearson(lx_n, ly) ** 2 > pearson(lx_dn, ly) ** 2
                   else "dimension-dependent sqrt(d/N)",
    }

    # ------------------------------------------------------------------
    # Proposition 3: random-direction inertness.
    # (i) rank-deficient constructions, (ii) exactly isotropic control.
    # ------------------------------------------------------------------
    d, n = 128, 4000
    p3 = []
    for rank in (1, 2, 4, 8, 16, 32, 64, 128):
        B = rng.normal(size=(d, rank))
        G = rng.normal(size=(n, rank)) @ B.T
        W = effect_gramian(G)
        lam1 = float(np.linalg.eigvalsh(W)[-1])
        Wd = rng.normal(size=(20000, d))
        Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
        s2sq = np.einsum("ij,jk,ik->i", Wd, W, Wd)
        p3.append({
            "construction_rank": rank,
            "r_eff": r_eff(G),
            "r_eff_over_d": stable_rank_ratio(G),
            "E_S2sq_monte_carlo": float(s2sq.mean()),
            "E_S2sq_monte_carlo_sem": float(s2sq.std() / math.sqrt(len(s2sq))),
            "E_S2sq_closed_form_trW_over_d": float(np.trace(W) / d),
            "rel_error": float(abs(s2sq.mean() - np.trace(W) / d) / (np.trace(W) / d)),
            "ratio_to_lambda1_closed_form": float(np.trace(W) / (d * lam1)),
        })
    # isotropic control: W_g = I exactly -> r_eff/d must equal 1
    Giso = rng.normal(size=(200000, 32))
    p3_iso = {
        "d": 32, "r_eff": r_eff(Giso), "r_eff_over_d": stable_rank_ratio(Giso),
        "expected_r_eff_over_d": 1.0,
    }
    out["checks"]["P3_random_direction_inertness"] = p3
    out["checks"]["P3_isotropic_control"] = p3_iso

    # ------------------------------------------------------------------
    # Proposition 5: closed-form dark direction is the constrained optimum
    # ------------------------------------------------------------------
    d, n = 48, 3000
    B = rng.normal(size=(d, 5))
    G = rng.normal(size=(n, 5)) @ B.T + rng.normal(size=(n, d)) * 0.05
    O = rng.normal(size=(d, 2))
    v_star = dark_direction(G, O)
    s_star = S_bar(G, v_star)
    Q = np.linalg.qr(O)[0]
    Z = rng.normal(size=(400000, d))
    Z = Z - (Z @ Q) @ Q.T
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    rand_s = np.abs(Z @ gbar(G))
    out["checks"]["P5_dark_direction_optimality"] = {
        "Sbar_closed_form": s_star,
        "Sbar_best_of_400k_random_in_complement": float(rand_s.max()),
        "closed_form_is_optimal": bool(s_star >= float(rand_s.max()) - 1e-9),
        "margin": float(s_star - rand_s.max()),
    }

    # ------------------------------------------------------------------
    # Corollary 2.1: the (AUC, Sbar) plane is fully occupied.
    # ------------------------------------------------------------------
    d, n, sigma = 32, 8000, 1.0
    e1 = np.zeros(d); e1[0] = 1.0
    e2 = np.zeros(d); e2[1] = 1.0
    grid = []
    for theta_deg in (0, 15, 30, 45, 60, 75, 90):
        th = math.radians(theta_deg)
        mean_dir = math.cos(th) * e1 + math.sin(th) * e2
        for dm in (0.25, 0.5, 1.0, 2.0, 4.0):
            yv = rng.integers(0, 2, size=n)
            Xv = rng.normal(0.0, sigma, size=(n, d)) + np.outer(yv, dm * mean_dir)
            Gv = np.tile(e1, (n, 1))
            wp = fisher_probe(Xv, yv, gamma=1e-6)
            grid.append({
                "theta_deg": theta_deg, "delta_mu": dm,
                "probe_auc": direction_auc(Xv, yv, wp),
                "Sbar_probe": S_bar(Gv, wp),
                "S2_probe": S_two(Gv, wp),
                "rho": rho_alignment(Gv, wp),
                "Sbar_max": float(np.linalg.norm(gbar(Gv))),
            })
    out["sweep"]["auc_vs_certificate_grid"] = grid
    aucs = np.array([g["probe_auc"] for g in grid])
    sbars = np.array([g["Sbar_probe"] for g in grid])
    out["sweep"]["auc_vs_Sbar_pearson"] = pearson(aucs, sbars)
    out["sweep"]["auc_vs_Sbar_spearman"] = spearman(aucs, sbars)

    path = os.path.join(RESULTS, "e1_exact_decoupling.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------------------------------------------
    print("=" * 80)
    print("E1  EXACT DECOUPLING  (Theorem 2, Props 3-5, 7)")
    print("=" * 80)
    for tag in ("axis_aligned", "random_rotation"):
        c = out["checks"][tag]
        print(f"\n[{tag}]  d={c['d']} N={c['n']}")
        print(f"  T2(a) probe AUC  pop / closed form : {c['probe_auc_population']:.6f} / "
              f"{c['probe_auc_closed_form']:.6f}  (err {c['probe_auc_abs_error_population']:.2e})")
        print(f"  T2(b) max|effect| POPULATION probe : {c['T2b_max_abs_effect_population_probe']:.3e}   <- theory: 0")
        print(f"  T2(b) max|effect| empirical probe  : {c['T2b_max_abs_effect_empirical_probe']:.3e}   <- finite-sample leak")
        print(f"  T2(c) dark AUC (pop)               : {c['T2c_dark_auc_population']:.6f} "
              f"(|err| {c['T2c_dark_auc_abs_error_from_half']:.2e})   <- theory: 0.5")
        print(f"  T2(c) dark effect / max effect     : {c['T2c_dark_effect_per_unit_alpha']:.6f} / "
              f"{c['T2c_max_attainable_effect_norm_gbar']:.6f}  (gap {c['T2c_dark_effect_gap']:.2e})")
        print(f"  P4    rho  pop / empirical         : {c['P4_rho_population_probe']:.3e} / "
              f"{c['P4_rho_empirical_probe']:.3e}")

    lk = out["checks"]["P7_finite_sample_leakage"]
    print("\n[P7 finite-sample spurious causal leakage]  population truth rho = 0")
    print("     d       N    rho_hat    1/sqrt(N)   rho*sqrt(N)   sqrt(d/N)")
    for r in lk["rows"]:
        print(f"   {r['d']:>3}  {r['n']:>6}   {r['rho_hat_mean']:.5f}     {r['inv_sqrt_n']:.5f}"
              f"      {r['rho_times_sqrt_n']:.3f}       {r['sqrt_d_over_n']:.5f}")
    mn, md = lk["model_inv_sqrt_n"], lk["model_sqrt_d_over_n"]
    print(f"   model  rho ~ N^(-1/2)  : slope {mn['loglog_slope']:.4f} (pred 1.0)  R^2 {mn['r2']:.4f}")
    print(f"   model  rho ~ sqrt(d/N) : slope {md['loglog_slope']:.4f} (pred 1.0)  R^2 {md['r2']:.4f}")
    print(f"   verdict: {lk['verdict']}")

    print("\n[P3 random-direction inertness]   E[S2^2] = tr(W)/d")
    print("   rank   r_eff   r_eff/d      MC        closed     rel.err")
    for r in out["checks"]["P3_random_direction_inertness"]:
        print(f"   {r['construction_rank']:>4}  {r['r_eff']:>6.2f}   {r['r_eff_over_d']:>6.4f}  "
              f"{r['E_S2sq_monte_carlo']:>9.4f}  {r['E_S2sq_closed_form_trW_over_d']:>9.4f}   "
              f"{r['rel_error']:.2e}")
    iso = out["checks"]["P3_isotropic_control"]
    print(f"   isotropic control: r_eff/d = {iso['r_eff_over_d']:.4f} (expected 1.0)")

    p5 = out["checks"]["P5_dark_direction_optimality"]
    print(f"\n[P5] closed form Sbar={p5['Sbar_closed_form']:.6f}  vs best of 400k random "
          f"in complement={p5['Sbar_best_of_400k_random_in_complement']:.6f}  "
          f"optimal={p5['closed_form_is_optimal']}")

    print(f"\n[Cor 2.1] across the (theta, delta_mu) grid: corr(probe AUC, Sbar) "
          f"pearson={out['sweep']['auc_vs_Sbar_pearson']:.4f} "
          f"spearman={out['sweep']['auc_vs_Sbar_spearman']:.4f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
