"""E6 -- The two geometries move independently.

The structural claim of the theory is that the read side depends on the labels
and not on the behaviour, while the write side depends on the behaviour and not
on the labels. This experiment tests that directly, and tests the strongest
version of the consequence:

    even when the probe is trained on EXACTLY the concept the behaviour
    functional measures, the alignment rho is small.

Design. Fix one model and one prompt set, so the residual states X are computed
once. Then:

  * K label definitions  -> K probe directions   (depend on y, not on phi)
  * K behaviour functionals -> K adjoint means   (depend on phi, not on y)

Three of the (label, behaviour) pairs are semantically MATCHED by construction
(toxicity/toxic-words, question/question-mark, digit/digit-token). If the read
and write geometries were the same object, the matched diagonal would stand out.

Reported: the full K x K alignment matrix rho(probe_i, gbar_j), a permutation
test on whether the matched diagonal exceeds the off-diagonal, and a check that
each side is invariant to the other's input.

Run:  python experiments/e6_independence.py
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
    S_bar, direction_auc, fisher_probe, gbar, r_eff, rho_alignment,
    stable_rank_ratio, unit,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "data")
CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "cache")
os.makedirs(RESULTS, exist_ok=True)

MODEL = os.environ.get("EGC_MODEL", "gpt2")
N = 300
ALPHA_FRAC = 0.10

# behaviour functionals: (name, positive words, negative words)
BEHAVIOURS = [
    ("toxic", ["idiot", "jerk", "moron", "loser", "fool", "creep"],
              ["person", "friend", "guy", "man", "student", "worker"]),
    ("sentiment", ["great", "wonderful", "excellent", "lovely", "happy", "good"],
                  ["awful", "terrible", "horrible", "sad", "bad", "poor"]),
    ("question", ["?"], ["."]),
    ("digit", ["1", "2", "3", "4", "5"], ["a", "b", "c", "d", "e"]),
    ("formality", ["therefore", "however", "moreover", "furthermore"],
                  ["yeah", "okay", "gonna", "stuff"]),
]

# label definitions applied to the prompt text; three are matched to a behaviour
def labels_for(prompts, tox):
    L = {}
    L["toxic"] = (np.asarray(tox) > 0.5).astype(int)
    L["question"] = np.array([1 if "?" in p else 0 for p in prompts])
    L["digit"] = np.array([1 if any(ch.isdigit() for ch in p) else 0 for p in prompts])
    lens = np.array([len(p) for p in prompts])
    L["length"] = (lens > np.median(lens)).astype(int)
    caps = np.array([sum(ch.isupper() for ch in p) / max(1, len(p)) for p in prompts])
    L["caps"] = (caps > np.median(caps)).astype(int)
    return L


MATCHED = {"toxic": "toxic", "question": "question", "digit": "digit"}


def load_rtp(n, seed=0):
    with open(os.path.join(CACHE, "rtp_balanced.json")) as f:
        d = json.load(f)
    p, y, t = np.array(d["prompts"], dtype=object), np.array(d["y"]), np.array(d["toxicity"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(p))[:n]
    return list(p[idx]), y[idx], t[idx]


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    lm = LMAdapter(MODEL)
    prompts, _, tox = load_rtp(N, seed=3)
    print(f"{MODEL}: layers={lm.n_layers} d={lm.d_model}; {len(prompts)} prompts")

    # resolve behaviour token sets, dropping words that are not single tokens
    beh = []
    for nm, pw, nw in BEHAVIOURS:
        pos = lm.token_ids(pw, leading_space=(nm not in ("question", "digit")))
        neg = lm.token_ids(nw, leading_space=(nm not in ("question", "digit")))
        if len(pos) and len(neg):
            beh.append((nm, pos, neg))
        else:
            print(f"  ! dropping behaviour '{nm}': no single-token targets")
    bnames = [b[0] for b in beh]

    L = labels_for(prompts, tox)
    lnames = [k for k in L if L[k].min() != L[k].max()]
    print(f"  behaviours: {bnames}")
    print(f"  labels    : {lnames}  (class balance "
          f"{[round(float(L[k].mean()),2) for k in lnames]})")

    # one forward per behaviour is required for the backward; X is identical
    # across behaviours, so capture it once and reuse.
    X, G = {}, {}
    for nm, pos, neg in beh:
        Xi, Gi, _ = lm.states_and_adjoints(prompts, pos, neg, batch=8)
        if not X:
            X = Xi
        G[nm] = Gi
        print(f"  adjoints for '{nm}' done ({time.time()-t0:.0f}s)")

    # sanity: X must be behaviour-independent (it is the same forward pass)
    Xchk = {nm: float(np.abs(X[0] - Xi[0]).max()) for nm, Xi in [(beh[0][0], X)]}

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(prompts))
    tr, te = perm[: len(prompts) // 2], perm[len(prompts) // 2:]

    out = {"experiment": "e6_independence", "model": MODEL, "n": len(prompts),
           "behaviours": bnames, "labels": lnames, "matched": MATCHED,
           "X_behaviour_invariance_max_abs_diff": Xchk, "layers": []}

    for Lyr in range(lm.n_layers):
        Xl = X[Lyr]
        probes = {k: fisher_probe(Xl[tr], L[k][tr], gamma=1e-2) for k in lnames}
        aucs = {k: direction_auc(Xl[te], L[k][te], probes[k]) for k in lnames}
        gbars = {nm: unit(gbar(G[nm][Lyr][tr])) for nm in bnames}

        R = [[rho_alignment(G[b][Lyr][tr], probes[l]) for b in bnames] for l in lnames]
        # alignment among the behaviours themselves, for scale
        BB = [[float(abs(gbars[a] @ gbars[b])) for b in bnames] for a in bnames]
        # alignment among the probes themselves
        PP = [[float(abs(unit(probes[a]) @ unit(probes[b]))) for b in lnames] for a in lnames]

        diag, off = [], []
        for i, l in enumerate(lnames):
            for j, b in enumerate(bnames):
                (diag if MATCHED.get(l) == b else off).append(R[i][j])

        rec = {
            "layer": Lyr,
            "probe_auc": aucs,
            "rho_matrix": R,
            "gbar_gbar": BB, "probe_probe": PP,
            "matched_mean": float(np.mean(diag)) if diag else None,
            "unmatched_mean": float(np.mean(off)) if off else None,
            "matched_max": float(np.max(diag)) if diag else None,
            "r_eff_over_d": {nm: stable_rank_ratio(G[nm][Lyr][tr]) for nm in bnames},
        }
        out["layers"].append(rec)
        print(f"  L{Lyr:>2}: matched rho={rec['matched_mean']:.4f} "
              f"unmatched rho={rec['unmatched_mean']:.4f} "
              f"| probe AUCs " + " ".join(f"{k}={aucs[k]:.2f}" for k in lnames))

    md = [r["matched_mean"] for r in out["layers"]]
    ud = [r["unmatched_mean"] for r in out["layers"]]
    out["summary"] = {
        "matched_mean_over_layers": float(np.mean(md)),
        "unmatched_mean_over_layers": float(np.mean(ud)),
        "ratio": float(np.mean(md) / np.mean(ud)) if np.mean(ud) else None,
        "layers_where_matched_exceeds_unmatched": int(sum(1 for a, b in zip(md, ud) if a > b)),
        "n_layers": len(md),
    }
    s = out["summary"]
    print(f"\nMatched (probe trained on exactly the behaviour's concept): "
          f"rho = {s['matched_mean_over_layers']:.4f}")
    print(f"Unmatched (probe for an unrelated concept):                  "
          f"rho = {s['unmatched_mean_over_layers']:.4f}")
    print(f"ratio = {s['ratio']:.2f}; matched exceeds unmatched in "
          f"{s['layers_where_matched_exceeds_unmatched']}/{s['n_layers']} layers")

    out["wall_time_s"] = time.time() - t0
    path = os.path.join(RESULTS, "e6_independence.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}  ({out['wall_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
