"""Offline, family-held-out tests of corpus selection and source exclusion.

No competition embeddings or labels are read. Source heads and their class
moments are fitted only on training families; validation labels are used only
for reporting. Recipe and distance are fixed before running the experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from probe.finalize import fit_head
from probe.select import (Recipe, clean_sources, resample_to_prior,
                          quota_accuracy, auroc, _stratified_cap)
from probe.submission import build_artifact
from probe.routing import candidate as routing_candidate, package_router


def score(head, x):
    # All heads use std|lda|0.6, folded into raw coordinates for cheap scoring.
    return x @ (head["coef"] / head["scale"]) + (
        head["intercept"] - head["center"] @ (head["coef"] / head["scale"]))


def moments(x, y):
    return [(x[y == c].mean(0, dtype=np.float64),
             x[y == c].var(0, dtype=np.float64)) for c in (0, 1)]


def mixture(stats, prior):
    (m0, v0), (m1, v1) = stats
    mean = (1-prior)*m0 + prior*m1
    var = (1-prior)*v0 + prior*v1 + prior*(1-prior)*(m1-m0)**2
    return mean, np.sqrt(np.maximum(var, 1e-12))


def distance(stats, x, scale, prior):
    mean, std = mixture(stats, prior)
    # Diagonal Gaussian Wasserstein distance in training-standardized units.
    return float(np.mean(((x.mean(0)-mean)/scale)**2 +
                         ((x.std(0)-std)/scale)**2))


def prediction(scores, prior):
    k = int(round(len(scores)*prior))
    out = np.zeros(len(scores), dtype=np.int64)
    if k:
        out[np.argpartition(scores, len(scores)-k)[len(scores)-k:]] = 1
    return out


def measure(scores, y, prior, baseline):
    p = prediction(scores, prior)
    right, base = p == y, baseline == y
    return {"accuracy": float(right.mean()), "auroc": auroc(scores, y),
            "gains": int((right & ~base).sum()), "losses": int((base & ~right).sum())}


def run(bundle, out, max_train=14000, source_cap=8000, within=False):
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(bundle, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    sources = data["sources"]
    family_map = dict(zip(sources.tolist(), data["families"].tolist()))
    x, y, sources, hygiene = clean_sources(x, y, sources, family_map)
    fam = np.asarray([family_map[s] for s in sources])
    names = sorted(set(fam.tolist()))
    recipe = Recipe("std", "lda", 0.6)
    heads, stats = {}, {}
    for name in names:
        idx = np.flatnonzero(fam == name)
        idx = _stratified_cap(idx, y[idx], min(source_cap, len(idx)), seed=23)
        heads[name] = fit_head(x[idx], y[idx], recipe, dtype=np.float32)
        stats[name] = moments(x[idx], y[idx])
        print(f"source fitted {name}: {len(idx)}", flush=True)
    rows = []
    for held in names:
        train = np.flatnonzero(fam != held)
        train = _stratified_cap(train, y[train], min(max_train, len(train)), seed=23)
        pooled = fit_head(x[train], y[train], recipe, dtype=np.float32)
        clean_idx = np.flatnonzero((fam != held) & ~np.isin(fam, ["hatecheck", "chat_aegis"]))
        clean_idx = _stratified_cap(clean_idx, y[clean_idx], min(max_train, len(clean_idx)), seed=23)
        clean = fit_head(x[clean_idx], y[clean_idx], recipe, dtype=np.float32)
        # Include pooled head as a candidate, providing a pre-fitted fallback.
        pool_stats = moments(x[train], y[train])
        eligible = [s for s in names if s != held]
        xt, yt = x[fam == held], y[fam == held]
        for prior in (0.5, 1200/1700):
            for seed in (17, 29, 41):
                idx = resample_to_prior(yt, prior, seed)
                if len(idx) > 1700:
                    idx = _stratified_cap(idx, yt[idx], 1700, seed)
                xx, yy = xt[idx], yt[idx]
                ps = score(pooled, xx)
                bp = prediction(ps, prior)
                candidates = {s: score(heads[s], xx) for s in eligible}
                dd = {s: distance(stats[s], xx, pooled["scale"], prior) for s in eligible}
                dd["pooled"] = distance(pool_stats, xx, pooled["scale"], prior)
                chosen = min(dd, key=dd.get)
                source_only = min(eligible, key=dd.get)
                candidates["pooled"] = ps
                measurements = {s: measure(z, yy, prior, bp) for s, z in candidates.items()}
                corr = spearmanr([dd[s] for s in eligible],
                                 [measurements[s]["auroc"] for s in eligible]).statistic
                row = {"held_family": held, "prior": prior, "seed": seed,
                       "n": len(yy), "train_n": len(train), "chosen": chosen,
                       "chosen_source_only": source_only, "distances": dd,
                       "distance_auc_spearman": None if not np.isfinite(corr) else float(corr),
                       "pooled": measurements["pooled"],
                       "routed": measurements[chosen],
                       "routed_source_only": measurements[source_only],
                       "clean_pool": measure(score(clean, xx), yy, prior, bp),
                       "source_results": measurements}
                rows.append(row)
        print(f"held out {held}: complete", flush=True)
        (out / "transfer_partial.json").write_text(json.dumps(rows, indent=1))
    summary = []
    for prior in (0.5, 1200/1700):
        subset = [r for r in rows if r["prior"] == prior]
        for method in ("pooled", "routed", "routed_source_only", "clean_pool"):
            summary.append({"prior": prior, "method": method,
                            "mean_accuracy": float(np.mean([r[method]["accuracy"] for r in subset])),
                            "mean_auroc": float(np.mean([r[method]["auroc"] for r in subset])),
                            "gains": sum(r[method]["gains"] for r in subset),
                            "losses": sum(r[method]["losses"] for r in subset)})
    within_rows = []
    if within:
        train_parts, eval_parts = [], []
        within_heads, within_stats = {}, {}
        rng = np.random.default_rng(317)
        for name in names:
            tr, te = [], []
            for label in (0, 1):
                idx = rng.permutation(np.flatnonzero((fam == name) & (y == label)))
                cut = int(0.8*len(idx))
                tr.extend(idx[:cut]); te.extend(idx[cut:])
            tr, te = np.asarray(tr), np.asarray(te)
            train_parts.append(tr); eval_parts.append(te)
            within_heads[name] = fit_head(x[tr], y[tr], recipe, dtype=np.float32)
            within_stats[name] = moments(x[tr], y[tr])
        tr = np.concatenate(train_parts)
        baseline = fit_head(x[tr], y[tr], recipe, dtype=np.float32)
        bstats = moments(x[tr], y[tr])
        for name, te in zip(names, eval_parts):
            for prior in (0.5, 1200/1700):
                idx = te[resample_to_prior(y[te], prior, seed=17)]
                if len(idx)>1700:
                    idx = _stratified_cap(idx, y[idx], 1700, seed=17)
                xx, yy = x[idx], y[idx]
                ps = score(baseline, xx)
                bp = prediction(ps, prior)
                dd = {s: distance(within_stats[s],xx,baseline['scale'],prior) for s in names}
                dd['pooled'] = distance(bstats,xx,baseline['scale'],prior)
                chosen = min(dd,key=dd.get)
                z = ps if chosen=='pooled' else score(within_heads[chosen],xx)
                within_rows.append({'held_family':name,'prior':prior,'n':len(yy),
                                    'chosen':chosen,'distances':dd,
                                    'pooled':measure(ps,yy,prior,bp),
                                    'routed':measure(z,yy,prior,bp)})
        print('WITHIN_SOURCE', json.dumps(within_rows), flush=True)
    result = {"recipe": recipe.key, "source_cap": source_cap, "max_train": max_train,
              "seeds_are_overlapping_resamples_not_independent_replications": True,
              "families": names, "hygiene": hygiene, "rows": rows, "summary": summary,
              "within_source_holdout": within_rows}
    if within:
        # Full-data deployment artifact. The pooled head is both the measured
        # incumbent recipe and a conservative router fallback.
        pooled_full = fit_head(x, y, recipe, dtype=np.float32)
        router_candidates = [routing_candidate(name, heads[name], x[fam==name], y[fam==name])
                             for name in names]
        router_candidates.append(routing_candidate('pooled', pooled_full, x, y))
        artifact = build_artifact([pooled_full], version='router', combine='score',
                                  positive_rate=1200/1700, meta={
                                      'recipe':recipe.key,
                                      'training_rows':len(y),
                                      'routing':'prior-adjusted diagonal Gaussian Wasserstein',
                                      'validation':'family LOSO plus source-stratified 80/20'})
        zip_path = out/'corpus_router.zip'
        result['candidate_checks'] = package_router(artifact, {
            'scale':pooled_full['scale'], 'candidates':router_candidates}, zip_path)
        result['candidate_zip'] = str(zip_path)
        result['candidate_sha256'] = __import__('hashlib').sha256(zip_path.read_bytes()).hexdigest()
    (out / "transfer_results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    print("TRANSFER_OK", flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/corpus_transfer"))
    p.add_argument("--max-train", type=int, default=14000)
    p.add_argument("--source-cap", type=int, default=8000)
    a = p.parse_args()
    run(a.bundle, a.out, a.max_train, a.source_cap)
