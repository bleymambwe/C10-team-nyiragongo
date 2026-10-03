"""Leakage-safe optimizer for the Latent Probe Challenge.

The competition supplies a single ``[n, 2304]`` layer-14 masked-mean matrix.
This runner also accepts ``[n, layers, 2304]`` arrays for offline layer audits.
Every scaler, PCA, feature selector, and classifier is fitted inside each CV
fold.  It deliberately never reads leaderboard or reference labels.

Bundle format (``.npz``): ``X``, ``y`` and optional ``texts``/``groups``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class Config:
    layer: int | None = None
    rank: int | None = None
    scaler: str = "none"
    row_l2: bool = False
    remove_top_pc: int = 0
    selector: str = "none"
    features: int | None = None
    probe: str = "logreg"
    C: float = 0.01
    alpha: float = 1.0
    threshold: float = 0.5


def canonical(value: str) -> str:
    return " ".join(str(value).lower().split())


def duplicate_groups(texts: np.ndarray) -> np.ndarray:
    """Stable exact-duplicate groups; near-duplicate review is reported separately."""
    return np.asarray([hashlib.sha1(canonical(x).encode()).hexdigest() for x in texts])


def load_bundle(path: Path):
    with np.load(path, allow_pickle=True) as z:
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.int64)
        texts = np.asarray(z["texts"], dtype=object) if "texts" in z else None
        groups = np.asarray(z["groups"], dtype=object) if "groups" in z else None
    if X.ndim not in (2, 3) or len(X) != len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError("X must be [n,d] or [n,l,d], and y must contain 0 and 1")
    if texts is not None and groups is None:
        groups = duplicate_groups(texts)
    return X, y, texts, groups


def select_layer(X: np.ndarray, layer: int | None) -> np.ndarray:
    if X.ndim == 2:
        if layer is not None:
            raise ValueError("layer was requested but X is already two-dimensional")
        return X
    return X[:, 0 if layer is None else layer, :]


def fit_transform(Xtr, Xva, cfg: Config, ytr=None):
    a, b = np.asarray(Xtr, dtype=np.float32), np.asarray(Xva, dtype=np.float32)
    if cfg.row_l2:
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    if cfg.scaler == "standard":
        t = StandardScaler().fit(a)
        a, b = t.transform(a), t.transform(b)
    elif cfg.scaler == "robust":
        t = RobustScaler().fit(a)
        a, b = t.transform(a), t.transform(b)
    if cfg.rank is not None or cfg.remove_top_pc:
        d = a.shape[1]
        keep = min(d, cfg.rank or d)
        ncomp = min(d, max(keep + cfg.remove_top_pc, cfg.remove_top_pc + 1))
        pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=0).fit(a)
        a, b = pca.transform(a), pca.transform(b)
        if cfg.remove_top_pc:
            a, b = a[:, cfg.remove_top_pc:], b[:, cfg.remove_top_pc:]
        if cfg.rank is not None:
            a, b = a[:, :keep], b[:, :keep]
    if cfg.selector != "none":
        k = min(int(cfg.features or a.shape[1]), a.shape[1])
        score = f_classif if cfg.selector == "anova" else mutual_info_classif
        # Fit the supervised selector on training rows only.
        if ytr is None:
            raise ValueError("ytr is required for supervised feature selection")
        selector = SelectKBest(score, k=k).fit(a, ytr)
        a, b = selector.transform(a), selector.transform(b)
    return a, b


def make_probe(cfg: Config, seed: int):
    if cfg.probe == "logreg":
        return LogisticRegression(C=cfg.C, penalty="l2", solver="lbfgs", max_iter=2500, random_state=seed)
    if cfg.probe == "l1":
        return LogisticRegression(C=cfg.C, penalty="l1", solver="liblinear", max_iter=2500, random_state=seed)
    if cfg.probe == "elastic":
        return LogisticRegression(C=cfg.C, penalty="elasticnet", l1_ratio=0.15, solver="saga", max_iter=4000, random_state=seed)
    if cfg.probe == "svm":
        return LinearSVC(C=cfg.C, dual=False, max_iter=4000, random_state=seed)
    if cfg.probe == "ridge":
        return RidgeClassifier(alpha=cfg.alpha)
    if cfg.probe == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    raise ValueError(f"unknown probe {cfg.probe}")


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X).ravel()


def threshold_for_accuracy(y, p):
    candidates = np.unique(np.r_[0.5, np.quantile(p, np.linspace(0.01, 0.99, 199))])
    acc = np.asarray([accuracy_score(y, p >= t) for t in candidates])
    return float(candidates[int(np.argmax(acc))])


def evaluate(X, y, cfg: Config, groups=None, seeds=(17, 29, 41), folds=5):
    X = select_layer(X, cfg.layer)
    fold_rows, oof_sum, oof_count = [], np.zeros(len(y)), np.zeros(len(y))
    for seed in seeds:
        if groups is None:
            splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=1, random_state=seed)
            split_iter = splitter.split(X, y)
        else:
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(X, y, groups)
        for fold, (tr, va) in enumerate(split_iter):
            A, B = fit_transform(X[tr], X[va], cfg, y[tr])
            model = make_probe(cfg, seed).fit(A, y[tr])
            p = scores(model, B)
            oof_sum[va] += p
            oof_count[va] += 1
            fold_rows.append({"seed": seed, "fold": fold, "accuracy": float(accuracy_score(y[va], p >= cfg.threshold)), "n": len(va)})
    p = oof_sum / np.maximum(oof_count, 1)
    chosen = threshold_for_accuracy(y, p)
    pred = p >= chosen
    fold_acc = np.asarray([r["accuracy"] for r in fold_rows])
    row = asdict(cfg)
    row.update({
        "mean_cv": float(fold_acc.mean()), "std_cv": float(fold_acc.std(ddof=1) if len(fold_acc) > 1 else 0),
        "worst_fold": float(fold_acc.min()), "oof_accuracy": float(accuracy_score(y, pred)),
        "oof_threshold": chosen, "oof_auroc": float(roc_auc_score(y, p)),
        "oof_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "oof_f1": float(f1_score(y, pred, zero_division=0)), "n_rows": int(len(y)),
        "duplicate_groups": int(len(np.unique(groups))) if groups is not None else None,
    })
    return row, p, fold_rows


def coarse_configs(d):
    return [Config(rank=k, scaler=s, row_l2=l2, probe=p, C=c)
            for k in (None, 256, 512, 768, 1024, 1200, 1536)
            for s in ("none", "standard") for l2 in (False, True)
            for p in ("logreg", "svm", "ridge", "lda")
            for c in ((0.001, 0.01, 0.1) if p in ("logreg", "svm") else (1.0,))
            if k is None or k <= d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/optimization"))
    ap.add_argument("--layers", default="all", help="all or comma-separated layer indices")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="17,29,41")
    ap.add_argument("--max-configs", type=int, default=0)
    args = ap.parse_args()
    X, y, texts, groups = load_bundle(args.bundle)
    args.out.mkdir(parents=True, exist_ok=True)
    d = X.shape[-1]
    cfgs = coarse_configs(d)
    if X.ndim == 3:
        layers = range(X.shape[1]) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
        cfgs = [Config(**{**asdict(c), "layer": l}) for l in layers for c in cfgs]
    if args.max_configs:
        cfgs = cfgs[:args.max_configs]
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    rows, oof = [], {}
    manifest = {"bundle": str(args.bundle), "shape": list(X.shape), "folds": args.folds, "seeds": seeds, "configs": len(cfgs), "grouped": groups is not None}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for i, cfg in enumerate(cfgs, 1):
        started = time.perf_counter()
        row, p, folds = evaluate(X, y, cfg, groups, seeds, args.folds)
        row.update({"id": i, "seconds": time.perf_counter() - started})
        rows.append(row); oof[str(i)] = p.astype(np.float32)
        print(f"[{i}/{len(cfgs)}] layer={cfg.layer} rank={cfg.rank} {cfg.probe} -> {row['oof_accuracy']:.4f} (cv {row['mean_cv']:.4f} +/- {row['std_cv']:.4f})")
        pd.DataFrame(rows).sort_values(["oof_accuracy", "worst_fold"], ascending=False).to_csv(args.out / "results.csv", index=False)
    np.savez_compressed(args.out / "oof_predictions.npz", y=y, **oof)
    print(json.dumps({"results": str(args.out / 'results.csv'), "oof": str(args.out / 'oof_predictions.npz'), "top": sorted(rows, key=lambda r: (r['oof_accuracy'], r['worst_fold']), reverse=True)[:20]}, indent=2, default=str))


if __name__ == "__main__":
    main()
