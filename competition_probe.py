"""Leakage-safe, competition-oriented ensembles for latent toxicity embeddings."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import expit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC


def _candidate_models(n_features: int, seed: int, calibration_cv: int = 3):
    k = min(n_features, 256)
    return {
        "logreg_l2": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000, random_state=seed),
        ),
        "logreg_elastic": make_pipeline(
            RobustScaler(), SelectKBest(f_classif, k=k),
            LogisticRegression(
                C=0.1, penalty="elasticnet", l1_ratio=0.15, solver="saga",
                class_weight="balanced", max_iter=8000, random_state=seed,
            ),
        ),
        "shrinkage_lda": make_pipeline(
            StandardScaler(), SelectKBest(f_classif, k=k),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(), SelectKBest(f_classif, k=k),
            CalibratedClassifierCV(
                # ``dual="auto"`` is only accepted by newer sklearn releases.
                # The explicit primal solver keeps this artifact portable across
                # the versions commonly used by notebooks and submission runners.
                LinearSVC(C=0.02, class_weight="balanced", dual=False, random_state=seed),
                method="sigmoid", cv=calibration_cv,
            ),
        ),
    }


def _probability(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return expit(model.decision_function(X))


def _metric_score(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    parts = [
        roc_auc_score(y, p), average_precision_score(y, p),
        balanced_accuracy_score(y, pred), f1_score(y, pred, zero_division=0),
        (matthews_corrcoef(y, pred) + 1) / 2,
    ]
    # Harmonic mean penalizes a model that wins one metric but fails another.
    return float(len(parts) / np.sum(1 / np.clip(parts, 1e-6, None)))


def _splitter(y, groups, seed, repeats):
    minority = int(np.bincount(np.asarray(y, dtype=int)).min())
    if minority < 2:
        raise ValueError("At least two examples from each class are required for cross-validation")
    n_splits = max(2, min(5, minority))
    if groups is not None:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed), repeats
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=seed), 1


def fit_competition_ensemble(X, y, *, groups=None, seed=17, repeats=5, max_models=3):
    """Fit a diverse linear ensemble and return a joblib-portable plain dictionary.

    Pass conversation/author/template IDs as ``groups`` whenever available. The
    returned artifact contains only sklearn estimators and primitive metadata.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or len(X) != len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError("X must be 2-D and y must contain both binary classes")
    minority = int(np.bincount(y).min())
    calibration_cv = max(2, min(3, minority))
    models = _candidate_models(X.shape[1], seed, calibration_cv=calibration_cv)
    splitter, passes = _splitter(y, groups, seed, repeats)
    oof_sum = {name: np.zeros(len(y)) for name in models}
    oof_count = np.zeros(len(y))

    for _ in range(passes):
        splits = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
        for train_idx, valid_idx in splits:
            oof_count[valid_idx] += 1
            for name, estimator in models.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = clone(estimator).fit(X[train_idx], y[train_idx])
                oof_sum[name][valid_idx] += _probability(fitted, X[valid_idx])
    if np.any(oof_count == 0):
        raise RuntimeError("Cross-validation did not score every training row")
    oof = {name: values / oof_count for name, values in oof_sum.items()}
    quality = {name: _metric_score(y, p) for name, p in oof.items()}

    # Greedy quality-diversity selection: reward complementary probability errors.
    selected = [max(quality, key=quality.get)]
    while len(selected) < min(max_models, len(models)):
        ensemble_error = y - np.mean([oof[name] for name in selected], axis=0)
        remaining = [name for name in models if name not in selected]
        gain = {}
        for name in remaining:
            corr = abs(np.corrcoef(ensemble_error, y - oof[name])[0, 1])
            gain[name] = quality[name] * (1.0 - 0.35 * np.nan_to_num(corr, nan=1.0))
        selected.append(max(gain, key=gain.get))

    raw_weights = np.array([max(quality[name] - 0.45, 0.01) ** 2 for name in selected])
    weights = raw_weights / raw_weights.sum()
    ensemble_oof = sum(weight * oof[name] for weight, name in zip(weights, selected))
    thresholds = np.linspace(0.15, 0.85, 281)
    threshold_scores = []
    for threshold in thresholds:
        pred = ensemble_oof >= threshold
        threshold_scores.append(
            0.4 * balanced_accuracy_score(y, pred)
            + 0.3 * f1_score(y, pred, zero_division=0)
            + 0.3 * ((matthews_corrcoef(y, pred) + 1) / 2)
        )
    threshold = float(thresholds[int(np.argmax(threshold_scores))])

    fitted_models = []
    for name in selected:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_models.append(clone(models[name]).fit(X, y))
    return {
        "format_version": 1, "models": fitted_models, "model_names": selected,
        "weights": weights, "threshold": threshold, "n_features": X.shape[1],
        "oof_score": _metric_score(y, ensemble_oof, threshold), "candidate_scores": quality,
        "seed": seed,
    }


def predict_artifact(artifact, X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != artifact["n_features"]:
        raise ValueError(f"Expected [n, {artifact['n_features']}] embeddings; got {X.shape}")
    if artifact.get("format_version", 1) >= 2:
        probability = sum(
            weight * _portable_probability(model, X)
            for weight, model in zip(artifact["weights"], artifact["models"])
        )
        return (probability >= artifact["threshold"]).astype(int)
    probability = sum(
        weight * _probability(model, X)
        for weight, model in zip(artifact["weights"], artifact["models"])
    )
    return (probability >= artifact["threshold"]).astype(int)


def _portable_probability(model, X):
    """Evaluate a serialized linear model using NumPy only.

    This deliberately avoids pickling sklearn estimators. Codabench currently
    evaluates submissions with sklearn 1.5.1, while local environments may
    have a newer version; estimator pickles are not a stable interchange
    format across those versions.
    """
    scale = np.asarray(model["scale"], dtype=np.float32)
    center = np.asarray(model["center"], dtype=np.float32)
    transformed = (X - center) / scale
    support = np.asarray(model["support"], dtype=bool)
    transformed = transformed[:, support]
    logit = transformed @ np.asarray(model["coef"], dtype=np.float32)
    logit = logit + float(model["intercept"])
    return expit(np.clip(logit, -40, 40))


def _portable_linear_model(pipeline):
    """Extract a Standard/RobustScaler + optional selector + linear model."""
    steps = dict(pipeline.named_steps)
    scaler = steps.get("standardscaler") or steps.get("robustscaler")
    selector = steps.get("selectkbest")
    estimator = steps.get("logisticregression") or steps.get("lineardiscriminantanalysis")
    if scaler is None or estimator is None:
        raise TypeError("Only fitted scaler + linear classifier pipelines are portable")
    center = getattr(scaler, "mean_", None)
    if center is None:
        center = getattr(scaler, "center_", None)
    scale = np.asarray(getattr(scaler, "scale_", np.ones_like(center)), dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    scale = np.where(np.isfinite(scale) & (np.abs(scale) > 1e-12), scale, 1.0)
    support = (selector.get_support() if selector is not None
               else np.ones(center.shape[0], dtype=bool))
    coef = np.asarray(estimator.coef_, dtype=np.float32)
    intercept = np.asarray(estimator.intercept_, dtype=np.float32)
    if coef.shape[0] == 2:
        coef = coef[1] - coef[0]
        intercept = intercept[1] - intercept[0]
    else:
        coef = coef[0]
        intercept = intercept[0]
    return {
        "kind": "linear_sigmoid",
        "center": center,
        "scale": scale,
        "support": np.asarray(support, dtype=bool),
        "coef": coef,
        "intercept": float(intercept),
    }


def make_portable_artifact(artifact, X_reference=None, y_reference=None):
    """Convert a fitted competition artifact to a sklearn-free joblib dict.

    The elastic-net, LDA, and logistic pipelines are representable exactly as
    NumPy affine transforms followed by a sigmoid. Calibrated SVMs are omitted
    because their sklearn calibration internals are version-sensitive; the
    remaining weights are renormalized. If reference data is supplied, the
    decision threshold is reselected on that artifact's predictions.
    """
    portable_models = []
    portable_names = []
    portable_weights = []
    for name, weight, fitted in zip(artifact["model_names"], artifact["weights"], artifact["models"]):
        try:
            portable_models.append(_portable_linear_model(fitted))
        except TypeError:
            continue
        portable_names.append(name)
        portable_weights.append(float(weight))
    if not portable_models:
        raise ValueError("No fitted linear models could be converted to a portable artifact")
    weights = np.asarray(portable_weights, dtype=float)
    weights /= weights.sum()
    portable = {
        "format_version": 2,
        "portable_numpy": True,
        "models": portable_models,
        "model_names": portable_names,
        "weights": weights,
        "threshold": float(artifact["threshold"]),
        "n_features": int(artifact["n_features"]),
        "seed": artifact.get("seed"),
        "candidate_scores": artifact.get("candidate_scores", {}),
    }
    if X_reference is not None and y_reference is not None:
        X_reference = np.asarray(X_reference, dtype=np.float32)
        y_reference = np.asarray(y_reference, dtype=int)
        probability = sum(
            weight * _portable_probability(model, X_reference)
            for weight, model in zip(weights, portable_models)
        )
        thresholds = np.linspace(0.15, 0.85, 281)
        scores = []
        for threshold in thresholds:
            pred = (probability >= threshold).astype(int)
            scores.append(
                0.4 * balanced_accuracy_score(y_reference, pred)
                + 0.3 * f1_score(y_reference, pred, zero_division=0)
                + 0.3 * ((matthews_corrcoef(y_reference, pred) + 1) / 2)
            )
        portable["threshold"] = float(thresholds[int(np.argmax(scores))])
    return portable
