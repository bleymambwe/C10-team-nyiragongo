"""Build and verify a CodaBench 17670 submission zip.

The platform loads `classifier.py` and calls `Classifier().predict(X)` once with
the whole evaluation batch.  The zip must hold exactly `classifier.py` and
`trained_probe.joblib` at its root, and nothing may be fitted at inference time.

The emitted predictor is plain NumPy: the scoring container has sklearn, but a
pickled sklearn estimator would tie the submission to a version we cannot see.
Serialising the coefficients and doing the dot product by hand removes that
whole class of failure.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np

HIDDEN_SIZE = 2304
TARGET_POSITIVE_RATE = 1200.0 / 1700.0


CLASSIFIER_SOURCE = '''"""Latent Probe Challenge submission: a pre-fitted linear probe.

Nothing here is trained. `trained_probe.joblib` holds coefficients fitted
offline; `predict` applies fixed arithmetic to them.

Two steps read the incoming batch, and neither is training:

  * optional per-dimension standardisation using the batch's own mean and
    standard deviation. This is feature normalisation, uses no labels, and
    fits no parameters that are kept.
  * the decision rule labels the highest-scoring `positive_rate` fraction of
    the batch positive. The class prior is a published property of the task,
    so this is a fixed operating point, not a fitted one.
"""

import joblib
import numpy as np
from pathlib import Path


class Classifier:
    def __init__(self):
        self.artifact = joblib.load(Path(__file__).with_name("trained_probe.joblib"))

    @staticmethod
    def _rank01(values):
        """Ranks scaled to [0, 1], with ties averaged."""
        values = np.asarray(values, dtype=np.float64)
        n = len(values)
        if n <= 1:
            return np.zeros(n)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        sorted_values = values[order]
        start = 0
        for i in range(1, n + 1):
            if i == n or sorted_values[i] != sorted_values[start]:
                if i - start > 1:
                    ranks[order[start:i]] = ranks[order[start:i]].mean()
                start = i
        return ranks / (n - 1.0)

    @staticmethod
    def _top_components(z, k):
        """Leading k right singular vectors of a centred matrix, shaped [k, d]."""
        k = min(k, min(z.shape) - 1)
        if k <= 0:
            return None
        gram = (z.T @ z) / max(len(z), 1)
        _values, vectors = np.linalg.eigh(gram)
        return np.ascontiguousarray(vectors[:, -k:].T[::-1])

    def _head_score(self, head, X):
        z = X
        if head["row_l2"]:
            z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
        mode = head["standardize"]
        if mode == "batch":
            center = z.mean(axis=0)
            scale = np.maximum(z.std(axis=0), 1e-6)
            z = (z - center) / scale
        elif mode == "train":
            z = (z - head["center"]) / head["scale"]
        remove_top = int(head.get("remove_top", 0))
        if remove_top:
            components = (self._top_components(z, remove_top) if mode == "batch"
                          else head.get("components"))
            if components is not None:
                z = z - (z @ components.T) @ components
        return z @ head["coef"] + head["intercept"]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        expected = int(self.artifact["n_features"])
        if X.ndim != 2 or X.shape[1] != expected:
            raise ValueError(f"Expected [n, {expected}] embeddings; got {X.shape}")
        n = len(X)
        if n == 0:
            return np.empty(0, dtype=np.int64)

        heads = self.artifact["heads"]
        weights = np.array([h["weight"] for h in heads], dtype=np.float64)
        weights = weights / weights.sum()
        combine = self.artifact.get("combine", "rank")

        if combine == "rank":
            # Averaging ranks rather than logits keeps one head with a wide
            # score range from dominating heads fitted on other geometries.
            score = sum(w * self._rank01(self._head_score(h, X))
                        for w, h in zip(weights, heads))
        elif combine == "probability":
            score = sum(w * _sigmoid(self._head_score(h, X))
                        for w, h in zip(weights, heads))
        else:
            score = sum(w * self._head_score(h, X) for w, h in zip(weights, heads))

        if self.artifact.get("decision", "quota") == "threshold":
            # The heads were fitted on a class-balanced corpus, so their output
            # estimates P(toxic | x) under a 50/50 prior. Re-weighting to the
            # task's prior moves the 0.5 boundary to 1 - prior. Unlike a quota
            # this makes no assumption about how many positives are in *this*
            # batch, only about the prior -- it degrades smoothly if the batch
            # composition is not what we expect.
            return (score >= float(self.artifact["threshold"])).astype(np.int64)

        rate = float(self.artifact["positive_rate"])
        count = int(round(n * rate))
        count = min(max(count, 0), n)
        prediction = np.zeros(n, dtype=np.int64)
        if count:
            prediction[np.argpartition(score, n - count)[n - count:]] = 1
        return prediction


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
'''


def make_head(coef, intercept, *, row_l2=False, standardize="train",
              center=None, scale=None, weight=1.0, remove_top=0,
              components=None) -> dict:
    if standardize == "train" and (center is None or scale is None):
        raise ValueError("standardize='train' needs center and scale")
    if remove_top and standardize != "batch" and components is None:
        raise ValueError("remove_top with fixed statistics needs stored components")
    return {
        "coef": np.asarray(coef, dtype=np.float64).ravel(),
        "intercept": float(intercept),
        "row_l2": bool(row_l2),
        "standardize": standardize,
        "center": None if center is None else np.asarray(center, dtype=np.float64),
        "scale": None if scale is None else np.asarray(scale, dtype=np.float64),
        "remove_top": int(remove_top),
        "components": None if components is None else np.asarray(components, dtype=np.float64),
        "weight": float(weight),
    }


def build_artifact(heads: list[dict], *, version: str,
                   positive_rate: float = TARGET_POSITIVE_RATE,
                   combine: str = "rank", decision: str = "quota",
                   threshold: float | None = None, n_features: int = HIDDEN_SIZE,
                   meta: dict | None = None) -> dict:
    for head in heads:
        if head["coef"].shape != (n_features,):
            raise ValueError(f"head coef must be [{n_features}], got {head['coef'].shape}")
    if decision not in ("quota", "threshold"):
        raise ValueError(f"unknown decision rule {decision!r}")
    if decision == "threshold":
        if combine != "probability":
            raise ValueError("a threshold decision needs combine='probability'; "
                             "rank scores are already a quota by construction")
        if threshold is None:
            # Heads fitted on balanced data estimate P(toxic|x) at a 50/50
            # prior; re-weighting to `positive_rate` moves the boundary here.
            threshold = 1.0 - float(positive_rate)
    return {
        "version": version,
        "n_features": int(n_features),
        "positive_rate": float(positive_rate),
        "combine": combine,
        "decision": decision,
        "threshold": None if threshold is None else float(threshold),
        "heads": heads,
        "meta": meta or {},
    }


def package(artifact: dict, zip_path: Path) -> Path:
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        blob = Path(tmp) / "trained_probe.joblib"
        joblib.dump(artifact, blob, compress=3)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("classifier.py", CLASSIFIER_SOURCE)
            archive.write(blob, "trained_probe.joblib")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    if sorted(names) != ["classifier.py", "trained_probe.joblib"]:
        raise ValueError(f"zip root must hold exactly the two required files, got {names}")
    return zip_path


def smoke_test(zip_path: Path, n: int = 1700, expect_rate: float | None = None,
               n_features: int = HIDDEN_SIZE) -> dict:
    """Import the zip in isolation, exactly as the platform does."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)
        spec = importlib.util.spec_from_file_location(
            "submitted_classifier", Path(tmp) / "classifier.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        source = (Path(tmp) / "classifier.py").read_text(encoding="utf-8")
        if ".fit(" in source:
            raise ValueError("classifier.py calls .fit(); training on the platform is forbidden")

        classifier = module.Classifier()
        rng = np.random.default_rng(7)
        X = rng.normal(size=(n, n_features)).astype(np.float32)
        started = time.perf_counter()
        prediction = classifier.predict(X)
        duration = time.perf_counter() - started

        if prediction.shape != (n,):
            raise ValueError(f"predict returned {prediction.shape}, expected {(n,)}")
        if not set(np.unique(prediction)).issubset({0, 1}):
            raise ValueError("predict must return only 0 and 1")
        positives = int(prediction.sum())
        if expect_rate is not None:
            want = int(round(n * expect_rate))
            if positives != want:
                raise ValueError(f"expected {want} positives, got {positives}")
        # The platform allows 600 s; anything near that means something is wrong.
        if duration > 60:
            raise ValueError(f"predict took {duration:.1f}s, far too slow for {n} rows")
        return {"rows": n, "positives": positives, "seconds": round(duration, 4),
                "bytes": Path(zip_path).stat().st_size}


def build_and_verify(heads: list[dict], zip_path: Path, *, version: str,
                     positive_rate: float = TARGET_POSITIVE_RATE,
                     combine: str = "rank", decision: str = "quota",
                     threshold: float | None = None, n_features: int = HIDDEN_SIZE,
                     meta: dict | None = None) -> dict:
    artifact = build_artifact(heads, version=version, positive_rate=positive_rate,
                              combine=combine, decision=decision, threshold=threshold,
                              n_features=n_features, meta=meta)
    path = package(artifact, zip_path)
    checks = {}
    # The dev phase scores 1700 rows and the final phase about 1360, so both
    # sizes must run. Only a quota fixes the positive count; a threshold rule
    # deliberately lets it float, so it is not asserted there.
    for n in (1700, 1360):
        checks[str(n)] = smoke_test(path, n=n, n_features=n_features,
                                    expect_rate=positive_rate if decision == "quota" else None)
    return {"zip": str(path), "version": version, "positive_rate": positive_rate,
            "combine": combine, "decision": decision, "heads": len(heads),
            "checks": checks}


if __name__ == "__main__":
    # Self-check with a random probe: proves the packaging path works end to end.
    rng = np.random.default_rng(0)
    head = make_head(rng.normal(size=HIDDEN_SIZE), 0.0,
                     row_l2=True, standardize="batch", weight=1.0)
    out = build_and_verify([head], Path("artifacts/_selftest/probe.zip"),
                           version="selftest")
    print(json.dumps(out, indent=1))
