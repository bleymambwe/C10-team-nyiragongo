"""The submitted classifier must reproduce the training-time transform exactly.

The feature pipeline is written twice: once in `select.Transform`, where probes
are fitted, and once inside `CLASSIFIER_SOURCE`, where the platform runs it. A
silent divergence between the two would not fail anything -- it would just score
badly on the leaderboard and look like a bad model. These tests compare the two
implementations directly, for every transform in the grid.
"""

import importlib.util
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe.finalize import fit_head  # noqa: E402
from probe.select import TRANSFORMS, Recipe  # noqa: E402
from probe.submission import (  # noqa: E402
    build_artifact,
    package,
    smoke_test,
)

WIDTH = 96


def _load_classifier(zip_path: Path):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(tmp)
    spec = importlib.util.spec_from_file_location("parity_classifier", Path(tmp) / "classifier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Classifier()


def _data(seed=5, n=900, d=WIDTH):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=d)
    y = rng.integers(0, 2, size=n)
    X = (rng.normal(size=(n, d)) * np.exp(rng.normal(size=d) * 0.4)
         + np.outer(y, direction) * 0.6 + rng.normal(size=d) * 2.0)
    return X.astype(np.float32), y


@pytest.mark.parametrize("transform", TRANSFORMS, ids=lambda t: t.name)
def test_classifier_head_score_matches_training_transform(tmp_path, transform):
    X, y = _data()
    recipe = Recipe(transform.name, "logistic", 0.05)
    head = fit_head(X, y, recipe)

    artifact = build_artifact([head], version="parity", n_features=WIDTH)
    zip_path = package(artifact, tmp_path / f"{transform.name.replace('+', '_')}.zip")
    classifier = _load_classifier(zip_path)

    # A held-out batch, deliberately shifted, so batch-statistic transforms are
    # actually exercised rather than seeing the training distribution again.
    Xte, _ = _data(seed=9, n=400)
    Xte = Xte * 1.7 + 3.0

    state = transform.fit(X)
    expected = transform.apply(state, Xte) @ head["coef"] + head["intercept"]
    actual = classifier._head_score(head, np.asarray(Xte, dtype=np.float64))

    # Principal components are sign-ambiguous, but projecting them out is not,
    # so the scores must agree to numerical precision either way.
    assert np.allclose(actual, expected, atol=1e-8, rtol=1e-6), (
        f"{transform.name}: max |diff| = {np.abs(actual - expected).max():.3e}")


def test_quota_and_threshold_rules_differ_only_in_the_operating_point(tmp_path):
    X, y = _data()
    head = fit_head(X, y, Recipe("std", "logistic", 0.05))
    Xte, _ = _data(seed=11, n=1700)

    quota_zip = package(build_artifact([head], version="q", n_features=WIDTH), tmp_path / "q.zip")
    thresh_zip = package(
        build_artifact([head], version="t", n_features=WIDTH,
                       combine="probability", decision="threshold"),
        tmp_path / "t.zip")

    quota_pred = _load_classifier(quota_zip).predict(Xte)
    thresh_pred = _load_classifier(thresh_zip).predict(Xte)

    assert quota_pred.sum() == 1200  # the quota fixes the count exactly
    # The threshold rule must be free to disagree about *how many* are positive.
    # It must still order the batch identically, so its positives are a nested
    # set: whichever rule predicts fewer positives, they are a subset.
    if thresh_pred.sum() <= quota_pred.sum():
        assert set(np.flatnonzero(thresh_pred)).issubset(set(np.flatnonzero(quota_pred)))
    else:
        assert set(np.flatnonzero(quota_pred)).issubset(set(np.flatnonzero(thresh_pred)))


def test_threshold_rule_rejects_a_rank_combination():
    X, y = _data()
    head = fit_head(X, y, Recipe("std", "logistic", 0.05))
    with pytest.raises(ValueError, match="combine='probability'"):
        build_artifact([head], version="bad", n_features=WIDTH, decision="threshold")


def test_zip_layout_and_no_fit_call(tmp_path):
    X, y = _data()
    head = fit_head(X, y, Recipe("std", "logistic", 0.05))
    zip_path = package(build_artifact([head], version="layout", n_features=WIDTH),
                       tmp_path / "layout.zip")
    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == ["classifier.py", "trained_probe.joblib"]
    # smoke_test refuses a classifier that calls .fit() on the platform.
    result = smoke_test(zip_path, n=1700, n_features=WIDTH, expect_rate=1200 / 1700)
    assert result["positives"] == 1200
