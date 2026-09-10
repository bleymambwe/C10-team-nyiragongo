import numpy as np
from sklearn.datasets import make_classification

from competition_probe import fit_competition_ensemble, predict_artifact


def test_ensemble_is_deterministic_and_portable_shape():
    X, y = make_classification(n_samples=80, n_features=24, n_informative=8, random_state=7)
    first = fit_competition_ensemble(X, y, seed=9, repeats=2)
    second = fit_competition_ensemble(X, y, seed=9, repeats=2)
    assert np.array_equal(predict_artifact(first, X), predict_artifact(second, X))
    assert 0.15 <= first["threshold"] <= 0.85
    assert np.isclose(np.sum(first["weights"]), 1)
    assert len(np.unique(predict_artifact(first, X))) == 2


def test_feature_contract_is_checked():
    X, y = make_classification(n_samples=50, n_features=12, random_state=3)
    artifact = fit_competition_ensemble(X, y, repeats=1)
    try:
        predict_artifact(artifact, X[:, :-1])
    except ValueError as exc:
        assert "Expected" in str(exc)
    else:
        raise AssertionError("feature mismatch should fail")
