import zipfile

import numpy as np

from train_probe_v004 import (
    CandidateSpec,
    fit_probe_artifact,
    package_submission,
    predict_artifact,
)


def test_v004_multiclass_artifact_is_portable(tmp_path):
    rng = np.random.default_rng(4)
    n, d = 700, 96
    source_labels = np.arange(n) % 7
    rng.shuffle(source_labels)
    class_directions = rng.normal(size=(7, d)).astype(np.float32)
    X = class_directions[source_labels] + rng.normal(scale=0.8, size=(n, d)).astype(np.float32)
    texts = [f"unique training statement {i}" for i in range(n)]
    specs = [
        CandidateSpec("binary", False, 64, 0.1),
        CandidateSpec("multiclass", False, 64, 0.1),
        CandidateSpec("multiclass", True, 96, 0.1),
    ]
    artifact, report = fit_probe_artifact(
        X, source_labels, texts, specs=specs, max_tuning_rows=10_000
    )
    prediction = predict_artifact(artifact, X)
    assert prediction.shape == (n,)
    assert set(np.unique(prediction)) == {0, 1}
    assert artifact["format_version"] == 4
    assert report["validation_weighted_accuracy"] > 0.8

    zip_path = package_submission(artifact, tmp_path)
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["classifier.py", "trained_probe.joblib"]
