"""Build competition-ready quota variants from the reproduced v005 probe.

These variants change only the decision policy.  The learned 2,304-dimensional
layer-14 masked-mean probe is copied byte-for-byte from v005; no hidden labels
or platform data are read.  ``--quota`` is the number of positive predictions
for a 1,700-row development batch and scales proportionally for other sizes.
"""

from __future__ import annotations

import argparse
import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import joblib
import numpy as np


SOURCE = Path("artifacts/v005/toxicity_probe_v005_layer14_mean64.zip")


def classifier_source(mode: str, quota: int | None) -> str:
    quota_code = "None" if quota is None else str(int(quota))
    return f'''import joblib
import numpy as np
from pathlib import Path


class Classifier:
    def __init__(self):
        self.artifact = joblib.load(Path(__file__).with_name("trained_probe.joblib"))

    @staticmethod
    def _probability(component, X):
        z = X
        if component["row_l2"]:
            z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
        z = z[:, component["features"]]
        z = (z - component["center"]) / component["scale"]
        logits = z @ component["coef"].T + component["intercept"]
        return 1.0 / (1.0 + np.exp(-np.clip(logits.ravel(), -40, 40)))

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        expected = int(self.artifact["n_features"])
        if X.ndim != 2 or X.shape[1] != expected:
            raise ValueError(f"Expected [n, {{expected}}] embeddings; got {{X.shape}}")
        probability = np.zeros(len(X), dtype=np.float64)
        for weight, component in zip(self.artifact["weights"], self.artifact["components"]):
            probability += float(weight) * self._probability(component, X)
        if "threshold" == "{mode}":
            return (probability >= float(self.artifact["threshold"])).astype(np.int64)
        count = int(round(len(X) * ({quota_code} / 1700.0)))
        count = min(max(count, 0), len(X))
        prediction = np.zeros(len(X), dtype=np.int64)
        if count:
            prediction[np.argpartition(probability, len(X) - count)[-count:]] = 1
        return prediction
'''


def build(out_dir: Path, variants: list[tuple[str, str, int | None]]):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE) as src:
        model_bytes = src.read("trained_probe.joblib")
    exact_v006 = Path("artifacts/v006/toxicity_probe_v006_rank1200_layer14_mean64.zip")
    with zipfile.ZipFile(exact_v006) as src:
        v006_classifier = src.read("classifier.py")
    for name, mode, quota in variants:
        path = out_dir / f"toxicity_probe_{name}.zip"
        classifier = v006_classifier.decode("utf-8") if name.startswith("candidate_a_") else classifier_source(mode, quota)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("classifier.py", classifier)
            z.writestr("trained_probe.joblib", model_bytes)
        with zipfile.ZipFile(path) as z:
            assert z.namelist() == ["classifier.py", "trained_probe.joblib"]
            assert z.read("trained_probe.joblib") == model_bytes
        print(path)


def smoke(path: Path):
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(path) as z:
            z.extractall(td)
        import importlib.util
        spec = importlib.util.spec_from_file_location("portfolio_classifier", Path(td) / "classifier.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        pred = module.Classifier().predict(np.random.default_rng(0).normal(size=(1700, 2304)).astype(np.float32))
        assert pred.shape == (1700,)
        assert set(np.unique(pred)).issubset({0, 1})
        print(path.name, "smoke positives", int(pred.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/portfolio"))
    args = ap.parse_args()
    variants = [
        ("candidate_a_v006_rank1200", "quota", 1200),
        ("candidate_b_rank1250", "quota", 1250),
        ("candidate_c_rank1292", "quota", 1292),
        ("candidate_d_threshold_v005", "threshold", None),
    ]
    build(args.out_dir, variants)
    for path in sorted(args.out_dir.glob("toxicity_probe_*.zip")):
        smoke(path)


if __name__ == "__main__":
    main()
