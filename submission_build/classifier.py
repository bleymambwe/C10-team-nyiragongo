import joblib
import numpy as np
from pathlib import Path


class Classifier:
    def __init__(self):
        self.artifact = joblib.load(Path(__file__).with_name("trained_probe.joblib"))

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        expected = int(self.artifact["n_features"])
        if X.ndim != 2 or X.shape[1] != expected:
            raise ValueError(f"Expected [n, {expected}] embeddings; got {X.shape}")
        probability = np.zeros(len(X), dtype=float)
        for weight, model in zip(self.artifact["weights"], self.artifact["models"]):
            z = ((X - model["center"]) / model["scale"])[:, model["support"]]
            z = z @ model["coef"] + float(model["intercept"])
            probability += float(weight) / (1.0 + np.exp(-np.clip(z, -40, 40)))
        return (probability >= float(self.artifact["threshold"])).astype(int)
