"""Train the v004 CodaBench probe with the organizer's exact Gemma contract.

Competition contract (CodaBench competition 17670):
  * google/gemma-2-2b
  * hidden-state index 14
  * masked mean pooling
  * maximum sequence length 64
  * 0 = Normal, 1 = any mental-health distress signal

The source dataset retains seven labels.  A multinomial linear head can model
the heterogeneous positive class while the submitted interface remains a
small, sklearn-free NumPy predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


VERSION = "v004"
MODEL_ID = "google/gemma-2-2b"
DATASET_ID = "moujar/MentalHealth-Darija"
HIDDEN_STATE_INDEX = 14
MAX_LENGTH = 64
HIDDEN_SIZE = 2304
NORMAL_SOURCE_LABEL = 1
TARGET_POSITIVE_PRIOR = 1200 / 1700
SEED = 34


def load_hf_token(env_path: Path = Path("BMToken.env")) -> str | None:
    """Read HF_TOKEN without printing or persisting it anywhere else."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*(?:export\s+)?HF_TOKEN\s*=\s*['\"]?([^'\"\s#]+)", line)
        if match:
            return match.group(1)
    return None


def canonical_text(value: str) -> str:
    return " ".join(str(value).lower().split())


def load_training_rows(max_rows: int | None = None):
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split="train")
    texts = np.asarray(dataset["text_en"], dtype=object)
    source_labels = np.asarray(dataset["label"], dtype=np.int64)
    keep = np.asarray([bool(canonical_text(text)) for text in texts])
    texts, source_labels = texts[keep], source_labels[keep]

    # Keep one copy of exact duplicate text/label pairs. Conflicting duplicate
    # labels are retained because they encode genuine annotation ambiguity.
    seen: set[tuple[str, int]] = set()
    selected: list[int] = []
    for index, (text, label) in enumerate(zip(texts, source_labels)):
        key = (canonical_text(text), int(label))
        if key not in seen:
            seen.add(key)
            selected.append(index)
    texts, source_labels = texts[selected], source_labels[selected]

    if max_rows and max_rows < len(texts):
        # Deterministic stratified cap for cheap smoke runs.
        from sklearn.model_selection import train_test_split

        indices, _ = train_test_split(
            np.arange(len(texts)),
            train_size=max_rows,
            random_state=SEED,
            stratify=source_labels,
        )
        indices.sort()
        texts, source_labels = texts[indices], source_labels[indices]
    return texts.tolist(), source_labels


class _LayerCaptured(RuntimeError):
    pass


def _dtype_for_device(torch, device):
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def extract_embeddings(
    texts: list[str],
    output_path: Path,
    *,
    batch_size: int = 32,
    env_path: Path = Path("BMToken.env"),
) -> np.ndarray:
    """Extract index-14 masked-mean embeddings with resumable disk caching."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    token = load_hf_token(env_path)
    if not token:
        raise RuntimeError("HF_TOKEN was not found in the environment or BMToken.env")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_for_device(torch, device)
    print({"device": str(device), "dtype": str(dtype), "rows": len(texts)})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModel.from_pretrained(
            MODEL_ID,
            token=token,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except TypeError:  # transformers 4.x used torch_dtype instead of dtype
        model = AutoModel.from_pretrained(
            MODEL_ID,
            token=token,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    model = model.to(device).eval()
    model.requires_grad_(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(".progress.json")
    if output_path.exists():
        embeddings = np.lib.format.open_memmap(output_path, mode="r+")
        if embeddings.shape != (len(texts), HIDDEN_SIZE):
            raise ValueError(f"Existing cache has shape {embeddings.shape}, expected {(len(texts), HIDDEN_SIZE)}")
        start = 0
        if progress_path.exists():
            start = int(json.loads(progress_path.read_text(encoding="utf-8"))["next_index"])
    else:
        embeddings = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=(len(texts), HIDDEN_SIZE)
        )
        start = 0

    # A pre-hook on decoder block 14 sees exactly hidden_states[14]. Raising a
    # private exception skips blocks 14-25 and approximately halves extraction
    # compute. The parity check below verifies this against output_hidden_states.
    capture: dict[str, torch.Tensor] = {}
    active_mask: dict[str, torch.Tensor] = {}

    def capture_hook(_module, args):
        hidden = args[0]
        mask = active_mask["value"].to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        capture["value"] = pooled.float().cpu()
        raise _LayerCaptured

    target_layer = model.layers[HIDDEN_STATE_INDEX]

    # One-batch exactness check before enabling the accelerated path.
    check = tokenizer(
        texts[: min(2, len(texts))],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        full = model(**check, output_hidden_states=True, use_cache=False, return_dict=True)
    check_mask = check["attention_mask"].to(full.hidden_states[HIDDEN_STATE_INDEX].dtype).unsqueeze(-1)
    expected = (
        (full.hidden_states[HIDDEN_STATE_INDEX] * check_mask).sum(dim=1)
        / check_mask.sum(dim=1).clamp_min(1)
    ).float().cpu()
    del full
    active_mask["value"] = check["attention_mask"]
    handle = target_layer.register_forward_pre_hook(capture_hook)
    try:
        with torch.inference_mode():
            try:
                model(**check, use_cache=False, return_dict=True)
            except _LayerCaptured:
                pass
        max_error = float((capture["value"] - expected).abs().max())
        if max_error > 1e-5:
            raise RuntimeError(f"Early-exit parity check failed: max error {max_error}")
        print(f"Layer-14 early-exit parity max error: {max_error:.3g}")

        # Blocks after the capture point can never execute. Releasing them
        # substantially reduces host RAM as well as GPU memory pressure.
        model.layers = torch.nn.ModuleList(list(model.layers[: HIDDEN_STATE_INDEX + 1]))

        started = time.time()
        for batch_start in range(start, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            encoded = tokenizer(
                texts[batch_start:batch_end],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            active_mask["value"] = encoded["attention_mask"]
            capture.clear()
            with torch.inference_mode():
                try:
                    model(**encoded, use_cache=False, return_dict=True)
                except _LayerCaptured:
                    pass
            if "value" not in capture:
                raise RuntimeError("Layer hook did not capture an activation")
            embeddings[batch_start:batch_end] = capture["value"].numpy()
            embeddings.flush()
            progress_path.write_text(
                json.dumps({"next_index": batch_end, "rows": len(texts), "version": VERSION}),
                encoding="utf-8",
            )
            if batch_end == len(texts) or batch_end % max(batch_size * 20, 1) == 0:
                elapsed = max(time.time() - started, 1e-6)
                rate = (batch_end - start) / elapsed
                remaining = (len(texts) - batch_end) / max(rate, 1e-6)
                print(f"embedded {batch_end}/{len(texts)} | {rate:.1f} rows/s | ETA {remaining/60:.1f} min")
    finally:
        handle.remove()

    progress_path.write_text(
        json.dumps({"next_index": len(texts), "rows": len(texts), "complete": True, "version": VERSION}),
        encoding="utf-8",
    )
    return np.asarray(embeddings)


def _row_transform(X: np.ndarray, row_l2: bool) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if not row_l2:
        return X
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def _component_probability(component: dict, X: np.ndarray) -> np.ndarray:
    z = _row_transform(X, component["row_l2"])
    z = z[:, component["features"]]
    z = (z - component["center"]) / component["scale"]
    logits = z @ component["coef"].T + component["intercept"]
    if component["kind"] == "binary":
        return 1.0 / (1.0 + np.exp(-np.clip(logits.ravel(), -40, 40)))
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(np.clip(logits, -40, 0))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    normal_column = int(np.flatnonzero(component["classes"] == NORMAL_SOURCE_LABEL)[0])
    return 1.0 - probabilities[:, normal_column]


def predict_artifact(artifact: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != artifact["n_features"]:
        raise ValueError(f"Expected [n, {artifact['n_features']}] embeddings; got {X.shape}")
    probability = np.zeros(len(X), dtype=np.float64)
    for weight, component in zip(artifact["weights"], artifact["components"]):
        probability += float(weight) * _component_probability(component, X)
    return (probability >= float(artifact["threshold"])).astype(np.int64)


def _hash_validation_mask(texts: list[str], fraction: float = 0.2) -> np.ndarray:
    cutoff = int(fraction * 10_000)
    values = []
    for text in texts:
        digest = hashlib.sha1(canonical_text(text).encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:4], "big") % 10_000 < cutoff)
    return np.asarray(values, dtype=bool)


def _best_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    # Reweight validation examples to the known 1200/1700 hidden-set prior.
    y = np.asarray(y, dtype=np.int64)
    source_prior = float(y.mean())
    weights = np.where(
        y == 1,
        TARGET_POSITIVE_PRIOR / max(source_prior, 1e-8),
        (1 - TARGET_POSITIVE_PRIOR) / max(1 - source_prior, 1e-8),
    )
    thresholds = np.unique(np.quantile(probability, np.linspace(0.01, 0.99, 399)))
    thresholds = np.unique(np.r_[0.5, thresholds])
    scores = np.asarray([
        np.average((probability >= threshold) == y, weights=weights)
        for threshold in thresholds
    ])
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


@dataclass(frozen=True)
class CandidateSpec:
    kind: str
    row_l2: bool
    k: int
    C: float


def _fit_component(X: np.ndarray, source_y: np.ndarray, spec: CandidateSpec) -> dict:
    from sklearn.feature_selection import f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    target = (source_y != NORMAL_SOURCE_LABEL).astype(np.int64) if spec.kind == "binary" else source_y
    transformed = _row_transform(X, spec.row_l2)
    scores, _ = f_classif(transformed, target)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(np.float32).max)
    k = min(spec.k, transformed.shape[1])
    features = np.sort(np.argpartition(scores, -k)[-k:]).astype(np.int32)
    selected = transformed[:, features]
    scaler = StandardScaler().fit(selected)
    selected = scaler.transform(selected)
    model = LogisticRegression(
        C=spec.C,
        solver="lbfgs",
        max_iter=800,
        tol=2e-5,
        random_state=SEED,
    ).fit(selected, target)
    return {
        "kind": spec.kind,
        "row_l2": spec.row_l2,
        "features": features,
        "center": scaler.mean_.astype(np.float32),
        "scale": np.maximum(scaler.scale_, 1e-8).astype(np.float32),
        "coef": model.coef_.astype(np.float32),
        "intercept": model.intercept_.astype(np.float32),
        "classes": model.classes_.astype(np.int64),
        "spec": {"kind": spec.kind, "row_l2": spec.row_l2, "k": spec.k, "C": spec.C},
    }


def fit_probe_artifact(
    X: np.ndarray,
    source_labels: np.ndarray,
    texts: list[str],
    *,
    specs: list[CandidateSpec] | None = None,
    max_tuning_rows: int = 25_000,
) -> tuple[dict, dict]:
    X = np.asarray(X, dtype=np.float32)
    source_labels = np.asarray(source_labels, dtype=np.int64)
    y = (source_labels != NORMAL_SOURCE_LABEL).astype(np.int64)
    valid = _hash_validation_mask(texts)
    train = ~valid
    if len(np.unique(source_labels[valid])) < 7:
        raise RuntimeError("Validation split is missing a source class")

    if specs is None:
        specs = [
            CandidateSpec(kind, row_l2, k, C)
            for kind in ("binary", "multiclass")
            for row_l2 in (False, True)
            for k in ((512, 1024, HIDDEN_SIZE) if kind == "binary" else (512, 1024))
            for C in ((0.02, 0.1, 0.5) if kind == "binary" else (0.02, 0.1))
        ]
    tuning_indices = np.flatnonzero(train)
    if len(tuning_indices) > max_tuning_rows:
        from sklearn.model_selection import train_test_split

        tuning_indices, _ = train_test_split(
            tuning_indices,
            train_size=max_tuning_rows,
            random_state=SEED,
            stratify=source_labels[tuning_indices],
        )
    evaluations: list[dict] = []
    print(f"Fitting {len(specs)} validation candidates on {len(tuning_indices)} rows")
    for index, spec in enumerate(specs, 1):
        component = _fit_component(X[tuning_indices], source_labels[tuning_indices], spec)
        probability = _component_probability(component, X[valid])
        threshold, score = _best_threshold(y[valid], probability)
        evaluations.append({
            "spec": spec,
            "component": component,
            "probability": probability,
            "threshold": threshold,
            "accuracy": score,
        })
        print(f"[{index:02d}/{len(specs)}] {spec} -> weighted accuracy {score:.4f}")

    evaluations.sort(key=lambda item: item["accuracy"], reverse=True)
    selected = [evaluations[0]]
    ensemble_probability = evaluations[0]["probability"].copy()
    ensemble_weights = np.asarray([1.0])
    threshold, ensemble_score = _best_threshold(y[valid], ensemble_probability)

    # Greedily admit only candidates that improve accuracy and make different
    # errors. At most three matrix products keeps platform prediction fast.
    for candidate in evaluations[1:12]:
        best_trial = None
        for candidate_weight in (0.15, 0.25, 0.35, 0.5):
            trial_probability = (1 - candidate_weight) * ensemble_probability + candidate_weight * candidate["probability"]
            trial_threshold, trial_score = _best_threshold(y[valid], trial_probability)
            if best_trial is None or trial_score > best_trial[0]:
                best_trial = (trial_score, trial_threshold, candidate_weight, trial_probability)
        if best_trial[0] > ensemble_score + 0.0002:
            ensemble_score, threshold, new_weight, ensemble_probability = best_trial
            ensemble_weights *= 1 - new_weight
            ensemble_weights = np.r_[ensemble_weights, new_weight]
            selected.append(candidate)
        if len(selected) == 3:
            break

    # Refit selected specifications on every public example.
    components = [_fit_component(X, source_labels, item["spec"]) for item in selected]
    artifact = {
        "format_version": 4,
        "version": VERSION,
        "portable_numpy": True,
        "n_features": int(X.shape[1]),
        "components": components,
        "weights": ensemble_weights.astype(np.float64),
        "threshold": float(threshold),
        "target_positive_prior": TARGET_POSITIVE_PRIOR,
        "contract": {
            "model": MODEL_ID,
            "hidden_state_index": HIDDEN_STATE_INDEX,
            "pooling": "masked_mean",
            "max_length": MAX_LENGTH,
        },
    }
    report = {
        "version": VERSION,
        "rows": int(len(X)),
        "validation_rows": int(valid.sum()),
        "validation_positive_prior": float(y[valid].mean()),
        "target_positive_prior": TARGET_POSITIVE_PRIOR,
        "selected": [item["component"]["spec"] for item in selected],
        "weights": ensemble_weights.tolist(),
        "threshold": float(threshold),
        "validation_weighted_accuracy": float(ensemble_score),
        "top_candidates": [
            {**item["component"]["spec"], "accuracy": float(item["accuracy"]), "threshold": float(item["threshold"])}
            for item in evaluations[:10]
        ],
    }
    return artifact, report


CLASSIFIER_SOURCE = '''import joblib
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
        if component["kind"] == "binary":
            return 1.0 / (1.0 + np.exp(-np.clip(logits.ravel(), -40, 40)))
        logits = logits - logits.max(axis=1, keepdims=True)
        probability = np.exp(np.clip(logits, -40, 0))
        probability /= probability.sum(axis=1, keepdims=True)
        normal_column = int(np.flatnonzero(component["classes"] == 1)[0])
        return 1.0 - probability[:, normal_column]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        expected = int(self.artifact["n_features"])
        if X.ndim != 2 or X.shape[1] != expected:
            raise ValueError(f"Expected [n, {expected}] embeddings; got {X.shape}")
        probability = np.zeros(len(X), dtype=np.float64)
        for weight, component in zip(self.artifact["weights"], self.artifact["components"]):
            probability += float(weight) * self._probability(component, X)
        return (probability >= float(self.artifact["threshold"])).astype(np.int64)
'''


def package_submission(artifact: dict, output_dir: Path) -> Path:
    build_dir = output_dir / f"submission_build_{VERSION}"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    (build_dir / "classifier.py").write_text(CLASSIFIER_SOURCE, encoding="utf-8")
    joblib.dump(artifact, build_dir / "trained_probe.joblib", compress=3)
    zip_path = output_dir / f"mental_health_probe_{VERSION}_layer14_mean64.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(build_dir / "classifier.py", "classifier.py")
        archive.write(build_dir / "trained_probe.joblib", "trained_probe.joblib")
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["classifier.py", "trained_probe.joblib"]
    return zip_path


def smoke_test_zip(zip_path: Path):
    import importlib.util
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temp_dir)
        spec = importlib.util.spec_from_file_location("submitted_classifier", Path(temp_dir) / "classifier.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classifier = module.Classifier()
        rng = np.random.default_rng(7)
        sample = rng.normal(size=(1700, HIDDEN_SIZE)).astype(np.float32)
        started = time.perf_counter()
        prediction = classifier.predict(sample)
        duration = time.perf_counter() - started
        assert prediction.shape == (1700,)
        assert set(np.unique(prediction)).issubset({0, 1})
        print(f"Isolated ZIP smoke test: {duration:.4f}s for 1700 rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=0, help="0 uses the full public corpus")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / VERSION)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    texts, source_labels = load_training_rows(args.max_rows or None)
    embedding_path = args.embeddings or args.output_dir / f"gemma2_layer14_mean64_{len(texts)}.npy"
    labels_path = args.output_dir / f"source_labels_{len(texts)}.npy"
    np.save(labels_path, source_labels)

    if args.skip_extraction:
        X = np.load(embedding_path, mmap_mode="r")
    else:
        X = extract_embeddings(texts, embedding_path, batch_size=args.batch_size)
    if X.shape != (len(texts), HIDDEN_SIZE) or not np.isfinite(X).all():
        raise ValueError(f"Invalid embedding matrix: {X.shape}")

    artifact, report = fit_probe_artifact(X, source_labels, texts)
    artifact_path = args.output_dir / f"trained_probe_{VERSION}.joblib"
    report_path = args.output_dir / f"training_report_{VERSION}.json"
    joblib.dump(artifact, artifact_path, compress=3)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    zip_path = package_submission(artifact, args.output_dir)
    smoke_test_zip(zip_path)
    print(json.dumps({"artifact": str(artifact_path), "report": str(report_path), "submission": str(zip_path)}, indent=2))


if __name__ == "__main__":
    main()
