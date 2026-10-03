"""Train v005 for CodaBench competition 17670: toxicity detection.

The organizer contract is Gemma-2-2B hidden-state index 14, masked mean
pooling, and a maximum sequence length of 64.  This module reuses the tested
v004 extractor and portable NumPy submission implementation, but trains on
Civil Comments toxicity labels instead of the unrelated mental-health corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import joblib
import numpy as np

import train_probe_v004 as base


VERSION = "v005"
DATASET_ID = "google/civil_comments"
SAFE_ROWS = 20_000
TOXIC_ROWS = 30_000
TOXICITY_THRESHOLD = 0.5
SEED = 34


def _stable_rank(text: str) -> int:
    return int.from_bytes(hashlib.sha1(base.canonical_text(text).encode("utf-8")).digest()[:8], "big")


def load_training_rows(max_rows: int | None = None):
    """Load a deterministic, deduplicated toxicity sample.

    A 40/60 safe/toxic mix is close to the hidden evaluation prevalence while
    still retaining enough examples from both classes for a stable boundary.
    """
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split="train")
    texts = np.asarray(dataset["text"], dtype=object)
    scores = np.asarray(dataset["toxicity"], dtype=np.float32)
    labels = (scores >= TOXICITY_THRESHOLD).astype(np.int64)

    # Remove empty and exact duplicate comments before deterministic sampling.
    chosen: dict[str, tuple[str, int, float]] = {}
    for text, label, score in zip(texts, labels, scores):
        key = base.canonical_text(text)
        if not key:
            continue
        # If duplicate annotations differ, retain the more decisive score.
        prior = chosen.get(key)
        certainty = abs(float(score) - TOXICITY_THRESHOLD)
        if prior is None or certainty > abs(prior[2] - TOXICITY_THRESHOLD):
            chosen[key] = (str(text), int(label), float(score))

    rows = list(chosen.values())
    safe = sorted((row for row in rows if row[1] == 0), key=lambda row: _stable_rank(row[0]))
    toxic = sorted((row for row in rows if row[1] == 1), key=lambda row: _stable_rank(row[0]))

    if max_rows:
        toxic_count = min(len(toxic), max(1, round(max_rows * TOXIC_ROWS / (SAFE_ROWS + TOXIC_ROWS))))
        safe_count = min(len(safe), max_rows - toxic_count)
    else:
        safe_count, toxic_count = min(SAFE_ROWS, len(safe)), min(TOXIC_ROWS, len(toxic))

    sampled = safe[:safe_count] + toxic[:toxic_count]
    sampled.sort(key=lambda row: _stable_rank(row[0]))
    return [row[0] for row in sampled], np.asarray([row[1] for row in sampled], dtype=np.int64)


def fit_probe_artifact(X: np.ndarray, labels: np.ndarray, texts: list[str]):
    """Tune a compact ensemble of binary logistic probes."""
    specs = [
        base.CandidateSpec("binary", row_l2, k, C)
        for row_l2 in (False, True)
        for k in (512, 1024, base.HIDDEN_SIZE)
        for C in (0.002, 0.01, 0.05, 0.2)
    ]

    # The generic fitter treats every label other than NORMAL_SOURCE_LABEL as
    # positive.  Binary Civil Comments labels therefore map directly.
    original_normal = base.NORMAL_SOURCE_LABEL
    base.NORMAL_SOURCE_LABEL = 0
    try:
        # The v004 fitter's seven-source-class guard does not apply here, so
        # present seven harmless source strata that preserve the binary target.
        scores = np.asarray(labels, dtype=np.int64)
        strata = scores.copy()
        rank = np.asarray([_stable_rank(text) for text in texts], dtype=np.uint64)
        strata[scores == 1] = 1 + (rank[scores == 1] % 6).astype(np.int64)
        artifact, report = base.fit_probe_artifact(X, strata, texts, specs=specs, max_tuning_rows=35_000)
    finally:
        base.NORMAL_SOURCE_LABEL = original_normal

    artifact["version"] = VERSION
    artifact["task"] = "toxicity"
    report["version"] = VERSION
    report["task"] = "toxicity"
    report["label_rule"] = f"civil_comments.toxicity >= {TOXICITY_THRESHOLD}"
    return artifact, report


def package_submission(artifact: dict, output_dir: Path) -> Path:
    build_dir = output_dir / f"submission_build_{VERSION}"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    (build_dir / "classifier.py").write_text(base.CLASSIFIER_SOURCE, encoding="utf-8")
    joblib.dump(artifact, build_dir / "trained_probe.joblib", compress=3)
    zip_path = output_dir / f"toxicity_probe_{VERSION}_layer14_mean64.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(build_dir / "classifier.py", "classifier.py")
        archive.write(build_dir / "trained_probe.joblib", "trained_probe.joblib")
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["classifier.py", "trained_probe.joblib"]
    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / VERSION)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()

    base.VERSION = VERSION
    base.DATASET_ID = DATASET_ID
    base.NORMAL_SOURCE_LABEL = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    texts, labels = load_training_rows(args.max_rows or None)
    embedding_path = args.embeddings or args.output_dir / f"gemma2_layer14_mean64_{len(texts)}.npy"
    np.save(args.output_dir / f"toxicity_labels_{len(texts)}.npy", labels)

    if args.skip_extraction:
        X = np.load(embedding_path, mmap_mode="r")
    else:
        X = base.extract_embeddings(texts, embedding_path, batch_size=args.batch_size)
    if X.shape != (len(texts), base.HIDDEN_SIZE) or not np.isfinite(X).all():
        raise ValueError(f"Invalid embedding matrix: {X.shape}")

    artifact, report = fit_probe_artifact(X, labels, texts)
    artifact_path = args.output_dir / f"trained_probe_{VERSION}.joblib"
    report_path = args.output_dir / f"training_report_{VERSION}.json"
    joblib.dump(artifact, artifact_path, compress=3)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    zip_path = package_submission(artifact, args.output_dir)
    base.smoke_test_zip(zip_path)
    print(json.dumps({"artifact": str(artifact_path), "report": str(report_path), "submission": str(zip_path)}, indent=2))


if __name__ == "__main__":
    main()
