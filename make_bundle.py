"""Combine extracted ``.npy`` arrays into the optimizer's bundle format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_lines(path: Path | None, n: int):
    if path is None:
        return None
    values = path.read_text(encoding="utf-8").splitlines()
    if len(values) != n:
        raise ValueError(f"{path} has {len(values)} rows; expected {n}")
    return np.asarray(values, dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("embeddings", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--texts", type=Path)
    ap.add_argument("--groups", type=Path)
    args = ap.parse_args()
    X = np.load(args.embeddings, mmap_mode="r")
    y = np.asarray(np.load(args.labels), dtype=np.int64)
    if X.ndim not in (2, 3) or len(X) != len(y):
        raise ValueError(f"incompatible arrays: X={X.shape}, y={y.shape}")
    fields = {"X": np.asarray(X, dtype=np.float32), "y": y}
    texts = load_lines(args.texts, len(y))
    groups = load_lines(args.groups, len(y))
    if texts is not None:
        fields["texts"] = texts
    if groups is not None:
        fields["groups"] = groups
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **fields)
    print({"output": str(args.output), "shape": list(X.shape), "rows": len(y), "texts": texts is not None, "groups": groups is not None})


if __name__ == "__main__":
    main()
