"""Repackage the router with the exact leaderboard incumbent as its fallback."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import zipfile
from pathlib import Path

import joblib
import numpy as np

from src.probe.routing import package_router


def _load(zip_path: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extract("trained_probe.joblib", tmp)
        return joblib.load(Path(tmp) / "trained_probe.joblib")


def finalize(router_zip: Path, incumbent_zip: Path, output_zip: Path,
             source_only: bool = False) -> dict:
    routed = _load(router_zip)
    incumbent = _load(incumbent_zip)
    if len(incumbent["heads"]) != 1:
        raise ValueError("Expected a single-head incumbent")

    incumbent_head = incumbent["heads"][0]
    candidates = []
    found_pooled = False
    for item in routed["router"]["candidates"]:
        copy = dict(item)
        if copy["name"] == "pooled":
            copy["head"] = incumbent_head
            found_pooled = True
        candidates.append(copy)
    if not found_pooled:
        raise ValueError("Router is missing its pooled fallback")
    if source_only:
        candidates = [c for c in candidates if c["name"] != "pooled"]

    artifact = dict(incumbent)
    artifact["meta"] = dict(incumbent.get("meta", {})) | {
        "routing": "prior-adjusted diagonal Gaussian batch distance",
        "fallback_sha256": hashlib.sha256(incumbent_zip.read_bytes()).hexdigest(),
    }
    checks = package_router(
        artifact,
        {"scale": np.asarray(incumbent_head["scale"]), "candidates": candidates},
        output_zip,
    )
    return {
        "path": str(output_zip),
        "sha256": hashlib.sha256(output_zip.read_bytes()).hexdigest(),
        "incumbent_sha256": hashlib.sha256(incumbent_zip.read_bytes()).hexdigest(),
        "checks": checks,
        "candidate_names": [c["name"] for c in candidates],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("router_zip", type=Path)
    parser.add_argument("incumbent_zip", type=Path)
    parser.add_argument("output_zip", type=Path)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    print(finalize(args.router_zip, args.incumbent_zip, args.output_zip,
                   source_only=args.source_only))
