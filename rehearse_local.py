"""Rehearse the whole pipeline on CPU with a small cached model.

The competition contract needs Gemma-2-2b, which this machine cannot run at
corpus scale (2 cores, 8 GiB, no CUDA).  But the *method* -- which transform,
which probe, whether leave-one-source-out selection separates recipes at all --
does not depend on which decoder produced the embeddings.  Rehearsing with gpt2
on real toxicity text answers those questions for the price of ~20 minutes of
CPU, so the one GPU session is spent executing a plan rather than discovering
that the plan has a bug in it.

What transfers from this run: the ranking of transforms and probes, whether the
LOSO spread is large enough to choose on, and that every module runs on real
data. What does not transfer: the absolute accuracies. gpt2 is a far weaker
representation than Gemma-2-2b, so treat these numbers as a floor, not a
forecast.

    python rehearse_local.py --cap-scale 0.15 --model gpt2 --layer 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from probe.corpus import SOURCES_BY_NAME, build_corpus  # noqa: E402
from probe.extract import extract  # noqa: E402
from probe.finalize import build_candidates, report as candidate_report  # noqa: E402
from probe.select import (  # noqa: E402
    evaluate_cross_source,
    evaluate_recipes,
    report as loso_report,
)

# Sources chosen for a fast rehearsal: small downloads, five distinct families.
REHEARSAL_SOURCES = [
    "jigsaw_wiki",
    "davidson",
    "tweeteval_offensive",
    "tweeteval_hate",
    "hatecheck",
    "aegis",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-scale", type=float, default=0.15)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default="results/rehearsal")
    parser.add_argument("--sources", default=",".join(REHEARSAL_SOURCES))
    parser.add_argument("--skip-candidates", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in args.sources.split(",") if n]

    print("== corpus ==")
    started = time.perf_counter()
    texts, labels, sources = build_corpus(names, cap_scale=args.cap_scale, positive_rate=0.5)
    families = [SOURCES_BY_NAME[s].group for s in sources]
    labels = np.asarray(labels, dtype=np.int64)
    print(f"   {len(texts)} rows in {time.perf_counter() - started:.0f}s")
    if len(set(families)) < 3:
        print("   need at least 3 families for leave-one-source-out; aborting")
        return 1

    print("\n== extract ==")
    tag = f"{args.model.replace('/', '_')}_l{args.layer}_{len(texts)}"
    started = time.perf_counter()
    X = np.asarray(extract(texts, out / f"{tag}.npy", model_id=args.model,
                           layer=args.layer, batch_size=args.batch_size))
    elapsed = time.perf_counter() - started
    print(f"   {X.shape} in {elapsed:.0f}s ({len(texts) / max(elapsed, 1e-6):.1f} rows/s)")

    print("\n== leave-one-source-out ==")
    groups = dict(zip(sources, families))
    summary = evaluate_recipes(X, labels, np.asarray(sources), groups, verbose=True)
    (out / f"loso_{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print()
    print(loso_report(summary, top=20))

    print("\n== cross-source diagnostic (positives and negatives from different corpora) ==")
    cross = evaluate_cross_source(X, labels, np.asarray(sources), groups, verbose=False)
    (out / f"cross_{tag}.json").write_text(json.dumps(cross, indent=1), encoding="utf-8")
    print(f"   pairs: {', '.join(cross['pairs'])}")
    for row in cross["results"][:10]:
        print(f"   {row['key']:<30} worst={row['worst_accuracy']:.4f} "
              f"mean={row['mean_accuracy']:.4f}")

    if not args.skip_candidates:
        print("\n== candidates ==")
        manifest = build_candidates(X, labels, np.asarray(sources), groups, summary,
                                    out / f"submissions_{tag}", cross=cross,
                                    per_source=True)
        print(candidate_report(manifest))

    print(f"\nwrote {out / f'loso_{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
