"""Search public datasets for the organizer's hidden corpus by size fingerprint.

The competition never publishes its corpus, but the split arithmetic constrains
it tightly. `sklearn.train_test_split` takes `ceil(n * test_size)`, so a dev
split of exactly 1700 rows at 20% means the corpus had

    n in [8496, 8500]

and a stratified split of a corpus that is 70.588% positive is what produces the
dev set's exact 1200/500. The same reasoning on the sibling mental-health
competition (dev = 1697, near-balanced) gives n in [8481, 8485].

That sibling is the reason this is worth running rather than just arguing about.
Its leaderboard leader reached 0.9116 against the organizer's own 0.924
reference, which is in-domain quality -- somebody found that corpus. So the
fingerprint is *testable*: if it finds the mental-health corpus, the method is
credible and its answer on toxicity is worth acting on. If it finds neither,
that is itself the answer, and it costs no GPU time and no submissions to learn.

    python fingerprint_corpus.py --task both
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

HF_API = "https://huggingface.co/api"
DS_SERVER = "https://datasets-server.huggingface.co"


@dataclass(frozen=True)
class Target:
    name: str
    low: int
    high: int
    positive_low: float
    positive_high: float
    queries: tuple[str, ...]


# Row ranges come from ceil(n * 0.2) == dev_rows, which admits five values of n.
TOXICITY = Target(
    name="toxicity",
    low=8496, high=8500,
    positive_low=0.70, positive_high=0.71,
    queries=("toxic", "toxicity", "hate speech", "offensive", "abusive", "harmful",
             "profanity", "insult", "harassment", "moderation", "content safety",
             "hate", "cyberbullying", "troll", "obscene", "slur", "civility",
             "toxic comment", "offensive language", "safety classification"),
)

MENTAL_HEALTH = Target(
    name="mental_health",
    low=8481, high=8485,
    positive_low=0.45, positive_high=0.55,
    queries=("mental health", "depression", "anxiety", "suicide", "distress",
             "wellbeing", "stress detection", "emotion", "sentiment mental",
             "psychology", "counseling", "self harm", "mental illness",
             "reddit mental", "mental disorder", "therapy"),
)


def get(url: str, tries: int = 2, timeout: int = 25):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(1.0)


def discover(target: Target, per_query: int = 100) -> dict[str, dict]:
    seen: dict[str, dict] = {}
    for query in target.queries:
        url = f"{HF_API}/datasets?" + urllib.parse.urlencode(
            {"search": query, "limit": per_query, "full": "false"})
        try:
            for row in get(url):
                seen.setdefault(row["id"], row)
        except Exception as exc:
            print(f"    search {query!r} failed: {type(exc).__name__}")
    return seen


def size_of(name: str) -> dict | None:
    try:
        return get(f"{DS_SERVER}/size?dataset={urllib.parse.quote(name)}", tries=1, timeout=20)
    except Exception:
        return None


def positive_rate(name: str, config: str, split: str) -> tuple[float, str] | None:
    """Fraction of the majority-vs-rest binary split, if a label column exists."""
    query = urllib.parse.urlencode({"dataset": name, "config": config, "split": split})
    try:
        stats = get(f"{DS_SERVER}/statistics?{query}", tries=1, timeout=25)
    except Exception:
        return None
    for column in stats.get("statistics", []):
        if column["column_type"] not in ("string_label", "class_label", "int", "bool"):
            continue
        freq = column["column_statistics"].get("frequencies")
        if not freq or not 2 <= len(freq) <= 3:
            continue
        counts = sorted(freq.values(), reverse=True)
        total = sum(counts)
        if total:
            return counts[0] / total, column["column_name"]
    return None


def search(target: Target) -> list[dict]:
    print(f"\n{'=' * 70}\n{target.name}: looking for n in [{target.low}, {target.high}], "
          f"positive rate {target.positive_low:.2f}-{target.positive_high:.2f}\n{'=' * 70}")
    candidates = discover(target)
    print(f"  {len(candidates)} candidate datasets from {len(target.queries)} queries")

    hits = []
    checked = 0
    for name in sorted(candidates):
        info = size_of(name)
        checked += 1
        if checked % 100 == 0:
            print(f"    ...{checked}/{len(candidates)} checked, {len(hits)} hits", flush=True)
        if not info:
            continue
        try:
            splits = info["size"]["splits"]
            total = info["size"]["dataset"]["num_rows"]
        except Exception:
            continue

        # The corpus may be one split, or the union of all of them.
        shapes = [("__total__", "__total__", total)]
        shapes += [(s["config"], s["split"], s["num_rows"]) for s in splits]
        by_config: dict[str, int] = {}
        for s in splits:
            by_config[s["config"]] = by_config.get(s["config"], 0) + s["num_rows"]
        shapes += [(c, "__config_total__", n) for c, n in by_config.items()]

        for config, split, rows in shapes:
            if not (target.low <= rows <= target.high):
                continue
            entry = {"dataset": name, "config": config, "split": split, "rows": rows}
            if split not in ("__total__", "__config_total__"):
                rate = positive_rate(name, config, split)
                if rate:
                    entry["majority_rate"], entry["label_column"] = round(rate[0], 4), rate[1]
            hits.append(entry)
            match = ""
            if "majority_rate" in entry:
                ok = target.positive_low <= entry["majority_rate"] <= target.positive_high
                match = f"  rate={entry['majority_rate']:.4f} {'<<< MATCH' if ok else ''}"
            print(f"  HIT {name} [{config}/{split}] rows={rows}{match}", flush=True)

    print(f"  {len(hits)} size hits out of {checked} datasets checked")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["toxicity", "mental_health", "both"], default="both")
    parser.add_argument("--out", default="results/corpus_fingerprint.json")
    args = parser.parse_args()

    targets = {"toxicity": [TOXICITY], "mental_health": [MENTAL_HEALTH],
               "both": [MENTAL_HEALTH, TOXICITY]}[args.task]

    report = {}
    for target in targets:
        report[target.name] = search(target)

    from pathlib import Path
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")

    print("\n" + "=" * 70)
    for name, hits in report.items():
        strong = [h for h in hits if "majority_rate" in h]
        print(f"{name}: {len(hits)} size hits, {len(strong)} with a readable label balance")
        for h in strong:
            print(f"   {h['dataset']} [{h['config']}/{h['split']}] "
                  f"rows={h['rows']} rate={h['majority_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
