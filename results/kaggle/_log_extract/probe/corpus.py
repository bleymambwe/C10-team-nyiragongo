"""Multi-source toxicity corpus for the Latent Probe Challenge (CodaBench 17670).

The organizer's corpus is not published.  Rather than bet the submission on one
guess, this module assembles several public toxicity corpora and keeps the
source name attached to every row.  Source identity then serves two purposes:

  * it is the CV grouping for leave-one-source-out selection, which is the only
    honest local simulation of "score on a corpus you have never trained on";
  * it is the unit of the leaderboard probe in G4 of COMPETITION_GOALS.md.

Every loader returns raw text.  Embedding extraction is a separate step because
it needs a GPU; see colab_extract_embeddings.ipynb.

Schemas were read from the HF datasets-server on 2026-09-06, not assumed.
Large corpora are streamed with a bounded scan so a dry run does not pull
gigabytes; the scan order is the dataset's own order, so the selection is
reproducible.

Dependency-light and self-contained on purpose: the Colab notebook writes this
file out verbatim rather than uploading the repository.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Iterator

_WS = re.compile(r"\s+")


def canonical_text(value: object) -> str:
    """Normalisation used for de-duplication only, never for extraction."""
    return _WS.sub(" ", str(value).lower()).strip()


def stable_rank(text: str) -> int:
    """Deterministic, seed-free shuffle key so sampling is reproducible."""
    return int.from_bytes(hashlib.sha1(canonical_text(text).encode("utf-8")).digest()[:8], "big")


def _stream(name: str, config: str | None, split: str) -> Iterator[dict]:
    from datasets import load_dataset

    return iter(load_dataset(name, config, split=split, streaming=True))


def _take(iterable: Iterator[dict], limit: int) -> Iterator[dict]:
    for index, row in enumerate(iterable):
        if index >= limit:
            return
        yield row


# --------------------------------------------------------------------------
# Loaders.  Each yields (text, label) with label 1 = toxic / hateful / unsafe.
#
# Ambiguous middle bands are dropped rather than guessed.  For a linear probe a
# mislabelled row near the boundary costs more than a missing one, because it
# pulls the direction rather than merely failing to inform it.
# --------------------------------------------------------------------------


def load_civil_comments(scan: int):
    for row in _take(_stream("google/civil_comments", None, "train"), scan):
        score = row.get("toxicity")
        if score is None:
            continue
        if score >= 0.5:
            yield row["text"], 1
        elif score <= 0.05:
            yield row["text"], 0


def load_jigsaw_wiki(scan: int):
    """Original Jigsaw Wikipedia talk-page comments (the 2018 Kaggle train set)."""
    for row in _take(_stream("tasksource/jigsaw_toxicity", None, "train"), scan):
        yield row["comment_text"], int(row["toxic"])


def load_jigsaw_unintended(scan: int):
    """A second, differently-sampled view of the Civil Comments family."""
    for row in _take(_stream("mteb/toxic_conversations_50k", None, "train"), scan):
        yield row["text"], int(row["label"])


def load_davidson(scan: int):
    # class: 0 = hate speech, 1 = offensive language, 2 = neither.
    for row in _take(_stream("tdavidson/hate_speech_offensive", None, "train"), scan):
        yield row["tweet"], (0 if int(row["class"]) == 2 else 1)


def _tweeteval(config: str):
    for split in ("train", "validation", "test"):
        try:
            for row in _stream("cardiffnlp/tweet_eval", config, split):
                yield row["text"], int(row["label"])
        except Exception:
            continue


def load_tweeteval_offensive(scan: int):
    yield from _take(_tweeteval("offensive"), scan)


def load_tweeteval_hate(scan: int):
    yield from _take(_tweeteval("hate"), scan)


def load_hatecheck(scan: int):
    for row in _take(_stream("Paul/hatecheck", None, "test"), scan):
        yield row["test_case"], (1 if str(row["label_gold"]).strip().lower() == "hateful" else 0)


def load_toxic_chat(scan: int):
    for split in ("train", "test"):
        try:
            for row in _take(_stream("lmsys/toxic-chat", "toxicchat0124", split), scan):
                yield row["user_input"], int(row["toxicity"])
        except Exception:
            continue


def load_berkeley_hate(scan: int):
    """Berkeley measuring-hate-speech.

    Rows are *annotator* rows, not comments: 135,556 rows cover roughly 39,000
    comments.  Averaging per comment before thresholding is required, otherwise
    the same text enters the corpus several times with conflicting labels and
    the de-duplicator keeps whichever annotator happened to come first.
    """
    totals: dict[int, list[float]] = {}
    texts: dict[int, str] = {}
    for row in _take(_stream("ucberkeley-dlab/measuring-hate-speech", None, "train"), scan):
        key = int(row["comment_id"])
        totals.setdefault(key, []).append(float(row["hate_speech_score"]))
        texts.setdefault(key, row["text"])
    for key, scores in totals.items():
        mean = sum(scores) / len(scores)
        # The scale is calibrated so >0.5 is hate speech and <-1 is clearly
        # neutral or supportive.  The band between the two is genuinely
        # ambiguous, so it is dropped.
        if mean > 0.5:
            yield texts[key], 1
        elif mean < -1.0:
            yield texts[key], 0


def load_real_toxicity_prompts(scan: int):
    for row in _take(_stream("allenai/real-toxicity-prompts", None, "train"), scan):
        prompt = row.get("prompt") or {}
        score, text = prompt.get("toxicity"), prompt.get("text")
        if score is None or not text:
            continue
        if score >= 0.5:
            yield text, 1
        elif score <= 0.05:
            yield text, 0


def load_offensivelang(scan: int):
    """OffensiveLang: community-authored *implicit* offensive statements.

    Included because its size is close to the deduced hidden corpus (8,270 rows
    at 78.9% offensive, against a deduced ~8,500 at 70.6%) and because implicit
    offensiveness without profanity is a flavour none of the other corpora
    cover. The size resemblance is a hint, not a match -- see G0b.
    """
    for split in ("train", "test"):
        try:
            for row in _take(_stream("AmitDasRup123/OffensiveLang", None, split), scan):
                yield row["Text"], (1 if str(row["Offensive"]).strip().lower() == "yes" else 0)
        except Exception:
            continue


def load_hu_berlin_toxicity(scan: int):
    """A balanced 8,513-row toxicity set, comment-shaped text."""
    for split in ("train", "test"):
        try:
            for row in _take(_stream("HU-Berlin-ML-Internal/toxicity-dataset", None, split), scan):
                yield row["text"], int(row["label"])
        except Exception:
            continue


def load_aegis(scan: int):
    """NVIDIA Aegis: five annotator label columns, majority vote on 'Safe'."""
    for row in _take(_stream("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", None, "train"), scan):
        votes = [str(row.get(f"labels_{i}") or "").strip() for i in range(5)]
        votes = [v for v in votes if v]
        if not votes:
            continue
        safe = sum(1 for v in votes if v.lower() == "safe")
        if safe * 2 > len(votes):
            yield row["text"], 0
        elif safe * 2 < len(votes):
            yield row["text"], 1


@dataclass(frozen=True)
class Source:
    name: str
    load: Callable[[int], Iterator[tuple[str, int]]]
    cap: int
    # Rows to read from the remote corpus. Streaming stops here, so a small
    # rehearsal does not pay for a full pass over a 1.8M-row dataset.
    scan: int
    note: str = ""
    # Corpora drawn from the same underlying data must share a family, or a
    # leave-one-source-out split would train on a near-copy of its own test set
    # and the "unseen corpus" claim would be false.
    family: str = field(default="")

    @property
    def group(self) -> str:
        return self.family or self.name


SOURCES: tuple[Source, ...] = (
    Source("civil_comments", load_civil_comments, cap=20_000, scan=400_000,
           note="Civil Comments, toxicity>=0.5 vs <=0.05", family="civil"),
    Source("jigsaw_unintended", load_jigsaw_unintended, cap=10_000, scan=50_000,
           note="mteb/toxic_conversations_50k, a second Civil Comments sample", family="civil"),
    Source("jigsaw_wiki", load_jigsaw_wiki, cap=20_000, scan=60_000,
           note="Jigsaw 2018 Wikipedia talk-page comments", family="wiki"),
    Source("davidson", load_davidson, cap=16_000, scan=25_000,
           note="Davidson tweets, hate+offensive vs neither", family="twitter"),
    Source("tweeteval_offensive", load_tweeteval_offensive, cap=12_000, scan=20_000,
           note="OffensEval tweets", family="twitter"),
    Source("tweeteval_hate", load_tweeteval_hate, cap=10_000, scan=20_000,
           note="TweetEval hate", family="twitter"),
    Source("hatecheck", load_hatecheck, cap=3_700, scan=4_000,
           note="HateCheck functional tests, templated, 68.8% hateful", family="hatecheck"),
    Source("toxic_chat", load_toxic_chat, cap=8_000, scan=12_000,
           note="LMSYS user prompts to a chat model", family="chat"),
    Source("aegis", load_aegis, cap=8_000, scan=12_000,
           note="NVIDIA Aegis content safety, majority vote", family="chat_aegis"),
    Source("offensivelang", load_offensivelang, cap=8_000, scan=9_000,
           note="OffensiveLang, implicit offensiveness without profanity", family="offensivelang"),
    Source("hu_berlin_toxicity", load_hu_berlin_toxicity, cap=8_000, scan=9_600,
           note="HU-Berlin toxicity-dataset, balanced comment text", family="hu_berlin"),
    Source("berkeley_hate", load_berkeley_hate, cap=16_000, scan=136_000,
           note="Berkeley measuring-hate-speech, per-comment mean score", family="berkeley"),
    Source("real_toxicity_prompts", load_real_toxicity_prompts, cap=14_000, scan=100_000,
           note="RealToxicityPrompts prompt halves", family="rtp"),
)

SOURCES_BY_NAME = {s.name: s for s in SOURCES}


def _balance(rows: list[tuple[str, int]], cap: int, positive_rate: float
             ) -> list[tuple[str, int]]:
    """Deterministically down-sample to `cap` rows at a target positive rate.

    Every source is forced to the same prior so that leave-one-source-out
    differences reflect representation transfer rather than each corpus's own
    base rate.
    """
    pos = sorted((t for t, y in rows if y == 1), key=stable_rank)
    neg = sorted((t for t, y in rows if y == 0), key=stable_rank)
    want_pos = min(len(pos), int(round(cap * positive_rate)))
    want_neg = min(len(neg), cap - want_pos)
    # If one class ran out, spend the remaining budget on the other.
    if want_pos + want_neg < cap:
        if want_pos == len(pos):
            want_neg = min(len(neg), cap - want_pos)
        else:
            want_pos = min(len(pos), cap - want_neg)
    out = [(t, 1) for t in pos[:want_pos]] + [(t, 0) for t in neg[:want_neg]]
    out.sort(key=lambda r: stable_rank(r[0]))
    return out


def build_corpus(
    names: list[str] | None = None,
    *,
    positive_rate: float = 0.5,
    cap_scale: float = 1.0,
    scan_scale: float | None = None,
    min_chars: int = 3,
    max_chars: int = 2000,
    verbose: bool = True,
) -> tuple[list[str], list[int], list[str]]:
    """Return (texts, labels, sources) with global de-duplication.

    De-duplication is global and source-ordered: a text present in two corpora
    is kept only in the first one listed in SOURCES.  Leaving copies in both
    would leak across the leave-one-source-out boundary and inflate the local
    number in exactly the place it must not be inflated.
    """
    # The scan budget follows the cap budget by default, so a small rehearsal
    # does not stream a full pass over a 1.8M-row corpus to keep 400 rows. The
    # floor keeps enough headroom above the cap for balancing to have a choice.
    if scan_scale is None:
        scan_scale = min(1.0, max(cap_scale * 2.0, 0.02))

    chosen = [s for s in SOURCES if names is None or s.name in names]
    seen: set[str] = set()
    all_t: list[str] = []
    all_y: list[int] = []
    all_s: list[str] = []

    for source in chosen:
        scan = max(source.cap, int(round(source.scan * scan_scale)))
        try:
            raw = list(source.load(scan))
        except Exception as exc:  # one unavailable corpus must not sink the run
            if verbose:
                print(f"  SKIP {source.name}: {type(exc).__name__}: {str(exc)[:140]}")
            continue

        kept: list[tuple[str, int]] = []
        for text, label in raw:
            text = str(text).strip()
            if not (min_chars <= len(text) <= max_chars):
                continue
            key = canonical_text(text)
            if not key or key in seen:
                continue
            seen.add(key)
            kept.append((text, int(label)))

        cap = max(1, int(round(source.cap * cap_scale)))
        kept = _balance(kept, cap, positive_rate)
        all_t += [r[0] for r in kept]
        all_y += [r[1] for r in kept]
        all_s += [source.name] * len(kept)
        if verbose:
            pos = sum(r[1] for r in kept)
            n = max(1, len(kept))
            print(f"  {source.name:<22} n={len(kept):>6}  pos={pos:>6} ({pos / n:.3f})  "
                  f"scanned={len(raw):>7}  [{source.note}]")

    if verbose:
        print(f"  {'TOTAL':<22} n={len(all_t):>6}  pos={sum(all_y):>6}  "
              f"sources={len(set(all_s))}")
    return all_t, all_y, all_s


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build the multi-source toxicity corpus")
    parser.add_argument("--out", default="corpus.jsonl")
    parser.add_argument("--cap-scale", type=float, default=1.0)
    parser.add_argument("--positive-rate", type=float, default=0.5)
    parser.add_argument("--sources", default="", help="comma-separated subset")
    args = parser.parse_args()

    names = [n for n in args.sources.split(",") if n] or None
    texts, labels, sources = build_corpus(names, positive_rate=args.positive_rate,
                                          cap_scale=args.cap_scale)
    with open(args.out, "w", encoding="utf-8") as handle:
        for text, label, source in zip(texts, labels, sources):
            handle.write(json.dumps({"text": text, "label": label, "source": source}) + "\n")
    print(f"wrote {len(texts)} rows -> {args.out}")
