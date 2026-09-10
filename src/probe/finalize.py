"""Turn selection results into verified submission zips.

Two kinds of candidate come out of here.

`best_single` / `best_ensemble` are the models we actually want to score: the
recipes whose *worst* leave-one-source-out fold was strongest, refitted on every
labelled row.

`source_<family>` candidates exist for a different reason.  The organizer's
corpus is unknown, and one dev submission per training source turns the
leaderboard into a measurement of which public corpus the hidden data resembles.
That answer transfers to the final phase, which is a different sample of the
same unknown corpus; a threshold tuned against dev feedback would not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .select import ALL_POSITIVE_BASELINE, PROBES, TRANSFORMS, Recipe
from .submission import build_and_verify, make_head

TRANSFORMS_BY_NAME = {t.name: t for t in TRANSFORMS}


def fit_head(X: np.ndarray, y: np.ndarray, recipe: Recipe, weight: float = 1.0,
             dtype=np.float64) -> dict:
    """Fit one linear head and fold the transform into a serialisable form.

    `dtype` is a memory control, not a modelling choice: the final fit runs on
    every row, and at 108k x 2304 a float64 working copy is 2.0 GB. float32
    halves that and is well inside the precision of embeddings that were stored
    as float16.
    """
    transform = TRANSFORMS_BY_NAME[recipe.transform]
    state = transform.fit(X)
    Z = transform.apply(state, X, dtype=dtype)
    fit_fn, _grid, _needs = PROBES[recipe.probe]
    coef, intercept = fit_fn(Z, y, recipe.hyper)

    center = scale = None
    if transform.standardize == "train":
        center, scale = state["center"], state["scale"]
    return make_head(coef, intercept, row_l2=transform.row_l2,
                     standardize=transform.standardize,
                     center=center, scale=scale, weight=weight,
                     remove_top=transform.remove_top,
                     components=state.get("components"))


def _recipe_from_row(row: dict) -> Recipe:
    return Recipe(row["transform"], row["probe"], float(row["hyper"]))


def build_candidates(
    X: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    families: dict[str, str],
    summary: dict,
    out_dir: Path | str,
    *,
    cross: dict | None = None,
    ensemble_size: int = 3,
    per_source: bool = True,
    min_source_rows: int = 1500,
) -> dict:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    sources = np.asarray(sources)
    family_of = np.array([families.get(s, s) for s in sources])
    out_dir = Path(out_dir)
    rate = float(summary["positive_rate"])
    width = int(X.shape[1])
    results = summary["results"]
    if not results:
        raise ValueError("selection summary has no results")

    manifest: list[dict] = []
    skipped: list[dict] = []

    # 1. The single strongest recipe by worst-fold accuracy, refit on everything.
    best = _recipe_from_row(results[0])
    head = fit_head(X, y, best)
    manifest.append({
        "name": "best_single",
        "recipe": best.key,
        "loso_worst": results[0]["worst_accuracy"],
        "loso_mean": results[0]["mean_accuracy"],
        "trained_rows": int(len(y)),
        **build_and_verify([head], out_dir / "best_single.zip", version="best_single",
                           positive_rate=rate, n_features=width,
                           meta={"recipe": best.key, "loso": results[0]["worst_accuracy"]}),
    })

    # 1b. The same ranking with a prior-corrected probability threshold instead
    #     of a quota. The quota assumes this batch has the dev set's class
    #     balance; the threshold assumes only the prior, so it degrades smoothly
    #     if the final phase is composed differently. Submitting both in the dev
    #     phase measures what that hedge costs. Only logistic heads are eligible:
    #     an LDA score is not on a probability scale, so a sigmoid of it means
    #     nothing.
    logistic_rows = [r for r in results if r["probe"] == "logistic"]
    if logistic_rows:
        recipe = _recipe_from_row(logistic_rows[0])
        head = fit_head(X, y, recipe)
        manifest.append({
            "name": "best_threshold",
            "recipe": recipe.key,
            "loso_worst": logistic_rows[0]["worst_accuracy"],
            "loso_mean": logistic_rows[0]["mean_accuracy"],
            "trained_rows": int(len(y)),
            **build_and_verify([head], out_dir / "best_threshold.zip",
                               version="best_threshold", positive_rate=rate,
                               combine="probability", decision="threshold",
                               n_features=width, meta={"recipe": recipe.key}),
        })

    # 1c. The recipe that survives the cross-source diagnostic best. This is a
    #     different selection criterion, not a better one: it optimises for the
    #     hypothesis that the hidden corpus draws its positives and negatives
    #     from different sources. Building both lets the leaderboard decide
    #     which criterion was right, in one submission each.
    if cross and cross.get("results"):
        cross_best = _recipe_from_row(cross["results"][0])
        cross_score = cross["results"][0]["mean_accuracy"]
        if cross_score < ALL_POSITIVE_BASELINE:
            # On the gpt2 rehearsal this candidate scored 0.5827, below even the
            # random-ranking quota of 0.5848, and the runbook had it queued as a
            # day-one submission. Submissions are capped at five a day and are
            # irreversible; spending one on a model that loses to `predict all
            # ones` is the most expensive kind of mistake available here.
            skipped.append({"name": "best_cross_source", "recipe": cross_best.key,
                            "score": cross_score,
                            "reason": f"below all-positive baseline "
                                      f"{ALL_POSITIVE_BASELINE:.4f}"})
        elif cross_best.key != best.key:
            head = fit_head(X, y, cross_best)
            manifest.append({
                "name": "best_cross_source",
                "recipe": cross_best.key,
                "loso_worst": cross["results"][0]["worst_accuracy"],
                "loso_mean": cross["results"][0]["mean_accuracy"],
                "selected_by": "cross-source worst fold",
                "trained_rows": int(len(y)),
                **build_and_verify([head], out_dir / "best_cross_source.zip",
                                   version="best_cross_source", positive_rate=rate,
                                   n_features=width, meta={"recipe": cross_best.key}),
            })

    # 2. A rank ensemble over the top recipes, one per transform family. Two
    #    recipes that differ only in a regularisation constant make nearly the
    #    same errors, so they are filtered out: an ensemble is only worth
    #    building from members whose geometry actually differs.
    picked: list[dict] = []
    seen_transforms: set[str] = set()
    for row in results:
        if row["transform"] in seen_transforms:
            continue
        seen_transforms.add(row["transform"])
        picked.append(row)
        if len(picked) == ensemble_size:
            break
    if len(picked) > 1:
        heads = [fit_head(X, y, _recipe_from_row(row)) for row in picked]
        manifest.append({
            "name": "best_ensemble",
            "recipe": " + ".join(row["key"] for row in picked),
            "loso_worst": min(row["worst_accuracy"] for row in picked),
            "loso_mean": float(np.mean([row["mean_accuracy"] for row in picked])),
            "trained_rows": int(len(y)),
            **build_and_verify(heads, out_dir / "best_ensemble.zip",
                               version="best_ensemble", positive_rate=rate,
                               n_features=width,
                               meta={"recipes": [row["key"] for row in picked]}),
        })

    # 3. One probe per training source, for the leaderboard source probe.
    if per_source:
        for family in sorted(set(family_of.tolist())):
            mask = family_of == family
            if mask.sum() < min_source_rows:
                continue
            if len(set(y[mask].tolist())) < 2:
                continue
            head = fit_head(X[mask], y[mask], best)
            fold = results[0]["per_family"].get(family, {})
            manifest.append({
                "name": f"source_{family}",
                "recipe": best.key,
                # There is no leave-one-source-out number for a single-source
                # probe: it was trained on one corpus, not selected across
                # several. What is recorded instead is how the *pooled* model
                # scored when this family was held out, which says how unlike
                # the others this corpus is -- a different quantity, and it must
                # not be read as this candidate's expected accuracy.
                "loso_worst": None,
                "pooled_accuracy_on_this_family": fold.get("accuracy"),
                "trained_rows": int(mask.sum()),
                **build_and_verify([head], out_dir / f"source_{family}.zip",
                                   version=f"source_{family}", positive_rate=rate,
                                   n_features=width,
                                   meta={"recipe": best.key, "family": family}),
            })

    # Refuse candidates that lose to `predict all ones` -- but judge that on the
    # *mean* fold, not the worst.
    #
    # The hidden set is one unknown corpus, so the expected score is the mean
    # over held-out corpora; the worst fold is a risk measure, not a forecast.
    # Gating on the worst is far too strict, and it showed: on the real Gemma
    # run the best recipe had mean accuracy 0.7934 with 8 of 9 folds above the
    # baseline, and a single bad fold (offensivelang, 0.6252) caused every
    # pooled candidate to be refused. Nothing shippable survived.
    for row in list(manifest):
        score = row.get("loso_mean")
        if isinstance(score, float) and score < ALL_POSITIVE_BASELINE:
            manifest.remove(row)
            skipped.append({"name": row["name"], "recipe": row["recipe"], "score": score,
                            "reason": f"mean fold {score:.4f} below all-positive "
                                      f"baseline {ALL_POSITIVE_BASELINE:.4f}"})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if skipped:
        (out_dir / "skipped.json").write_text(json.dumps(skipped, indent=1), encoding="utf-8")
    return {"out_dir": str(out_dir), "candidates": manifest, "skipped": skipped}


def report(manifest: dict) -> str:
    """Render candidates, and say plainly what was refused and why."""
    # The "worst fold" column is not one quantity: most rows report the
    # leave-one-source-out worst fold, best_cross_source reports the
    # cross-source worst fold, and single-source probes have neither. Naming the
    # criterion per row stops the three being compared as if they were the same.
    lines = [f"{'candidate':<26}{'rows':>8}{'mean':>8}{'worst':>8}  {'selected by':<25}recipe"]
    for row in manifest["candidates"]:
        worst = row.get("loso_worst")
        mean = row.get("loso_mean")
        worst_text = f"{worst:.4f}" if isinstance(worst, float) else "    -"
        mean_text = f"{mean:.4f}" if isinstance(mean, float) else "    -"
        criterion = row.get("selected_by", "LOSO worst fold"
                            if row["name"].startswith("best") else "single corpus")
        lines.append(f"{row['name']:<26}{row['trained_rows']:>8}{mean_text:>8}{worst_text:>8}  "
                     f"{criterion:<25}{row['recipe']}")
    return "\n".join(lines)
