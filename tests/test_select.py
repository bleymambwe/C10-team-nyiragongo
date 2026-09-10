"""Correctness tests for the leave-one-source-out selection harness.

These run without any embeddings so a bug cannot survive until the GPU run.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe.select import (  # noqa: E402
    TARGET_POSITIVE_RATE,
    Recipe,
    auroc,
    best_threshold_accuracy,
    evaluate_recipes,
    quota_accuracy,
    resample_to_prior,
)


def test_quota_accuracy_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # 70% positive
    scores = np.arange(10, dtype=float)  # ranking exactly matches labels
    assert quota_accuracy(scores, y, 0.7) == pytest.approx(1.0)


def test_quota_accuracy_reversed_ranking():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    scores = np.arange(10, dtype=float)[::-1]
    # Top 7 by score are the first 7 rows: 3 negatives + 4 positives.
    # Predicted positive: rows 0-6 (4 correct), predicted negative: rows 7-9 (0 correct).
    assert quota_accuracy(scores, y, 0.7) == pytest.approx(0.4)


def test_quota_accuracy_matches_leaderboard_arithmetic():
    """The all-zeros submission scored exactly 500/1700 on the dev set."""
    y = np.concatenate([np.ones(1200, dtype=int), np.zeros(500, dtype=int)])
    scores = np.zeros(1700)
    assert quota_accuracy(scores, y, 0.0) == pytest.approx(500 / 1700)


def test_auroc_perfect_and_reversed_and_constant():
    y = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.0, 1.0, 2.0, 3.0]), y) == pytest.approx(1.0)
    assert auroc(np.array([3.0, 2.0, 1.0, 0.0]), y) == pytest.approx(0.0)
    # A constant score must score 0.5, not 1.0 -- this is the tie correction.
    assert auroc(np.zeros(4), y) == pytest.approx(0.5)


def test_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    scores = rng.normal(size=400) + y * 0.8
    from sklearn.metrics import roc_auc_score

    assert auroc(scores, y) == pytest.approx(roc_auc_score(y, scores), abs=1e-12)


def test_oracle_threshold_is_an_upper_bound_on_quota():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < TARGET_POSITIVE_RATE).astype(int)
    scores = rng.normal(size=500) + y * 0.5
    assert best_threshold_accuracy(scores, y) >= quota_accuracy(scores, y, TARGET_POSITIVE_RATE) - 1e-12


def test_resample_to_prior_hits_the_target_rate():
    y = np.concatenate([np.ones(900, dtype=int), np.zeros(900, dtype=int)])
    idx = resample_to_prior(y, TARGET_POSITIVE_RATE, seed=3)
    got = y[idx].mean()
    assert got == pytest.approx(TARGET_POSITIVE_RATE, abs=1e-3)
    # With 900 of each class, the positive side is the binding constraint at a
    # 70.6% target rate: 900 positives support 900/0.70588 = 1275 rows total.
    assert len(idx) >= 900 / TARGET_POSITIVE_RATE - 2


def test_evaluate_recipes_recovers_a_planted_direction():
    """Four synthetic sources share one toxic direction plus per-source shift.

    A probe that generalises must find the shared direction; a probe that
    latches onto the per-source offset must fail the held-out fold. The test
    asserts the harness can tell the difference.
    """
    rng = np.random.default_rng(7)
    d, per_source = 48, 400
    shared = rng.normal(size=d)
    shared /= np.linalg.norm(shared)

    X, y, sources = [], [], []
    for i, name in enumerate(["a", "b", "c", "d"]):
        offset = rng.normal(size=d) * 3.0  # a large, source-specific nuisance shift
        labels = rng.integers(0, 2, size=per_source)
        feats = rng.normal(size=(per_source, d)) + offset + np.outer(labels, shared) * 2.0
        X.append(feats)
        y.append(labels)
        sources += [name] * per_source

    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y)
    sources = np.array(sources)
    groups = {n: n for n in ["a", "b", "c", "d"]}

    recipes = [Recipe("std", "logistic", 0.05), Recipe("batchstd", "logistic", 0.05)]
    summary = evaluate_recipes(X, y, sources, groups, recipes=recipes, verbose=False)

    assert summary["families"] == ["a", "b", "c", "d"]
    assert len(summary["results"]) == 2
    for row in summary["results"]:
        assert row["n_folds"] == 4
        assert 0.0 <= row["worst_accuracy"] <= row["mean_accuracy"] <= 1.0
    # The planted direction is strong, so the best recipe must clear chance by a lot.
    assert summary["results"][0]["worst_accuracy"] > 0.8
    # Results are ranked by mean AUROC across folds, not by the worst fold. The
    # min over a handful of noisy folds is a min-of-noise: on the gpt2 rehearsal
    # it chose a recipe by a 0.0033 margin inside a ~0.011 standard error and
    # gave up 0.050 of mean AUROC for it.
    aurocs = [row["mean_auroc"] for row in summary["results"]]
    assert aurocs == sorted(aurocs, reverse=True)


def test_recipes_that_error_on_a_fold_rank_below_complete_ones():
    """A recipe summarised over fewer folds must never outrank a complete one.

    Its worst fold was never computed, so comparing it against a recipe that
    survived every fold compares different quantities. Latent on small data;
    live at 2304 dimensions, where LDA can hit a singular covariance.
    """
    X, y, sources, groups = _shifted_sources(gain=lambda rng, d: np.ones(d),
                                             offset_scale=0.5)

    good = Recipe("std", "logistic", 0.05)
    # A shrinkage outside [0, 1] makes the LDA solve fail on every fold it is
    # asked for, which is exactly the "errored on a fold" shape.
    bad = Recipe("std", "lda", float("nan"))
    summary = evaluate_recipes(X, y, sources, groups, recipes=[good, bad], verbose=False)

    by_key = {row["key"]: row for row in summary["results"]}
    if bad.key in by_key and by_key[bad.key]["n_folds"] < summary["expected_folds"]:
        order = [row["key"] for row in summary["results"]]
        assert order.index(good.key) < order.index(bad.key)


def _shifted_sources(gain, offset_scale, seed=11, d=32, per_source=600):
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=d)
    shared /= np.linalg.norm(shared)

    X, y, sources = [], [], []
    for name in ["p", "q", "r"]:
        held_out = name == "r"
        offset = np.zeros(d) if not held_out else rng.normal(size=d) * offset_scale
        scale = np.ones(d) if not held_out else gain(rng, d)
        labels = rng.integers(0, 2, size=per_source)
        feats = (rng.normal(size=(per_source, d)) + np.outer(labels, shared) * 1.5)
        X.append(feats * scale + offset)
        y.append(labels)
        sources += [name] * per_source

    return (np.vstack(X).astype(np.float32), np.concatenate(y), np.array(sources),
            {n: n for n in ["p", "q", "r"]})


def _accuracy_pair(args):
    summary = evaluate_recipes(
        *args,
        recipes=[Recipe("std", "logistic", 0.05), Recipe("batchstd", "logistic", 0.05)],
        verbose=False,
    )
    by_key = {row["key"]: row for row in summary["results"]}
    return (by_key["std|logistic|0.05"]["per_family"]["r"]["accuracy"],
            by_key["batchstd|logistic|0.05"]["per_family"]["r"]["accuracy"])


def test_quota_rule_is_already_immune_to_isotropic_shift():
    """A uniform shift and a single scalar gain cannot change a quota decision.

    The score is affine in X, so a batch-wide affine change of X is a monotone
    change of the score, and the quota rule only reads the ranking. This is why
    the top-1200 rule was worth +6 points on the leaderboard, and it means
    batch standardisation has nothing to add in this case.
    """
    args = _shifted_sources(gain=lambda rng, d: np.full(d, 4.0), offset_scale=0.0)
    args = (args[0] + 6.0, *args[1:])  # uniform offset on every source and dim
    train_std, batch_std = _accuracy_pair(args)
    assert train_std == pytest.approx(batch_std, abs=1e-12)


def test_batch_standardisation_beats_train_standardisation_under_anisotropic_shift():
    """Per-dimension rescaling is the shift that a quota alone cannot absorb.

    When the held-out source stretches some coordinates and squashes others,
    the score is no longer a monotone function of the training-domain score.
    Standardising with the evaluation batch's own statistics restores the
    geometry the probe was fitted in.
    """
    args = _shifted_sources(
        gain=lambda rng, d: np.exp(rng.normal(size=d) * 1.2),
        offset_scale=3.0,
    )
    train_std, batch_std = _accuracy_pair(args)
    assert batch_std > train_std + 0.02
