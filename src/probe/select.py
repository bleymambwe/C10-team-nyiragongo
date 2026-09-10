"""Leave-one-source-out probe selection for CodaBench 17670.

Why not ordinary CV
-------------------
v005 scored 0.836 on a random split of its own training corpus and 0.767 on the
hidden dev set.  Random-split CV measures "can the probe separate held-out rows
of a corpus it was trained on", which is not the question being asked.  The
hidden set is a corpus the probe has never seen.  The only local simulation of
that is to hold out an entire source.

Every held-out source is resampled to the hidden set's known positive rate
(1200/1700 = 0.70588) before scoring, so the accuracy printed here is on the
same footing as a leaderboard number.

Decision rule
-------------
The class prior is known exactly, so the operating point is a *quota*: label the
top `positive_rate` fraction of scores positive.  This is invariant to any
monotone shift of the score distribution, which is precisely what domain shift
produces.  On the real leaderboard, switching v005 from a fitted probability
threshold to this rule moved 0.7047 -> 0.7671 with identical model weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np

TARGET_POSITIVE_RATE = 1200.0 / 1700.0  # measured, see COMPETITION_GOALS.md G0

# Submitting all ones scores 1200/1700 on the dev set. Any probe that does not
# clear this is worse than a one-line submission that reads none of the
# features, and there is no point shipping it. It is a surprisingly high bar
# because the set is 70.6% positive, and it is easy to miss when reading
# accuracies in isolation: the gpt2 rehearsal cleared it on only 1 of 5 folds.
ALL_POSITIVE_BASELINE = TARGET_POSITIVE_RATE

# A quota rule applied to a random ranking. Below this there is no signal at all.
RANDOM_QUOTA_BASELINE = (2 * TARGET_POSITIVE_RATE * TARGET_POSITIVE_RATE
                         - TARGET_POSITIVE_RATE + (1 - TARGET_POSITIVE_RATE))


# --------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------


def quota_accuracy(scores: np.ndarray, y: np.ndarray, positive_rate: float) -> float:
    """Accuracy when exactly `positive_rate` of the batch is labelled positive."""
    n = len(scores)
    k = int(round(n * positive_rate))
    k = min(max(k, 0), n)
    pred = np.zeros(n, dtype=np.int64)
    if k:
        pred[np.argpartition(scores, n - k)[n - k:]] = 1
    return float((pred == y).mean())


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUROC with tie correction; no sklearn dependency."""
    y = np.asarray(y)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks within ties so a constant score scores 0.5, not 1.0.
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(scores) + 1):
        if i == len(scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def best_threshold_accuracy(scores: np.ndarray, y: np.ndarray) -> float:
    """Oracle upper bound for any threshold rule. Diagnostic only, never used
    to pick an operating point."""
    order = np.argsort(scores, kind="mergesort")
    ys = y[order]
    # correct(k) = (# negatives in first k) + (# positives in the rest)
    neg_prefix = np.concatenate([[0], np.cumsum(1 - ys)])
    pos_suffix = np.concatenate([np.cumsum(ys[::-1])[::-1], [0]])
    return float((neg_prefix + pos_suffix).max() / len(y))


def clean_sources(X, y, sources, groups, *, min_per_class: int = 200,
                  rebalance: bool = True, target_rate: float = 0.5, seed: int = 5,
                  verbose: bool = True):
    """Drop degenerate sources and even out per-source class balance.

    Two problems the corpus builder cannot prevent, both seen in the real run:

    * `hu_berlin_toxicity` came out with **3,828 rows and zero positives**. Its
      toxic half duplicates Civil Comments rows already taken, so global
      de-duplication removed them. A single-class source makes `ClassStats`
      average an empty slice, gives its leave-one-source-out fold no meaning,
      and injects a source-shaped cluster of negatives into pooled training.
    * `_balance` targets 50/50 but can only work with what survives the scan and
      the de-duplicator, so real rates ranged 0.087 (toxic_chat) to 0.781
      (offensivelang). Leaving that alone means leave-one-source-out differences
      partly measure each corpus's base rate rather than transfer, which is the
      one thing the design is supposed to isolate.

    Returns filtered (X, y, sources) plus a report of what was dropped.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(np.int64)
    sources = np.asarray(sources)
    rng = np.random.default_rng(seed)

    keep_mask = np.zeros(len(y), dtype=bool)
    report = {"dropped": [], "rebalanced": [], "kept": []}

    for name in sorted(set(sources.tolist())):
        idx = np.flatnonzero(sources == name)
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        if len(pos) < min_per_class or len(neg) < min_per_class:
            report["dropped"].append({"source": name, "n": int(len(idx)),
                                      "pos": int(len(pos)), "neg": int(len(neg)),
                                      "reason": f"fewer than {min_per_class} in a class"})
            continue
        if rebalance:
            # Largest subset of this source at `target_rate`, keeping as much as
            # the minority class allows.
            n = min(int(len(pos) / target_rate), int(len(neg) / (1 - target_rate)))
            take_pos = int(round(n * target_rate))
            chosen = np.concatenate([rng.permutation(pos)[:take_pos],
                                     rng.permutation(neg)[:n - take_pos]])
            if len(chosen) < len(idx):
                report["rebalanced"].append(
                    {"source": name, "from": int(len(idx)), "to": int(len(chosen)),
                     "rate_before": round(float(len(pos) / len(idx)), 4)})
        else:
            chosen = idx
        keep_mask[chosen] = True
        report["kept"].append({"source": name, "n": int(keep_mask[idx].sum())})

    if verbose:
        for d in report["dropped"]:
            print(f"  DROPPED {d['source']:<22} n={d['n']:>6} pos={d['pos']:>6} "
                  f"neg={d['neg']:>6}  ({d['reason']})")
        for r in report["rebalanced"]:
            print(f"  rebalanced {r['source']:<19} {r['from']:>6} -> {r['to']:>6} "
                  f"rows (was {r['rate_before']:.3f} positive)")
        remaining = sorted({groups.get(s, s) for s in sources[keep_mask].tolist()})
        print(f"  kept {int(keep_mask.sum())}/{len(y)} rows, "
              f"{len(set(sources[keep_mask].tolist()))} sources, "
              f"{len(remaining)} families: {', '.join(remaining)}")

    return X[keep_mask], y[keep_mask], sources[keep_mask], report


def _stratified_cap(idx: np.ndarray, y: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """Deterministically keep `cap` indices with the class ratio preserved."""
    rng = np.random.default_rng(seed)
    out = []
    for label in (0, 1):
        pool = idx[y == label]
        share = int(round(cap * len(pool) / len(idx)))
        pool = pool.copy()
        rng.shuffle(pool)
        out.append(pool[:max(1, share)])
    kept = np.concatenate(out)
    rng.shuffle(kept)
    return kept


def resample_to_prior(y: np.ndarray, positive_rate: float, seed: int = 0
                      ) -> np.ndarray:
    """Indices of the largest subset achieving `positive_rate`, deterministically."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    # Largest n with round(n*rate) <= len(pos) and n - round(n*rate) <= len(neg).
    by_pos = int(len(pos) / positive_rate) if positive_rate > 0 else 0
    by_neg = int(len(neg) / (1.0 - positive_rate)) if positive_rate < 1 else 0
    n = max(1, min(by_pos, by_neg))
    take_pos = int(round(n * positive_rate))
    take_neg = n - take_pos
    idx = np.concatenate([pos[:take_pos], neg[:take_neg]])
    rng.shuffle(idx)
    return idx


# --------------------------------------------------------------------------
# Feature transforms.  `fit` sees training rows only; `apply` may additionally
# use the evaluation batch's own statistics, which is inference-time
# normalisation, not training -- no labels are involved.
# --------------------------------------------------------------------------


@dataclass
class Transform:
    name: str
    row_l2: bool = False
    standardize: str = "train"  # "train" | "batch" | "none"
    # "All but the top": drop the leading principal components. In a mean-pooled
    # residual stream the first few directions carry sequence length, register
    # and topic -- large, dominant, and mostly nuisance. They are also the
    # directions most likely to differ between corpora, so removing them is a
    # domain-shift defence as much as a denoising step.
    remove_top: int = 0

    def fit(self, X: np.ndarray) -> dict:
        Z = _row_l2(X) if self.row_l2 else np.asarray(X, dtype=np.float64)
        if self.standardize == "train":
            center = Z.mean(axis=0, dtype=np.float64)
            scale = np.maximum(Z.std(axis=0, dtype=np.float64), 1e-6)
        else:
            center = np.zeros(Z.shape[1])
            scale = np.ones(Z.shape[1])
        components = None
        if self.remove_top and self.standardize != "batch":
            components = _top_components((Z - center) / scale, self.remove_top)
        return {"center": center, "scale": scale, "components": components}

    def apply(self, state: dict, X: np.ndarray, dtype=np.float64) -> np.ndarray:
        """Apply the fitted transform.

        `dtype` exists purely for memory. At d=2304 a float64 copy of a 19k-row
        held-out fold is 326 MB and that is what runs an 8 GiB machine out of
        room. float32 halves it and is far finer than the signal, which came
        from float16 embeddings. The default stays float64 so the parity test
        against the submitted classifier -- which works in float64 -- compares
        like with like.
        """
        Z = _row_l2(X).astype(dtype, copy=False) if self.row_l2 else np.asarray(X, dtype=dtype)
        if self.standardize == "batch":
            center = Z.mean(axis=0, dtype=np.float64)
            scale = np.maximum(Z.std(axis=0, dtype=np.float64), 1e-6)
        elif self.standardize == "train":
            center, scale = state["center"], state["scale"]
        else:
            center, scale = 0.0, 1.0
        Z = ((Z - center) / scale).astype(dtype, copy=False)
        if self.remove_top:
            components = (_top_components(Z, self.remove_top)
                          if self.standardize == "batch" else state["components"])
            if components is not None:
                Z = Z - (Z @ components.T) @ components
        return Z


def _row_l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def _top_components(Z: np.ndarray, k: int) -> np.ndarray:
    """Leading `k` right singular vectors of an already-centred matrix, [k, d]."""
    k = min(k, min(Z.shape) - 1)
    if k <= 0:
        return None
    # Eigendecomposition of the d x d scatter is far cheaper than an SVD of the
    # full matrix when n >> d, and only the top few vectors are wanted.
    gram = (Z.T @ Z) / max(len(Z), 1)
    values, vectors = np.linalg.eigh(gram)
    return np.ascontiguousarray(vectors[:, -k:].T[::-1])


TRANSFORMS = (
    Transform("raw", row_l2=False, standardize="none"),
    Transform("std", row_l2=False, standardize="train"),
    Transform("l2", row_l2=True, standardize="none"),
    Transform("l2+std", row_l2=True, standardize="train"),
    Transform("batchstd", row_l2=False, standardize="batch"),
    Transform("l2+batchstd", row_l2=True, standardize="batch"),
    Transform("std+abtt1", row_l2=False, standardize="train", remove_top=1),
    Transform("std+abtt4", row_l2=False, standardize="train", remove_top=4),
    # `batchstd+abtt*` is deliberately absent. It recomputed the top components
    # separately on the training batch and on the evaluation batch, so the
    # coefficients were fitted in one residual subspace and applied in another,
    # and at inference in a third. That is a basis mismatch dressed up as domain
    # adaptation. Fixed components (std+abtt*) keep train and test in the same
    # subspace and stay in the grid, though they measured worse than plain `std`
    # on the rehearsal too -- the grid keeps them so the Gemma data can decide.
)


# --------------------------------------------------------------------------
# Probes.  Each returns a callable scoring function plus a serialisable weight
# vector, because the submission must be a plain NumPy dot product.
# --------------------------------------------------------------------------


class ClassStats:
    """Pooled within-class statistics, computed once and reused.

    At d=2304 the within-class covariance costs ~n*d^2 flops, which dominates
    everything else in the harness. LDA at five shrinkage values needs the same
    covariance five times, and every transform/fold pair needs it once, so
    building it eagerly and solving repeatedly turns hours into minutes.
    """

    __slots__ = ("mu1", "mu0", "delta", "cov", "trace_term", "_eye")

    def __init__(self, Z: np.ndarray, y: np.ndarray):
        Z1, Z0 = Z[y == 1], Z[y == 0]
        self.mu1 = Z1.mean(axis=0, dtype=np.float64)
        self.mu0 = Z0.mean(axis=0, dtype=np.float64)
        self.delta = self.mu1 - self.mu0
        d = Z.shape[1]
        # Pooled within-class scatter: centre each class on its own mean so the
        # between-class direction does not leak into the covariance.
        #
        # Accumulated in row blocks rather than centring the whole class at once.
        # At d=2304 a single float64 copy of a 40k-row fold is 737 MB, and on an
        # 8 GiB machine that is what actually fails -- this raised
        # ArrayMemoryError on a 176 MiB request because the peak had already been
        # spent. Blocking caps the temporary at `chunk x d` regardless of fold
        # size, and the arithmetic is identical.
        cov = np.zeros((d, d), dtype=np.float64)
        total = 0
        chunk = max(1, min(4096, int(3e7 // max(d, 1))))
        for block, mu in ((Z1, self.mu1), (Z0, self.mu0)):
            if len(block) == 0:
                continue
            for start in range(0, len(block), chunk):
                centred = block[start:start + chunk].astype(np.float64, copy=False) - mu
                cov += centred.T @ centred
                del centred
            total += len(block)
        self.cov = cov / max(total, 1)
        self.trace_term = float(np.trace(self.cov) / d)
        self._eye = None

    def eye(self) -> np.ndarray:
        if self._eye is None:
            self._eye = np.eye(self.cov.shape[0])
        return self._eye


def fit_logistic(Z: np.ndarray, y: np.ndarray, C: float, stats: ClassStats | None = None
                 ) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", tol=1e-4)
    model.fit(Z, y)
    return model.coef_.ravel().astype(np.float64), float(model.intercept_[0])


def fit_lda_shrinkage(Z: np.ndarray, y: np.ndarray, shrinkage: float,
                      stats: ClassStats | None = None) -> tuple[np.ndarray, float]:
    """Shrinkage LDA, written out so the covariance can be shared.

    Equivalent to `LinearDiscriminantAnalysis(solver='lsqr', shrinkage=s)`, but
    sklearn rebuilds the covariance for every value of `s`.
    """
    stats = stats or ClassStats(Z, y)
    reg = (1 - shrinkage) * stats.cov + shrinkage * stats.trace_term * stats.eye()
    w = np.linalg.solve(reg, stats.delta)
    b = -float(w @ (stats.mu1 + stats.mu0) / 2.0)
    return w, b


def fit_diff_of_means(Z: np.ndarray, y: np.ndarray, _: float = 0.0,
                      stats: ClassStats | None = None) -> tuple[np.ndarray, float]:
    stats = stats or ClassStats(Z, y)
    w = stats.delta
    b = -float(w @ (stats.mu1 + stats.mu0) / 2.0)
    return w, b


# `needs_stats` marks the probes that can reuse the shared covariance.
PROBES: dict[str, tuple[Callable, tuple, bool]] = {
    "lda": (fit_lda_shrinkage, (0.02, 0.1, 0.3, 0.6, 0.9), True),
    "dom": (fit_diff_of_means, (0.0,), True),
    "logistic": (fit_logistic, (0.002, 0.05), False),
}


# --------------------------------------------------------------------------
# Leave-one-source-out evaluation
# --------------------------------------------------------------------------


@dataclass
class Recipe:
    transform: str
    probe: str
    hyper: float

    @property
    def key(self) -> str:
        return f"{self.transform}|{self.probe}|{self.hyper:g}"


def evaluate_recipes(
    X: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    groups: dict[str, str],
    *,
    positive_rate: float = TARGET_POSITIVE_RATE,
    recipes: list[Recipe] | None = None,
    max_train_rows: int = 40_000,
    verbose: bool = True,
) -> dict:
    """Train on every source outside a group, score the sources inside it.

    `groups` maps source name -> family.  Holding out a *family* rather than a
    single source keeps two samples of the same underlying corpus (Civil
    Comments and its mteb re-release, say) on the same side of the split.

    `max_train_rows` caps the training fold during *selection only*, so the
    grid finishes in minutes rather than hours; the chosen recipe is refitted on
    every row in `finalize.build_candidates`.  The cap is applied with class
    balance preserved and deterministically, so two runs compare like with like.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    sources = np.asarray(sources)
    families = np.array([groups.get(s, s) for s in sources])
    held_out = sorted(set(families.tolist()))

    if recipes is None:
        recipes = [
            Recipe(t.name, p, h)
            for t in TRANSFORMS
            for p, (_, grid, _needs) in PROBES.items()
            for h in grid
        ]

    per_recipe: dict[str, dict[str, dict]] = {r.key: {} for r in recipes}
    wanted = {(r.transform, r.probe, r.hyper) for r in recipes}

    for family in held_out:
        test_mask = families == family
        train_mask = ~test_mask
        if test_mask.sum() < 50 or train_mask.sum() < 200:
            continue

        train_idx = np.flatnonzero(train_mask)
        if len(train_idx) > max_train_rows:
            train_idx = _stratified_cap(train_idx, y[train_idx], max_train_rows, seed=23)

        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_mask], y[test_mask]
        keep = resample_to_prior(yte, positive_rate, seed=17)
        Xte, yte = Xte[keep], yte[keep]

        if verbose:
            print(f"\n  hold out {family:<18} train={len(ytr):>6} "
                  f"test={len(yte):>5} (pos {yte.mean():.3f})", flush=True)

        for transform in TRANSFORMS:
            if not any(r.transform == transform.name for r in recipes):
                continue
            state = transform.fit(Xtr)
            Ztr = transform.apply(state, Xtr, dtype=np.float32)
            Zte = transform.apply(state, Xte, dtype=np.float32)

            stats: ClassStats | None = None
            if any(needs and (transform.name, name, h) in wanted
                   for name, (_fn, grid, needs) in PROBES.items() for h in grid):
                stats = ClassStats(Ztr, ytr)

            for probe_name, (fit_fn, grid, needs) in PROBES.items():
                for hyper in grid:
                    recipe = Recipe(transform.name, probe_name, hyper)
                    if recipe.key not in per_recipe:
                        continue
                    try:
                        w, b = fit_fn(Ztr, ytr, hyper, stats if needs else None)
                    except Exception as exc:
                        per_recipe[recipe.key][family] = {"error": f"{type(exc).__name__}"}
                        continue
                    scores = Zte @ w + b
                    per_recipe[recipe.key][family] = {
                        "accuracy": quota_accuracy(scores, yte, positive_rate),
                        "auroc": auroc(scores, yte),
                        "oracle": best_threshold_accuracy(scores, yte),
                    }
            del Ztr, Zte, stats
            if verbose:
                best = max(
                    (per_recipe[r.key].get(family, {}).get("accuracy", 0.0)
                     for r in recipes if r.transform == transform.name),
                    default=0.0,
                )
                print(f"    {transform.name:<12} best quota-accuracy {best:.4f}", flush=True)

    summary = []
    for recipe in recipes:
        rows = [v for v in per_recipe[recipe.key].values() if "accuracy" in v]
        if not rows:
            continue
        accs = np.array([r["accuracy"] for r in rows])
        aucs = np.array([r["auroc"] for r in rows])
        summary.append({
            **asdict(recipe),
            "key": recipe.key,
            "mean_accuracy": float(accs.mean()),
            "worst_accuracy": float(accs.min()),
            "std_accuracy": float(accs.std()),
            "mean_auroc": float(np.nanmean(aucs)),
            "worst_auroc": float(np.nanmin(aucs)),
            "n_folds": len(rows),
            "per_family": {k: v for k, v in per_recipe[recipe.key].items()},
        })

    # Ranking key. This was `worst_accuracy` and that was a mistake worth
    # measuring: across 80 recipes x 5 noisy folds, the minimum is a
    # min-of-noise, and near the compressed region around the all-positive line
    # it selects for the *least informative* recipe. On the gpt2 rehearsal it
    # picked std+abtt4|lda|0.6 over std|lda|0.6 by 0.0033 -- well inside the
    # ~0.011 fold standard error -- and paid 0.050 of mean AUROC and 0.035 of
    # mean accuracy for it, dropping from 3 of 5 folds above the all-positive
    # line to 1 of 5.
    #
    # Mean AUROC over folds is prior-free, far less noisy than a min, and does
    # not reward a recipe for being uninformative. A recipe that errored on any
    # fold sorts last regardless: its `worst_accuracy` was never computed, so
    # ranking it against complete recipes compares different quantities.
    complete = max((r["n_folds"] for r in summary), default=0)
    summary.sort(key=lambda r: (r["n_folds"] >= complete, r["mean_auroc"],
                                r["worst_auroc"], r["mean_accuracy"]), reverse=True)
    return {"positive_rate": positive_rate, "families": held_out,
            "expected_folds": complete, "results": summary}


def evaluate_cross_source(
    X: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    groups: dict[str, str],
    *,
    positive_rate: float = TARGET_POSITIVE_RATE,
    recipes: list[Recipe] | None = None,
    max_train_rows: int = 40_000,
    verbose: bool = True,
) -> dict:
    """Score on folds whose positives and negatives come from different corpora.

    This tests a *hypothesis*, not a measurement: the organizer reached 94.9%
    with a linear probe trained on only ~5,440 rows, which is more than a
    semantically hard boundary usually gives up. The likeliest explanation is
    that their toxic and safe halves are drawn from visibly different
    distributions, so part of what their probe separates is provenance.

    If that is right, the recipe that transfers is the one that still ranks
    correctly when the held-out positives and negatives come from two corpora it
    has never seen -- which is what this measures. Families are paired
    cyclically rather than exhaustively, so this costs k fits, the same as the
    ordinary leave-one-source-out pass, instead of k^2.

    Read it as a tiebreaker next to `evaluate_recipes`, never instead of it.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    sources = np.asarray(sources)
    families = np.array([groups.get(s, s) for s in sources])
    names = sorted(set(families.tolist()))
    if len(names) < 3:
        return {"positive_rate": positive_rate, "pairs": [], "results": []}

    if recipes is None:
        recipes = [
            Recipe(t.name, p, h)
            for t in TRANSFORMS
            for p, (_, grid, _needs) in PROBES.items()
            for h in grid
        ]
    per_recipe: dict[str, dict[str, dict]] = {r.key: {} for r in recipes}
    pairs = [(names[i], names[(i + 1) % len(names)]) for i in range(len(names))]

    for positive_family, negative_family in pairs:
        held = (families == positive_family) | (families == negative_family)
        # Positives from one unseen corpus, negatives from another unseen one.
        take_pos = np.flatnonzero(held & (families == positive_family) & (y == 1))
        take_neg = np.flatnonzero(held & (families == negative_family) & (y == 0))
        if len(take_pos) < 40 or len(take_neg) < 20:
            continue
        n = min(int(len(take_pos) / positive_rate), int(len(take_neg) / (1 - positive_rate)))
        rng = np.random.default_rng(31)
        rng.shuffle(take_pos)
        rng.shuffle(take_neg)
        want_pos = int(round(n * positive_rate))
        test_idx = np.concatenate([take_pos[:want_pos], take_neg[:n - want_pos]])
        yte = y[test_idx]
        Xte = X[test_idx]

        train_idx = np.flatnonzero(~held)
        if len(train_idx) > max_train_rows:
            train_idx = _stratified_cap(train_idx, y[train_idx], max_train_rows, seed=23)
        Xtr, ytr = X[train_idx], y[train_idx]

        label = f"{positive_family}+/{negative_family}-"
        if verbose:
            print(f"\n  {label:<34} train={len(ytr):>6} test={len(yte):>5} "
                  f"(pos {yte.mean():.3f})", flush=True)

        for transform in TRANSFORMS:
            if not any(r.transform == transform.name for r in recipes):
                continue
            state = transform.fit(Xtr)
            Ztr = transform.apply(state, Xtr, dtype=np.float32)
            Zte = transform.apply(state, Xte, dtype=np.float32)
            stats = ClassStats(Ztr, ytr)
            for probe_name, (fit_fn, grid, needs) in PROBES.items():
                for hyper in grid:
                    recipe = Recipe(transform.name, probe_name, hyper)
                    if recipe.key not in per_recipe:
                        continue
                    try:
                        w, b = fit_fn(Ztr, ytr, hyper, stats if needs else None)
                    except Exception as exc:
                        per_recipe[recipe.key][label] = {"error": type(exc).__name__}
                        continue
                    scores = Zte @ w + b
                    per_recipe[recipe.key][label] = {
                        "accuracy": quota_accuracy(scores, yte, positive_rate),
                        "auroc": auroc(scores, yte),
                    }
            del Ztr, Zte, stats

    summary = []
    for recipe in recipes:
        rows = [v for v in per_recipe[recipe.key].values() if "accuracy" in v]
        if not rows:
            continue
        accs = np.array([r["accuracy"] for r in rows])
        summary.append({
            **asdict(recipe),
            "key": recipe.key,
            "mean_accuracy": float(accs.mean()),
            "worst_accuracy": float(accs.min()),
            "n_folds": len(rows),
            "per_pair": dict(per_recipe[recipe.key]),
        })
    summary.sort(key=lambda r: (r["worst_accuracy"], r["mean_accuracy"]), reverse=True)
    return {"positive_rate": positive_rate,
            "pairs": [f"{a}+/{b}-" for a, b in pairs], "results": summary}


def report(summary: dict, top: int = 15) -> str:
    rate = summary["positive_rate"]
    lines = [
        f"Leave-one-source-out, quota at positive_rate={rate:.5f}",
        f"families: {', '.join(summary['families'])}",
        "",
        f"baselines:  all-positive {ALL_POSITIVE_BASELINE:.4f}   "
        f"random-ranking quota {RANDOM_QUOTA_BASELINE:.4f}",
        "a recipe below the all-positive line is worse than submitting all ones",
        "",
        f"{'recipe':<34}{'worst':>8}{'mean':>8}{'std':>8}{'mAUC':>8}{'wAUC':>8}  vs base",
    ]
    for row in summary["results"][:top]:
        delta = row["worst_accuracy"] - ALL_POSITIVE_BASELINE
        flag = f"{delta:+.4f}" + ("" if delta >= 0 else " BELOW")
        lines.append(
            f"{row['key']:<34}{row['worst_accuracy']:>8.4f}{row['mean_accuracy']:>8.4f}"
            f"{row['std_accuracy']:>8.4f}{row['mean_auroc']:>8.4f}{row['worst_auroc']:>8.4f}"
            f"  {flag}"
        )
    beating = sum(1 for r in summary["results"]
                  if r["worst_accuracy"] >= ALL_POSITIVE_BASELINE)
    lines += ["",
              f"{beating}/{len(summary['results'])} recipes clear the all-positive "
              f"baseline on their WORST fold"]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Leave-one-source-out probe selection")
    parser.add_argument("bundle", help=".npz with X, y, sources (and optional families)")
    parser.add_argument("--out", default="loso_results.json")
    parser.add_argument("--positive-rate", type=float, default=TARGET_POSITIVE_RATE)
    args = parser.parse_args()

    data = np.load(args.bundle, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    sources = data["sources"]
    if "families" in data:
        groups = dict(zip(sources.tolist(), data["families"].tolist()))
    else:
        import sys
        sys.path.insert(0, "src")
        from probe.corpus import SOURCES_BY_NAME

        groups = {n: s.group for n, s in SOURCES_BY_NAME.items()}

    summary = evaluate_recipes(X, y, sources, groups, positive_rate=args.positive_rate)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    print()
    print(report(summary))
    print(f"\nwrote {args.out}")
