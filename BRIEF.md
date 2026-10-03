# Engineering brief: maximise accuracy on CodaBench 17670

For the engineer or agent taking this over. Written 2026-09-08 22:00 UTC.

**Read [HANDOVER.md](HANDOVER.md) §0 first** — it names the one file to ship and
the deadline that forfeits everything if missed. This brief is what to do *with*
the remaining time, not a replacement for it.

---

## 1. Mission and the clock

Produce the highest possible accuracy. Two deadlines, both hard:

| | UTC | remaining at time of writing |
|---|---|---|
| Development phase ends | 2026-09-09 22:00 | **24 h** |
| **Testing phase (the one that counts)** | 2026-09-10 00:00 → **2026-09-12 00:00** | 74 h |

**Submission budget: 3.** Spend them like they cost money, because information is
the scarce resource here, not compute.

| | accuracy | correct/1700 | implied held-out AUROC |
|---|---:|---:|---:|
| thylinao (leader) | **0.9412** | 1600 | ≈0.983 |
| **us** | 0.8976 | 1526 | ≈0.953 |
| amanpawar | 0.8859 | 1506 | ≈0.945 |
| organizer's own linear probe | 0.949 | — | ≈0.986 |
| all-positive baseline | 0.7059 | 1200 | 0.715 |

---

## 2. The one inference that should drive everything

**thylinao is 0.008 from the organizer's own reference. That is in-domain
quality on frozen features.**

The representation is fixed — everyone gets identical 2304-dim vectors. On frozen
features, every classifier-side knob we have measured is worth ±0.03 AUROC and
most are negative (§4). Nothing in transform, probe, regularisation or ensemble
space moves 0.03 AUROC, let alone closes a gap to 0.983.

A jump from 0.8624 to 0.9412 in one day is not what better regularisation looks
like. **It is what training on the right corpus looks like.**

So the ranking of effort is not close:

1. **Find or approximate the organizer's corpus** — the only thing that plausibly
   reaches 0.94.
2. **Adapt to the test batch at inference** — legal forms only, §6.
3. **Improve label quality** in the corpus we already have.
4. Everything else is noise-chasing.

---

## 3. Non-negotiables

1. **Ship `artifacts/kaggle/select/submissions/best_single.zip` in the Testing
   Phase** unless something has measurably beaten it.
   `sha256 0bd11ac11c0a09a8b38d834e06c558f0e17236dae680c215a5e33af8c746a601`
2. **`Force_Last`**: the board ranks your *most recent* submission, not your
   best. Always end on the best-scoring zip. This already cost us 1st place once.
3. **Never submit a zip that has not passed `smoke_test`** and cleared the 0.7059
   all-positive baseline. `finalize` refuses to build such candidates; do not
   bypass it.
4. **A 1700-row set has ±1.1% standard error (~±19 rows).** Differences under
   ~2.5 points are not evidence.

---

## 4. Known dead ends — do not spend a submission re-testing these

All measured, most on the real leaderboard:

| approach | result |
|---|---|
| rank ensembling | **−10 rows** on the real board (0.8918 vs 0.8976) |
| `raw` instead of `std` transform | **−36 rows** on the real board (0.8765) |
| ABTT / top-PC removal | −0.049 AUROC; 32 ABTT recipes 0.6859 vs 48 non-ABTT 0.7181 |
| nonlinear head (MLP) | −0.022 AUROC, worse on 4/5 folds |
| CORAL covariance alignment | −0.032 AUROC, worse on 5/5 folds |
| quota micro-tuning (±20 rows) | inside the noise floor; does not transfer to the final phase |

**And the meta-lesson: local selection does not predict the leaderboard.**
Leave-one-source-out under-predicted by ~0.10 (forecast 0.7934, real 0.8976).
A "fixed" variant that averaged only leaderboard-like folds got the ranking
*backwards* — preferred `raw|lda|0.6` by 0.010, real result went the other way by
0.021. Three points, all anti-correlated.

**Corollary: a local number may justify *building* a candidate. Only a different
hypothesis justifies *submitting* one.**

---

## 5. Strategies, ranked by expected value

### S1 — Identify the organizer's corpus (highest value, highest variance)

The leader almost certainly did this. Everything else is a consolation prize.

What has already been tried and failed: a size-and-balance fingerprint over
~1,000 HF datasets (`fingerprint_corpus.py`). It found **0 hits on the
mental-health sibling**, where the corpus is provably findable — so the method
is not the one that works. Do not simply re-run it wider without changing it.

Better angles, in order:

- **Solve the sibling first.** Competition 17669 (mental health, layer 23) has a
  leader at 0.9116 against an organizer reference of 0.924. Somebody found *that*
  corpus. It is an easier search space — far fewer mental-health datasets than
  toxicity ones. If you identify it, you learn the organizer's **selection
  habit** (top HF search result? a popular Kaggle set? a specific paper's
  release?), and that habit points straight at the toxicity corpus.
  Its fingerprint: dev 1697 rows, 849/848, so n ∈ [8481, 8485] near-balanced.
- **Research the organizer.** "TRI AI" created both competitions on 2026-07-25
  with copy-pasted task descriptions — the toxicity page still says
  "1: mental health distress signal". Use browse automation to look for: a TRI AI
  site, GitHub org, blog post, tutorial, Colab, or paper that the challenge was
  built from. The starting kit mentions a `latent_probe_pipeline.py` we have
  never seen — **find it.** It may name the dataset outright.
- **Search by content, not size.** Toxicity corpora with ~70.6% positive rate are
  unusual (most are majority-safe). Enumerate HF datasets by
  `task_categories:text-classification` + toxicity tags and filter on class
  balance rather than row count. The row-count assumption rests on an unverified
  "dev is 20% of the corpus" premise (COMPETITION_GOALS.md G0b flags this).
- **Ask.** The competition has a forum. Asking the organizer which corpus is used
  is legitimate and costs nothing.

**Validation before spending a submission:** if you find a candidate corpus,
extract it, train, and check that a probe trained *only* on it separates our
existing 9 held-out families unusually well. A corpus that is genuinely the
source should look different from the 13 we have.

### S2 — Inference-time adaptation to the test batch (high value, rules-sensitive)

`predict()` receives **all 1700 rows at once**. That is a large, unlabelled,
perfectly in-domain sample of exactly the distribution we are failing to match.
Exploiting it is the most powerful lever that does not require finding the
corpus.

**Read §6 before implementing any of this.** The rule is "no training on the
platform", and some of these cross that line.

Clearly safe (no parameters fitted, no labels used):
- batch standardisation — already in `Transform(standardize="batch")`.
- **Pre-fitted head selection.** Ship *n* heads in the joblib, each trained
  offline on a different corpus, plus each corpus's feature mean/covariance
  summary. At inference, compute an unsupervised distance (e.g. Fréchet /
  Bhattacharyya on the batch's mean and covariance) between the test batch and
  each training corpus, and **select the head whose training corpus is closest**.
  Nothing is fitted; it is a lookup over pre-computed models. **This is the
  single best idea in this brief that is unambiguously legal**, and it directly
  attacks the "which corpus" problem the leader appears to have solved by other
  means. The `source_*` heads already exist.
- **Prototype refinement without gradients** (T3A-style): read the literature
  before assuming this is safe — it updates class prototypes from test data,
  which is arguably fitting.

Ambiguous, needs the owner's explicit decision:
- self-training / pseudo-labelling on the test batch, then refitting. Powerful,
  and in my reading it *is* training on the platform. Do not ship it silently.

### S3 — Label quality in the corpus we have (moderate, safe, cheap)

Our 149k rows come from 13 corpora with heterogeneous, hand-chosen thresholds.
Civil Comments uses `toxicity >= 0.5`, which at exactly 0.5 is 2-of-4 annotators
— coin flips entering as labels, and a mislabelled row near the boundary rotates
the decision direction rather than merely failing to inform it.

- Tighten to `>= 0.7` / `<= 0.02`.
- **Relabel the whole corpus with a strong external toxicity classifier**
  (Detoxify / `unitary/toxic-bert`) and keep only rows where the external model
  and the corpus label agree. This is ordinary offline data cleaning, completely
  legal, and it is the most reliable of the safe ideas.
- Drop `hatecheck` (templated synthetic, worst training source on 4/5 folds) and
  `aegis` (~575-char mean, always truncated at the contract's 64 tokens).

### S4 — Corpus weighting (moderate, cheap)

Measured spread across single training sources was **0.618 → 0.924 AUROC** — ten
times the recipe lever. The `source_*.zip` candidates are built and unspent.
Submitting two of them answers "which corpus does the hidden set resemble", and
that answer transfers to the final phase.

Given only 3 submissions, this competes directly with S1/S2 for budget. Prefer it
only if S1 and S2 produce nothing.

---

## 6. The rules boundary — read before implementing S2

From the competition terms and the Participation page:

> "Trained models must be produced by you before submission; **no training is
> permitted on the competition platform itself.**"
> "Do not call `.fit()` on the platform."

My reading, which the owner should confirm:

| operation | verdict |
|---|---|
| batch mean/std normalisation | safe — label-free feature scaling, fits nothing that is kept |
| selecting among pre-fitted heads by unsupervised distance | safe — a lookup, no parameters created |
| a quota decision rule at a published class prior | safe as a fixed operating point |
| clustering the test batch and assigning labels to clusters | **not safe** — this is fitting a model on the platform |
| pseudo-labelling then refitting a classifier | **not safe** |
| encoding measurements of the input batch into output bits to read them back | **flatly prohibited** — a side channel; do not reinvent this under deadline pressure |

`submission.smoke_test` refuses any `classifier.py` containing `.fit(`. That
catches the letter, not the spirit — you are responsible for the spirit.

One open item the owner should decide: the `1200/1700` quota constant was derived
by reading `triai`'s all-zeros leaderboard score. It is one public aggregate, not
per-row labels, but it is the sharpest terms exposure in the project. Mitigation
costs ~1 point: use `best_threshold.zip`, which needs no such constant.

---

## 7. The loop: cheap gates before expensive ones

Never run an expensive step before the cheap one that could have rejected it.

```
GATE 0  desk check         minutes, free
        Does this change a *hypothesis* (training corpus, decision rule)
        or just a recipe? If it is only a recipe -> STOP. §4.

GATE 1  local sanity       minutes, free
        python -m pytest tests/ -q                       # 30 tests
        python iterate_local.py --max-train 14000        # bundle already local
        Reject anything below the 0.7059 all-positive baseline.
        Do NOT rank-order candidates on these numbers. §4.

GATE 2  small extraction   ~5 min GPU
        run_kaggle.py extract --cap-scale 0.15
        For a NEW corpus only: confirm the loader, labels and contract before
        paying for the full run. A wrong label mapping is invisible later.

GATE 3  full extraction    ~40 min GPU, free on Kaggle
        run_kaggle.py extract --cap-scale 1.0

GATE 4  critic review      §8. Must pass before any submission.

GATE 5  submit             1 of 3. Dry-run first (costs nothing):
        python codabench_submit.py submit <zip>            # dry run
        python codabench_submit.py submit <zip> --yes --label <name>
        Then ALWAYS re-check the board and end on the best zip.
```

The embedding bundle is already at
`artifacts/kaggle/extract/bundle_l14.npz` (638.8 MB), so Gates 0–1 need no GPU
and no network. Keep `--max-train ≤ 16000`: this machine has 8 GiB and three OOM
crashes came from exceeding it.

---

## 8. The critic agent

Spawn a critic before every submission. Give it **veto** authority over
candidates; it cannot approve an upload on its own.

Brief it with:

- the current board, our score, the all-positive baseline, and the AUROC
  calibration table (COMPETITION_GOALS.md);
- the dead-end table in §4;
- the specific candidate: what hypothesis it tests, what changed, what the local
  numbers are, and why the local numbers should not be trusted;
- the instruction to **argue the strongest case against submitting**, and to say
  VETO plainly if the candidate is not clearly better than 0.8976.

Ask it specifically to check:

1. Is this a new hypothesis or a recipe tweak in disguise?
2. Does the expected gain exceed ±19 rows of noise?
3. Does anything in it risk the terms in §6?
4. Arithmetic and code errors — it has found real ones before, including a
   ranking key that cost 0.051 AUROC and a baseline gate that refused the model
   which went on to score 0.8976.

The critic has already earned its place on this project. It found that **0 of 80
recipes cleared the all-positive baseline** on the rehearsal, which reframed the
whole effort. Do not skip it to save time.

---

## 9. Research directions worth browsing

Use the `browsepilot-research` skill (see the project CLAUDE.md — BrowsePilot
first, generic browser automation only as a fallback). Worthwhile targets:

- **The organizer.** TRI AI, the two competitions, the missing
  `latent_probe_pipeline.py`, any tutorial or repo the challenge was built from.
  This is the highest-value browsing by a wide margin.
- **Linear probing under distribution shift.** Deep Feature Reweighting; last-layer
  retraining; "probing classifiers" surveys; what actually helps when the encoder
  is frozen and only the head is learnable.
- **Training-free / gradient-free test-time adaptation.** T3A, prototype methods,
  BN-statistics adaptation. Read specifically to determine which of these fit
  *no* parameters, since that is the rules boundary in §6.
- **Toxicity dataset surveys.** A survey paper enumerating toxicity corpora is a
  better index than HF keyword search, and may name a corpus with ~70% positive
  rate and ~8.5k rows.
- **Mean-pooled sentence embeddings.** Known nuisance structure (length, register,
  frequency). Note ABTT measured *negative* here, so treat with suspicion.

---

## 10. Commands and state

```bash
python -m pytest tests/ -q                          # 30 tests
python codabench_submit.py status                   # board, deadlines, budget
PYTHONUTF8=1 python run_kaggle.py extract --wait    # ~40 min GPU
PYTHONUTF8=1 python run_kaggle.py select  --wait    # 30-90 min
PYTHONUTF8=1 python run_kaggle.py pull select --out artifacts/kaggle/select
python iterate_local.py --max-train 14000           # local, no GPU
```

`PYTHONUTF8=1` is required for anything reading Kaggle logs (progress-bar glyphs
crash cp1252 on Windows).

Ready-built, unspent candidates in
`artifacts/kaggle/select/submissions/`: `best_single` (0.8976, our best),
`best_threshold`, and eight `source_*` probes.

Credentials and Kaggle gotchas: HANDOVER.md §7. The P100/torch and Kaggle-secrets
traps are solved there; do not rediscover them.

---

## 11. Honest expectations

Beating 0.8976 is plausible. **Reaching 0.9412 without finding the corpus is
not**, on the evidence: it needs held-out AUROC ≈0.983, and every classifier-side
lever measured is worth ±0.03 with most negative.

So the realistic outcomes are:

- **S1 lands** → 0.93–0.95, competitive for first.
- **S2 (pre-fitted head selection) works** → 0.90–0.92 plausibly.
- **Neither** → we finish around 0.90 and second place.

Say which of these has happened rather than blurring it. A clear "the corpus is
not findable and 0.90 is the ceiling" is a more useful result than an optimistic
report, and it lets the owner decide how to spend the final submission.

**The failure mode that actually costs everything is not a low score — it is
missing the Testing Phase submission, or ending on a worse zip.**
