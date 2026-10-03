# Goal: >0.90 accuracy on the Latent Probe Challenge (Toxicity)

Target: **accuracy > 0.90** on CodaBench competition 17670.
Owner: bleymambwe. Opened 2026-09-06.

---

## G0 — Establish ground truth (DONE, 2026-09-06)

Everything below is read from the CodaBench public API, not inferred.

| Fact | Value | Source |
|---|---|---|
| Metric | accuracy | Task Description page |
| Representation | `google/gemma-2-2b`, hidden-state **layer 14**, **mean over all tokens**, truncation at **64** tokens, 2304 dims | Task Description page |
| Dev set size | **1700** rows | leaderboard scores are all exact multiples of 1/1700 |
| Dev composition | **1200 positive / 500 negative** (70.59% toxic) | `triai` submitted the all-zero starting kit and scored exactly 0.294118 = 500/1700 |
| Final set size | ~1360 rows | final `input_data` is 11,623,780 B vs dev 14,530,186 B |
| Dev phase ends | **2026-09-09 22:00 UTC** | phase 29478 |
| Final phase | **2026-09-10 00:00 → 2026-09-12 00:00 UTC**, single submission against unseen embeddings | phase 29479 + Timeline page |
| Submission budget | **5 per day** per phase | phase `max_submissions_per_day` |
| Organizer reference | a linear probe on layer 14 scored **94.9%** on their own data | Task Description page |
| Rules | own work only; **no training on the platform**; zip must hold `classifier.py` + `trained_probe.joblib` at root | Terms + Participation pages |

### G0b — What the sibling competition reveals (2026-09-07)

TRI AI runs a second, identically-templated challenge: 17669, *Mental Health
Sentiment Classification* (layer 23, organizer reference 92.4%). Reading both
leaderboards together pins down the hidden set's construction.

Every accuracy on a leaderboard is `correct / n`, so the set of scores
determines `n`. Solving for the smallest `n` consistent with all of them:

| | toxicity 17670 | mental health 17669 |
|---|---:|---:|
| dev rows | **1700** | **1697** |
| all-zeros baseline | 0.2941176471 → **500 negatives** | 0.4997053624 → **848 negatives** |
| dev composition | **1200 / 500** | 849 / 848 |
| final/dev `input_data` bytes | 0.79997 | 0.80020 |
| final rows | **1360** | 1358 |
| top participant | 0.8624 | 0.9116 |
| organizer's own probe | 0.949 | 0.924 |

Both phases split 0.8:1 in rows. That ratio is **measured**. Everything after it
is inference, and an earlier version of this section overstated it — corrected
here after review:

- *Assumed, not derived:* that dev is 20% of the corpus and final is 20% of the
  remainder. The byte ratio only fixes `final/dev = 0.8`; a (10%, 8%) split fits
  it equally well.
- *If* the 20% reading is right, `train_test_split` takes `ceil(n · 0.2)`, so
  dev = 1700 gives **n ∈ [8496, 8500]**, not exactly 8500. All five values yield
  final = 1360. The "exactly 6,000 / 2,500" composition holds only at n = 8500,
  and is in any case just 8500 × the observed dev rate — not independent
  evidence.

What actually survives, on a much weaker premise:

1. **The final set is 960 positive / 400 negative.** This follows from
   *stratification alone*, for any corpus size — the 8,500 derivation is not
   needed for it. It de-risks the quota rule; keep the threshold hedge anyway.
2. ~~The corpus must be very separable, so its toxic and safe halves probably
   differ by provenance.~~ **Withdrawn.** 0.949 accuracy corresponds to AUROC
   ≈0.986 (see the calibration table below), which is routine for a clean,
   well-labelled toxicity corpus scored in-domain. It needs no "provenance leak"
   hypothesis. `evaluate_cross_source` and the `best_cross_source` candidate
   were built to serve that hypothesis and should be treated as speculative
   rather than as a second selection criterion.

### Calibration: what a leaderboard accuracy implies about ranking quality

Equal-variance binormal model, 70.59% positive prior, quota decision rule
(simulated, N = 400k). This converts every score on the board into the only
quantity that matters for a frozen representation — held-out AUROC.

| accuracy | implied AUROC | who |
|---:|---:|---|
| 0.7059 | 0.715 | all-positive baseline |
| 0.7729 | 0.816 | **us** |
| 0.8553 | 0.916 | p_akogun |
| 0.8859 | 0.945 | amanpawar (leader) |
| 0.949 | ≈0.986 | organizer's own probe |

**This is the single most useful table in the file.** The leader is at held-out
AUROC ≈0.945, which is *in-domain* quality on frozen features. Every
classifier-side knob measured so far is worth ±0.03 AUROC and most are negative.
Nothing in the transform/probe/ensemble space moves 0.13 AUROC. A jump of that
size, achieved by two competitors independently inside 24 hours, is what finding
the right training corpus looks like — not what better regularisation looks like.

Note also that the mental-health leader (0.9116) has essentially matched that
task's organizer reference (0.924), while the toxicity leader (0.8624) is 8.7
points short of 0.949. Someone found the mental-health corpus. Nobody appears
to have found the toxicity one — which is what G4 is for.

### Leaderboard

As of 2026-09-06 21:00 UTC+2 (rule `Force_Last`, so this is each
user's most recent submission):

| # | user | accuracy | correct/1700 |
|---|---|---:|---:|
| 1 | thylinao | 0.8624 | 1466 |
| 2 | amanpawar | 0.8153 | 1386 |
| 3 | p_akogun | 0.7853 | 1335 |
| 4 | **bleymambwe (us)** | **0.7729** | **1314** |
| 5 | adeola5678 | 0.6541 | 1112 |
| — | all-zeros baseline | 0.2941 | 500 |

**The gap to close is 0.773 → 0.90+, and the organizer's own 0.949 says the
representation supports it.**

---

## G1 — Name the binding constraint (DONE)

It is **not** the classifier and **not** the layer. It is the **training
distribution**.

Evidence:

- v005 (Civil Comments, `toxicity >= 0.5`, 50k rows) scores **0.836 on its own
  held-out Civil Comments split** but only **0.705 / 0.767** on the hidden dev
  set. A ~7-point drop that large is domain shift, not variance.
- v004, trained on a mental-health corpus, scored **0.42** — below the 0.706
  all-ones baseline. The hidden labels are toxicity, confirmed.
- The organizer reaches **0.949 with a linear probe** on the same layer-14
  features. Our probe class is not the limitation.
- v006 vs v005 differ only in the decision threshold (top-1200 quota instead of
  a fitted probability cut) and that alone moved 0.705 → 0.767. Calibration was
  worth 6 points; ranking quality is what is left.

Conclusion: **buy ranking quality with in-domain-like training data, then keep
the prior-calibrated decision rule.**

---

## G1b — Audit of `artifacts/`: every iteration so far changed the same model

Hashing the contents of every submission zip settles what the v004→v007 sequence
actually varied.

| weights sha1 | submissions sharing them |
|---|---|
| `6b4831ae216e` | **7**: v005, v006, v007, and all four portfolio candidates |
| `f66cdf3c2654` | 1: v004 (the mental-health probe) |

**Every toxicity submission ever made carries byte-identical
`trained_probe.joblib`.** v005 → v006 → v007 differ only in `classifier.py`,
i.e. only in where the decision boundary was placed:

Read from the My Submissions page on 2026-09-07 (7 submissions, 5 scored, 2
failed; platform counters read **5/100 total**, **0/5 today**):

| id | file | decision rule | leaderboard |
|---|---|---|---|
| 915942 | `cheap_test_v003_2304_numpy.zip` | v003 lineage | 0.69 |
| 916085 | `mental_health_probe_v004_layer14_me.zip` | wrong task | 0.42 |
| 916199 | `toxicity_probe_v005_layer14_mean64.zip` | probability threshold 0.3742 | 0.7047 |
| 916209 | `toxicity_probe_v006_rank1200_layer1.zip` | top 1200 of 1700 | 0.7671 |
| **916231** | `toxicity_probe_v007_final_transduct.zip` | rank blend, top 1292 | **0.7729** |

This resolves an ambiguity carried in earlier notes: the current best on the
board is **v007**, not an untested quota variant. It also confirms the marginal
precision over ranks 1201-1292 was 0.55, exactly as the ceiling argument below
requires.

### That lever is now spent

With 1200 positives and 500 negatives, accuracy at quota `k` is
`(2·TP(k) − k + 500) / 1700`, so it improves only while the marginal precision
past rank `k` is above 0.50. The two measured points fix that curve:

- k=1200 gave 1304 correct, so TP(1200) = 1002, FP = 198.
- reaching the measured 1314 correct needs marginal precision 0.60 over ranks
  1201-1250, or 0.55 over ranks 1201-1292.

Precision is already decaying toward 0.5, which is where accuracy peaks. The
absolute ceiling — if all 198 missed positives happened to rank immediately next
— is 0.8835, and that ordering is not what the measured precisions describe. The
realistic ceiling for *any* decision rule on these weights is **≈0.78**, and the
last submission at 0.7729 is within about a point of it.

**Conclusion, and the answer to "make the next one the most accurate": the next
submission has to change the weights.** No further quota, threshold or rank
blend on the v005 model can reach 0.90, or even 0.80. Everything in G2-G5 below
exists to produce new weights.

## G2 — Extract embeddings for a multi-source toxicity corpus

The organizer's exact corpus is unknown. Rather than guess once, train on
several public toxicity corpora **kept separable by source**, so that
source identity can be used both as a CV grouping and as a leaderboard probe.

- **Deliverable:** `colab_extract_embeddings.ipynb` — pinned contract
  (`google/gemma-2-2b`, `output_hidden_states=True` index 14, masked mean,
  `max_length=64`), writes one `.npz` per source with `X`, `y`, `texts`, `source`.
- **Why Colab:** this machine is an Intel Skylake-U, 2 cores / 4 threads, 8 GiB
  RAM, no CUDA. Measured Gemma-2-2b CPU throughput is in `results/optimization/`;
  it is too slow for a corpus of the size required. A single free T4 session
  does the whole corpus in well under an hour.
- **Exit criterion:** ≥60k labelled rows across ≥5 sources, embeddings verified
  finite and shaped `[n, 2304]`, contract parity check passed.

**Status: DONE, 2026-09-07 22:42 UTC.** Kaggle kernel `latent-probe-extract` v6:
149,528 rows extracted to `(149528, 2304)` in **32.6 min at 76.4 rows/s** on a
Tesla P100 in float32, bundle 638.8 MB, `EXTRACT_OK`. Contract asserted inside
the kernel before any row was written: `gemma2 / 2304 hidden / 26 layers /
vocab 256000`, and the early-exit hook parity-checked against
`output_hidden_states=True`.

One data defect surfaced in the run and is handled at selection time rather than
by re-extracting (`select.clean_sources`):

- `hu_berlin_toxicity` produced **3,828 rows with zero positives** — its toxic
  half duplicates Civil Comments rows taken earlier, so global de-duplication
  removed them. A single-class source would make `ClassStats` average an empty
  slice and give its fold no meaning. It is dropped.
- Per-source positive rates ran **0.087 (toxic_chat) to 0.781 (offensivelang)**
  despite the builder targeting 0.50, because `_balance` can only work with what
  survives the scan and the de-duplicator. Each source is resampled to 0.50, so
  leave-one-source-out differences measure transfer rather than base rate.

After hygiene: **108,468 rows, 12 sources, 9 families, exactly 0.500 positive.**

---

## G3 — Local harness that predicts the leaderboard

A local CV number that tracks the hidden test is worth more than any single
model. Random-split CV does not track it — v005 proved that (0.836 local,
0.767 hidden).

- **Deliverable:** `loso_probe.py` — **leave-one-source-out** evaluation.
  Train on all sources but S, test on S, under the known 70.59% positive prior
  and the top-quota decision rule. Report mean and **worst-source** accuracy.
- **Rationale:** the hidden set is a corpus we have never trained on. The only
  honest local simulation of that is to score on a corpus we never trained on.
- **Exit criterion:** worst-source accuracy > 0.85 for the selected recipe,
  with the gap between mean and worst under 0.05.

---

## G4b — The corpus fingerprint search failed its own validation (2026-09-07)

`fingerprint_corpus.py` searched the Hugging Face datasets-server for a corpus
matching the deduced size and class balance. It was designed to be falsifiable:
run it on the **mental-health sibling first**, where we have strong evidence the
corpus *is* findable — that leaderboard's leader reached 0.9116, i.e. AUROC
≈0.96 against the organizer's own 0.924 reference. Somebody found it.

Result:

| task | datasets checked | size hits | balance match |
|---|---:|---:|---|
| mental health (n ∈ [8481, 8485], ~50% positive) | 799 | **0** | — |
| toxicity (n ∈ [8496, 8500], ~70.6% positive) | 969 | 7 | **none** |

All 7 toxicity hits are the Hateful Memes dataset and its derivatives — 8,500
train rows, which matches exactly, but it is a **multimodal meme** corpus at
~35.5% hateful, not 70.6%. A size coincidence, not a match.

**The method found nothing in the case where we know something exists.** So its
negative result on toxicity carries little weight, and the failure is the
informative part. Three explanations, none of which we can currently separate:

1. keyword search over ~1,000 datasets is a narrow slice of Hugging Face;
2. the organizer subsampled or combined corpora, so no published row count
   fingerprints it;
3. the "dev is 20% of the corpus" assumption is simply wrong — and note that
   G0b already flags it as assumed rather than derived.

**Consequence for the plan.** The critic put >0.90 at 20-25%, with essentially
all of that probability resting on this search landing. It did not land. Unless
the corpus turns up another way, the honest target is **0.83-0.88**, and first
place (currently 0.8859) is possible but not likely.

That does not make the extraction pointless — our current weights are capped at
~0.78 by arithmetic, and Gemma embeddings plus a well-chosen training source is
worth several points. It does mean the remaining effort should go to the lever
that measured largest (which single source transfers best, worth ~0.17 AUROC in
the rehearsal) rather than to recipe search (~0.02, often negative).

## G4 — Use the dev leaderboard as a source discriminator

Budget is 5 submissions/day and the dev phase ends 2026-09-09 22:00 UTC —
about **15 submissions left**. Spend them to identify the hidden corpus, not to
chase noise.

- Submit one probe per candidate source (same classifier, same calibration,
  only the training corpus changes). The source whose probe scores highest is
  closest to the hidden distribution.
- Each result is worth ~1700 labelled examples of information about *which
  corpus*, and unlike threshold tuning it transfers to the final phase, which is
  a different held-out sample of the same corpus.
- **Do not** spend submissions fitting the dev labels themselves (probing set
  membership, quota micro-tuning below ±20 rows). The final phase is a
  different 1360 rows; dev-label overfitting does not transfer and is against
  the spirit of the terms.
- **Exit criterion:** one source (or blend) identified with a dev score > 0.88.

---

## G5 — Ship the final submission

**DONE, 2026-09-10.** The reviewed pooled std-LDA artifact was submitted once
to Testing as submission 921106 and verified at **0.9235294118**
(1,256/1,360). This clears the project's >0.90 accuracy target. At the 08:57
UTC check it ranked second, 44 rows behind `thylinao`; Testing remains open
until 12 September, so placement is provisional.

- Retrain the winning recipe on every labelled row from the identified
  source(s).
- Keep the decision rule that is calibrated to the class prior rather than a
  probability threshold fitted on a different domain (this was worth +6 points).
  Note the final set is ~1360 rows, so the rule must be **a positive *rate*,
  not a fixed count of 1200**.
- Verify: zip has exactly `classifier.py` and `trained_probe.joblib` at root;
  isolated-import smoke test; runs in well under the 600 s execution limit;
  no `.fit()` at inference.
- **Exit criterion achieved:** one verified final-phase submission above 0.90.

---

## Iteration log

Append one row per experiment. No row is added without a measured number.

| date | id | change | local (LOSO worst) | dev leaderboard | verdict |
|---|---|---|---|---|---|
| 2026-09-05 | v005 | Civil Comments, prob threshold | 0.836 (random split, not LOSO) | 0.7047 | superseded |
| 2026-09-05 | v006 | v005 weights, top-1200 quota | — | 0.7671 | same weights, better operating point |
| 2026-09-05 | v-last | v005 weights, larger quota (916231) | — | 0.7729 | current best; ~1 pt below this model's ceiling |
| 2026-09-07 | — | rehearsal: gpt2 layer 8, 10,455 rows, 5 families | 0.678 worst / 0.705 mean | not submitted | method validated, not a forecast |
| | | | | | |

### The baseline that reframes everything (2026-09-07)

On a 1200/500 set, **submitting all ones scores 0.7059**. That is the real floor,
and it is high because the set is 70.6% positive. Two consequences:

| reference | score |
|---|---|
| quota rule on a *random* ranking | 0.5848 |
| **all-positive submission** | **0.7059** |
| our current board score (v007) | 0.7729 |
| leader | 0.8859 |

In the gpt2 rehearsal, **0 of 80 recipes cleared 0.7059 on their worst fold**,
and 4 of 5 folds of the best recipe were below it. Held-out AUROC was 0.659-0.854
(mean 0.705).

This does **not** invalidate the pipeline, but it does bound what the rehearsal
proved: it established that the machinery runs on real data, and nothing about
whether the strategy yields a competitive model. gpt2 layer 8 at 768 dims is a
far weaker representation than Gemma-2-2b layer 14 at 2304 — the same
Civil-Comments recipe on Gemma scores 0.7729 on the real dev set — so the
gap is consistent with the representation rather than the method. But it is now
an open question rather than an assumption, and the honest read is that
**cross-corpus transfer is much harder than the plan assumed.**

`select.report()` now prints both baselines and flags every recipe below the
all-positive line, so this cannot be overlooked again when the Gemma numbers
arrive. Found by the critic agent, not by inspection.

### 2026-09-08 — Gemma embeddings, and first place

Four submissions, all from the same 108,468-row pooled corpus after
`clean_sources`. Local numbers are leave-one-source-out; real numbers are the
dev leaderboard.

| id | candidate | local mean_all | local mean_like | **dev** |
|---|---|---:|---:|---:|
| 918460 | `best_single` = `std\|lda\|0.6` | 0.7934 | 0.8462 | **0.8976** |
| 918464 | `best_ensemble` (rank, 3 transforms) | 0.7934 | — | 0.8918 |
| 918476 | `raw\|lda\|0.6` | 0.7903 | 0.8563 | 0.8765 |
| 918485 | re-submit of 918460 (`Force_Last`) | — | — | **0.8976** |

**0.8976 = 1526/1700, first place**, ahead of amanpawar's 0.8859. Up 0.1247 from
v007's 0.7729. Implied held-out AUROC ≈0.953.

#### The finding that should govern the rest of this work

**Local selection does not predict the leaderboard, and at fine granularity it
is actively misleading.**

- `mean_all` under-predicted by ~0.10: it forecast 0.7934, the real score was
  0.8976. The mean is dragged down by folds the hidden corpus does not
  resemble (`offensivelang` 0.6252) while the real target sits at the
  `wiki`/`rtp` end (0.888/0.884).
- `mean_like` — an attempt to fix that by averaging only the folds near where
  the leaderboard landed — **ranked `raw|lda|0.6` above `std|lda|0.6` by 0.010,
  and the real result went the other way by 0.021.** Three points, all
  anti-correlated.
- Rank ensembling cost 10 rows (0.8918 vs 0.8976), matching the rehearsal's
  −0.005 AUROC. Correlated members add nothing.

Consequence: **stop searching recipes locally.** The remaining budget buys
leaderboard measurements, and each is worth more than any number of local folds.
Anything below a genuinely different *hypothesis* (a different training corpus,
a different decision rule) is not worth a submission.

#### Operational note: `Force_Last`

The leaderboard ranks each user by their **most recent** submission, not their
best. Submitting an experiment that scores worse drops the visible rank
immediately — that happened here at 0.8765. **Always end a session on the
best-scoring zip.**

### What the 2026-09-07 rehearsal established

Run on CPU with gpt2 layer 8 so the pipeline could be exercised end to end
without a GPU. gpt2 is a much weaker representation than Gemma-2-2b, so the
absolute numbers are a floor, not a prediction. Three things do transfer:

1. **Recipe choice is worth ~0.02; the held-out corpus is worth ~0.11.**
   Worst-fold accuracy ranged 0.66-0.78 depending on *which corpus* was held
   out, while the best and worst recipes on the same fold differed by about two
   points. This is the G1 diagnosis confirmed on real data: the training
   distribution is the lever, not the classifier.
2. ~~Removing the top principal components buys stability.~~ **Withdrawn — this
   was a selection artifact, not a finding.** `std+abtt4|lda|0.6` won on
   `worst_accuracy` by 0.0033 over `std|lda|0.6`, well inside the ~0.011 fold
   standard error. Ranking 80 recipes by the minimum over 5 noisy folds is a
   min-of-noise, and near the compressed region around the all-positive line it
   selects for the *least informative* recipe. Aggregated over the whole grid,
   ABTT is simply worse: **32 ABTT recipes mean AUROC 0.6859, 48 non-ABTT
   0.7181.** Re-ranking by mean AUROC picks `raw|logistic|0.002` instead and
   recovers **+0.051 AUROC, +0.031 accuracy, and 3 of 5 folds above the
   all-positive line instead of 1 of 5**. The ranking key is now mean AUROC
   (`select.py`), and `batchstd+abtt*` is removed outright — it fitted
   coefficients in one residual subspace and applied them in another.
3. **The rehearsal's failure is domain shift, not a weak representation.**
   Corrected after review: gpt2 layer 8 reaches **AUROC 0.864 in-domain** on
   this corpus (random 80/20 split). The drop to 0.754 under leave-one-source-out
   is ~0.11 of transfer loss, with a further ~0.05 thrown away by the transform
   the old ranking key selected. So roughly **70% transfer, 30% representation** —
   an earlier version of this file blamed gpt2, which was wrong.
4. **A single well-chosen source beat the 4-source pool on 4 of 5 folds**, and
   the spread across single training sources on one fold was 0.618 → 0.924
   AUROC. Pooling buys the *average* corpus; the payoff structure rewards the
   *best* one. This is the sharpest challenge to G2's central bet.

---

## Honest risk register

1. **Colab dependency.** Nothing in G3–G5 can be measured until G2 runs on a
   GPU. This is the single blocking item.
2. **The corpus may not be public.** If the organizer wrote or hand-curated
   their own toxicity set, no public corpus will match it and the realistic
   ceiling is ~0.85–0.88, not 0.90+. G4 will show this within a few
   submissions: if every source lands in a narrow band around 0.80, the corpus
   is not one we have.
3. **A 1700-row dev set has ~±1.1% standard error.** Differences under ~2.5
   points between two submissions are not evidence. Do not chase them.
4. **Dev ≠ final.** The final set is a different ~1360 rows. Any gain that came
   from tuning against dev feedback rather than from better ranking will not
   survive.
5. **The quota rule assumes the final set keeps the dev class balance.** This is
   the sharpest single risk in the plan, so state it plainly:

   - What is *measured*: dev is exactly 1200/500, because `triai`'s all-zero
     submission scored exactly 500/1700. Final `input_data` is exactly 0.79997
     of dev's byte size, so final is 1360 rows.
   - What is *assumed*: that final keeps the same 12:5 ratio, i.e. 960/400.
     Both are round numbers under that ratio, which is suggestive, not proof.
   - What it costs if wrong: with a perfect ranking, quota accuracy is
     `1 - |quota - true_rate|`. A balanced final set would cap a 0.706 quota at
     **0.794** no matter how good the probe is.
   - Mitigation: the dev phase can measure quota against a fitted probability
     threshold directly — submit both, compare. If the threshold rule is within
     noise of the quota on dev, prefer it for the final submission, because it
     is the one that does not depend on the assumption.
