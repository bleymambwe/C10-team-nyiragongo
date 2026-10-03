# Runbook: getting to >0.90

Goals and measured facts live in [COMPETITION_GOALS.md](COMPETITION_GOALS.md).
This file is the operating procedure.

## The one thing that needs a human

**Run `colab_probe_pipeline.ipynb` on a GPU.** Everything else is built, tested
and waiting on its output.

### Why not locally — measured, not estimated

Two CPU extraction runs on this machine, same 200 texts, same code path:

| model | params used to the capture layer | rows/s |
|---|---:|---:|
| gpt2, layer 8 | 57 M | 42.5 |
| pythia-410m, layer 14 | 210 M | 6.2 |

Throughput falls as `params^-1.48` here — worse than linear, because 2 cores and
8 GiB make this memory-bandwidth bound. Gemma-2-2b to layer 14 uses **1.09 B**
parameters, 19x gpt2. Against the full-corpus rehearsal rate (13 rows/s on real
text lengths) that extrapolates to **0.17-0.68 rows/s**, so the 153,700-row
corpus takes **2.5 to 10 days**.

Memory settles it independently: gemma-2-2b is 10.4 GB in float32, or 6.7 GB
after dropping every block above 14 — against 8 GiB of total system RAM. bfloat16
fits at 3.4 GB but this CPU (Skylake, no AVX512-BF16) emulates it, which is
slower still. Three background jobs were already killed for memory this session.

**Local extraction is not viable. This is not a tuning problem.**

### Where to run it instead

| option | wall clock | cost | setup | notes |
|---|---|---|---|---|
| **Colab free (T4)** | ~45-60 min | **$0** | none | first choice; GPU occasionally denied at peak |
| Kaggle notebooks | ~45-60 min | **$0** | none | 30 GPU-h/week, good fallback if Colab denies |
| RunPod community (3090/4090) | ~20-30 min + 10 min setup | **~$0.25-0.50** | small | best paid option; per-second billing |
| Vast.ai | similar | ~$0.15-0.40 | small | cheapest, more variable reliability |
| Colab Pro | ~15-25 min | $9.99/mo | none | only worth it if the free tier keeps refusing |
| GCP T4 on-demand | ~30 min + 30-60 min setup | ~$0.55/h | large | **new-project GPU quota defaults to 0 and approval can take 24-48 h** — do not start here with the phase ending 2026-09-09 |

Recommendation: **Colab free, then RunPod if the GPU is denied.** The job is one
hour on any 16 GB+ card; paying more buys nothing. GCP is the wrong shape for a
one-hour job against a two-day deadline.

1. Open <https://colab.research.google.com/> → Upload → `colab_probe_pipeline.ipynb`
2. Runtime → Change runtime type → **T4 GPU**
3. Left sidebar → the key icon → add a secret named `HF_TOKEN`, paste the token
   from `BMToken.env`, toggle notebook access on.
   (The account must have accepted the `google/gemma-2-2b` licence — this one has.)
4. Runtime → Run all. Expect ~50-70 minutes end to end.
5. It downloads `probe_outputs.zip`. Unzip it into `artifacts/colab/` here.

Kaggle notebooks work too; the Colab-specific cells are all wrapped in
`try/except` and degrade to a manual download from the Files pane.

Rerunning is cheap: with Drive mounted, the embeddings are cached and only the
selection and packaging steps repeat.

## What comes back

| file | what it is |
|---|---|
| `loso_results.json` | every recipe's leave-one-source-out score, worst fold first |
| `cross_source_results.json` | the same recipes scored on folds whose positives and negatives come from two different unseen corpora |
| `best_single.zip` | strongest recipe by worst LOSO fold, refit on the whole corpus |
| `best_ensemble.zip` | rank ensemble over the top three distinct transforms |
| `best_threshold.zip` | same ranking, prior-corrected threshold instead of a quota |
| `best_cross_source.zip` | strongest recipe by the cross-source criterion instead |
| `source_*.zip` | one probe per training corpus — the G4 experiment |
| `manifest.json` | recipe, training rows and smoke-test result for each zip |

Every zip is already verified: exactly two files at the root, isolated import,
correct output dtype, sane positive count at both 1700 and 1360 rows, and a
check that `classifier.py` contains no `.fit(` call.

## Reading the result before submitting

The number that matters in `loso_results.json` is **`worst_accuracy`**, not
`mean_accuracy`. The hidden corpus is one unknown draw, so the recipe that
survives its weakest held-out corpus is the one to trust.

Rough calibration for what to expect on the leaderboard, given our 0.7729 now
and the leader's 0.8624:

| LOSO worst | reading |
|---|---|
| < 0.75 | the corpus mix is not helping; something is wrong upstream, do not submit |
| 0.78 – 0.85 | normal; expect a leaderboard score in a similar band |
| > 0.88 | on track for the goal |

If **every** source lands in a narrow band near 0.80, that is the signal from
risk 2 in the goals file: the organizer's corpus is not one we have, and the
realistic ceiling is ~0.85-0.88. Say so rather than burning submissions.

## Spending the submission budget

5 per day. The dev phase ends **2026-09-09 22:00 UTC**, so about 12 remain.
The final phase (2026-09-10 → 2026-09-12) takes a single submission.

**Day 1 — settle the method**

1. `best_single` — the headline number
2. `best_cross_source` — the competing selection criterion. Submitting both
   settles, in one pair of numbers, whether the hidden corpus really does draw
   its positives and negatives from different sources (the G0b inference).
3. `best_threshold` — quota vs prior-corrected threshold, same ranking. This
   also buys the answer to risk 5: if the threshold rule is within noise on
   dev, prefer it for the final, because it does not assume the final set keeps
   the dev class balance.
4. + 5. the two `source_*` probes whose training corpora are least alike
   (a forum corpus and a Twitter corpus, say)

`best_ensemble` is deliberately not in day 1. On the rehearsal it scored *below*
`best_single` on the worst fold, which is what usually happens when the members
are highly correlated. Submit it only if day 1 leaves the method question open.

**Day 2 — finish the source probe**

The remaining `source_*` zips. Each one answers "does the hidden corpus look
like this one", and that answer transfers to the final phase.

**Day 3 — retrain on the answer**

Reweight the corpus toward whichever sources scored highest, rerun selection,
submit the result. Keep one submission in reserve.

## Rules to hold to

- 1700 rows is ±1.1% standard error. **Do not chase a 2-point difference.**
- Do not spend submissions tuning the quota by ±20 rows. The final phase is a
  different 1360 rows; that gain does not transfer, and it is the kind of
  dev-set fitting the terms are aimed at.
- Do not try to recover the hidden labels, from the leaderboard or from the
  scoring container. Beyond being against the terms, dev labels say nothing
  about the final set.
- Everything submitted must be trained offline. The only things the classifier
  computes at inference are label-free feature normalisations, which is why
  `smoke_test` refuses any `classifier.py` containing `.fit(`.

## Submitting without the click-through

`codabench_submit.py` drives a real Chrome against a persistent profile, so the
loop is: get candidates → dry-run → submit → read the score, without hand-editing
anything.

```bash
python codabench_submit.py login                       # once; you sign in yourself
python codabench_submit.py status                      # leaderboard, deadlines, budget
python codabench_submit.py submit <zip>                # DRY RUN by default
python codabench_submit.py submit <zip> --yes --label best_single
```

Design points that matter:

- **No credentials pass through the tool.** `login` opens a headed browser and
  waits for you to sign in; the session lives in a Chrome profile under
  `%LOCALAPPDATA%\codabench_probe_profile`, outside the repository.
- **`submit` is a dry run unless you pass `--yes`.** The dry run walks the whole
  path, locates the file input, screenshots the page and stops. It costs nothing
  from the 5/day, so selector breakage is found before a submission is spent.
- **It refuses to upload a zip it has not verified**: root must be exactly
  `classifier.py` + `trained_probe.joblib`, and the isolated-import smoke test
  must pass at 1700 rows.
- **It refuses a sixth submission in 24 h**, tracking `results/submission_log.json`.

**State as of 2026-09-07: logged in, approved participant, dry run green.** The
last dry run reported: phase `Development Phase` already active, 1 file input
located, 7 existing submissions read, platform counters **0/5 today, 5/100
total**. Everything up to the upload itself is verified; the upload step cannot
be rehearsed without spending a submission.

Three selector bugs were found by dry runs rather than by burning submissions,
which is the point of having them:

- The phase switch is `<div class="ui button">Development Phase </div>` — a div,
  so no button role, and a trailing space, so an exact text match misses it.
- The page has **seven** tables, so `inner_text("table")` raised on the
  ambiguity and silently returned no submission ids, disabling the "did a new
  submission appear" check. One of the seven is the leaderboard, carrying other
  people's ids — scraping it would have reported a stranger's upload as ours.
  The submissions table is now identified by its "File name" header.
- There is no Submit button: the upload fires on the file input's change event.
  Clicking one of the twelve "Submit" strings on the page would have done
  something else entirely.

The tool now refuses to upload if it could not positively select the phase,
because an upload into the wrong phase cannot be undone.

## Logging results

Append every submitted score to the iteration log in COMPETITION_GOALS.md —
date, candidate name, LOSO worst, leaderboard accuracy. A row without a
measured leaderboard number does not go in.
