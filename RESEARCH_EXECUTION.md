# Competition execution record

Started 2026-09-08 UTC (2026-09-09 locally). This record supersedes numerical
assumptions in BRIEF.md where direct evidence disagrees. Historical files and
the incumbent submission are preserved.

## Verified starting state

- Competition: https://www.codabench.org/competitions/17670/
- Public API and signed-in upload dry run confirm the development phase ends
  2026-09-09 22:00 UTC. Testing Phase opens 2026-09-10 00:00 UTC and closes
  2026-09-12 00:00 UTC; automatic phase migration is disabled.
- Incumbent submission 918485: 0.8976470588 accuracy, 0.0630 seconds.
- Leader at inspection: thylinao, 0.9411764706. Ranking uses Force_Last.
- At 2026-09-09 06:52 UTC the rolling local budget was 0/5 submissions in the
  preceding 24 hours. Preserve capacity to restore the incumbent after every
  experimental upload because the board uses Force_Last.
- Incumbent artifact SHA-256:
  `0bd11ac11c0a09a8b38d834e06c558f0e17236dae680c215a5e33af8c746a601`.
- Test suite after routing additions: 31 passed.

## Corrections required for a defensible preprint

1. Accuracy does not determine AUROC. The brief's accuracy-to-AUROC table
   assumes a score-distribution model; it is not an observed leaderboard metric.
   Constant prediction scores have AUROC 0.5.
2. For 1526/1700 correct, ordinary binomial SE is approximately 0.00735,
   not 0.011. Comparing two models on the same examples requires paired errors.
   Repeated submissions of identical predictions are not independent trials.
3. Strong competitor accuracy does not identify the training corpus or prove
   that it is publicly available. There is no demonstrated 0.90 ceiling.
4. `best_threshold.zip` changes from standardized LDA to unscaled logistic
   regression, and its threshold still depends on the inferred prior 1200/1700.
   It is neither a controlled decision-rule ablation nor prior-independent.
5. A source candidate's recorded `pooled_accuracy_on_this_family` evaluates a
   different pooled model. Existing source ZIPs have no baseline validation of
   their own. Rejected candidates may remain as ZIPs after manifest filtering.
6. Final class balance and row count are inferred, not observed ground truth.
   Report the quota's dependence on that assumption.

## Corpus discovery

BrowsePilot was used first. Its search returned only Reddit results even for
broad dataset queries, and no results for explicit GitHub/Hugging Face site
filters (jobs 799-806). General search was used as a source-discovery fallback.
BrowsePilot extraction remains functional (jobs 807, 808, 811).

The organizer is TRI AI / AI Saturdays Lagos. Its official project page says
dataset sourcing is part of the challenge and supplies no training corpus:
https://aisaturdayslagos.github.io/cohort_structure/cohort10/projects.html
The official curriculum identifies the capstone team and mentors:
https://aisaturdayslagos.github.io/cohort_structure/cohort10/index.html
Public repository trees for cohort_structure, the Week 6 teaching repository,
and cohort10_leaderboard did not expose the missing latent_probe_pipeline.py.
The competition forum is disabled. No messages were sent to anyone.

This is an unsuccessful exact-corpus search, not proof of nonexistence.

New source examined: https://huggingface.co/datasets/s-nlp/paradetox
Human toxic-to-neutral rewrites provide a distinct contrastive training signal.
We remove normalized-text overlaps and exclude Civil/Wikipedia families from
claims of independent cross-corpus transfer because ParaDetox shares their
ancestry. Strict de-duplication against the existing corpus excluded 10,156
overlaps and produced a balanced 3,000-row sample (fingerprint
`bce9331fed368fb3`).

ToxiGen was inspected but not added: its card requests a data-access signup.
WildGuardMix returned an authorization error and was not downloaded or bypassed.

Offline teacher: https://huggingface.co/unitary/toxic-bert
Use its pinned model revision to score training texts. Keep original labels
only when positive score >=0.7 or negative score <=0.3; rebalance each retained
source. Do not filter evaluation examples. This teacher may have trained on
Civil/Wikipedia, limiting independence claims for those evaluation families.
The model card warns that its weights differ from the Detoxify library; do not
describe these as interchangeable implementations.

## Completed first experiment

`experiments/corpus_transfer.py`, private Kaggle kernel
https://www.kaggle.com/code/bleymambwe/latent-probe-corpus-transfer (version 1).
Results: `artifacts/kaggle/corpus-transfer/transfer_results.json`.

Nine held-out families; fixed std|lda|0.6; source cap 8,000 and pooled cap 14,000.
Three overlapping resamples (17,29,41), at each of two class priors. These are
resampling sensitivity checks, not independent replicates. All source heads,
moments, and pooled scales exclude the held-out family. Selection uses fixed
diagonal Gaussian Wasserstein distance, with source class moments mixed at
the configured prior. Nothing is fitted during inference.

| Method | Accuracy at 0.5 prior | Accuracy at 12/17 prior | AUROC at 12/17 |
|---|---:|---:|---:|
| Pooled baseline | 0.752723 | 0.786891 | 0.824032 |
| Routing with pooled fallback | 0.763330 | 0.796002 | 0.836201 |
| Routing without pooled fallback | 0.768253 | 0.797309 | 0.834598 |
| Drop HateCheck and Aegis | 0.750689 | 0.784788 | 0.823098 |

The critic recommends retaining the pooled fallback and vetoes source-only
routing for submission. The small macro gain does not establish superiority
to the full-data incumbent.

The uncapped full-data rerun produced the following 12/17-prior macro results:

| Method | Accuracy | AUROC |
|---|---:|---:|
| Pooled baseline | 0.795946 | 0.832204 |
| Routing with pooled fallback | 0.799313 | 0.838130 |
| Routing without pooled fallback | 0.798921 | 0.838564 |
| Drop HateCheck and Aegis | 0.793399 | 0.829022 |

In source-stratified 80/20 holdouts, the router selected the matching source in
all nine families and improved macro accuracy from 0.829456 to 0.870673. This
supports routing on an in-family batch but does not reveal the hidden corpus.

## Teacher and ParaDetox experiment

The pinned toxicity teacher scored all 149,528 original rows. Agreement
filtering retained only high-confidence labels, and evaluation labels were never
teacher-filtered. Full LOSO results at the 12/17 prior were:

| Training method | Accuracy | AUROC | Same-family baseline accuracy |
|---|---:|---:|---:|
| Original pooled baseline | 0.795946 | 0.832204 | 0.795946 |
| Teacher agreement filter | 0.785709 | 0.815232 | 0.795946 |
| ParaDetox only | 0.702109 | 0.696101 | 0.781455 |
| ParaDetox augmented | 0.781849 | 0.811965 | 0.781455 |

Teacher filtering was decisively worse. ParaDetox augmentation was essentially
flat in accuracy and worse in AUROC, so neither candidate qualifies for a live
submission.

## Development submission results

The critic independently cleared each experimental upload and retained veto
authority. The packaged router with the exact incumbent as its pooled fallback
had SHA-256
`a934eb3b2a73c53b37719e1ca0c0e3d6f8591b35b5aca25a9b15808e7d131a42`.
It scored 0.8976470588 (1526/1700), an exact tie with the incumbent, in
submission 919789. The tie does not reveal whether it selected the pooled head
or merely produced an equally accurate alternative prediction vector.

The source-only router had SHA-256
`07b18ae692d17b5bbd3aa305d71f712ebb3189d77ee629ad78c3f0f00556a263`.
It scored 0.8494117647 (1444/1700), an 82-row regression. This directly rejects
replacing the pooled model with the nearest stored public-source head on the
development distribution.

The exact incumbent was immediately restored as submission 919793 and verified
at 0.8976470588. Because the pooled-fallback router did not improve and the
source-only router regressed sharply, `best_single.zip` remains the Testing
Phase choice.

The critic approved that exact incumbent for the single Testing Phase attempt.
Windows task `LatentProbeFinalSubmission` was scheduled for 2026-09-10 02:05
Africa/Windhoek (00:05 UTC), five minutes after the phase opened. The guarded
script rechecked the exact ZIP hash and live phase window, used absolute paths
and the existing interactive browser profile, and prohibited automatic retry.
Its launch failed before creating a submission; the audited recovery and final
verified result are recorded below.

## Rule interpretation

The platform terms require our own offline-trained model; no platform training.
Batch routing chooses a stored head using fixed arithmetic. Its permissibility
is an interpretation of the terms, not an organizer endorsement.

`allow_robot_submissions=false` concerns the special admin-designated bot-user
feature, not ordinary participant UI automation. Verified against official docs:
https://docs.codabench.org/dev/Developers_and_Administrators/Robot-submissions/
and https://github.com/codalab/codabench/pull/2499
The ordinary admitted-participant upload interface is used, with its limits.

## Testing submission result

The scheduled 2026-09-10 00:05 UTC attempt passed its hash and phase-window
guards but exited before creating a submission. Its log stopped immediately
after the guard record, Task Scheduler returned exit code 1, the local ledger
had no Testing entry, and no new browser screenshot existed. This evidence was
treated as a failed pre-upload launch, not as permission to retry blindly.
The root cause was the invalid PowerShell parameter-set combination
`Tee-Object -LiteralPath ... -Append`; `-Append` is supported with `-FilePath`
but not the `-LiteralPath` parameter set. Both logging pipelines were corrected.

At 08:52 UTC, the public competition API confirmed phase 29479 (`Testing Phase`)
was current. An authenticated dry run then positively selected the Testing
panel and showed platform counters of `0/5` for the day and `0/1` total. The
critic gate was rerun: all 31 tests passed; the ZIP had exactly
`classifier.py` and `trained_probe.joblib`; its CRC was clean; and SHA-256 was
`0bd11ac11c0a09a8b38d834e06c558f0e17236dae680c215a5e33af8c746a601`.

One recovery upload was made with label
`final_testing_pooled_std_lda06_manual`. CodaBench created submission 921106 at
2026-09-10 08:54:49 UTC. The read-only submission API verified phase 29479,
owner `bleymambwe`, file `best_single.zip`, status `Finished`, accuracy
**0.9235294118 (1,256/1,360)**, and duration 0.0520865917 seconds. The platform
counter is now `1/1`; no further Testing upload is possible or appropriate.

At the 08:57 UTC leaderboard check, this result was second behind `thylinao` at
0.9558823529 (1,300/1,360), a gap of 44 rows. Because Testing remains open until
12 September 00:00 UTC, placement is provisional even though our submission is
final.
