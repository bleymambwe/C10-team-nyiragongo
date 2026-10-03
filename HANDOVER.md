# Engineering Handover: Latent Probing for Toxicity

**Competition:** CodaBench 17670  
**Owner:** Blessing Mambwe (`bleymambwe`)  
**Updated:** 10 September 2026, 11:05 CAT (UTC+2)  
**Status:** Testing submission 921106 finished at 0.9235294118 (1,256/1,360); currently second

This document is the operational source of truth. `BRIEF.md` records the original
strategy, and `RESEARCH_EXECUTION.md` records the corrected research history.
Where they differ, follow this handover and the saved platform evidence.

## 1. Final outcome

The exact reviewed artifact was submitted to the Testing Phase:

```text
File:   artifacts/kaggle/select/submissions/best_single.zip
SHA256: 0bd11ac11c0a09a8b38d834e06c558f0e17236dae680c215a5e33af8c746a601
Model:  standardized shrinkage LDA, lambda 0.6
Rule:   label the top 12/17 of batch scores positive
Dev:    0.8976470588 = 1,526/1,700 correct
Test:   0.9235294118 = 1,256/1,360 correct
ID:     921106
```

Do not replace it with either router:

| Candidate | Development score | Decision |
|---|---:|---|
| Pooled `best_single.zip` | **0.8976470588** | Final choice |
| Router with pooled fallback | 0.8976470588 | Tied; added complexity without gain |
| Source-only router | 0.8494117647 | Rejected; lost 82 rows |

The submission is finished and verified through the read-only CodaBench API.
At 2026-09-10 08:57 UTC it ranked second behind `thylinao` at 0.9558823529.
Testing remains open until 12 September, so this is a current rather than final
competition placement.

## 2. Deadlines in UTC and Central Africa Time

Central Africa Time (CAT) is UTC+2. Windhoek uses the same offset.

| Event | UTC | CAT / Windhoek |
|---|---|---|
| Development closes | 9 Sep, 22:00 | **10 Sep, 00:00** |
| Testing opens | 10 Sep, 00:00 | **10 Sep, 02:00** |
| Guarded upload runs | 10 Sep, 00:05 | **10 Sep, 02:05** |
| Testing closes | 12 Sep, 00:00 | **12 Sep, 02:00** |

Development and Testing use separate submission panels. Development uploads do
not migrate automatically. A development upload made before 02:00 CAT cannot
count as the Testing submission.

## 3. Testing upload execution

Windows Task Scheduler ran this task:

```text
Task:          LatentProbeFinalSubmission
Result:        Failed with exit code 1 after the hash and time guards
Local trigger: 2026-09-10 02:05 CAT
Logon type:    Interactive
Script:        scheduled_testing_submit.ps1
```

The scheduled attempt created no CodaBench submission and no ledger entry. The
failure occurred after the guards because `Tee-Object -LiteralPath ... -Append`
combines parameters from incompatible PowerShell parameter sets, so the upload
pipeline never launched. The script now uses `-FilePath ... -Append`. Recovery
followed the documented no-duplicate procedure:

1. Confirmed the Testing phase was current through the public competition API.
2. Ran an authenticated dry run that showed `0/1` Testing submissions used.
3. Rechecked 31 tests, ZIP CRC, the two-member root, and the approved SHA-256.
4. Uploaded once through the Testing panel with audit label
   `final_testing_pooled_std_lda06_manual`.
5. CodaBench created submission 921106; the platform counter changed to `1/1`.
6. Verified the finished result through `/api/submissions/921106/`.

The guarded script was designed to perform these steps once:

1. Recomputes the ZIP SHA-256 and aborts unless it matches the approved hash.
2. Confirms that current UTC time falls inside the Testing window.
3. Selects the CodaBench `Testing Phase` panel explicitly.
4. Makes one upload attempt. It never retries an ambiguous result.
5. Reads the returned submission ID from `results/submission_log.json`.
6. Verifies that ID through the public API against Testing phase ID `29479`.
7. Polls until the submission finishes or the 12-minute verification limit ends.
8. Writes the verification record to `artifacts/testing_submission_result.json`.

The task runs only in the same signed-in Windows session because the browser
driver is graphical. Wake-on-run is enabled, but wake-on-run cannot sign in to
Windows. Leave the account signed in; locking the session is safer than logging
out.

The task history can still be inspected with:

```powershell
$task = Get-ScheduledTask -TaskName 'LatentProbeFinalSubmission'
$info = $task | Get-ScheduledTaskInfo
$task | Select-Object TaskName, State
$info | Select-Object LastRunTime, NextRunTime, LastTaskResult
```

## 4. Verified Testing result

Do not use `python codabench_submit.py status` to verify Testing. That command
reads development leaderboard ID `19655`.

Use the generated record:

```powershell
Get-Content artifacts/testing_submission_result.json
Get-Content results/scheduled_testing_submit.log -Tail 120
```

`artifacts/testing_submission_result.json` contains:

```text
phase:          29479
phase_name:     Testing Phase
filename:       best_single.zip
owner:          bleymambwe
status:         Finished
phase_verified: true
accuracy:       0.9235294118 = 1,256/1,360
duration:       0.0520865917 seconds
submission_id:  921106
```

To recheck a known submission ID without changing platform state:

```powershell
python verify_codabench_submission.py <SUBMISSION_ID> `
  --expected-phase 29479 `
  --output artifacts/testing_submission_result.json `
  --max-wait 720
```

## 5. Recovery record

The scheduled task did fail, and the recovery procedure below was completed
successfully. It is retained as the audit trail; do not upload again because
the Testing counter is now `1/1`.

1. Read `results/scheduled_testing_submit.log`.
2. Read the newest Testing entry in `results/submission_log.json`.
3. If an ID exists, verify it with `verify_codabench_submission.py`.
4. If no ID exists, inspect `%LOCALAPPDATA%\codabench_after_submit.png`.
5. Confirm that Testing remains open.
6. Obtain the required critic review before any manual upload.
7. Make one manual attempt only if the evidence shows that no submission exists.

Manual command used after those checks (do not run again):

```powershell
python codabench_submit.py submit `
  artifacts/kaggle/select/submissions/best_single.zip `
  --yes `
  --phase 'Testing Phase' `
  --label final_testing_pooled_std_lda06_manual
```

The competition brief gives a critic veto authority before every submission.
That rule also applies to recovery uploads.

## 6. Current platform state

At the last verified development status:

| Item | Value |
|---|---|
| Leader | `thylinao`, 0.9411764706 = 1,600/1,700 |
| Our position | Second |
| Our restored score | **0.8976470588 = 1,526/1,700** |
| Restored submission | `919793` |
| Development uploads in last 24 hours | 3/5 |
| Ranking rule | `Force_Last` |

At the Testing status check on 10 September 2026 at 08:57 UTC:

| Item | Value |
|---|---|
| Current leader | `thylinao`, 0.9558823529 = 1,300/1,360 |
| Our current position | Second |
| Our Testing score | **0.9235294118 = 1,256/1,360** |
| Testing submission | `921106` |
| Testing submissions used | 1/1 |
| Testing closes | 12 September 2026, 00:00 UTC |

`Force_Last` displays the most recent submission, not the best historical one.
The source-only router temporarily dropped the account to 0.8494. Submission
919793 restored the exact incumbent and returned the board to 0.8976470588.

## 7. Submission history that matters

| Submission | Artifact | Score | Result |
|---:|---|---:|---|
| 918460 | `best_single.zip` | 0.8976470588 | First successful pooled model |
| 918464 | `best_ensemble.zip` | 0.8918 | Rank ensemble lost 10 rows |
| 918476 | `raw_lda06.zip` | 0.8765 | Raw features lost 36 rows |
| 918485 | `best_single.zip` | 0.8976470588 | First restoration |
| 919789 | `corpus_router_incumbent_fallback.zip` | 0.8976470588 | Exact tie |
| 919792 | `corpus_router_source_only.zip` | 0.8494117647 | Lost 82 rows |
| 919793 | `best_single.zip` | 0.8976470588 | Current restored model |

The local submission ledger is `results/submission_log.json`.

## 8. Final model

The platform supplies a matrix shaped `[n_examples, 2304]` to
`Classifier.predict(X)`. The submitted classifier applies fixed arithmetic; it
does not fit parameters on the evaluation batch.

Training pipeline:

```text
13 public toxicity sources
  -> 149,528 extracted examples
  -> global normalized-text de-duplication
  -> remove one single-class source
  -> rebalance each retained source to 50/50
  -> 108,468 examples, 12 sources, 9 families
  -> Gemma 2 2B hidden_states[14]
  -> attention-mask-weighted mean, max_length=64
  -> 2,304-dimensional float32 vectors
  -> train-statistics standardization
  -> shrinkage LDA, lambda=0.6
  -> fixed quota at 12/17
```

The ZIP contains exactly:

```text
classifier.py
trained_probe.joblib
```

The classifier uses NumPy rather than a pickled sklearn estimator, which avoids
scoring-environment version mismatches.

## 9. Corpus and validation evidence

The retained families are Civil Comments, Twitter, Jigsaw Wikipedia, Berkeley
hate speech, RealToxicityPrompts, Aegis, OffensiveLang, HateCheck, and ToxicChat.
Related datasets remain in the same validation family to reduce leakage.

Full-data leave-one-family-out results at the inferred 12/17 prior:

| Method | Accuracy | AUROC |
|---|---:|---:|
| Pooled baseline | 0.795946 | 0.832204 |
| Router with pooled fallback | 0.799313 | 0.838130 |
| Source-only router | 0.798921 | 0.838564 |
| Pool without HateCheck and Aegis | 0.793399 | 0.829022 |

The router recognized all nine families in source-stratified holdouts and raised
macro accuracy from 0.829456 to 0.870673. That result did not transfer into a
leaderboard gain. The source-only live regression provides stronger evidence
against source specialization on the hidden batch.

The exact organizer corpus remains unidentified. The official TRI AI project
page says participants must source, scrape, or generate their own datasets; it
does not publish a training corpus. Treat any statement that the leader found
the organizer corpus as speculation.

## 10. Rejected experiments

Do not spend another submission on these ideas without new evidence:

| Experiment | Evidence |
|---|---|
| Source-only batch router | 0.849412 live; 82-row regression |
| Pooled-fallback router | Exact live tie; no benefit for extra complexity |
| Teacher-agreement filtering | 0.785709 vs 0.795946 offline baseline |
| ParaDetox only | 0.702109 offline accuracy |
| ParaDetox augmentation | 0.781849 vs 0.781455 accuracy, lower AUROC |
| Rank ensemble | 0.8918 live; 10-row regression |
| Raw LDA | 0.8765 live; 36-row regression |
| ABTT / top-PC removal | Mean AUROC fell by about 0.049 |
| Nonlinear MLP | AUROC fell by about 0.022 |
| CORAL alignment | AUROC fell by about 0.032 |

ParaDetox research produced 3,000 balanced, strictly de-duplicated rows after
excluding 10,156 overlaps. Its fingerprint is `bce9331fed368fb3`. Preserve the
corpus and negative result; do not present it as the competition's source.

## 11. Statistical cautions

- Accuracy does not determine AUROC. The leaderboard exposes accuracy only.
- The 0.897647 score has ordinary binomial standard error about 0.00735.
- An approximate Wilson 95% interval is 0.8823 to 0.9112 under an IID model.
- Compare two models on the same hidden rows with paired disagreements; the
  public leaderboard does not expose them.
- The development quota of 1,200/1,700 is inferred from platform outcomes.
- The expected Testing quota of 960/1,360 is inferred, not observed.
- Seeded prior resamples overlap. They are sensitivity checks, not independent
  replications.
- Adaptive leaderboard selection makes the development score optimistic.

## 12. Safety rules

1. Never upload a ZIP unless its SHA-256 matches the reviewed artifact.
2. Never retry an upload until the submission table and API prove that no record
   exists.
3. Never rely on the development status command for a Testing result.
4. Never print Hugging Face, Kaggle, browser-profile, or environment secrets.
5. Never train, optimize, or update parameters on platform evaluation data.
6. Keep the ZIP root to the two required files.
7. Run the critic review before every submission and honor a veto.
8. Preserve the incumbent when an experiment fails.

## 13. Verification commands

Run from the repository root in PowerShell:

```powershell
python -m pytest tests/ -q

Get-FileHash `
  artifacts/kaggle/select/submissions/best_single.zip `
  -Algorithm SHA256

python codabench_submit.py status

python -c "import zipfile; p='artifacts/kaggle/select/submissions/best_single.zip'; print(zipfile.ZipFile(p).namelist())"
```

Expected results:

```text
31 tests pass
SHA256 = 0BD11AC11C0A09A8B38D834E06C558F0E17236DAE680C215A5E33AF8C746A601
ZIP members = ['classifier.py', 'trained_probe.joblib']
```

Warnings from `tests/test_instruments.py` concern tests that return values rather
than using assertions. They do not indicate a model failure.

## 14. Repository map

| Path | Purpose |
|---|---|
| `HANDOVER.md` | This operational handover |
| `RESEARCH_EXECUTION.md` | Corrected research and submission record |
| `BRIEF.md` | Original competition strategy and critic rule |
| `artifacts/kaggle/select/submissions/best_single.zip` | Final reviewed model |
| `artifacts/final_submission_result.json` | Development outcome summary |
| `artifacts/testing_submission_result.json` | Testing result after upload |
| `results/scheduled_testing_submit.log` | Scheduled upload log |
| `results/submission_log.json` | Local upload ledger and IDs |
| `scheduled_testing_submit.ps1` | Guarded one-shot Testing upload |
| `verify_codabench_submission.py` | Read-only phase, status, and score verifier |
| `codabench_submit.py` | Browser upload driver; dry-run by default |
| `src/probe/corpus.py` | Corpus loaders and global de-duplication |
| `src/probe/extract.py` | Gemma layer-14 masked-mean extraction |
| `src/probe/select.py` | Grouped validation and model selection |
| `src/probe/submission.py` | Portable classifier and ZIP smoke test |
| `src/probe/routing.py` | Frozen batch router used in rejected experiments |
| `experiments/corpus_transfer.py` | Router validation experiment |
| `experiments/teacher_select.py` | Teacher and ParaDetox experiment |
| `output/pdf/latent_probe_toxicity_preprint.pdf` | Nine-page competition preprint |

## 15. Environment

- Python: `C:\Python312\python.exe`
- Shell: PowerShell
- CodaBench browser profile:
  `%LOCALAPPDATA%\codabench_probe_profile`
- BrowsePilot project:
  `C:\Users\BLESSING MAMBWE\Pictures\Projects\Browse Automation`
- Kaggle kernels:
  `bleymambwe/latent-probe-corpus-transfer` and
  `bleymambwe/latent-probe-teacher-select`

Credentials remain outside the repository. Do not display `.env` files, tokens,
or browser-session contents.

## 16. Definition of done

The competition handoff is complete. Verified:

- `artifacts/testing_submission_result.json` exists.
- The record names phase `29479` and `Testing Phase`.
- The owner is `bleymambwe`.
- The filename is `best_single.zip`.
- The status is `Finished`.
- An accuracy score is present.
- The final score and submission ID are recorded in this handover,
  `RESEARCH_EXECUTION.md`, `artifacts/final_submission_result.json`, and the
  regenerated competition preprint.
