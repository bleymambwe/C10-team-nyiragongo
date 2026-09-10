# Audit Log

## 2026-08-15

- Created a minimal public repository scope for proposal-stage publication only.
- Intentionally excluded local research archives, rendered media, notebooks, large artifacts, and unrelated course materials.
- Added restrictive ignore rules to reduce risk of accidental publication of non-public assets.
- Reserved `paper/` and `artifacts/` for future curated public materials.

## Ongoing log format

Add future entries with:

- date,
- decision,
- reason,
- files added or excluded,
- any publication-risk note.

## 2026-08-24

**Decision.** Moved the repository from proposal-stage scaffolding to an
executed research program, and changed the research target.

**Reason.** A novelty scan run today found that several headline directions in
the `Research/` package have been claimed since it was written: circuit
non-identifiability (ICLR 2025 *Everything, Everywhere, All at Once*; ICLR 2026
uniqueness paper; *Certified Circuits*; *Many Circuits, One Mechanism*),
control-theoretic interpretability (*From Black-Box to White-Box*;
*Controllability Analysis of State-Space LMs*), and toxicity steering
(*CausalDetox*; linear-optimal-control steering). The decodability/control gap
is *observed* in 2026 work but never *predicted*. That gap was taken instead.

**Files added.**

- `docs/00_GAP_ANALYSIS.md` — novelty scan, registered predictions P1–P8
- `docs/01_THEORY.md` — Theorems 1–2, Propositions 3–7, proofs, and a
  post-hoc section recording which claims survived measurement
- `docs/02_PAPER_PLAN.md` — submission plan and reviewer-objection answers
- `src/egc/` — certificate library, LM adapters, toy transformer, prompt data
- `experiments/e1..e6`, `run_all.py` — one script per claim, fixed seeds
- `tests/test_instruments.py` — pins `auc`, `fisher_probe`, `spearman`,
  `pearson` to sklearn/scipy references
- `results/data/*.json` — raw results behind every figure
- `dashboard/` — reproducible HTML report

**Publication-risk notes.**

- `results/cache/rtp_balanced.json` holds a 1200-prompt balanced sample of
  RealToxicityPrompts. It contains toxic text by construction. It is a cache
  of a public research dataset, not new content, but it should be excluded
  from any public release rather than shipped as a repository artifact.
- `src/egc/toxdata.py` contains identity terms and mild insult vocabulary used
  strictly as *logit measurement targets* for a bias audit. Nothing is
  generated or emitted. The identity-term list exists to measure false
  positives, which is its only purpose.
- Model weights and HF caches remain outside version control.

**Corrections made during the work, recorded rather than silently fixed.**

- Prompt-perplexity was adopted as a collateral-damage metric and returned
  identically zero; causal masking makes it vacuous for a final-position
  write. Replaced with next-token KL. Both the failure and the reason are
  documented in `docs/01_THEORY.md` §12.3.
- The pre-registered prediction that dataset confounding widens the
  probe/steer gap was **not confirmed** on GPT-2. Reported as a null result;
  the surviving explanation is the low effective rank of the effect Gramian.
- Proposition 5 predicts a probe-orthogonal direction has AUC exactly 0.5.
  Exact in the Gaussian construction, only approximate in GPT-2 (0.54–0.69).
  Stated as a limitation rather than rounded toward the theory.

## 2026-08-24 (later)

**Added.** A narrated video walkthrough and a preprint, both generated from the
same experiment outputs as the dashboard.

- `paper-video/` — Remotion composition, 17 scenes, 10:01 at 1920x1080/30fps.
  Narration synthesised locally with Kokoro-82M (`af_heart`); no API key and no
  network. Scene durations are derived from measured audio lengths, so the
  visuals cannot drift from the voiceover.
- `docs/03_VIDEO_TRANSCRIPT.md` — full script with timecodes.
- `out/seeing-is-not-steering.mp4` — rendered output (48 MB), excluded from
  version control.
- `paper/preprint.tex` + `preprint.pdf` — 12-page preprint.
- `paper/make_numbers.py`, `make_figures.py`, `check_macros.py` — the paper's
  numbers and figures are generated from `results/data/*.json`; nothing is
  hand-typed, and an undefined macro fails the check rather than disappearing.

**Publication-risk notes.**

- The preprint carries a boxed statement that its 2025–2026 citations were
  found by a literature scan, are cited by title, and require verification
  before submission. They fix the paper's position relative to concurrent work;
  they are not verified references.
- The video and paper both report E5 as two models rather than three, and make
  no claim resting on E2 or E4, which did not complete.
- Scene 07 of the video draws the "live fraction" grid at 40x true density,
  because 0.18% of 288 cells is invisible. The caption states the exaggeration
  on screen.

## 2026-08-26

**Fixed the data-loss flaw.** E2, E4 and E5 previously wrote their JSON only
after the whole suite finished, so stopping a run discarded everything. All
three now checkpoint incrementally (E4 per layer, E5 per model, E2 per trained
run), and E5 resumes from whatever is already on disk. `dashboard/build.py`
gained `EGC_DATA_DIR` / `EGC_OUT_HTML` overrides so the report can be built
against synthetic data for testing without touching real results.

**E4 completed** — prediction P5 now has a verdict, and the checkpointing
proved itself: the suite was stopped during E5's resume, and E4's completed
results survived.

- Ordering candidates by the certificate reaches 90% of the best attainable
  effect after 1.3 measurements of 64 on average (2/1/1 across layers 3/6/9),
  against 29.3 at random (46/22/20) and 14.7 by probe accuracy (32/6/6).
  Oracle is 1.0. That is a ~22x budget reduction against random.
- Spearman with true effect: certificate 0.761, |AUC-0.5| 0.313.

**Still outstanding.** E2 (from-scratch transformers with a confound dial) and
E5's third model (pythia-70m) have not run. Both are absent rather than
reported, in the dashboard, the preprint and the video.

**Known staleness.** The video walkthrough predates E4 and therefore does not
include the active-design result. It is not wrong, but it is now incomplete
relative to the dashboard and preprint. Re-rendering costs roughly one hour
(TTS plus render) and has not been done.

**Bugs found by verification, not by inspection.**

- The dashboard's new Outputs section referenced `C.s1`, a theme object that
  exists only in the Remotion project. This threw a ReferenceError inside the
  render IIFE and blanked every section below it. Caught by reading the browser
  console after the build, not by reading the code.
- Two LaTeX edits made through Python string replacement were silently
  corrupted: `\times` inside an f-string became a tab, and `\ref` became a
  carriage return, producing `\Sef{...}`. Both were found by rendering the PDF
  to images and reading them. The repository now avoids Python escapes for
  LaTeX edits.

## 2026-08-26 (evening)

**The experimental programme is complete.** All six experiments have run; every
registered prediction has a verdict.

E2 finished (12 runs, 115 min, 4 confound levels x 3 seeds). It splits the old
P6 into two claims and settles both:

- **P6a confirmed** — the gap is produced by gradient descent, not only by
  construction. At *zero* confounding, rho = 0.151 and the probe recovers 12.0%
  of available control.
- **P6b refuted** — training *under* a confound does not widen the gap. Across
  confound 0.50 -> 0.98, mean rho moves 0.151 -> 0.147, a change of 0.004, or
  0.2x the within-level seed SD (0.017). Probe efficiency is flat at
  12.0-12.1%; the probe-orthogonal direction holds at 98.7-98.8%. The
  manipulation demonstrably worked: accuracy against the marker tracks the dial
  at 0.504 / 0.703 / 0.899 / 0.979.

Two independent designs — one inheriting representations (E3), one creating
them (E2) — now refute the confound hypothesis. The surviving mechanism is the
low-rankness of the effect Gramian (Prop. 3), not label contamination.

E5 completed all three models. pythia-70m (GPTNeoX, d=512) replicates gpt2 and
distilgpt2: fisher 8.2%, diff-of-means 14.1%, orthogonal 98.6%.

**A finding on scale, from four independent points.** In absolute terms r_eff
is 1.6 (E2 toys, d=64), 2.3 (gpt2), 3.2 (distilgpt2), 4.6 (pythia-70m) — a
small constant across a 12x range of width, not growing with d. If it stays
O(1), r_eff/d shrinks with scale and the gap *widens*, opposite to the failure
mode named as the largest threat. Recorded as suggestive, not established.

**Correctness work.**

- `dashboard/build.py` now serialises with `allow_nan=False`. NaN is legal in
  Python's JSON output but not in JavaScript's `JSON.parse`, so one leaking
  through would blank the whole page silently. It now fails loudly instead.
- Investigated a NaN in the E2 pool statistics. It is not a bug: at layer 0 of
  the toy model every pooled direction scores AUC exactly 0.500, so the rank
  correlation is genuinely undefined. The aggregation filters non-finite
  values, so no figure or claim was affected.
- Table 1 of the preprint was hand-typed, contradicting the paper's own
  no-hand-typed-numbers rule. It is now generated into `paper/table_e5.tex`.
  Generating it surfaced a second problem: `\input`ting a row fragment inside a
  tabular raises "Misplaced \noalign", so the generator emits the whole
  tabular.
- A third `\r`-in-a-Python-string corruption hit `make_figures.py`. Repaired at
  byte level. LaTeX and matplotlib strings are no longer edited via Python
  string replacement.

**External edits preserved.** `paper/preprint.tex` was edited outside this
session, adding remarks relating the four-cell partition to the Kalman
canonical decomposition, Prop. 3 to Backus-Gilbert resolving power, the
certificate to Elfving c-optimal design, and a candidate non-normality
mechanism for small r_eff (Trefethen; two 2026 residual-stream Jacobian
papers), plus six bibliography entries. These were left intact; the E2
integration was applied on top and compiles cleanly at 16 pages.

**Still outstanding.** The video walkthrough predates E4 and E2 and is now
missing both. It is not wrong, but it is incomplete relative to the paper and
dashboard.

## 2026-09-10 - Cohort 10 release

**Decision.** Curated the public repository as the final AI Saturdays Lagos
Cohort 10 competition package for Team Latent Lens.

**Included.** The release contains the final Gemma latent-probe pipeline, cloud
notebook, source-aware evaluation, tests, exact CodaBench archive, verified
development and testing score records, and the four required challenge PDFs.
The README is below the 6,000-character limit before references and names
Blessing Mambwe and Adeola Fafemi as contributors and Moses Olafenwa as mentor.

**Excluded.** Raw corpus text, large activation arrays, authenticated browser
state, competition email exports, course archives, local caches, media, and
secrets remain outside version control. The external Google Form remains a
manual final step because it may require personal declarations.
