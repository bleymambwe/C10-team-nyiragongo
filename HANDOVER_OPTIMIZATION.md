# Latent Probe Toxicity Optimization Handover

Updated: 2026-09-07

## Mission

Improve hidden-test accuracy for the Latent Probe Challenge without using hidden
labels, reference data, leaderboard labels, or platform-side training. Select
models with leakage-safe out-of-fold (OOF) validation, then package a single
competition-valid ZIP for the final phase.

## Competition contract

The official [Codabench competition page](https://www.codabench.org/competitions/17670/)
defines the input and submission contract:

- Model: `google/gemma-2-2b`.
- Input: 2,304-dimensional vectors from hidden-state index 14.
- Pooling: mean over all non-padding tokens.
- Tokenization: truncation at 64 tokens.
- Metric: classification accuracy.
- Submission root: exactly `classifier.py` and `trained_probe.joblib`.
- Training must happen offline; the platform only calls `Classifier.predict`.
- Development phase permits repeated submissions; the final phase permits one.

The competition page calls the positive class a “mental health distress signal”
in one paragraph even though the title says toxicity. Empirical evidence favors
toxicity labels: Civil Comments training produced 0.7047, while the
MentalHealth-Darija model produced 0.42 on the development set.

Do not submit a layer-0/1/2 model as an official alternative. The platform
always supplies layer-14 vectors. Multi-layer inputs are useful only for offline
diagnostics if they are available.

## Repository map

- `train_probe_v004.py`: original extractor and portable probe fitter.
- `train_probe_v005.py`: Civil Comments toxicity training path.
- `build_probe_v006.py`: exact top-1,200 prediction-quota wrapper.
- `build_probe_v007.py`: unsubmitted transductive rank/centroid experiment.
- `competition_probe.py`: broader linear ensemble implementation.
- `optimize_probe.py`: new leakage-safe CV optimizer.
- `make_bundle.py`: combines `.npy` features and labels into optimizer input.
- `build_portfolio.py`: creates four quota/threshold candidate ZIPs.
- `results/optimization/historical_results.csv`: append-only historical table.
- `OPTIMIZATION_REPORT.md`: detailed audit and evidence.
- `artifacts/v005/`: measured Civil Comments model and 0.7047 result.
- `artifacts/v006/`: measured 0.7671 result.
- `artifacts/portfolio/`: handoff candidate ZIPs.

Never read or print `BMToken.env`; it contains authentication material.

## Exact v006 finding

`v006` and `v005` contain byte-identical `trained_probe.joblib` files. The
learned model is unchanged:

- Three binary logistic heads.
- Full 2,304 features for two heads.
- 1,024 F-score-selected features for one head.
- Standard scaling; one head also uses row L2 normalization.
- Head weights: `0.6375`, `0.1125`, `0.25`.
- v005 threshold: approximately `0.374219`.

The only v006 change is prediction policy: label the highest 1,200 scores in a
1,700-row batch as positive. The controlled development ablation is:

| Version | Positive predictions | TP | FP | FN | TN | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| v005 threshold | 880 | 789 | 91 | 411 | 409 | 0.704706 |
| v006 top-1,200 | 1,200 | 1,002 | 198 | 198 | 302 | 0.767059 |

The 320 newly positive rows contain 213 positives and 107 negatives. The gain
is prior calibration, not PCA rank reduction, a layer change, pooling, or a
new classifier. In this repository, “rank1200” means a prediction quota. In
`optimize_probe.py`, `Config.rank` means PCA dimensionality. Keep those terms
separate.

## Current deliverables

Recommended measured candidate:

`artifacts/portfolio/toxicity_probe_candidate_a_v006_rank1200.zip`

Other portfolio controls:

- `toxicity_probe_candidate_b_rank1250.zip`
- `toxicity_probe_candidate_c_rank1292.zip`
- `toxicity_probe_candidate_d_threshold_v005.zip`

All ZIPs contain exactly the required two root files and pass isolated smoke
tests. Candidate A embeds the same classifier and model bytes as v006.

## Running the real optimizer

The workspace does not contain challenge embeddings or local Gemma weights. A
GPU/Colab run must first produce features and labels. The existing extractor
requires the Hugging Face token in `BMToken.env` and writes an embedding `.npy`
plus a labels `.npy` file:

```powershell
python train_probe_v005.py --output-dir artifacts/v005 --batch-size 8
```

Combine them into a bundle:

```powershell
python make_bundle.py `
  artifacts/v005/gemma2_layer14_mean64_50000.npy `
  artifacts/v005/toxicity_labels_50000.npy `
  embeddings_bundle.npz
```

Run a staged search:

```powershell
python optimize_probe.py embeddings_bundle.npz `
  --out results/optimization/run01 `
  --folds 5 `
  --seeds 17,29,41
```

For a multi-layer diagnostic tensor shaped `[n, layers, 2304]`:

```powershell
python optimize_probe.py multilayer_bundle.npz `
  --layers 0,1,2,3,4,5,6 `
  --out results/optimization/layers
```

The runner writes `results.csv`, `oof_predictions.npz`, and `manifest.json`.
It supports fold-local standard/robust scaling, row L2 normalization, PCA
ranks, top-PC removal, ANOVA or mutual-information selection, logistic/SVM/
ridge/shrinkage-LDA probes, thresholds, duplicate grouping, and repeated CV.

## Validation requirements

Use 5 folds and at least 3 seeds for the first pass; use 5 seeds for finalists.
If texts or author/template IDs exist, pass groups. Exact duplicate groups are
automatically created from normalized text when `texts` are included in the
bundle. Inspect near duplicates separately before trusting CV.

Select by OOF accuracy first, then require:

- low fold standard deviation;
- strong worst-fold accuracy;
- a stable threshold across seeds;
- no large train/OOF gap;
- no collapse on profanity-only benign text, implicit toxicity, negation,
  quoted text, identity language, long/short examples, and low-confidence rows.

Do not select a model from one lucky split. Do not fit PCA, scaling, or feature
selection before the fold split. Do not use development leaderboard scores as
training labels.

## Suggested experiment order

1. Reproduce the v005 linear ensemble on the new bundle.
2. Compare logistic C values over `1e-4` through `1e3`.
3. Compare PCA ranks 256, 512, 768, 1,024, 1,200, 1,536, and full.
4. Compare no scaling, standard scaling, robust scaling, and row L2.
5. Compare top-PC removal counts 1, 2, 4, 8, 16, and 32.
6. Compare ANOVA selection against unsupervised PCA.
7. Evaluate SVM, ridge, and shrinkage LDA only after the logistic baseline.
8. Build OOF error correlations before adding an ensemble member.
9. Tune a final threshold from OOF predictions only.
10. Retrain selected configurations on all permitted labelled rows and smoke-test
    the ZIP in an isolated directory.

Pooling alternatives and layer sweeps require token-level or multi-layer
embeddings. They cannot be recovered from the existing 2-D v005 artifact.

## Verification already run

```powershell
python -m py_compile optimize_probe.py make_bundle.py build_portfolio.py
python build_probe_v006.py
python build_portfolio.py
python -m pytest -q tests
```

The project test suite passes 8 tests. Repository-wide pytest also discovers an
unrelated pre-existing import-name collision between two research folders,
both containing `test_diagnostics.py`; do not treat that as a probe failure.

## Handoff decision

Submit Candidate A first if a development-phase submission is needed. It is the
only candidate with a measured leaderboard result. Candidate B/C are quota
sensitivity controls, not validated improvements. Candidate D is the original
threshold baseline.

Before the final one-submission phase, replace this recommendation only when a
real grouped/repeated OOF run shows a stable improvement and the candidate
remains compliant with the fixed layer-14 input contract.
