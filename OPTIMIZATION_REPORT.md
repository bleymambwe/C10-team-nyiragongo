# Competition Optimization Report

## Reproduction and rules audit

The reproduced `v006` ZIP has byte-identical `classifier.py` and
`trained_probe.joblib` contents to the saved artifact. Its contract is Gemma-2-2B,
hidden state 14, masked mean over all non-padding tokens, maximum length 64, and
2,304 input features. The saved model is the `v005` Civil Comments toxicity
ensemble: three binary logistic heads, full or 1,024 univariate F-score-selected
features, optional row L2 normalization, and weights 0.6375/0.1125/0.25.

The name `rank1200` is a prediction quota, not a feature rank. For a batch of
1,700 rows, v006 labels the top 1,200 ensemble scores as toxic. The competition
page confirms that official scoring always supplies layer-14 all-token means; a
layer-0/1/2 submission is therefore not a valid alternative representation for
the official interface. The page also states that the metric is accuracy, the
ZIP must contain `classifier.py` and `trained_probe.joblib` at its root, no
platform training is allowed, and the final phase allows one submission.

## What caused the 0.70 to 0.77 jump

This is a controlled one-variable ablation. v005 and v006 contain the same
serialized model bytes. Their prediction files differ only by 320 labels:
880 positives in v005 versus 1,200 in v006, with no positive-to-negative flips.
Using the known development composition (1,200 positive and 500 negative rows),
the confusion counts are:

| submission | TP | FP | FN | TN | correct | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| v005 threshold | 789 | 91 | 411 | 409 | 1,198 | 0.704706 |
| v006 top-1,200 | 1,002 | 198 | 198 | 302 | 1,304 | 0.767059 |

Of the 320 newly positive rows, 213 are positive and 107 negative, so the quota
adds 106 net correct predictions. The gain is prior/ranking calibration under
the development distribution, not a layer change, PCA rank change, pooling
change, or classifier change. The earlier leaderboard label “layer 1 + rank
1200” is inconsistent with the artifact and official task contract.

## Existing evidence table

See `results/optimization/historical_results.csv` for the append-only table.
The only genuine local OOF values currently available are the reports embedded
in v004/v005; no official layer/rank embedding tensor is present in this
workspace. The v005 0.8359 OOF value is from its deterministic 20% hash split
with a target-prior reweight, not repeated grouped CV. It must not be treated as
evidence that a hidden-test score above 0.83 is expected.

## Reproducible optimizer

`optimize_probe.py` accepts an `.npz` bundle with `X`, `y`, and optional
`texts`/`groups`. It evaluates fold-local transformations and probes, persists
`results.csv`, `oof_predictions.npz`, and `manifest.json`, and reports mean CV,
standard deviation, worst fold, OOF accuracy, AUROC, balanced accuracy, F1,
threshold, and duplicate-group count. Example:

```powershell
python optimize_probe.py embeddings_bundle.npz --out results/optimization/run01 --folds 5 --seeds 17,29,41
```

After running the existing Gemma extractor, combine its arrays with:

```powershell
python make_bundle.py artifacts/v005/gemma2_layer14_mean64_50000.npy artifacts/v005/toxicity_labels_50000.npy embeddings_bundle.npz
```

For a multi-layer audit tensor shaped `[n, layers, 2304]`:

```powershell
python optimize_probe.py multilayer_bundle.npz --layers 0,1,2,3,4,5,6 --out results/optimization/layers
```

The supervised selectors and every scaler/PCA are fitted only on the fold's
training rows. Near-duplicate review still requires a text bundle; exact
duplicates are automatically grouped by normalized-text SHA-1 when `texts` are
provided.

## Submission portfolio

`build_portfolio.py` creates four root-valid ZIPs from the exact v005 weights:

| candidate | policy | intended use |
|---|---|---|
| candidate A | top 1,200 / 1,700 | exact reproduced 0.767 dev winner |
| candidate B | top 1,250 / 1,700 | modestly higher positive prior |
| candidate C | top 1,292 / 1,700 | v007's unverified dev hypothesis |
| candidate D | original v005 threshold | calibration-control baseline |

All four smoke-test successfully and contain exactly the two required files.
Candidate A is the first submission recommendation because it is the only one
with a measured leaderboard result. Candidates B/C are deliberately portfolio
variants, not claims of improvement. Candidate D is useful only as a control.

The current machine has 8 GiB RAM, 4 logical CPUs, no CUDA device, no local
Gemma weights, and no cached challenge embeddings. Therefore a real layer/rank,
pooling, repeated-CV, error, stress, or seed sweep cannot be honestly reported
from this workspace. Run the optimizer after producing a bundle on a GPU/Colab
runtime; do not download or infer hidden/reference labels from CodaBench.
