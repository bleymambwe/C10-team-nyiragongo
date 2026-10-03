# Competition objective and iteration loop

## North-star objective

Maximize hidden-test generalization—not public-leaderboard luck—using a valid,
portable linear-probe submission. Optimize a balanced scorecard so gains in one
metric do not hide regressions in another.

## Success criteria

1. **Leakage-safe validation:** group by author, conversation, source, or template;
   keep a final untouched holdout when the data volume permits it.
2. **Metric breadth:** track AUROC, average precision, balanced accuracy, F1, MCC,
   confusion matrices, and subgroup/domain results.
3. **Robustness:** report repeated-fold mean and variance, seed variance, label-shuffle
   controls, and performance on minimal pairs and shifted domains.
4. **Model diversity:** combine only complementary linear probes whose out-of-fold
   errors differ; never add a model merely because it is different.
5. **Threshold discipline:** choose the operating threshold exclusively from
   out-of-fold predictions and align its objective with the competition metric.
6. **Submission integrity:** enforce feature width and binary output, train only
   offline, keep both required files at the ZIP root, and smoke-test isolated import.

## Elite iteration loop

1. Freeze a split manifest and duplicate/group audit.
2. Establish majority, lexical, and single-logistic baselines.
3. Extract identical Gemma representations under a pinned model/tokenizer/layer/
   pooling contract; cache them by content hash.
4. Compare layer and pooling choices using inner validation only. Never use public or
   final test feedback to choose them.
5. Fit the diverse ensemble in `competition_probe.py`; inspect per-model OOF scores,
   error correlation, weights, and threshold.
6. Stress-test nuisance features, quoted toxicity, negation, profanity without abuse,
   implicit threats, paraphrases, identity mentions, and domain shift.
7. Change one hypothesis at a time and retain it only when repeated grouped CV improves
   both central performance and worst-group performance.
8. Retrain on all permitted labelled data, create the ZIP, run the platform-interface
   smoke test, and record code/data/model hashes.

## Honest constraint

No technique guarantees first place. The notebook's synthetic 32-row training set is
for teaching and must be replaced with the official labelled embedding data. The main
competitive edge will come from representative data, correct groups, and validation
that matches the hidden test—not from adding unconstrained classifier complexity.
