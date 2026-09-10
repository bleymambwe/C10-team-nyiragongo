# Project Summary

## Latent Probing for Toxicity Detection

Team Latent Lens studies whether a compact probe can recover a binary toxicity signal from the hidden representations of a frozen language model.

The competition pipeline uses a frozen `google/gemma-2-2b` decoder-only Transformer, hidden-state index 14 masked-mean activations, standardization, and one shrinkage linear discriminant head. The selected configuration uses lambda 0.6 and a top-12/17 quota. Because LDA is fitted analytically, it has no learning rate, optimizer, or gradient-training epochs. The head was refitted once on 108,468 prepared rows after nine family-held-out validation folds. It achieved 89.76% accuracy on the 1,700-row CodaBench development split and 92.35% on the 1,360-row testing split.

The public release contains:

- reproducible data extraction and cleaning code;
- activation extraction and source-aware selection code;
- a cloud notebook and pinned dependency ranges;
- the exact final competition archive and verified score records;
- automated schema, parity, routing, and probe tests; and
- the four required Cohort 10 challenge documents.

Raw text, activation matrices, course exports, authenticated browser data, and large research media remain outside version control because of licensing, safety, privacy, or size.

## Scope

This is a research and benchmarking prototype, not a production moderation service. A binary toxicity label cannot fully represent intent, quotation, counterspeech, severity, target, culture, or policy. Operational use would require local validation, subgroup audits, calibrated thresholds, human review, appeals, security controls, and drift monitoring.

## Team

- Team leader: Blessings Mambwe - bleymambwe@gmail.com
- Fafemi Adeola - adeola5678@gmail.com
- Musonda Musunga - mamusonda@gmail.com
- Hamna Kaleem - hamnanmah@gmail.com
- Mentor: Moses - guymodscientist@gmail.com

All four team contributors receive equal contribution credit. Hamna Kaleem is also in another team; her work with Team Latent Lens was limited to shared research and Cohort Challenge activities.
