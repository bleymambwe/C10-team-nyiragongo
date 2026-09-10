# Project Summary

## Latent Probing for Toxicity Detection

Team Latent Lens studies whether a compact probe can recover a binary toxicity signal from the hidden representations of a frozen language model.

The competition pipeline uses Gemma 2 2B layer-14 mean-pooled activations, standardization, and a shrinkage linear discriminant probe. The selected configuration achieved 89.76% accuracy on the CodaBench development split and 92.35% on the testing split.

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

- Blessing Mambwe
- Adeola Fafemi
- Mentor: Moses Olafenwa
