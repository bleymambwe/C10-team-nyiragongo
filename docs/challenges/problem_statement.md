# Problem Statement

## Latent Probing for Toxicity Detection

**Team:** Latent Lens | **Programme:** AI Saturdays Lagos, Cohort 10

## The problem

Online platforms and conversational AI systems must identify attacks, threats, humiliation, and persistent harassment. Manual review does not scale, while keyword filters miss implicit hostility and may penalize quoted, reclaimed, or dialect-specific language.

This project asks a focused research question: **does a frozen language model's internal representation contain a signal that a small, interpretable classifier can use to distinguish toxic or hostile text from non-toxic text?**

## Who experiences the problem

Targets of abuse bear immediate safety and participation costs. Moderators face repeated exposure and inconsistent queues. Operators need reliable triage, while researchers and regulators need transparent evidence about failures. Communities whose dialects or identity terms are over-represented in toxic examples face a distinct false-positive risk.

## Proposed technical response

We use `google/gemma-2-2b`, a decoder-only Transformer with 26 decoder blocks and hidden width 2,304, as a frozen representation model. PyTorch and Hugging Face Transformers tokenize each input to at most 64 tokens and extract hidden state index 14. Masked mean pooling produces one 2,304-dimensional vector per text. NumPy/scikit-learn-compatible code standardizes the vector and fits one shrinkage linear discriminant head with lambda 0.6.

The decision rule ranks scores and labels the top 12/17 of each competition batch positive. This is a quota, not a 17-model ensemble. Gemma is not fine-tuned. Shrinkage LDA is fitted by a covariance estimate and linear solve, so there is no learning rate, optimizer, or gradient-training epoch count. Five LDA shrinkage values and other registered linear-probe recipes are compared over nine family-held-out validation folds; the selected head is refitted once on all 108,468 prepared examples.

## Success criteria

The primary competition measure is classification accuracy on held-out CodaBench data. The final system achieved 89.76% development accuracy (1,526 of 1,700) and 92.35% testing accuracy (1,256 of 1,360). Source-aware leave-one-source-out evaluation achieved pooled accuracy 0.795946 and AUROC 0.832204.

The repository includes extraction and selection code, a cloud notebook, dependencies, deterministic checks, score records, and the exact final archive.

## Boundaries

This binary English-language research prototype does not determine intent, truth, legality, or an appropriate moderation action. Labels reflect particular policies and annotator judgments; performance may vary across dialects, identity mentions, cultures, euphemisms, and conversation types. The model should support review, never act as the sole basis for sanctions. Deployment requires local validation, subgroup audits, appeals, human oversight, and drift monitoring.

**Deliverable.** The public repository reconstructs permitted sources, extracts activations, selects the probe, verifies the schema, and reproduces the competition archive. Raw text is excluded where licensing, safety, or size prevents redistribution.

## Team and contacts

**Mentor:** Moses (guymodscientist@gmail.com). **Team leader:** Blessings Mambwe (bleymambwe@gmail.com). **Team members:** Fafemi Adeola (adeola5678@gmail.com), Musonda Musunga (mamusonda@gmail.com), and research/challenge collaborator Hamna Kaleem (hamnanmah@gmail.com).

All four team contributors receive equal contribution credit. Hamna is also in another team; her Team Latent Lens role covered shared research and Cohort Challenge work and does not imply exclusive membership.
