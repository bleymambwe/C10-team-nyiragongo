# Problem Statement

## Latent Probing for Toxicity Detection

**Team:** Latent Lens | **Contributors:** Blessing Mambwe and Adeola Fafemi | **Mentor:** Moses Olafenwa | **Programme:** AI Saturdays Lagos, Cohort 10

## The problem

Online platforms and conversational AI systems must identify content that attacks, threatens, humiliates, or persistently harasses people. Manual review alone does not scale, while simple keyword filters are brittle: they miss implicit hostility, ignore conversational context, and may penalize quoted, reclaimed, or dialect-specific language.

This project asks a focused research question: **does a frozen language model's internal representation contain a signal that a small, interpretable classifier can use to distinguish toxic or hostile text from non-toxic text?**

## Who experiences the problem

People targeted by abuse bear the immediate safety and participation costs. Moderators face repeated exposure and inconsistent decision queues. Platform operators, community managers, and developers need reliable triage tools, while researchers and regulators need transparent evidence about failure modes. Communities whose dialects or identity terms are over-represented in toxic training examples face a distinct false-positive risk.

## Proposed technical response

We use Gemma 2 2B as a frozen representation model. Each input is tokenized to a maximum of 64 tokens. The hidden state at layer 14 is mean-pooled into a 2,304-dimensional vector, standardized, and classified with a shrinkage linear discriminant probe. The selected configuration uses shrinkage lambda 0.6 and a 12-of-17 quota ensemble.

Only the probe is fitted. The base language model is not fine-tuned. This makes the experiment cheaper to reproduce and isolates the information already present in the latent representation.

## Success criteria

The primary competition measure is classification accuracy on held-out CodaBench data. The final system achieved 89.76% development accuracy (1,526 of 1,700) and 92.35% testing accuracy (1,256 of 1,360). Source-aware leave-one-source-out evaluation achieved pooled accuracy 0.795946 and AUROC 0.832204.

Success also requires reproducibility and responsible interpretation. The repository therefore includes the end-to-end extraction and selection code, a cloud notebook, dependency versions, deterministic checks, curated result records, and the exact final submission archive.

## Boundaries

This binary English-language research prototype does not determine intent, truth, legality, or an appropriate moderation action. Labels reflect particular policies and annotator judgments; performance may vary across dialects, identity mentions, cultures, euphemisms, and conversation types. The model should support review, never act as the sole basis for sanctions. Deployment requires local validation, subgroup audits, appeals, human oversight, and drift monitoring.

**Deliverable.** The public repository reconstructs permitted sources, extracts activations, selects the probe, verifies the schema, and reproduces the competition archive. Raw text is excluded where licensing, safety, or size prevents redistribution.
