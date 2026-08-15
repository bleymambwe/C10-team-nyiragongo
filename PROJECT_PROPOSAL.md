# Project Proposal

## Title

Latent Probing for Toxicity

## Date

2026-08-15

## Objective

Build a small, reproducible software package for studying whether a simple linear probe can recover a narrowly defined toxicity label from hidden representations of a frozen language model.

The immediate target is a Gemma-based workflow that:

- loads a frozen causal language model,
- extracts hidden states layer by layer,
- applies a simple pooling rule to obtain per-example representations,
- trains linear probes on those representations,
- compares layerwise results against basic controls,
- records enough metadata to support reproducibility and later audit.

## Research question

The software is intended to support the narrow question:

Can a simple linear classifier predict a small, explicitly defined toxicity label from internal vectors of a frozen language model?

## Intended software scope

The planned software will eventually provide:

- deterministic data preparation for a small labelled demonstration dataset,
- hidden-state extraction from a frozen model,
- layerwise linear-probe training,
- baseline and shuffled-label controls,
- metric reporting,
- artifact export for fitted probes and metadata,
- simple audit outputs describing inputs, configuration, and results.

## Non-goals

This project does not currently claim to provide:

- a production moderation system,
- a causal account of toxic behaviour,
- fairness or deployment readiness,
- benchmark-grade empirical conclusions,
- a proof that any decodable direction is used by the model mechanistically.

## Publication scope for this repository

This public repository is intentionally limited to proposal-stage materials and audit scaffolding.

It does not publish:

- local research archives,
- rendered videos,
- audio files,
- course materials,
- unrelated papers or PDFs,
- large experimental artifacts,
- notebooks or media not explicitly curated for public release.

## Initial audit requirements

Future public software releases should preserve:

- model identifier and revision,
- dependency versions,
- pooling rule and probe configuration,
- train/test split policy,
- random seed configuration,
- control definitions,
- artifact manifest for outputs included in a release.

## Expected next repository milestones

1. Define a stable package layout.
2. Separate reusable code from notebook-only experimentation.
3. Add reproducible configuration files and testable entry points.
4. Add release-safe example artifacts.
5. Add JOSS paper assets once the software scope is stable.
