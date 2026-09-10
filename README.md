# Latent Probing for Toxicity Detection

**Team Latent Lens submission - AI Saturdays Lagos, Cohort 10**

This project detects toxic or hostile text by probing hidden representations from Gemma 2 2B. It converts an intermediate transformer activation into a compact linear decision rule, providing a reproducible baseline for studying what language models encode about toxicity.

## Results

| Split | Accuracy | Correct |
|---|---:|---:|
| CodaBench development | 89.76% | 1,526 / 1,700 |
| CodaBench testing | **92.35%** | **1,256 / 1,360** |

The final submitted artifact is [`artifacts/kaggle/select/submissions/best_single.zip`](artifacts/kaggle/select/submissions/best_single.zip). Its SHA-256 digest is `0bd11ac11c0a09a8b38d834e06c558f0e17236dae680c215a5e33af8c746a601`.

## Problem and approach

Online abuse is contextual, culturally variable, and costly to moderate at scale. Keyword filters miss implicit hostility and can over-flag reclaimed or quoted language. We test whether a frozen language model's latent state contains a useful toxicity signal.

The pipeline tokenizes each text to at most 64 tokens, extracts the mean-pooled hidden state from Gemma 2 2B layer 14, standardizes the 2,304-dimensional representation, and fits a shrinkage linear discriminant probe. The selected configuration uses shrinkage `lambda = 0.6` and a 12-of-17 quota ensemble.

## Dataset

We harmonize 149,528 records from 13 public sources, then clean and deduplicate them to 108,468 examples spanning 12 sources and nine dataset families. Sources include Civil Comments, toxic/offensive Twitter corpora, Wikipedia talk-page attacks, Berkeley D-Lab hate speech, RealToxicityPrompts, Aegis Safety, Offensive Language, HateCheck, and ToxicChat.

Labels are mapped to a binary task: `1` for toxic/hostile and `0` for non-toxic. Raw text is not committed because of licensing, size, and safety constraints; the extraction command downloads or reconstructs the permitted sources. See the [Data Card](docs/data_card.pdf) for provenance, risks, and exclusions.

## Training pipeline

1. `extract` loads, normalizes, and deduplicates the public datasets.
2. Gemma 2 2B produces layer-14 mean-pooled activations.
3. `select` compares reproducible linear-probe configurations.
4. The selected model produces the exact CodaBench submission archive.

The language model remains frozen; only the compact probe is fitted. Fixed seeds, saved configuration, parity checks, and archive hashing support reproducibility.

## Evaluation and quality assurance

Selection uses source-aware validation rather than relying only on a random split. Leave-one-source-out evaluation achieved pooled accuracy `0.795946` and AUROC `0.832204`, exposing domain-transfer weaknesses that a single aggregate score could hide. The repository also checks schema, row count, label domain, deterministic selection, archive integrity, and parity between notebook and script outputs.

Accuracy is the official leaderboard metric. It should be read alongside per-source behavior and false-positive/false-negative analysis; the system is a research prototype, not an autonomous moderation decision-maker.

## Reproduction

Python 3.10+ and a CUDA-capable GPU are recommended for activation extraction.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_kaggle.py extract --cap-scale 1.0
python run_kaggle.py status extract
python run_kaggle.py select --strategy default
python run_kaggle.py pull select --out artifacts/kaggle
python -m pytest tests -q
```

For a guided cloud run, open [`gemma_latent_toxicity_probe_competition.ipynb`](gemma_latent_toxicity_probe_competition.ipynb). Access to the gated Gemma checkpoint may require accepting its Hugging Face licence and supplying a personal token through the runtime's secret manager; never commit tokens.

## Repository guide

- `src/probe/` - corpus, embedding, probe, and selection code
- `run_kaggle.py` - reproducible orchestration entry point
- `experiments/` - transfer and model-selection experiments
- `tests/` - competition and submission-parity checks
- `docs/` - the four required Cohort Challenge documents
- `artifacts/` - curated score records and final submission archive

## Responsible use

Toxicity labels reflect dataset policies, annotator judgments, and social context. Performance can degrade across dialects, languages, identities, and emerging euphemisms. Use human review, appeals, subgroup audits, drift monitoring, and locally agreed thresholds. Do not use this prototype as the sole basis for sanctions or access decisions.

## Appendix: contributors and mentor

- **Blessing Mambwe** - research, implementation, experimentation, evaluation, and submission engineering
- **Adeola Fafemi** - behavioural-science research, benchmark synthesis, quality assurance, and stakeholder framing
- **Moses Olafenwa** - assigned Cohort 10 mentor

Required challenge documents: [Problem Statement](docs/problem_statement.pdf), [Data Card](docs/data_card.pdf), [Impact Statement Card](docs/impact_statement_card.pdf), and [Stakeholder Engagement](docs/stakeholder_engagement.pdf).

## References

1. Team Gemma. *Gemma 2: Improving Open Language Models at a Practical Size* (2024).
2. Borkan et al. *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification* (2019).
3. Gehman et al. *RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models* (2020).
4. Rottger et al. *HateCheck: Functional Tests for Hate Speech Detection Models* (2021).
5. Lin et al. *ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation* (2023).
