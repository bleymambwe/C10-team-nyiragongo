# C10-team-nyiragongo: Latent Probing for Toxicity Detection

**Team:** Nyiragongo
**Programme:** AI Saturdays Lagos, Cohort 10

This project tests whether a frozen language model representation contains a
useful signal for detecting toxic or hostile text with a small, interpretable
linear probe.

## Dataset

The pipeline harmonizes 13 public English-language sources. It extracted
149,528 records and retained 108,468 examples from 12 sources across nine
families after label mapping, filtering, balancing, and cross-source
deduplication. The upstream identifiers, mappings, access notes, and the
automatic loading command are in [`doc/DATASETS.md`](doc/DATASETS.md).

Raw text is not committed: the combined corpus contains harmful language and
has source-specific size, licence, and access restrictions. The repository
does include the loader, so the prepared text corpus can be rebuilt with:

```bash
python src/probe/corpus.py --out artifacts/corpus.jsonl --cap-scale 1.0
```

The competition split is supplied and scored by CodaBench, not included in
the public corpus. Final probe training uses 108,468 prepared rows. Internal
validation uses nine family-held-out folds. CodaBench development has 1,700
rows and testing has 1,360 rows; coefficients are never fitted on either.

## Training Pipeline

`src/probe/corpus.py` streams the listed datasets with Hugging Face Datasets,
maps each source to binary labels (`1` toxic/hostile, `0` non-toxic), removes
ambiguous or unusable rows, deduplicates globally, and balances each source.

The backbone is frozen `google/gemma-2-2b` (26 decoder blocks, hidden width
2,304). PyTorch and Hugging Face Transformers tokenize to 64 tokens and extract
hidden state index 14. Masked mean pooling produces one 2,304-dimensional
vector per text. The probe is a standardized shrinkage LDA head with
`lambda=0.6`, fitted by a covariance estimate and linear solve. Five shrinkage
values and other registered linear-probe recipes are compared using
family-held-out validation, then the selected head is refit once on all
prepared training rows. No model weights are updated; learning rate,
optimizer, and gradient-training epochs are not applicable.

The competition classifier ranks scores and labels the top `12/17`
(`70.588%`) of each batch positive. This is a fixed quota, not a 17-model
ensemble.

## Evaluation

| Split | Accuracy | Correct |
|---|---:|---:|
| CodaBench development | 89.76% | 1,526 / 1,700 |
| CodaBench testing | **92.35%** | **1,256 / 1,360** |

Family-held-out transfer achieved pooled accuracy `0.795946` and AUROC
`0.832204`, showing that performance is not uniform across domains. Tests
cover the model contract, deterministic extraction, label domain, source
hygiene, archive layout, quota behavior, and parity between the fitted head
and submitted classifier.

## Reproduction

Use Python 3.10+ and a CUDA-capable GPU for activation extraction. The
self-contained [`gemma_latent_toxicity_probe_competition.ipynb`](gemma_latent_toxicity_probe_competition.ipynb)
is the shortest route:

1. Install the dependencies with `pip install -r requirements.txt`.
2. Open the notebook on a GPU runtime, set an `HF_TOKEN` secret, and run all
   cells. The Hugging Face account must have accepted the Gemma licence.
3. Download `probe_outputs.zip` from the runtime and place its results under
   `artifacts/`. It contains selection reports and packaged candidates.
4. Run `python -m pytest tests -q` to verify the local implementation.

For Kaggle, the same pipeline is generated from `src/probe/` and run in two
stages because extraction is the expensive step:

```bash
python run_kaggle.py extract --cap-scale 1.0
python run_kaggle.py status extract
python run_kaggle.py select --strategy default
python run_kaggle.py pull select --out artifacts/kaggle
```

The exact final archive is [`artifacts/kaggle/select/submissions/best_single.zip`](artifacts/kaggle/select/submissions/best_single.zip).
SHA-256: `0bd11ac11c0a09a8b38d834e06c558f0e17236dae680c215a5e33af8c746a601`.

## Appendix

Required Cohort 10 challenge documents are in [`doc/`](doc/):
`problem_statement.pdf`, `data_card.pdf`, `impact_statement_card.pdf`, and
`stakeholder_engagement.pdf`. Editable challenge sources are in
[`docs/challenges/`](docs/challenges/).

Contributors: Blessings Mambwe (team leader), Fafemi Adeola, Musonda Musunga,
and Hamna Kaleem (research and challenge collaborator). Mentor: Moses.
All four contributors receive equal credit. Hamna also belongs to another
team; her work here was limited to shared research and Cohort Challenge work
and does not imply exclusive team membership.

Use this research prototype for benchmarking or moderation support with human
review, local validation, subgroup audits, appeals, and drift monitoring. Do
not use it as the sole basis for sanctions or access decisions.

## References

1. Team Gemma. *Gemma 2: Improving Open Language Models at a Practical Size* (2024).
2. Borkan et al. *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification* (2019).
3. Gehman et al. *RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models* (2020).
4. Rottger et al. *HateCheck: Functional Tests for Hate Speech Detection Models* (2021).
5. Lin et al. *ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation* (2023).
