# Latent Probing for Toxicity Detection

**Team Latent Lens submission - AI Saturdays Lagos, Cohort 10**

This project detects toxic or hostile text by probing hidden representations from a frozen Gemma model. It provides a reproducible, compact baseline for studying what language models encode about toxicity.

## Results

| Split | Role | Accuracy | Correct |
|---|---|---:|---:|
| CodaBench development | Candidate comparison | 89.76% | 1,526 / 1,700 |
| CodaBench testing | Final held-out evaluation | **92.35%** | **1,256 / 1,360** |

Final artifact: [`artifacts/kaggle/select/submissions/best_single.zip`](artifacts/kaggle/select/submissions/best_single.zip). SHA-256: `0bd11ac11c0a09a8b38d834e06c558f0e17236dae680c215a5e33af8c746a601`.

## Model architecture and framework

- **Backbone:** `google/gemma-2-2b`, a frozen decoder-only Gemma 2 Transformer with 26 decoder blocks and hidden width 2,304.
- **Framework:** PyTorch and Hugging Face Transformers for tokenization and activation extraction; NumPy/scikit-learn-compatible code for probe selection and fitting.
- **Representation:** masked mean of hidden state index 14 (the input to decoder block 14), with inputs truncated to 64 tokens.
- **Probe:** one standardized linear discriminant head with covariance shrinkage `lambda = 0.6`.
- **Decision rule:** rank scores and label the top `12/17` (70.588%) of each competition batch positive. This is a quota rule, not a 17-model ensemble.

Gemma's weights remain frozen. Shrinkage LDA uses a closed-form covariance estimate and linear solve, so **learning rate, optimizer, and gradient-training epochs are not applicable**. Candidate selection evaluates five LDA shrinkage values (`0.02`, `0.1`, `0.3`, `0.6`, `0.9`) and other registered linear-probe recipes. The selected probe is then fitted once on all 108,468 prepared training examples.

## Dataset and splits

We extracted 149,528 records from 13 public sources and retained 108,468 examples from 12 sources across nine dataset families after cleaning and source hygiene. Sources include Civil Comments, Twitter toxicity/offensiveness corpora, Wikipedia attacks, Berkeley hate speech, RealToxicityPrompts, Aegis Safety, Offensive Language, HateCheck, and ToxicChat. Labels are binary: `1` toxic/hostile and `0` non-toxic.

| Stage | Rows | Use |
|---|---:|---|
| Final probe training | 108,468 | Fit standardization and one LDA head |
| Internal validation | 9 family-held-out folds | Select robust recipe; each selection training fold capped at 40,000 rows |
| CodaBench development | 1,700 | Compare packaged candidates; never fit coefficients |
| CodaBench testing | 1,360 | Final evaluation only |

Raw text and activations are excluded because of licensing, size, and safety constraints. The extraction command downloads or reconstructs permitted sources. See the [Data Card](docs/data_card.pdf).

## Evaluation and quality assurance

Family-held-out evaluation achieved pooled accuracy `0.795946` and AUROC `0.832204`, showing that domain transfer is harder than the competition split. Checks cover model contract, deterministic extraction, label domain, row count, source hygiene, archive contents, quota behavior, and parity between the fitted head and submitted classifier.

Accuracy is the official competition metric. It should be read with per-source errors and false-positive/false-negative analysis.

## Reproduction

Python 3.10+ and a CUDA-capable GPU are recommended. Activation extraction uses batch size 64.

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

The guided notebook is [`gemma_latent_toxicity_probe_competition.ipynb`](gemma_latent_toxicity_probe_competition.ipynb). Gemma access may require accepting its licence and supplying a personal Hugging Face token through the runtime secret manager. Never commit tokens.

## Responsible use

Toxicity labels reflect dataset policies, annotator judgments, and context. Performance can degrade across dialects, languages, identities, and emerging euphemisms. Use human review, appeals, subgroup audits, drift monitoring, and locally agreed thresholds. Never use this prototype as the sole basis for sanctions or access decisions.

## Team, mentor, and contribution statement

- **Team leader:** Blessings Mambwe - <bleymambwe@gmail.com>
- **Team member:** Fafemi Adeola - <adeola5678@gmail.com>
- **Team member:** Musonda Musunga - <mamusonda@gmail.com>
- **Research and challenge collaborator:** Hamna Kaleem - <hamnanmah@gmail.com>
- **Mentor:** Moses - <guymodscientist@gmail.com>

All four team contributors receive equal contribution credit for this submission. Hamna Kaleem is also a member of another team; her participation with Team Latent Lens was limited to the shared research and Cohort Challenge work and does not imply exclusive team membership.

Required documents: [Problem Statement](docs/problem_statement.pdf), [Data Card](docs/data_card.pdf), [Impact Statement Card](docs/impact_statement_card.pdf), and [Stakeholder Engagement](docs/stakeholder_engagement.pdf).

## References

1. Team Gemma. *Gemma 2: Improving Open Language Models at a Practical Size* (2024).
2. Borkan et al. *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification* (2019).
3. Gehman et al. *RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models* (2020).
4. Rottger et al. *HateCheck: Functional Tests for Hate Speech Detection Models* (2021).
5. Lin et al. *ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation* (2023).
