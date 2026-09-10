# Data Card

## Latent Probing for Toxicity Detection

**Team:** Latent Lens | **Contributors:** Blessing Mambwe and Adeola Fafemi | **Mentor:** Moses Olafenwa

## Dataset summary

The project builds a harmonized binary toxicity corpus from public research datasets. Extraction produced 149,528 records from 13 sources. Cleaning, normalization, filtering, and cross-source deduplication retained 108,468 examples from 12 sources across nine dataset families.

The principal source counts after preparation were:

- Civil Comments: 26,218
- Twitter toxicity/offensiveness corpora: 27,572
- Wikipedia talk-page attacks: 11,314
- Berkeley D-Lab hate-speech data: 16,000
- RealToxicityPrompts: 14,000
- Aegis Safety: 6,152
- Offensive Language: 3,496
- HateCheck: 2,326
- ToxicChat: 1,390

Counts describe the reproducible prepared corpus used by this project and may differ from the full upstream datasets because of caps, filtering, unavailable rows, or deduplication.

## Intended task and labels

The intended task is binary text classification. Label `1` means the upstream annotation or policy mapping indicates toxic, hostile, hateful, threatening, abusive, or offensive content. Label `0` means the mapped example is non-toxic for this task.

Because upstream taxonomies differ, the binary mapping compresses distinctions such as hate speech, insult, threat, profanity, harassment, and unsafe conversational content. It must not be interpreted as a universal definition of toxicity.

## Collection and processing

The repository's extraction pipeline retrieves or reconstructs datasets from their permitted public sources, maps schemas to a common text-and-label format, removes unusable rows, normalizes fields, and deduplicates records. Fixed seeds and source identifiers support repeatability and source-aware evaluation.

Inputs are tokenized with the Gemma tokenizer and truncated to 64 tokens. The released submission does not include raw activation matrices or the full text corpus. Those artifacts are large, may contain harmful language, and can have redistribution limits.

## Splits and evaluation

Competition development and testing sets are evaluated by CodaBench using accuracy. For internal robustness checks, source membership is retained and leave-one-source-out evaluation tests transfer to a held-out data source. The pooled transfer result was accuracy 0.795946 and AUROC 0.832204, lower than the final testing score and therefore an important caution against assuming uniform domain performance.

## Known risks and limitations

- Toxicity is subjective and policy-dependent; annotators may disagree.
- Identity terms can become spurious shortcuts and create false positives.
- English-heavy sources under-represent many languages and cultural contexts.
- Platform-specific language, sampling, and annotation procedures cause domain shift.
- A binary label hides severity, target, intent, quotation, counterspeech, and context.
- Truncation can remove relevant context from long messages.
- Public datasets may contain personal information or distressing content despite upstream controls.

## Access, licensing, and governance

Users must follow every upstream dataset's licence, terms, and access conditions. Raw datasets are not redistributed in this repository. Where access permits, scripts automate retrieval; otherwise the README documents the required upstream step. Gemma access is governed by its model licence and may require the user to accept terms through Hugging Face.

Do not commit access tokens, local caches, raw text exports, or embeddings. Restrict access to sensitive working data, minimize retention, and delete caches when no longer required.

## Recommended use

Use this corpus for research, benchmarking, and supervised moderation support with human review. Before operational use, evaluate local data, disaggregated error rates, dialect and identity-term slices, emerging vocabulary, and false-positive/false-negative costs. Provide an appeals path and document threshold changes.

## Maintenance

Record upstream versions, extraction date, checksum, mapping changes, and known exclusions for each rebuild. Re-run source-aware tests when the corpus, tokenizer, model, or moderation policy changes. Treat drift and stakeholder feedback as reasons to revise the dataset rather than merely recalibrate a threshold.
