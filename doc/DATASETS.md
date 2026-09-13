# Dataset Access Manifest

The project does not redistribute the combined raw corpus. The source corpora
contain harmful language and have different licences, terms, and access
conditions. `src/probe/corpus.py` loads them on demand with the Hugging Face
Datasets streaming API and records the source name on every row.

Run from the repository root:

```bash
pip install -r requirements.txt
python src/probe/corpus.py --out artifacts/corpus.jsonl --cap-scale 1.0
```

The command may require an accepted upstream licence or authentication. Check
each upstream dataset card before downloading, redistributing, or using the
data outside this project. Do not commit raw text, caches, access tokens, or
activation matrices.

| Loader | Upstream dataset | Split/config | Mapping used here |
|---|---|---|---|
| `civil_comments` | [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | `train` | toxicity >= 0.5 -> 1; <= 0.05 -> 0 |
| `jigsaw_unintended` | [mteb/toxic_conversations_50k](https://huggingface.co/datasets/mteb/toxic_conversations_50k) | `train` | upstream binary label |
| `jigsaw_wiki` | [tasksource/jigsaw_toxicity](https://huggingface.co/datasets/tasksource/jigsaw_toxicity) | `train` | upstream `toxic` label |
| `davidson` | [tdavidson/hate_speech_offensive](https://huggingface.co/datasets/tdavidson/hate_speech_offensive) | `train` | hate/offensive -> 1; neither -> 0 |
| `tweeteval_offensive` | [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | `offensive` | upstream binary label |
| `tweeteval_hate` | [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | `hate` | upstream binary label |
| `hatecheck` | [Paul/hatecheck](https://huggingface.co/datasets/Paul/hatecheck) | `test` | `hateful` -> 1; other -> 0 |
| `toxic_chat` | [lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) | `toxicchat0124`, `train`/`test` | upstream `toxicity` label |
| `aegis` | [nvidia/Aegis-AI-Content-Safety-Dataset-1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0) | `train` | majority Safe -> 0; otherwise -> 1 |
| `offensivelang` | [AmitDasRup123/OffensiveLang](https://huggingface.co/datasets/AmitDasRup123/OffensiveLang) | `train`/`test` | `Offensive=yes` -> 1 |
| `hu_berlin_toxicity` | [HU-Berlin-ML-Internal/toxicity-dataset](https://huggingface.co/datasets/HU-Berlin-ML-Internal/toxicity-dataset) | `train`/`test` | upstream binary label |
| `berkeley_hate` | [ucberkeley-dlab/measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) | `train` | per-comment mean > 0.5 -> 1; < -1.0 -> 0 |
| `real_toxicity_prompts` | [allenai/real-toxicity-prompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts) | `train` | toxicity >= 0.5 -> 1; <= 0.05 -> 0 |

The builder applies source caps, deterministic SHA-1 text ordering, length
filters, and global case/whitespace-normalized deduplication. Its exact source
order and preprocessing are the reproducibility contract; upstream datasets
can change, so record the access date and dataset revisions for a new rebuild.
