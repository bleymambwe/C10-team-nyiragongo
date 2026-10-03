"""New public contrastive corpus and offline teacher-agreement experiment.

The teacher is used only to filter training rows. Evaluation retains original
labels, including disagreements. ParaDetox may share ancestry with Wikipedia
and Civil Comments; these families are excluded from its transfer evidence.
"""
import gc
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from probe.corpus import canonical_text, _balance
from probe.extract import extract, save_bundle
from probe.finalize import fit_head
from probe.select import Recipe, resample_to_prior, _stratified_cap, auroc, quota_accuracy
from probe.submission import build_and_verify


def run(model_path, cap=3000):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from huggingface_hub import model_info

    out = Path('/kaggle/working')
    corpus_path = Path(glob.glob('/kaggle/input/**/corpus.jsonl', recursive=True)[0])
    records = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]
    # A stricter overlap key collapses punctuation and case differences too.
    key = lambda t: re.sub(r'\W+', '', str(t).casefold())
    seen = {key(r['text']) for r in records}
    pool = []
    duplicates = 0
    ds = load_dataset('s-nlp/paradetox', split='train')
    for row in ds:
        for field, label in [('en_toxic_comment', 1), ('en_neutral_comment', 0)]:
            text = row[field].strip()
            k = key(text)
            if k in seen:
                duplicates += 1
                continue
            seen.add(k)
            if len(text) >= 3:
                pool.append((text, label))
    chosen = _balance(pool, cap, 0.5)
    assert len(chosen) == cap and sum(y for _, y in chosen) == cap // 2
    manifest = {'dataset': 's-nlp/paradetox', 'fingerprint': ds._fingerprint,
                'revision': ds.info.version.__str__(), 'raw_pairs': len(ds),
                'new_rows': len(chosen), 'excluded_duplicates': duplicates,
                'labels': {'en_toxic_comment': 1, 'en_neutral_comment': 0},
                'excluded_transfer_families': ['civil', 'wiki'],
                'note': 'New candidate corpus, not identified as competition source.'}
    (out/'new_corpus.jsonl').write_text(''.join(json.dumps({'text':t,'label':y,'source':'paradetox'})+'\n' for t,y in chosen))
    print('NEW_CORPUS', json.dumps(manifest), flush=True)
    xx = extract([t for t,_ in chosen], '/kaggle/temp/paradetox_l14.npy',
                 model_id=model_path, batch_size=64)
    save_bundle(out/'paradetox_l14.npz', xx, np.array([y for _,y in chosen]),
                ['paradetox']*len(chosen), families=['paradetox']*len(chosen))
    del xx
    gc.collect()
    torch.cuda.empty_cache()

    # Use an explicitly pinned public model revision; never treat teacher
    # confidence as ground truth or filter any held-out evaluation rows.
    teacher = 'unitary/toxic-bert'
    revision = model_info(teacher).sha
    tok = AutoTokenizer.from_pretrained(teacher, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(teacher, revision=revision).to('cuda').eval()
    names = {int(i):str(s).lower() for i,s in model.config.id2label.items()}
    indexes = [i for i,s in names.items() if s in ('toxic', 'toxicity')]
    assert len(indexes) == 1, names
    scores = np.zeros(len(records), dtype=np.float32)
    ordered = sorted(range(len(records)), key=lambda i:len(records[i]['text']))
    for start in range(0, len(ordered), 64):
        idx = ordered[start:start+64]
        inp = tok([records[i]['text'] for i in idx], padding=True, truncation=True,
                  max_length=128, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            pred = model(**inp).logits.sigmoid()[:,indexes[0]].cpu().numpy()
        scores[idx] = pred
        if start % 12800 == 0:
            print(f'TEACHER {start}/{len(records)}', flush=True)
    np.save(out/'teacher_scores.npy', scores)
    manifest['teacher'] = {'model':teacher,'revision':revision,'max_length':128,
                           'label_mapping':names,'corpus_sha256':hashlib.sha256(corpus_path.read_bytes()).hexdigest()}
    (out/'new_corpus_manifest.json').write_text(json.dumps(manifest, indent=1))
    print('TEACHER_CORPUS_OK', flush=True)

