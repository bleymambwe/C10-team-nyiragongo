"""Isolated GPU kernel for new corpus extraction and training-label audit."""
import argparse
import json
from pathlib import Path

import run_kaggle as runner

SLUG = 'latent-probe-teacher-corpus'
FOLDER = Path('results/kaggle/teacher-corpus')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action', choices=['build','push','status','pull'])
    p.add_argument('--cap', type=int, default=3000)
    a=p.parse_args()
    ref=f'{runner.USER}/{SLUG}'
    if a.action in ('build','push'):
        FOLDER.mkdir(parents=True, exist_ok=True)
        # Reuse the already validated GPU architecture and mounted-model setup.
        setup=runner.EXTRACT_BODY[:runner.EXTRACT_BODY.index('import probe.corpus as corpus')]
        body=Path('experiments/teacher_corpus.py').read_text()
        script=runner._inline_modules()+'\n'+setup+'\n'+body+f'\nrun(MODEL_PATH, cap={a.cap})\n'
        (FOLDER/'script.py').write_text(script, encoding='utf-8')
        meta=runner._metadata(SLUG, SLUG, True, [f'{runner.USER}/{runner.EXTRACT_SLUG}'],
                              models=[runner.GEMMA_KAGGLE_REF])
        (FOLDER/'kernel-metadata.json').write_text(json.dumps(meta, indent=1))
        print('built', FOLDER)
    if a.action=='build':
        return
    api=runner._api()
    if a.action=='push':
        r=api.kernels_push(str(FOLDER.resolve()))
        print({k:getattr(r,k,None) for k in ('ref','url','versionNumber','error')})
    elif a.action=='status':
        r=api.kernels_status(ref)
        print({'status':str(r.status),'failure':getattr(r,'failureMessage',None)})
    else:
        target=Path('artifacts/kaggle/teacher-corpus')
        target.mkdir(parents=True, exist_ok=True)
        api.kernels_output(ref, path=str(target.resolve()))
        print([str(x) for x in target.iterdir()])


if __name__=='__main__':
    main()
