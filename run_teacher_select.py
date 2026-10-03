"""CPU evaluation downstream of the completed GPU teacher/corpus job."""
import argparse
import json
from pathlib import Path
import run_kaggle as runner

SLUG='latent-probe-teacher-select'
FOLDER=Path('results/kaggle/teacher-select')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','push','status','pull']);a=p.parse_args()
    ref=f'{runner.USER}/{SLUG}'
    if a.action in ('build','push'):
        FOLDER.mkdir(parents=True,exist_ok=True)
        script=runner._inline_modules()+'\n'+Path('experiments/teacher_select.py').read_text()+'\nrun()\n'
        (FOLDER/'script.py').write_text(script,encoding='utf-8')
        meta=runner._metadata(SLUG,SLUG,False,[f'{runner.USER}/{runner.EXTRACT_SLUG}',f'{runner.USER}/latent-probe-teacher-corpus'])
        (FOLDER/'kernel-metadata.json').write_text(json.dumps(meta,indent=1))
        print('built',FOLDER)
    if a.action=='build':return
    api=runner._api()
    if a.action=='push':
        r=api.kernels_push(str(FOLDER.resolve()));print({k:getattr(r,k,None) for k in ('ref','url','versionNumber','error')})
    elif a.action=='status':
        r=api.kernels_status(ref);print({'status':str(r.status),'failure':getattr(r,'failureMessage',None)})
    else:
        target=Path('artifacts/kaggle/teacher-select');target.mkdir(parents=True,exist_ok=True)
        api.kernels_output(ref,path=str(target.resolve()));print([str(x) for x in target.iterdir()])


if __name__=='__main__':main()
