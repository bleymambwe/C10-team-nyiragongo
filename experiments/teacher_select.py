"""Evaluate offline label filtering and a new corpus on untouched family folds."""
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
from probe.finalize import fit_head
from probe.select import Recipe, clean_sources, resample_to_prior, _stratified_cap, auroc
from probe.submission import build_and_verify


def linear(head,x):
    w=head['coef']/head['scale']
    return x@w+head['intercept']-head['center']@w


def pred(z,p):
    n=len(z);k=int(round(n*p));y=np.zeros(n,dtype=int)
    if k:y[np.argpartition(z,n-k)[n-k:]]=1
    return y


def metrics(z,y,p,base):
    ok=pred(z,p)==y; b=base==y
    return {'accuracy':float(ok.mean()),'auroc':auroc(z,y),
            'gains':int((ok&~b).sum()),'losses':int((b&~ok).sum())}


def run():
    out=Path('/kaggle/working')
    old=np.load(glob.glob('/kaggle/input/**/bundle_l14.npz',recursive=True)[0],allow_pickle=True)
    y0=old['y'];src0=old['sources'];fam0=old['families']
    groups=dict(zip(src0.tolist(),fam0.tolist()))
    ind,_,_,hygiene=clean_sources(np.arange(len(y0))[:,None],y0,src0,groups)
    ind=ind[:,0]
    x=old['X'][ind].astype(np.float32);y=y0[ind];src=src0[ind];fam=fam0[ind]
    teacher_path=glob.glob('/kaggle/input/**/teacher_scores.npy',recursive=True)[0]
    teacher=np.load(teacher_path)
    assert len(teacher)==len(y0)
    teacher=teacher[ind]
    agree=((y==1)&(teacher>=.7))|((y==0)&(teacher<=.3))
    # Balance within each retained training source so the head cannot learn
    # source membership as a surrogate label from the filtering operation.
    ki,_,_,teacher_hygiene=clean_sources(np.flatnonzero(agree)[:,None],y[agree],src[agree],groups)
    clean=ki[:,0]
    new=np.load(glob.glob('/kaggle/input/**/paradetox_l14.npz',recursive=True)[0],allow_pickle=True)
    xn=new['X'].astype(np.float32);yn=new['y']
    recipe=Recipe('std','lda',.6)
    newhead=fit_head(xn,yn,recipe,dtype=np.float32)
    rows=[]
    for name in sorted(set(fam)):
        tr=np.flatnonzero(fam!=name)
        ct=clean[fam[clean]!=name]
        base=fit_head(x[tr],y[tr],recipe,dtype=np.float32)
        cleaned=fit_head(x[ct],y[ct],recipe,dtype=np.float32)
        # ParaDetox shares source ancestry with these two families. Do not
        # present those as unseen-source validation for ParaDetox.
        valid_new=name not in ('civil','wiki')
        augmented=(fit_head(np.concatenate([x[tr],xn]),np.concatenate([y[tr],yn]),recipe,dtype=np.float32)
                   if valid_new else None)
        for seed in (17,29,41):
            te=np.flatnonzero(fam==name)
            te=te[resample_to_prior(y[te],1200/1700,seed)]
            if len(te)>1700:te=_stratified_cap(te,y[te],1700,seed)
            xx,yy=x[te],y[te];p=1200/1700
            z=linear(base,xx);bp=pred(z,p)
            row={'held_family':str(name),'seed':seed,'n':len(yy),
                 'baseline_train_n':len(tr),'teacher_train_n':len(ct),
                 'baseline':metrics(z,yy,p,bp),
                 'teacher_agreement':metrics(linear(cleaned,xx),yy,p,bp)}
            if valid_new:
                row['paradetox_only']=metrics(linear(newhead,xx),yy,p,bp)
                row['paradetox_augmented']=metrics(linear(augmented,xx),yy,p,bp)
            rows.append(row)
        (out/'teacher_select_partial.json').write_text(json.dumps(rows,indent=1))
        print('SELECT_FAMILY',name,flush=True)
    summary={}
    for method in ('baseline','teacher_agreement','paradetox_only','paradetox_augmented'):
        rr=[r for r in rows if method in r]
        summary[method]={'mean_accuracy':float(np.mean([r[method]['accuracy'] for r in rr])),
                         'mean_auroc':float(np.mean([r[method]['auroc'] for r in rr])),
                         'baseline_on_same_families':float(np.mean([r['baseline']['accuracy'] for r in rr])),
                         'families':sorted(set(r['held_family'] for r in rr))}
    result={'rows':rows,'summary':summary,'teacher_hygiene':teacher_hygiene,
            'teacher_kept':len(clean),'baseline_kept':len(y),
            'external_teacher_may_have_seen_civil_and_wiki':True,
            'seeds_are_overlapping_resamples':True,'candidates':[]}
    # Gate BEFORE fitting or creating any candidate zip.
    candidates=[('teacher_agreement',x[clean],y[clean]),
                ('paradetox_augmented',None,None)]
    for method,xt,yt in candidates:
        s=summary[method]
        if s['mean_accuracy']<=max(1200/1700,s['baseline_on_same_families']):
            continue
        if method=='paradetox_augmented':
            xt=np.concatenate([x,xn]);yt=np.concatenate([y,yn])
        head=fit_head(xt,yt,recipe,dtype=np.float32)
        path=out/'submissions'/f'{method}.zip'
        packaged=build_and_verify([head],path,version=method,combine='score',
                                  meta={'recipe':recipe.key,'offline_training_rows':len(yt)})
        packaged['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        result['candidates'].append(packaged)
    (out/'teacher_select_results.json').write_text(json.dumps(result,indent=1))
    print(json.dumps(summary,indent=1),flush=True)
    print('TEACHER_SELECT_OK',flush=True)

