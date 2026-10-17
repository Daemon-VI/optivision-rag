import json,collections,re
import numpy as np, pandas as pd
P='C:/Users/Lenovo/projects/optivision-package/'
M=json.load(open(P+'src/optivision/presets/measured.json'))['models']
def dsname(d): return 'docvqa' if 'docvqa' in d else 'infovqa' if 'infovqa' in d else 'scifact'
def build():
    rows=[];base=[]
    for mid,dd in M.items():
        for d,v in dd.items():
            ds=dsname(d)
            base.append(dict(model=mid,ds=ds,nq=v['n_queries'],nd=v['n_docs'],baseline=v['baseline'],vpd=[r for r in v['rows'] if not r['pipeline']['stages']][0]['vectors_per_doc']))
            for r in v['rows']:
                st=r['pipeline']['stages']
                pr=[s for s in st if s['stage']=='project']
                names=[s['stage'] for s in st]
                rows.append(dict(model=mid,ds=ds,label=r['label'],names='>'.join(names),
                  k=pr[0]['dim'] if pr else None,method=pr[0].get('method') if pr else None,
                  ret=r['retention'],lo=r['retention_interval'][0],hi=r['retention_interval'][1],comp=r['compression'],vpd=r['vectors_per_doc']))
    return pd.DataFrame(rows),pd.DataFrame(base)
df,bs=build()
# model dimension: from registry
import sys,os
sys.path.insert(0,P+'src')
from optivision.adapters import MODELS
info={}
for mid in M:
    mi=MODELS.get(mid) if hasattr(MODELS,'get') else None
    info[mid]=mi
def mattr(mid,a):
    mi=info.get(mid); return getattr(mi,a,None) if mi is not None else None
dim=pd.Series({m:mattr(m,'dimension') for m in M}); fam=pd.Series({m:mattr(m,'family') for m in M}); mod=pd.Series({m:mattr(m,'modality') for m in M}); ev=pd.Series({m:mattr(m,'evidence') for m in M})
df['D']=df.model.map(dim); df['family']=df.model.map(fam); df['modality']=df.model.map(mod)
bs['D']=bs.model.map(dim); bs['family']=bs.model.map(fam); bs['modality']=bs.model.map(mod); bs['evidence']=bs.model.map(ev)
bs['fp16']=bs.evidence.fillna('').str.contains('float16')
pp=df[(df.names=='project')].copy(); pp['frac']=pp.k/pp.D

def mkey(l):
    m=re.match(r'(hierarchical_merge\([\d.]+\))',l); return m.group(1) if m else None
_d2=df[df.names.isin(['hierarchical_merge','hierarchical_merge>project'])].copy()
_d2['mk']=_d2.label.map(mkey)
_mo=_d2[_d2.names=='hierarchical_merge'].set_index(['model','ds','mk']).ret.to_dict()
mp=_d2[_d2.names=='hierarchical_merge>project'].copy()
mp['mret']=[_mo[(a,b,c)] for a,b,c in zip(mp.model,mp.ds,mp.mk)]
mp['rel']=mp.ret/mp.mret
mp['frac']=mp.k/mp.D
rng=np.random.default_rng(12345)
def cluster_boot(vals,groups,stat=np.mean,B=5000,seed=1):
    """bootstrap resampling groups (models) with all their rows; returns (est, lo, hi)"""
    vals=np.asarray(vals,float);groups=np.asarray(groups)
    ug=np.unique(groups);idx={g:np.where(groups==g)[0] for g in ug}
    r=np.random.default_rng(seed);out=[]
    for _ in range(B):
        gs=r.choice(ug,len(ug));v=np.concatenate([vals[idx[g]] for g in gs]);out.append(stat(v))
    return stat(vals),np.percentile(out,2.5),np.percentile(out,97.5)

# unit curves: per (model,ds,k) PCA relative retention
_mavg=mp[mp.k<mp.D].groupby(['model','ds','k']).rel.mean().rename('rel_merge')
_pure=pp[pp.method=='pca'].set_index(['model','ds','k']).ret.rename('rel_pure')
curve=pd.concat([_pure,_mavg],axis=1).reset_index()
curve['D']=curve.model.map(dim); curve['frac']=curve.k/curve.D
curve['rel']=curve.rel_pure.fillna(curve.rel_merge)
curve['family']=curve.model.map(fam); curve['modality']=curve.model.map(mod)
curve=curve[curve.k<curve.D]
