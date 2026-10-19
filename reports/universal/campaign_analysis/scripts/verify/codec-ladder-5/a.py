import json,numpy as np,re
from scipy import stats
from optivision.adapters import MODELS
m=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
rng=np.random.default_rng(0)
print(type(MODELS), list(MODELS)[:2] if not isinstance(MODELS,list) else MODELS[0])
reg={ (x.model_id if hasattr(x,'model_id') else x):x for x in (MODELS.values() if isinstance(MODELS,dict) else MODELS)}
rows=[]
for mid,ds in m.items():
    info=reg.get(mid)
    for d,v in ds.items():
        b=[r for r in v['rows'] if r['label']=='binary']
        if not b: continue
        rows.append((mid,d.split('/')[-1][:7],info.dimension if info else None,1-b[0]['retention'],info.family if info else None))
import collections
print(len(rows), collections.Counter(r[1] for r in rows), [r[0] for r in rows if r[2] is None])
def boot(x,y,g,B=2000):
    x=np.array(x);y=np.array(y);g=np.array(g);u=np.unique(g);out=[]
    for _ in range(B):
        s=rng.choice(u,len(u));idx=np.concatenate([np.where(g==k)[0] for k in s])
        if len(set(x[idx]))<2: continue
        out.append(stats.spearmanr(x[idx],y[idx])[0])
    return np.nanpercentile(out,[2.5,97.5])
for ds in ['docvqa','infovqa','scifact']:
    R=[r for r in rows if r[1]==ds[:7] or r[1].startswith(ds[:4])]
    if not R: print(ds,'none');continue
    x=np.log2([r[2] for r in R]);y=[r[3] for r in R]
    rho,p=stats.spearmanr(x,y); print(ds,len(R),rho,p,boot(x,y,[r[0] for r in R]))
    # drop dim128 / only non-128 
    print(' median by dim',{int(d):(round(float(np.median([r[3] for r in R if r[2]==d])),4),sum(1 for r in R if r[2]==d)) for d in sorted(set(r[2] for r in R))})
    # within-128
    r128=[r for r in R if r[2]==128]
    # family-excluded: leave-one-family-out
    fams=set(r[4] for r in R); mn=[];
    for f in fams:
        S=[r for r in R if r[4]!=f]
        if len(set(r[2] for r in S))>1: mn.append(stats.spearmanr([r[2] for r in S],[r[3] for r in S])[0])
    print(' LOFO rho range',min(mn),max(mn))
    # one rep per family
    ff=collections.defaultdict(list)
    for r in R: ff[r[4]].append(r)
    rep=[(np.median([q[2] for q in v]),np.median([q[3] for q in v])) for v in ff.values()]
    print(' family-level rho',stats.spearmanr(*zip(*rep)),len(rep))
