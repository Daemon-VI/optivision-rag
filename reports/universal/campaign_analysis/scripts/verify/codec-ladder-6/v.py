import json,numpy as np,pandas as pd
from scipy import stats
from optivision.adapters import MODELS
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
rows=[]
for m,dd in d.items():
    mi=MODELS.get(m) if isinstance(MODELS,dict) else None
    if mi is None:
        mi=next((x for x in MODELS if getattr(x,'model_id',x)==m),None) if not isinstance(MODELS,dict) else None
    for ds,e in dd.items():
        b=[r for r in e['rows'] if r['label']=='binary']
        if not b: continue
        b=b[0]
        rows.append(dict(m=m,ds=ds.split('/')[-1][:6],base=e['baseline'],loss=1-b['retention'],vpd=b['vectors_per_doc'],
          dim=getattr(mi,'dimension',None),fam=getattr(mi,'family',None),fp16='float16 inference' in e.get('note',''),nq=e['n_queries']))
df=pd.DataFrame(rows);print(df.groupby('ds').size());print(df.dim.isna().sum())
df.to_csv('t.csv',index=False)
def sp(x,y):
    r,p=stats.spearmanr(x,y);return round(r,3),round(p,4),len(x)
im=df[df.ds!='scifac']
print('pooled img',sp(im.loss,im.base))
for ds,g in df.groupby('ds'):
    print(ds,sp(g.loss,g.base), 'dim128',sp(g[g.dim==128].loss,g[g.dim==128].base) if (g.dim==128).sum()>3 else None,
     'abs',sp((g.base*g.loss)[g.dim==128],g.base[g.dim==128]) if (g.dim==128).sum()>3 else None,
     'vpd',sp(g.loss,g.vpd))
    print(' fp16',g[g.fp16].loss.mean(),g[~g.fp16].loss.mean(),stats.mannwhitneyu(g[g.fp16].loss,g[~g.fp16].loss).pvalue if g.fp16.any() and (~g.fp16).any() else None, g.fp16.sum())
# sibling/extreme robustness for dim128 within dataset
for ds,g in df.groupby('ds'):
    g=g[g.dim==128]
    if len(g)<4: continue
    print(ds,'dim128 n',len(g),'fams',g.fam.value_counts().to_dict())
    rs=[stats.spearmanr(g.drop(i).loss,g.drop(i).base)[0] for i in g.index]
    print(' LOO range',round(min(rs),2),round(max(rs),2))
    rng=np.random.default_rng(0);bs=[]
    for _ in range(2000):
        s=g.sample(len(g),replace=True,random_state=int(rng.integers(1e9)))
        if s.loss.nunique()>1: bs.append(stats.spearmanr(s.loss,s.base)[0])
    print(' boot',np.percentile(bs,[2.5,97.5]).round(2))
# dim effect among all image: dim vs base, loss
print(im.groupby(['ds','dim']).agg(n=('loss','size'),base=('base','mean'),loss=('loss','mean')))
