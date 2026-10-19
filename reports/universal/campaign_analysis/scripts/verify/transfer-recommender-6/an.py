import json,collections,math,re,sys
import numpy as np
sys.path.insert(0,'.')
from optivision.adapters import MODELS
P="C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"
d=json.load(open(P))["models"]
units=[]
for m,v in d.items():
    mi=MODELS[m]
    for ds,x in v.items():
        rows={json.dumps(r['pipeline'],sort_keys=True):r for r in x['rows']}
        units.append(dict(m=m,ds=ds,fam=mi.family,mod=mi.modality,dim=mi.dimension,rows=rows,
           vpd=rows['{"stages": []}']['vectors_per_doc']))
cnt=collections.Counter(l for u in units for l in u['rows'])
CAND=sorted(l for l,n in cnt.items() if n>=86 and l!='{"stages": []}')
print(len(units),len(CAND))
nU=len(units)
R=np.full((nU,len(CAND)),np.nan);C=R.copy()
for i,u in enumerate(units):
    for j,l in enumerate(CAND):
        r=u['rows'].get(l)
        if r: R[i,j]=r['retention'];C[i,j]=r['compression']
models=sorted(d)
midx={m:[i for i,u in enumerate(units) if u['m']==m] for m in models}
info={m:units[midx[m][0]] for m in models}
# sibling clusters
def cluster(m):
    n=m.lower()
    if 'colpali' in n and 'vidore' in n and 'colpali-v1.2' in n: return 'cp12'
    if 'colpali-v1.3' in n: return 'cp13'
    if n.endswith('colpali') or 'colpali-v1.1' in n: return m
    if 'tsystems/colqwen2.5-3b' in n or 'metric-ai/colqwen2.5-3b' in n: return 'qwen25_3b_multi'
    if 'colqwen2-v1.0' in n: return 'cq2v1'
    if 'tomoro' in n: return 'tomoro'
    if 'colbertv2.0' in n: return 'colbertv2'
    return m
for m in models: info[m]['cl']=cluster(m)
cl=collections.defaultdict(list)
for m in models: cl[info[m]['cl']].append(m)
print({k:v for k,v in cl.items() if len(v)>1})
lv=np.array([math.log2(info[m]['vpd']) for m in models]); ld=np.array([math.log2(info[m]['dim']) for m in models])
sdv,sdd=lv.std(),ld.std()
def knn(m,pool,k):
    a=info[m]
    ds=[]
    for p in pool:
        b=info[p]
        dd=((math.log2(b['dim'])-math.log2(a['dim']))/sdd)**2+((math.log2(b['vpd'])-math.log2(a['vpd']))/sdv)**2+100*(a['mod']!=b['mod'])
        ds.append(dd)
    return [pool[i] for i in np.argsort(ds,kind='stable')[:k]]
def peers(m,strat,co):
    oth=[p for p in models if p!=m and not(co and info[p]['cl']==info[m]['cl'])]
    if strat=='global': return oth
    if strat=='modality': return [p for p in oth if info[p]['mod']==info[m]['mod']]
    if strat=='family':
        ms=[p for p in oth if info[p]['fam']==info[m]['fam']]
        if len(ms)<2: ms=[p for p in oth if info[p]['mod']==info[m]['mod']]
        return ms
    return knn(m,oth,int(strat[3:]))
f16=CAND.index('{"stages": [{"stage": "float16"}]}')
def sel(rows,t,rule='min'):
    r=R[rows];c=C[rows];av=~np.isnan(r);n=av.sum(0)
    if rule=='min': ok=np.nanmin(np.where(av,r,np.inf),0)>=t
    else: ok=(np.where(av,r>=t,False).sum(0)/np.maximum(n,1))>=0.9
    ok&=n>=max(1,int(.8*len(rows)))
    if not ok.any(): return None
    med=np.nanmedian(np.where(av,c,np.nan),0)
    mn=np.nanmin(np.where(av,r,np.inf),0); return int(np.argmax(np.where(ok,med+1e-6*mn,-np.inf)))
res=[]
for strat in ['global','modality','family','knn5','knn10']:
  for co in (0,1):
    for m in models:
        ms=peers(m,strat,co)
        for ui in midx[m]:
            rows=[i for p in ms for i in midx[p]]
            for t in (.95,.97,.99):
              for rule in ('min','q90'):
                j=sel(rows,t,rule); fb=j is None
                if fb: j=f16
                res.append((strat,co,m,ui,t,rule,fb,R[ui,j]>=t,C[ui,j],R[ui,j]))
import pandas as pd
df=pd.DataFrame(res,columns='strat co m ui t rule fb hit comp ret'.split())
nanu=[i for i in range(nU) if np.isnan(R[i]).any()]
df=df[~df.ui.isin(nanu)]
df.to_pickle('res.pkl')
rng=np.random.default_rng(0)
def boot(sub,col='hit',B=2000):
    g=sub.groupby('m')[col].mean(); v=g.values
    bs=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(B)]
    return np.percentile(bs,[2.5,97.5])
for rule in ('min','q90'):
  print('RULE',rule)
  for co in (0,1):
    for t in (.95,.97,.99):
      line=[]
      for s in ['global','modality','family','knn5','knn10']:
        sub=df[(df.strat==s)&(df.co==co)&(df.t==t)&(df.rule==rule)]
        lo,hi=boot(sub)
        line.append(f"{s}: hit {sub.hit.mean():.3f} [{lo:.2f},{hi:.2f}] med{sub.comp.median():.1f} fb{sub.fb.mean():.2f}")
      print('co',co,'t',t,' | '.join(line))
print('=== paired deltas vs global (cluster bootstrap over models), rule min')
def pm(strat,co,t,rule='min'):
    s=df[(df.strat==strat)&(df.co==co)&(df.t==t)&(df.rule==rule)]
    return s.groupby('m').agg(hit=('hit','mean'),comp=('comp','median'),lc=('comp',lambda x: np.log(x).mean()))
for t in (.95,.97,.99):
  for s in ['family','knn5','knn10']:
    a=pm(s,0,t);g=pm('global',0,t)
    dh=(a.hit-g.hit).values; dl=(a.lc-g.lc).values
    ms=len(dh)
    b=[(dh[ix].mean(),dl[ix].mean()) for ix in (rng.integers(0,ms,ms) for _ in range(2000))]
    b=np.array(b)
    print(t,s,'dhit %.3f [%.3f,%.3f]'%(dh.mean(),*np.percentile(b[:,0],[2.5,97.5])),'dlogcomp x%.2f [%.2f,%.2f]'%(np.exp(dl.mean()),*np.exp(np.percentile(b[:,1],[2.5,97.5]))))
print('=== leave-one-model-out of the evaluation: drop most extreme / by modality, co=1')
for t in (.95,.97,.99):
  for s in ['global','family','knn10']:
    for mo in ('text','image'):
      sub=df[(df.strat==s)&(df.co==1)&(df.t==t)&(df.rule=='min')&(df.ui.map(lambda i:units[i]['mod'])==mo)]
      print(t,s,mo,'hit %.3f'%sub.hit.mean(),'med comp %.1f'%sub.comp.median(),'n',sub.m.nunique())
print('=== family misses at .95, co=0')
s=df[(df.strat=='family')&(df.co==0)&(df.t==.95)&(df.rule=='min')&(~df.hit)]
print(s.groupby('m').size())
print('=== peer set size / fam sizes')
print(collections.Counter(info[m]['fam'] for m in models))
