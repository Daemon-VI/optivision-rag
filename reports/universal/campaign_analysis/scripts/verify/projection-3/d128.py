from load import *
from scipy import stats
rng=np.random.default_rng(2)
def key(r):
    return ' > '.join(f"{s['stage']}:{s.get('method','')}:{s.get('dim','')}:{s.get('ratio','')}" for s in r['pipeline']['stages'])
res=[]
for (m,ds),v in sorted(U.items()):
    d=v[0][1]
    if d['baseline']['dim']!=128: continue
    rs=rows(d)
    rnd=[r for r in d['rows'] if len(r['pipeline']['stages'])==1 and r['pipeline']['stages'][0].get('method')=='random' and r['pipeline']['stages'][0]['dim']==32]
    pc64=[r for r in d['rows'] if r['label']=='project(64)']
    mg={l:rs[l][M] for l in rs if l.startswith('hierarchical_merge') and l.count('>')==1}
    if not rnd: continue
    res.append(dict(m=m,ds=ds,rnd=rnd[0][M],pc64=pc64[0][M],mg33_32=mg.get('hierarchical_merge(0.33) > project(32)'),mg25_32=mg.get('hierarchical_merge(0.25) > project(32)'),mg33_128=mg.get('hierarchical_merge(0.33) > project(128)'),mg25_128=mg.get('hierarchical_merge(0.25) > project(128)'),mg33_16=mg.get('hierarchical_merge(0.33) > project(16)'),vec=rnd[0]['vectors_per_doc']))
import pandas as pd
df=pd.DataFrame(res); print(len(df),df.m.nunique(),df.mg33_32.isna().sum(),df.mg25_32.isna().sum())
df['d33']=(df.mg33_32-df.rnd)*100; df['d25']=(df.mg25_32-df.rnd)*100
for c in ['d33','d25']:
    x=df[c].dropna(); print(c,len(x),x.mean(),(x>0).sum(),x.min(),x.max())
# model clusters
g=df.groupby('m').d33.mean().dropna(); print('models',len(g),g.mean(),(g>0).sum())
cb=[g.values[rng.integers(0,len(g),len(g))].mean() for _ in range(10000)]; print(np.percentile(cb,[2.5,97.5]))
# cluster boot over units using model clusters
print('merge cost via 128 (1-ret):',(1-df.mg33_128).mean()*100,(1-df.mg25_128).mean()*100)
print('random-32 retention mean',df.rnd.mean(),'pca64',df.pc64.mean(),'mg33_32',df.mg33_32.mean())
# vs pca64 not equal k; ratio-based: merge-controlled random impossible. Compare diff vs (pc64 - rnd)
print('pca64-rnd',((df.pc64-df.rnd)*100).mean())
# dataset split
for ds,s in df.groupby('ds'): print(ds,len(s),s.d33.mean())
# correlation of advantage with random loss
print(df[['d33']].describe().T)
print(df.sort_values('d33').head(8)[['m','ds','rnd','mg33_32','d33']])
# models where random wins
print('random wins',(df.d33<0).sum(), df[df.d33<0].m.unique()[:20])
# dependence on merge cost: advantage vs merge128 cost
print('corr d33 vs mg33_128 cost',np.corrcoef(df.d33.fillna(0),1-df.mg33_128)[0,1])
print('=====')
img=df[df.ds.isin(['docvqa','infovqa'])]; txt=df[~df.ds.isin(['docvqa','infovqa'])]
for n,s in [('image',img),('text',txt)]:
    g=s.groupby('m').d33.mean(); cb=[g.values[rng.integers(0,len(g),len(g))].mean() for _ in range(10000)]
    print(n,len(s),len(g),'mean units',s.d33.mean(),'model-mean',g.mean(),np.percentile(cb,[2.5,97.5]),'ahead',(s.d33>0).sum(),'median',s.d33.median())
    print(' random32 retention mean by model',1-s.groupby('m').rnd.mean().mean())
print('all random-32 loss, model-mean',1-df.groupby('m').rnd.mean().mean())
# random loss by k from wide
import collections
loss=collections.defaultdict(list)
for (m,ds),v in U.items():
    d=v[0][1]
    for r in d['rows']:
        st=r['pipeline']['stages']
        if len(st)==1 and st[0]['stage']=='project' and st[0].get('method')=='random':
            loss[(d['baseline']['dim'],st[0]['dim'])].append((m,1-r[M]))
for k,v in sorted(loss.items()):
    mods={}
    for m,x in v: mods.setdefault(m,[]).append(x)
    print(k,len(mods),round(100*np.mean([np.mean(x) for x in mods.values()]),2))
