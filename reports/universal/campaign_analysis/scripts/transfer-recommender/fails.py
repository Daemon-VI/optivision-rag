from lomo import *
import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
D=pd.read_pickle('frontier.pkl')
names=[key_str(k) for k in CAND]
D['pipe']=D.j.map(lambda j:names[j]); D['ret']=[R[u,j] for u,j in zip(D.ui,D.j)]
D['dim']=D.model.map(lambda m:minfo[m]['dim']); D['mod']=D.model.map(lambda m:minfo[m]['modality'])
D['ds']=[ds[u][-22:-14] for u in D.ui]
for strat,c in [('dimband_mod',0.9),('family',0.9),('global',0.9)]:
  for t in TARGETS:
    g=D[(D.strat==strat)&(D.q=='lo')&(D.c==c)&(D.t==t)]
    f=g[~g.hit].sort_values('short',ascending=False)
    print(f'--- {strat} lo>={c} t={t}: {len(f)}/{len(g)} fails')
    print(f[['model','ds','dim','mod','pipe','ret','short','comp','ocomp']].round(3).to_string(index=False))
# what pipelines are chosen, by dim band, for dimband_mod lo>=.9
for t in TARGETS:
    g=D[(D.strat=='dimband_mod')&(D.q=='lo')&(D.c==0.9)&(D.t==t)]
    print(t); print(g.groupby(['mod','dim'])['pipe'].agg(lambda s:s.value_counts().to_dict()).to_string())
# failure share of units in clusters (are failures concentrated in few models?)
g=D[(D.strat=='dimband_mod')&(D.q=='lo')&(D.c==0.9)]
print(g.groupby('t').apply(lambda x: x[~x.hit].model.nunique()))
