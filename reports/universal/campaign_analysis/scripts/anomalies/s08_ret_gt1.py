from common import *
import pandas as pd
from scipy import stats
df=pd.read_pickle(OUT+'/wide_long.pkl'); d=df[df.m=='ndcg@5'].copy()
d['eps']=d.ret-1
# group by label type (codec only rows etc)
base=d[d.label.isin(['float16','int8(per_vector)','int4','int4(mean)','binary','lloyd2(7)','project(128)','project(128) > float16','project(160)'])]
rng=np.random.default_rng(0)
print("label, n, n>1, n==1(|e|<1e-9), n<1, mean eps, sign-test p (>1 vs <1), cluster-boot 95% CI of mean eps (resample model)")
for lab,g in base.groupby('label'):
    up=(g.eps>1e-9).sum(); eq=(abs(g.eps)<=1e-9).sum(); dn=(g.eps<-1e-9).sum()
    p=stats.binomtest(up,up+dn,0.5).pvalue if up+dn>0 else np.nan
    mods=g.groupby('model').eps.mean(); 
    bs=[rng.choice(mods.values,len(mods)).mean() for _ in range(4000)]
    print(f"{lab:28s} n={len(g):3d} up={up:3d} eq={eq:3d} dn={dn:3d} mean={g.eps.mean():+.4f} p={p:.3g} CI=[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]")
# ret>1 overall by group/ds
print((d.assign(gt=d.ret>1.0001).groupby(['ds'])['gt'].mean()))
print(d.assign(gt=d.ret>1.0001).groupby('group')['gt'].agg(['mean','sum','size']))
# how many of the ret>1 rows have CI that includes 1 (i.e. lo<=1)
x=d[d.ret>1.0001]; print("ret>1 rows:",len(x),"CI excludes 1:",(x.lo>1).sum())
# rows with highest compression among those with ret>=1: "free lunch" claims
y=d[(d.ret>=1.0)&(d.comp>8)].sort_values('comp',ascending=False).head(12)
print(y[['model','ds','label','ret','lo','hi','comp']].to_string())
print("rows ret>=1 & comp>=8:",len(d[(d.ret>=1)&(d.comp>=8)]),"by label:",d[(d.ret>=1)&(d.comp>=8)].label.value_counts().head(6).to_dict())
