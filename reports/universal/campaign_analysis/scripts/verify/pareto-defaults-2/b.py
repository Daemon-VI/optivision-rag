import json,numpy as np
from scipy import stats
exec(open('a.py').read().split("labs=")[0])
miss=[m for m in d if D in d[m] and not get(m,D,"binary")];print(miss)
for m in miss: print(m,[r["label"] for r in d[m][D]["rows"] if 'inar' in r["label"] or 'bit' in r["label"]][:4], I in d[m], len(d[m][D]["rows"]))
print([m for m in d if D in d[m] and not get(m,D,"hierarchical_merge(0.25) > int4(mean)")])
# coverage of binary over 40 per ds with whatever label
def cov(ds,l,thr=0.95):
    rs=[get(m,ds,l) for m in d if get(m,ds,l)];return sum(r["retention_interval"][0]>=thr for r in rs),len(rs)
for ds in (D,I,S): print(ds,cov(ds,"binary"))
def boot(a,b,name):
    dd=a-b;rng=np.random.default_rng(1)
    bs=[rng.choice(dd,len(dd)).mean() for _ in range(10000)]
    print(name,len(dd),dd.mean().round(4),np.percentile(bs,[2.5,97.5]).round(4),(dd<0).sum(),stats.wilcoxon(a,b).pvalue)
img=[m for m in d if D in d[m] and I in d[m]]
def pv(l):
    ms=[m for m in img if get(m,D,l) and get(m,I,l) and get(m,D,"binary") and get(m,I,"binary")]
    return ms
hb="hierarchical_merge(0.25) > binary"
ms=pv(hb)
# interpretation A: (hm>bin - bin) Doc minus Info
A=np.array([get(m,D,hb)["retention"]-get(m,D,"binary")["retention"] for m in ms]);B=np.array([get(m,I,hb)["retention"]-get(m,I,"binary")["retention"] for m in ms])
boot(A,B,"hmbin-bin D vs I")
# interpretation B: hm>binary D-I
boot(np.array([get(m,D,hb)["retention"] for m in ms]),np.array([get(m,I,hb)["retention"] for m in ms]),"hm>bin D vs I")
for l in ["hierarchical_merge(0.25)","hierarchical_merge(0.25) > project(128) > binary","hierarchical_merge(0.25) > int4(mean)"]:
    ms=pv(l);boot(np.array([get(m,D,l)["retention"] for m in ms]),np.array([get(m,I,l)["retention"] for m in ms]),l)
