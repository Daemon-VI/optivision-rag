import json,numpy as np,collections
from scipy.stats import wilcoxon
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
D='vidore/docvqa_test_subsampled';I='vidore/infovqa_test_subsampled'
R={}
for m,v in d.items():
    if D in v and I in v:
        R[m]={ds:{r['label']:r['retention'] for r in v[ds]['rows']} for ds in(D,I)}
labs=collections.Counter(l for m in R for l in R[m][D] if l in R[m][I])
print(len(R),len(labs))
rng=np.random.default_rng(0)
res=[]
for l,c in labs.items():
    if l in('float32',) or c<30: continue
    ms=[m for m in R if l in R[m][D] and l in R[m][I]]
    x=np.array([R[m][D][l]-R[m][I][l] for m in ms])
    if np.all(x==0): continue
    bs=[rng.choice(x,len(x)).mean() for _ in range(2000)]
    try:p=wilcoxon(x).pvalue
    except: p=1
    res.append((l,len(x),x.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),(x<0).mean(),np.median(x),p))
n=len(res);print('pipelines',n)
sig=0
for r in sorted(res,key=lambda r:r[2]):
    s=r[7]*n<0.05;sig+=s
    print('%-40s n=%d mean=%.4f [%.4f,%.4f] lower=%.2f med=%.4f p=%.2g %s'%(r+(("*" if s else ""),))[:0] if False else '%-40s n=%d mean=%.4f [%.4f,%.4f] lower=%.2f med=%.4f p=%.2g %s'%(*r,"*" if s else ""))
print('sig',sig)
# sign-based: how many sig positive
print('pos mean',sum(r[2]>0 for r in res),'neg mean',sum(r[2]<0 for r in res))
# leave out: is effect driven by sibling/ extreme? per-model mean diff
pm=collections.defaultdict(list)
for m in R:
    for r in res:
        l=r[0]
        if l in R[m][D] and l in R[m][I]: pm[m].append(R[m][D][l]-R[m][I][l])
for m,v in sorted(pm.items(),key=lambda kv:np.mean(kv[1])): print('%.4f %s'%(np.mean(v),m))
# baselines
for m in list(d)[:0]:pass
fp16=[m for m in R if any(l=='float16' for l in R[m][D]) ]
print(sum(1 for m in d if 'vidore/docvqa_test_subsampled' in d[m]))
b=[(m,d[m][D]['baseline'],d[m][I]['baseline']) for m in R]
print(np.mean([x[1] for x in b]),np.mean([x[2] for x in b]))
