import json,glob,re,os,numpy as np
from scipy.stats import spearmanr
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
data={} # model -> ds -> k -> ret ; dim
dims={}
for p in glob.glob(R+'campaign/*/wide/*.json')+glob.glob(R+'wide/*_test_subsampled.json'):
    if 'smoke' in p: continue
    d=json.load(open(p))
    m=p.replace(chr(92),'/').split('/')
    model=m[-3] if 'campaign' in p else m[-1].split('_')[0]
    ds=d['dataset']
    dims[model]=d['baseline']['dim']
    for r in d['rows']:
        mm=re.fullmatch(r'project\((\d+)\)',r['label'])
        if mm: data.setdefault(model,{}).setdefault(ds,{})[int(mm.group(1))]=r['retention:ndcg@5']
print(len(data),'models')
def loss(model,k):
    v=[1-x[k] for x in data[model].values() if k in x]
    return np.mean(v) if v else None
rows=[]
for m in sorted(data,key=lambda m:dims[m]):
    ks=sorted({k for x in data[m].values() for k in x})
    print(m,dims[m],len(data[m]),{k:round(loss(m,k)*100,2) for k in ks if k<=256})
wide=[m for m in data if dims[m]>128 and loss(m,128) is not None]
print(len(wide))
x=[np.log(dims[m]) for m in wide];y=[loss(m,128) for m in wide]
rho=spearmanr(x,y)[0];print('rho',rho)
rng=np.random.default_rng(0);c=sum(abs(spearmanr(x,rng.permutation(y))[0])>=abs(rho) for _ in range(5000));print('p',(c+1)/5001)
# drop-one, drop sibling
for m in wide:
    xx=[a for a,b in zip(wide,y) if a!=m]
    print('drop',m, round(spearmanr([np.log(dims[a]) for a in xx],[loss(a,128) for a in xx])[0],2)) if False else None
# within-4096 and 320-640 groups
from collections import defaultdict
g=defaultdict(list)
for m in wide: g[dims[m]].append((m,round(loss(m,128)*100,2)))
print(dict(g))
# shares by group
def shares(lo,hi,k):
    u=[(m,ds) for m in data if lo<=dims[m]<=hi for ds in data[m] if k in data[m][ds]]
    v=np.array([data[m][ds][k] for m,ds in u]);return len(u),(v>=.95).sum(),(v>=.98).sum(),v.mean()
for lo,hi,k in [(128,128,64),(320,640,80),(320,640,128),(4096,4096,128),(4096,4096,256),(4096,4096,512),(2048,2560,128)]:
    print(lo,hi,k,shares(lo,hi,k))
# per-unit k=128 on 4096
for m in data:
    if dims[m]==4096: print(m,{ds:round(data[m][ds].get(128,np.nan),3) for ds in data[m]})
# pure PCA 1024/2048
for k in (1024,2048): print(k,[round(np.mean(list(x[k] for x in data[m].values() if k in x)),3) for m in data if dims[m]==4096])
# 128-d d=128 truth: non-wide excluded; 320-d truncated sets
print('---extra')
v=np.array([data[m][ds][64] for m in data if dims[m]==128 for ds in data[m] if 64 in data[m][ds]])
print('64: >=.98',(v>=.98).sum(),'>=.975',(v>=.975).sum(),'>=.95',(v>=.95).sum(),len(v))
print('80@320 mean loss',np.mean([loss(m,80) for m in data if dims[m]==320 and loss(m,80) is not None]))
xs=np.array([np.log(dims[m]) for m in wide]);ys=np.array(y)
for i,m in enumerate(wide):
    k=np.arange(len(wide))!=i
    print('drop',m[:30],round(spearmanr(xs[k],ys[k])[0],2))
# collapse: one value per d-group/family: groups mean
gd={}
for m in wide: gd.setdefault(dims[m],[]).append(loss(m,128))
gx=sorted(gd);gy=[np.mean(gd[a]) for a in gx];print(gx,np.round(gy,4),spearmanr(gx,gy))
# permutation with group-level (6 groups, exact)
import itertools
cnt=0;tot=0
for p in itertools.permutations(gy):
    tot+=1;cnt+=abs(spearmanr(gx,p)[0])>=abs(spearmanr(gx,gy)[0])-1e-12
print('exact group perm p',cnt/tot)
# 4096 k=128 per unit and ret ci ?
