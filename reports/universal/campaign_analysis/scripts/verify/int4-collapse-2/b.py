import json,numpy as np
from scipy import stats
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
R=[]
for m,ds in d.items():
    for dn,v in ds.items():
        L={r['label']:r['retention'] for r in v['rows']}
        if all(k in L for k in ['int4','binary','int4(mean)','int8(per_vector)']):
            R.append((m,dn.split('/')[-1],L['int4'],L['binary'],L['int4(mean)'],L['int8(per_vector)']))
print(len(R))
for r in sorted(R,key=lambda r:r[2])[:22]: print("%-55s %-18s int4 %.3f bin %.3f c4 %.3f i8 %.3f"%r)
txt=[r for r in R if 'docvqa' not in r[1] and 'infovqa' not in r[1]]
img=[r for r in R if r not in txt]
print(len(txt),len(img))
def sp(rs):
    a=np.array([[1-r[2],1-r[3]] for r in rs]);return stats.spearmanr(a[:,0],a[:,1]),stats.pearsonr(a[:,0],a[:,1])
print("text",sp(txt))
# image: per-model avg across datasets
from collections import defaultdict
g=defaultdict(list)
for r in img:g[r[0]].append(r)
im=[(k,'',np.mean([x[2] for x in v]),np.mean([x[3] for x in v])) for k,v in g.items()]
print(len(im),sp(im))
low=[r for r in txt if r[3]<=0.95]; print(len(low),sp(low))
low2=[r for r in low if 'mxbai' not in r[0]]; print("no mxbai",len(low2),sp(low2))
rng=np.random.default_rng(0);a=np.array([[1-r[2],1-r[3]] for r in txt]);bs=[]
for _ in range(5000):
    i=rng.integers(0,len(a),len(a));
    if np.ptp(a[i,0])>0 and np.ptp(a[i,1])>0: bs.append(stats.spearmanr(a[i,0],a[i,1])[0])
print(np.percentile(bs,[2.5,97.5]))
