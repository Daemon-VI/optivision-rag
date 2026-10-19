import json,re,numpy as np,collections
from optivision.adapters import MODELS
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
dim={k:v.dimension for k,v in MODELS.items()}
rng=np.random.default_rng(0)
def cboot(vals,groups,B=4000):
    g=collections.defaultdict(list)
    for v,k in zip(vals,groups):g[k].append(v)
    ks=list(g);ms=np.array([np.mean(g[k]) for k in ks]);ns=np.array([len(g[k]) for k in ks])
    out=[]
    for _ in range(B):
        i=rng.integers(0,len(ks),len(ks));out.append(np.sum(ms[i]*ns[i])/np.sum(ns[i]))
    return np.mean(vals),np.percentile(out,[2.5,97.5]),len(vals),len(ks)
# merge pairs
A=[];Ag=[];Ad=[]
codec={'int8':'int8(per_vector)','int4':'int4(mean)','binary':'binary'}
C={c:[] for c in codec};Cg={c:[] for c in codec};Cd={c:[] for c in codec};Ck={c:[] for c in codec}
for m,dd in d.items():
  for ds,e in dd.items():
    R={r['label']:r['retention'] for r in e['rows']}
    for l,v in R.items():
      mm=re.fullmatch(r'(hierarchical_merge\([\d.]+\)) > project\((\d+)\)',l)
      if mm and mm.group(1) in R and f'project({mm.group(2)})' in R:
        A.append(v/R[mm.group(1)]-R[f'project({mm.group(2)})']);Ag.append(m);Ad.append(dim.get(m))
      mm=re.fullmatch(r'project\((\d+)\) > (.+)',l)
      if mm and mm.group(2) in codec.values() and mm.group(2) in R and f'project({mm.group(1)})' in R:
        c=[k for k,x in codec.items() if x==mm.group(2)][0]
        C[c].append(v/R[f'project({mm.group(1)})']-R[mm.group(2)]);Cg[c].append(m);Cd[c].append(dim.get(m));Ck[c].append(int(mm.group(1)))
A=np.array(A)*100
print('merge pairs',cboot(A,Ag),'mean abs',np.abs(A).mean(),'max',np.abs(A).max())
print('n by model',collections.Counter(Ag))
# drop each model
for m in set(Ag):
    s=[a for a,g in zip(A,Ag) if g!=m];print(' drop',m[-30:],round(np.mean(s),2))
for c in codec:
    v=np.array(C[c])*100;dm=np.array(Cd[c]);g=np.array(Cg[c])
    print(c,'all',cboot(v,list(g)))
    for lab,mask in [('d<=128',dm<=128),('128<d<2048',(dm>128)&(dm<2048)),('d>=2048',dm>=2048)]:
        if mask.sum():print('  ',lab,cboot(v[mask],list(g[mask])))
    pm=collections.defaultdict(list)
    for a,b in zip(v,g):pm[b].append(a)
    print('   per-model means', {k[-25:]+str(dim.get(k)):round(np.mean(x),2) for k,x in pm.items() if dim.get(k,0)>=2048})
