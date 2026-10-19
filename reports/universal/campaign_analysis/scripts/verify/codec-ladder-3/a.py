import json,numpy as np
from scipy import stats
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
U=[]
for m in d:
  for ds,v in d[m].items():
    L={r['label']:r for r in v['rows']}
    if 'binary' in L and 'lloyd2(7)' in L:
      short='sci' if 'scifact' in ds.lower() else ('doc' if 'doc' in ds else 'info')
      U.append((m,short,L['binary']['retention'],L['lloyd2(7)']['retention'],L['binary']['retention_interval'],L['lloyd2(7)']['retention_interval']))
print(len(U))
b=np.array([u[2] for u in U]);l=np.array([u[3] for u in U]);g=np.array([u[1] for u in U])
print('spearman',stats.spearmanr(b,l))
def tab(b,l):
  lo=l<0.97;bl=b<0.95
  t=[[ (lo&bl).sum(),(lo&~bl).sum()],[(~lo&bl).sum(),(~lo&~bl).sum()]]
  return t,stats.fisher_exact(t)
print(tab(b,l))
for s in ['sci','doc','info']:
  k=g==s;print(s,k.sum(),stats.spearmanr(b[k],l[k]),tab(b[k],l[k]), 'ge.99 b',(b[k]>=.99).mean(),'l',(l[k]>=.99).mean())
# bootstrap by model clusters
ms=sorted(set(u[0] for u in U));rng=np.random.default_rng(0);idx={m:[i for i,u in enumerate(U) if u[0]==m] for m in ms}
r=[]
for _ in range(2000):
  s=rng.choice(len(ms),len(ms));ii=sum([idx[ms[j]] for j in s],[]);r.append(stats.spearmanr(b[ii],l[ii])[0])
print('cluster boot CI',np.percentile(r,[2.5,97.5]))
# one unit per model: mean of units
# drop sibling families: keep dedup by org prefix
fam={}
for i,u in enumerate(U): fam.setdefault(u[0].split('/')[0],[]).append(i)
r=[]
for _ in range(300):
  ii=[rng.choice(v) for v in fam.values()];r.append(stats.spearmanr(b[ii],l[ii])[0])
print('one unit per org',len(fam),np.mean(r),np.percentile(r,[2.5,97.5]))
# leave out extreme: remove units b<0.5
k=b>0.5;print('drop b<.5',k.sum(),stats.spearmanr(b[k],l[k]))
# partial within dataset: rank-demeaned
# flips
print('flips')
for u in sorted(U,key=lambda u:u[3]-u[2]):
  if u[3]-u[2]<-0.005: print(u[0],u[1],round(u[2],3),round(u[3],3),u[4],u[5], 'sig' if u[5][1]<u[4][0] else '')
print('lloyd2<.97 units')
for u in U:
  if u[3]<0.97: print(u[0],u[1],round(u[2],3),round(u[3],3))
print('binary<.95 but lloyd2>=.97')
for u in U:
  if u[2]<.95 and u[3]>=.97: print(u[0],u[1],round(u[2],3),round(u[3],3))
# lloyd2<binary-0.02 flips count with threshold
print([( t,int((l<b-t).sum())) for t in [0.005,0.02,0.05]])
print('lloyd2>binary share',(l>b).mean())
