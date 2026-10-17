import json,re,numpy as np,collections
from scipy import stats
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
rng=np.random.default_rng(0)
res=[]
for m in d:
  for ds,v in d[m].items():
    R={r['label']:r for r in v['rows']}
    for l,r in R.items():
      mm=re.fullmatch(r'project\((\d+)\) > binary',l)
      if mm and f'project({mm.group(1)})' in R and 'binary' in R:
        p=R[f'project({mm.group(1)})']['retention'];b=R['binary']['retention'];a=r['retention']
        res.append((m,ds,int(mm.group(1)),p,b,a,r['retention_interval']))
print(len(res))
txt=[x for x in res if 'scifact' in x[1].lower()];img=[x for x in res if 'scifact' not in x[1].lower()]
print(len(txt),len(img),collections.Counter(x[2] for x in txt),collections.Counter(x[2] for x in img))
def boot(arr,n=20000):
  arr=np.array(arr);bs=[rng.choice(arr,len(arr)).mean() for _ in range(n)];return arr.mean(),np.percentile(bs,[2.5,97.5])
for name,g in [('txt',txt),('img',img)]:
  # per-model mean over datasets/ k
  pm=collections.defaultdict(list)
  for x in g: pm[x[0]].append(x[5]-x[3]*x[4])
  pmm=[np.mean(v) for v in pm.values()]
  print(name,len(pm),boot(pmm), 'rows',boot([x[5]-x[3]*x[4] for x in g]))
  print(' gain vs binary', np.mean([x[5]>x[4] for x in g]),np.mean([x[5]-x[4] for x in g]),' pca worse frac',np.mean([x[5]<x[4] for x in g]))
  # per-model
  mods=list(pm)
  ex=[(m,np.mean(pm[m])) for m in mods]
  print(sorted(ex,key=lambda t:t[1]))
  if name=='txt':
    bl=[1-np.mean([x[4] for x in g if x[0]==m]) for m in mods]
    print(stats.spearmanr(bl,pmm))
    # leave-one-out
    for i,m in enumerate(mods):
      print('drop',m,np.mean([e for j,e in enumerate(pmm) if j!=i]))
    print('sign test',stats.binomtest(sum(e>0 for e in pmm),len(pmm)).pvalue, 'wilcoxon',stats.wilcoxon(pmm).pvalue)
  else:
    print('wilcoxon',stats.wilcoxon(pmm).pvalue, sum(e>0 for e in pmm))
    # per-dataset
    for ds in set(x[1] for x in g): 
      print(ds,boot([x[5]-x[3]*x[4] for x in g if x[1]==ds]))
for x in txt: print(x[0],x[2],round(x[3],3),round(x[4],3),round(x[5],3),x[6])
print('=====EXTRA')
import numpy as np
def cl(x):return np.array([x[5]-x[3]*x[4] for x in x]),np.array([1-x[4] for x in x])
for name,g in [('img all',img),('img k64',[x for x in img if x[2]==64]),('img k64 docvqa',[x for x in img if x[2]==64 and 'doc' in x[1]]),('img k64 info',[x for x in img if x[2]==64 and 'info' in x[1]]),('txt',txt)]:
  e,bl=cl(g);print(name,len(g),len(set(x[0] for x in g)),'mean',e.mean(),boot(e)[1],'binloss mean',bl.mean(),'max',bl.max(),'spearman',stats.spearmanr(bl,e))
# image rows with binary loss >0.05
hi=[x for x in img if 1-x[4]>0.05]
e,bl=cl(hi);print('img binloss>0.05',len(hi),e.mean(),boot(e)[1]);
for x in hi:print(x[0][:30],x[1][-12:],x[2],round(x[3],3),round(x[4],3),round(x[5],3))
# txt matched on binary loss: ratio a/b etc
e,bl=cl(txt);a=np.array([x[5] for x in txt]);b=np.array([x[4] for x in txt]);print('spearman a vs binloss',stats.spearmanr(a,bl),'a range',a.min(),a.max())
# pooled regression e ~ binloss with dataset dummy
g=img+txt
e,bl=cl(g);isT=np.array(['scifact' in x[1].lower() for x in g],float)
X=np.c_[np.ones(len(g)),bl,isT];beta=np.linalg.lstsq(X,e,rcond=None)[0];print('pooled beta const,binloss,text',beta)
# bootstrap by model
mods=sorted(set(x[0] for x in g));idx={m:[i for i,x in enumerate(g) if x[0]==m] for m in mods}
B=[]
for _ in range(3000):
  s=rng.choice(len(mods),len(mods));ii=sum([idx[mods[j]] for j in s],[])
  B.append(np.linalg.lstsq(X[ii],e[ii],rcond=None)[0])
print(np.percentile(np.array(B),[2.5,97.5],axis=0))
# image per-model at k=64 only
pm=collections.defaultdict(list)
for x in img:
  if x[2]==64:pm[x[0]].append(x[5]-x[3]*x[4])
print('img k64 per model',len(pm),boot([np.mean(v) for v in pm.values()]))
# dedupe siblings: image drop merged/hf duplicates
dup=['vidore/colpali-v1.2-hf','vidore/colpali-v1.2-merged','vidore/colpali-v1.3-hf','vidore/colqwen2-v1.0-hf','tsystems/colqwen2.5-3b-multilingual-v1.0-merged','Metric-AI/ColQwen2.5-3b-multilingual-v1.0','vidore/colpali-v1.1']
print('img dedupe',boot([x[5]-x[3]*x[4] for x in img if x[0] not in dup])) 
# float16 baseline models
print([ (m) for m in d if any('float16' in str(v.get('note','')) for v in d[m].values())][:30])
