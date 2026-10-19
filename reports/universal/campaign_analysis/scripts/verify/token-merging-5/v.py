import json,numpy as np,collections
from scipy.stats import wilcoxon
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
rng=np.random.default_rng(0)
cells=[];ad_all=[]
for m,dsd in d.items():
  for ds,e in dsd.items():
    rows=e['rows']
    base=[r for r in rows if r['pipeline']['stages']==[]]
    vb=base[0]['vectors_per_doc']
    hier=[];ads=[]
    for r in rows:
      st=r['pipeline']['stages']
      if len(st)==1 and st[0]['stage']=='hierarchical_merge' and st[0].get('max_distance') is None:
        hier.append((r['vectors_per_doc']/vb,1-r['retention']))
      if len(st)==1 and st[0]['stage']=='adaptive_merge':
        ads.append((st[0],r['vectors_per_doc']/vb,1-r['retention'],r['vectors_per_doc']))
    hier.sort()
    for s,k,l,v in ads:
      ad_all.append((m,ds,s.get('radius'),s.get('center'),k,l,v,len(hier)))
      if len(hier)<2 or k<0.1: continue
      x=np.log([h[0] for h in hier]);y=[h[1] for h in hier]
      if np.log(k)<x[0]-1e-9 or np.log(k)>x[-1]+1e-9: continue
      cells.append((m,ds,s.get('radius'),s.get('center'),k,l,float(np.interp(np.log(k),x,y))))
print(len(ad_all),len(cells))
print(collections.Counter((a[1][:10],a[2],a[3]) for a in ad_all))
for ds in ['vidore/docvqa_test_subsampled',None]:
  pass
dss=sorted(set(c[1] for c in cells));print(dss)
def rep(sub,name):
  a=np.array([c[5]-c[6] for c in sub])*100
  if len(a)<3: print(name,len(a));return
  bs=[np.median(rng.choice(a,len(a))) for _ in range(5000)]
  bm=[rng.choice(a,len(a)).mean() for _ in range(5000)]
  try:p=wilcoxon(a).pvalue
  except: p=float('nan')
  print(name,len(a),'meankeep',np.median([c[4] for c in sub]).round(3),'ad',np.mean([c[5] for c in sub])*100,'hi',np.mean([c[6] for c in sub])*100,'mean diff',a.mean().round(2),np.percentile(bm,[2.5,97.5]).round(2),'median',np.median(a).round(2),'p',round(p,3),'worse',(a>0).sum())
for ds in dss: rep([c for c in cells if c[1]==ds],ds)
print('--- radius 0.6 only, no center')
for ds in dss: rep([c for c in cells if c[1]==ds and c[2]==0.6],ds)
print('--- no center')
for ds in dss: rep([c for c in cells if c[1]==ds and c[3] is None],ds)
print('settings',collections.Counter((c[2],c[3]) for c in cells))
# per-model
for c in cells:
  if 'scifact' in c[1]: print(c[0],c[2],c[3],round(c[4],3),round(c[5]*100,2),round(c[6]*100,2))
print('answerai')
for a in ad_all:
  if 'answerai' in a[0]: print(a)
for m,dsd in d.items():
  if 'answerai' in m:
    for ds,e in dsd.items():
      for r in e['rows']:
        s=r['pipeline']['stages']
        if len(s)==1 and s[0]['stage']=='hierarchical_merge': print(s[0],r['retention'],r['vectors_per_doc'])
# robustness: exclude scifact extreme (the model with largest |diff|), per-family
sf=[c for c in cells if 'scifact' in c[1]]
for i in range(len(sf)):
  rep(sf[:i]+sf[i+1:],'leave-out '+sf[i][0][:30])
