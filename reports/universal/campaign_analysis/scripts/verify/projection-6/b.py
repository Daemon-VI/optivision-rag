import json,re,numpy as np
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
X=[];Y=[];P=[];M=[]
for m,dd in d.items():
  for ds,e in dd.items():
    R={r['label']:r['retention'] for r in e['rows']}
    for l,v in R.items():
      mm=re.fullmatch(r'(hierarchical_merge\([\d.]+\)) > project\((\d+)\)',l)
      if mm and mm.group(1) in R and f'project({mm.group(2)})' in R:
        X.append(R[mm.group(1)]*R[f'project({mm.group(2)})']);Y.append(v);P.append(R[f'project({mm.group(2)})']);M.append(R[mm.group(1)])
X,Y,P,M=map(np.array,(X,Y,P,M))
print(len(X),'corr product vs obs',np.corrcoef(X,Y)[0,1],'mean abs err pp',np.abs(X-Y).mean()*100)
print('project ret range',P.min(),np.median(P),P.max(),' merge ret',M.min(),np.median(M),M.max())
# subset where combined shortfall >2pp
s=(1-X)>0.03;print('n with product<0.97',s.sum(),'mean err pp',((Y-X)[s]).mean()*100,'abs',np.abs(Y-X)[s].mean()*100, 'corr of shortfall',np.corrcoef(1-X[s],1-Y[s])[0,1])
# nemotron-4B project>binary excess by k
for m in d:
  if 'nemotron-colembed-vl-4b' in m:
    for ds,e in d[m].items():
      R={r['label']:r['retention'] for r in e['rows']}
      print(ds,'binary alone',R.get('binary'),[(l,round(v/R[l.split(' > ')[0]]-R['binary'],3)) for l,v in R.items() if re.fullmatch(r'project\(\d+\) > binary',l)])
