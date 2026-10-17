import json
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
rows=[]
for m,ds in d.items():
    for k,v in ds.items():
        R={r["label"]:r for r in v["rows"]}
        if 'int4' in R and 'int4(mean)' in R:
            rows.append((m,k,R['int4']['retention'],R['int4(mean)']['retention'],R['int8(per_vector)']['retention'] if 'int8(per_vector)' in R else None, R.get('lloyd2(7)',{}).get('retention'),R.get('binary',{}).get('retention'),R['int4']['bytes_per_doc'],R['int4(mean)']['bytes_per_doc']))
print(len(rows), len([m for m in d]))
import numpy as np
a=np.array([r[2] for r in rows]);b=np.array([r[3] for r in rows])
print("int4<.97",(a<.97).sum(),"<.90",(a<.9).sum(),"mean<.97",(b<.97).sum(),"min mean",b.min(),"median int4",np.median(a))
for t in(.99,):
  print("tail<.99 int4",(a<t).sum(),"mean",(b<t).sum())
for n,i in(('int8',4),('lloyd',5),('bin',6)):
  x=np.array([r[i] for r in rows if r[i] is not None]);print(n,len(x),(x<.99).sum())
for r in sorted(rows,key=lambda r:r[2])[:14]: print(r[:4],r[7],r[8])
print(np.median([r[8]-r[7] for r in rows])/1024)
print([m for m in d if 'nemo' in m.lower() or 'nvidia' in m.lower()])
