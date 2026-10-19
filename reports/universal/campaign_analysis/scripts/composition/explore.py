import json,collections
m=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
print(len(m))
k=list(m)[0]; print(k, list(m[k]))
d=m[k][list(m[k])[0]]
print({a:b for a,b in d.items() if a!='rows'})
for r in d['rows'][:5]: print(r)
c=collections.Counter()
for mm in m.values():
  for ds in mm.values():
    for r in ds['rows']: c[r['label']]+=1
print(len(c)); 
for l,n in c.most_common(80): print(n,l)
