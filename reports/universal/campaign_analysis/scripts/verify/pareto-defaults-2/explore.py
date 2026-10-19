import json
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
from collections import Counter
c=Counter(); ds=Counter()
for m,v in d.items():
    for k,x in v.items():
        ds[k]+=1
        for r in x["rows"]: c[r["label"]]+=1
print(ds)
for l,n in c.most_common(60): print(n,l)
