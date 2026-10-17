import json,collections
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
c=collections.Counter()
ex={}
for m,ds in d.items():
    for k,v in ds.items():
        for r in v["rows"]:
            for s in r["pipeline"]["stages"]:
                c[s["stage"]]+=1; ex.setdefault(s["stage"],(m,r["label"],r["pipeline"]))
print(c)
for k,v in ex.items(): print(k,v)
print(len(d), [ (k,len(v)) for m,v in list(d.items())[:3] for k in v])
