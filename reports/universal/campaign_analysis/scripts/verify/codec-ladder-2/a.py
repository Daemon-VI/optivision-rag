import json,collections
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
labs=collections.Counter()
for m,ds in d.items():
    for k,v in ds.items():
        for r in v["rows"]: labs[r["label"]]+=1
print([l for l in labs if 'int' in l.lower() or 'binary' in l or 'lloyd' in l][:40])
print(len(d), sum(len(v) for v in d.values()))
