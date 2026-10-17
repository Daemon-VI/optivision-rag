import json,collections
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
labs=collections.Counter()
for m,ds in d.items():
    for k,v in ds.items():
        for r in v["rows"]:
            if "int4" in r["label"] and "+" not in r["label"] and "token" not in r["label"]: labs[r["label"]]+=1
print(labs.most_common(10))
