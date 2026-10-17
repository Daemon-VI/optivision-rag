import json,collections
d=json.load(open("src/optivision/presets/measured.json"))["models"]
c=collections.Counter()
for m,v in d.items():
    for ds,x in v.items():
        for r in x["rows"]: c[r["label"]]+=1
for l,n in sorted(c.items(), key=lambda t:-t[1]): print(n,l)
print(len(c))
