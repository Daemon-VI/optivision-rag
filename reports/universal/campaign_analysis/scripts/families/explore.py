import json
d=json.load(open("measured.json"))["models"]
print(len(d))
for m,v in d.items():
    for ds,x in v.items():
        print(m,ds,x["baseline"],x["n_docs"],x["n_queries"],len(x["rows"]),str(x.get("note"))[:50])
r=list(d.values())[0]
x=list(r.values())[0]
print(x["rows"][:3])
print(list(x.keys())); print(x["rows"][:2])
