import json
j=json.load(open("C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/lightonai--LateOn/wide/LateOn_scifact.json"))
print(j["baseline"])
for r in j["rows"]:
    if r["label"] in("int4","int4(mean)","float32","binary"): print({k:v for k,v in r.items() if k.startswith(("ndcg","retention","label"))})
m=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]["lightonai/LateOn"]
for d,v in m.items():
    print(d,v["baseline"],v.get("note"))
    for r in v["rows"]:
        if r["label"] in ("int4","int4(mean)"): print(r)
