import json,glob,collections
R="C:/Users/Lenovo/projects/optivision-package/reports/universal/"
fs=glob.glob(R+"campaign/*/wide/*.json")+glob.glob(R+"wide/*.json")
print(len(fs))
c=collections.Counter()
for f in fs:
    try: d=json.load(open(f,encoding="utf8"))
    except Exception as e: continue
    if not isinstance(d,dict) or "rows" not in d: continue
    for r in d["rows"]:
        if "int4" in r["label"] and r["group"]=="codec": c[r["label"]]+=1
print(c)
d=json.load(open(fs[0],encoding="utf8")); print(d["rows"][0].keys()); print([r["label"] for r in d["rows"]][:60])
