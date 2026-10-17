import json
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
for m,dd in d.items():
  for ds,x in dd.items():
    L={r["label"]:r for r in x["rows"]}
    if "int4" in L and L["int4"]["retention"]<.95 or "int4(mean)" in L and L["int4(mean)"]["retention_interval"][0]<.95:
        print(m,ds[:10],L["int4"]["retention"],L["int4(mean)"]["retention"],L["int4(mean)"]["retention_interval"][0])
