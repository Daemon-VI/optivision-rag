import json
from optivision.adapters import MODELS
M=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
print(len(M))
for mid,ds in M.items():
    labs=set(x["label"] for v in ds.values() for x in v["rows"])
    if "int4" not in labs: print(mid,sorted(labs)[:12], mid in MODELS)
