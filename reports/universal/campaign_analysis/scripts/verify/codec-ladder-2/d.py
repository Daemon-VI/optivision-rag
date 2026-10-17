import json,numpy as np
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
import sys
sys.path.insert(0,"x")
try:
    from optivision.adapters import MODELS
except Exception as e: MODELS={};print(e)
for m,ds in sorted(d.items(), key=lambda kv: min(v['rows'][0]['retention'] for v in kv[1].values())):
    for k,v in ds.items():
        R={r["label"]:r for r in v["rows"]}
        if 'int4' in R:
            mi=MODELS.get(m); 
            print(f"{m[:48]:48s} {k[-12:]:12s} {getattr(mi,'dimension','?')} int4={R['int4']['retention']:.3f} mean={R['int4(mean)']['retention']:.3f} vpd={R['int4']['vectors_per_doc']}")
