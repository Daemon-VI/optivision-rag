import json,glob,itertools,random
import numpy as np
from scipy.stats import fisher_exact
base="C:/Users/Lenovo/projects/optivision-package/reports/universal"
# raw wide files
R={}
for f in glob.glob(base+"/campaign/*/wide/*.json")+glob.glob(base+"/wide/*.json"):
    try: j=json.load(open(f))
    except: continue
    if "rows" not in j: continue
    m=f.replace("\\","/").split("/")
    for r in j["rows"]:
        if r.get("label")=="int4":
            R.setdefault(f.split("/")[-1].rsplit("_",1)[0] if "campaign" not in f else m[-3],{})[j["dataset"]]=r["retention:ndcg@5"] if "retention:ndcg@5" in r else None
print(len(R))
for k,v in sorted(R.items(), key=lambda kv:min(x for x in kv[1].values() if x is not None) if any(x is not None for x in kv[1].values()) else 9)[:12]: print(k,v)
print([k for k,v in R.items() if all(x is None for x in v.values())])
