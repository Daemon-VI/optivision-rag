import json
from scipy.stats import spearmanr
M=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
T={m:{x["label"]:x["retention"] for x in ds["BeIR/scifact"]["rows"]} for m,ds in M.items() if "BeIR/scifact" in ds}
print([l for l in next(iter(T.values())) if "lloyd" in l.lower() or "(" not in l][:20])
import re
def lab(m,pat):
    c=[l for l in T[m] if re.fullmatch(pat,l)]; return T[m][c[0]] if c else None
for o in("binary","lloyd2\(7\)"):
    ks=[m for m in T if "int4" in T[m] and lab(m,o) is not None]
    x=[T[m]["int4"] for m in ks];y=[lab(m,o) for m in ks]
    print(o,len(ks),spearmanr(x,y))
    k2=[m for m in ks if T[m]["int4"]>0.9]
    print(" excl collapsed",len(k2),spearmanr([T[m]["int4"] for m in k2],[lab(m,o) for m in k2]))
