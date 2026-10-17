import json,numpy as np
from scipy.stats import spearmanr
M=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
out={}
for m,ds in M.items():
    k=list(ds)
    if len(k)<2: continue
    A={r['label']:r for r in ds[k[0]]['rows']};B={r['label']:r for r in ds[k[1]]['rows']}
    for l in set(A)&set(B): out.setdefault(m,[]).append((A[l]['compression'],A[l]['retention'],B[l]['retention']))
bins=[(0,16),(16,64),(64,1e9)]
for lo,hi in bins:
    r=[]
    for m,rows in out.items():
        x=[(a,b) for c,a,b in rows if lo<=c<hi]
        if len(x)>=8: r.append(spearmanr(*zip(*x))[0])
    print(lo,hi,len(r),np.nanmedian(r))
# rank-by-compression baseline vs own docvqa: predict InfoVQA rank by compression alone
r=[spearmanr([c for c,a,b in rows],[b for c,a,b in rows])[0] for rows in out.values()]
print("compression-only rho median",np.median(r))
