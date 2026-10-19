import json,numpy as np
from scipy.stats import spearmanr,wilcoxon
M=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
D={}
for m,ds in M.items():
    k=list(ds)
    if len(k)<2: continue
    a,b=[{r['label']:r['retention'] for r in ds[x]['rows']} for x in k[:2]]
    D[m]=(k[:2],a,b)
print(len(D), {tuple(v[0]) for v in D.values()})
rho={}
for m,(k,a,b) in D.items():
    L=sorted(set(a)&set(b))
    rho[m]=spearmanr([a[l] for l in L],[b[l] for l in L])[0]
v=np.array(list(rho.values()));print("median rho",np.median(v),v.min(),v.max(),len(v))
rng=np.random.default_rng(0);bs=[np.median(rng.choice(v,len(v))) for _ in range(5000)];print(np.percentile(bs,[2.5,97.5]))
# LOMO
ms=list(D); 
def run(models,tag):
    R1=[];R2=[];E1=[];E2=[]
    for m in models:
        k,a,b=D[m];L=sorted(set(a)&set(b))
        pred=[]
        for l in L:
            o=[D[x][2][l] for x in D if x!=m and l in D[x][2] and x in models]
            pred.append(np.median(o) if o else np.nan)
        pred=np.array(pred);y=np.array([b[l] for l in L]);own=np.array([a[l] for l in L])
        ok=~np.isnan(pred)
        R1.append(spearmanr(pred[ok],y[ok])[0]);R2.append(spearmanr(own[ok],y[ok])[0])
        E1.append(np.abs(pred[ok]-y[ok]).mean());E2.append(np.abs(own[ok]-y[ok]).mean())
    R1,R2,E1,E2=map(np.array,(R1,R2,E1,E2))
    print(tag,len(models),"rho other",np.median(R1),"own",np.median(R2),"wilc",wilcoxon(R1,R2).pvalue,"MAE",np.median(E1),np.median(E2),wilcoxon(E1,E2).pvalue,"own wins",(R2>R1).sum())
    return R1,R2
run(ms,"all")
# fix: which models have both
# dedupe siblings by family prefix
import re
def fam(m): return re.sub(r'[-_ ]?(\d+(\.\d+)?b|base|large|small|flash|4b|8b|3b|2b).*$','',m.split('/')[-1].lower())
seen={};
for m in ms: seen.setdefault(fam(m),m)
print(len(seen),list(seen))
run(list(seen.values()),"dedup")
# drop top extreme: models with lowest rho
ex=[m for m in ms if rho[m]>0.6]; run(ex,"drop low-rho")
# partial: within-compression band
