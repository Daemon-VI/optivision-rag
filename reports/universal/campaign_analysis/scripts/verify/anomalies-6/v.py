import json,glob,os,numpy as np
from scipy import stats
R="C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/"
rows=[]
for f in sorted(glob.glob(R+"*/run.json")):
    d=json.load(open(f));c=d.get("check_retriever")
    if not c: continue
    m=d["model"];own=c["query_ranks_own_document_first"]
    # baseline recall@1
    r1=[]
    for w in glob.glob(os.path.dirname(f)+"/wide/*.json"):
        try:r1.append((json.load(open(w))["baseline"]["recall@1"],os.path.basename(w)))
        except Exception as e:pass
    rows.append((m,d["kind"],c["pairs"],own,c.get("ok"),d["spec"].get("dtype"),r1,d["started_utc"]))
for r in rows:
    if r[3]<=10 or r[2]!=12: print(r[:6],r[7])
print(len(rows))
from collections import Counter
for k in("image","text"):
    print(k,sorted(Counter(r[2]-r[3] for r in rows if r[1]==k).items()))
cv=[r for r in rows if "ColVec1.1" in r[0]]
print([(r[0],r[3],r[7]) for r in cv])
ot=[r for r in rows if "ColVec1.1-4b" not in r[0]]
n=sum(r[2] for r in ot);k=sum(r[3] for r in ot);print(k,n,k/n)
p=k/n
print("tail9 binom",stats.binom.cdf(9,12,p), "empirical pooled p of <=9:",)
print("models with <=9:",[r[0] for r in rows if r[3]<=9])
a=[(r[0],r[3]) for r in ot]
# fisher
cvk=sum(r[3] for r in cv);cvn=sum(r[2] for r in cv);print(cvk,cvn)
print(stats.fisher_exact([[cvk,cvn-cvk],[sum(r[3] for r in rows if r not in cv),sum(r[2] for r in rows if r not in cv)-sum(r[3] for r in rows if r not in cv)]]))
# overdispersion: fraction of models with <=10 vs binomial expectation
misses=np.array([r[2]-r[3] for r in ot]);print("mean miss",misses.mean(),"var",misses.var(ddof=1),"binom var",12*(1-p)*p)
nm=len(ot);print("expected #models with >=2 misses",nm*(1-stats.binom.cdf(9,12,p)),"observed",(misses>=2).sum(), "n",nm)
# spearman
for k in("text","image"):
    for ds in("scifact","docvqa","infovqa"):
        x=[];y=[]
        for r in rows:
            if r[1]!=k: continue
            for v,fn in r[6]:
                if ds in fn.lower(): x.append(r[2]-r[3]);y.append(v)
        if len(x)>3: print(k,ds,len(x),stats.spearmanr(x,y))
# mean-over-datasets for text
