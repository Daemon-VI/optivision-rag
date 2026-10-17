import json,glob,os,numpy as np
R="C:/Users/Lenovo/projects/optivision-package/reports/universal/"
fs=glob.glob(R+"campaign/*/wide/*.json")+glob.glob(R+"wide/*.json")
M="ndcg@5"; P={}
for f in fs:
    d=json.load(open(f,encoding="utf8"))
    if "rows" not in d: continue
    lab={r["label"]:r for r in d["rows"]}
    if "int4" not in lab: continue
    mid=f.replace("\\","/").split("/")
    key=f.replace("\\","/")
    P.setdefault(key,[]).append((f,lab,d))
print(len(P), [k for k,v in P.items() if len(v)>1])
rows=[]
for k,v in P.items():
    f,lab,d=v[0]
    a,b=lab["int4"],lab["int4(mean)"]
    rows.append(dict(k=os.path.basename(k)[:45],ds=d["dataset"][:8],dim=a["dim"],ra=a["retention:"+M],rb=b["retention:"+M],la=a["retention_ci:"+M][0],lb=b["retention_ci:"+M][0]))
print(len(rows))
ra=np.array([r["ra"] for r in rows]);rb=np.array([r["rb"] for r in rows]);la=np.array([r["la"] for r in rows]);lb=np.array([r["lb"] for r in rows])
print("lb<.95 plain",(la<.95).sum(),"centred",(lb<.95).sum())
for r in rows:
    if r["la"]<.95 or r["lb"]<.95: print(r)
dlt=rb-ra
print("mean",dlt.mean(),"median",np.median(dlt),"better",(dlt>0).sum(),"within.003",(abs(dlt)<=.003).sum(), "worse",(dlt<-.003).sum())
rng=np.random.default_rng(0)
print("excl failing",np.mean(dlt[la>=.95]),np.median(dlt[la>=.95]), "n",(la>=.95).sum())
print("worst centred-minus-plain",sorted(dlt)[:5])
bs=[np.median(rng.choice(dlt,len(dlt))) for _ in range(5000)]; print("median CI",np.percentile(bs,[2.5,97.5]))
# sign test on non-ties
nz=dlt[abs(dlt)>.003]; print("beyond .003: pos",(nz>0).sum(),"neg",(nz<0).sum())
# compounded
def cov(lbl,th,sub=None):
    n=c=0
    for k,v in P.items():
        f,lab,d=v[0]
        if sub and not sub(d): continue
        if lbl in lab: n+=1; c+= lab[lbl]["retention_ci:"+M][0]>=th
    return c,n
for th in (.95,.99):
  for s,name in ((None,"all"),(lambda d:"scifact" not in d["dataset"].lower(),"image")):
    print(th,name,[ (l,cov(l,th,s)) for l in ("hierarchical_merge(0.25) > int4","hierarchical_merge(0.25) > int4(mean)")])
print("--- point-retention coverage")
def covp(lbl,th,sub=None):
    n=c=0
    for k,v in P.items():
        f,lab,d=v[0]
        if sub and not sub(d): continue
        if lbl in lab: n+=1; c+= lab[lbl]["retention:"+M]>=th
    return c,n
for th in (.95,.99):
  for s,name in ((None,"all"),(lambda d:"scifact" not in d["dataset"].lower(),"image")):
    print(th,name,[ (l,covp(l,th,s)) for l in ("hierarchical_merge(0.25) > int4","hierarchical_merge(0.25) > int4(mean)")])
# remove sibling/nemotron, extremes
idx=[i for i,r in enumerate(rows)]
keep=[i for i,r in enumerate(rows) if "nemotron" not in r["k"] and "mxbai" not in r["k"]and "LateOn" not in r["k"]]
print("excluding 3 clusters",len(keep),"plain lb<.95",(la[keep]<.95).sum(),"mean d",dlt[keep].mean(),"median",np.median(dlt[keep]))
sc=[i for i,r in enumerate(rows) if "scifact" in r["k"]]; im=[i for i in range(len(rows)) if i not in sc]
for n,ii in (("sci",sc),("img",im)): print(n,len(ii),"plain<.95",(la[ii]<.95).sum(),"cent<.95",(lb[ii]<.95).sum(),"med d",np.median(dlt[ii]),"better",(dlt[ii]>0).sum())
