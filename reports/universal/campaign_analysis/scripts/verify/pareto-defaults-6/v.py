import os,json,glob,numpy as np,sys
R="C:/Users/Lenovo/projects/optivision-package/reports/universal/"
files=glob.glob(R+"campaign/*/wide/*.json")+glob.glob(R+"wide/*.json")
rng=np.random.default_rng(0)
def load(files):
    P=[]
    for f in files:
        d=json.load(open(f))
        if 'rows' not in d: continue
        rows=d['rows']; b=[r for r in rows if r['label']=='float32']
        if not b: continue
        b=b[0]; X=[]
        for r in rows:
            if r['label']=='float32' or not r.get('scan_ms_per_query'): continue
            X.append((r['group'],r['label'],r['scan_ms_per_query']/b['scan_ms_per_query'],r['vectors_per_doc']/b['vectors_per_doc'],r['dim']/b['dim'],r['bytes_per_doc']/b['bytes_per_doc']))
        P.append((f,X,b['scan_ms_per_query']))
    return P
def fit(X):
    A=np.array([[1,np.log(x[3]),np.log(x[4])] for x in X]);y=np.log([x[2] for x in X])
    c,*_=np.linalg.lstsq(A,y,rcond=None);res=y-A@c;return c,1-res.var()/y.var()
def boot(P,B=1000):
    out=[]
    for _ in range(B):
        idx=rng.integers(0,len(P),len(P));X=[x for i in idx for x in P[i][1]];out.append(fit(X)[0])
    return np.percentile(out,[2.5,97.5],axis=0)
def rep(P,name):
    X=[x for p in P for x in p[1]];c,r2=fit(X);print(name,len(P),len(X),c,r2);print(boot(P)[:,1:])
P=load(files);rep(P,'all')
# float32 rows scan spread
print('f32 scan medians',np.median([p[2] for p in P]))
# within-file float32 vs baseline scan
# drop each model dup: dedupe by file basename
seen={};
for p in P: seen[p[0].split('/')[-1]]=p
rep(list(seen.values()),'dedup')
X=[x for p in P for x in p[1]]
for g in sorted(set(x[0] for x in X)):
    s=[x[2] for x in X if x[0]==g];print(g,len(s),np.median(s),np.percentile(s,[25,75]))
cod={}
for x in X:
    if x[0]=='codec': cod.setdefault(x[1],[]).append(x[2])
for k,v in cod.items(): print(k,len(v),np.median(v))
for lab in ['hierarchical_merge(0.25) > int8(per_vector)']:
    s=[x[2] for x in X if x[1]==lab];print(lab,len(s),np.median(s),np.percentile(s,[25,75]),np.median([x[3] for x in X if x[1]==lab]))
# vectors-only fit, residual: codec rows slowdown
for g in ['token','token+codec']:
    s=[(x[2]/x[3]) for x in X if x[0]==g];print(g,'scan/vec ratio',np.median(s),np.percentile(s,[5,95]))
# time vs absolute within-file noise: float32 vs float16 row
r=[]
for f in files:
    d=json.load(open(f));
    if 'rows' in d:
        L={x['label']:x for x in d['rows']}
        if 'float16' in L: r.append(L['float16']['scan_ms_per_query']/L['float32']['scan_ms_per_query'])
print('f16/f32',np.median(r),np.percentile(r,[5,95]))
from scipy.stats import spearmanr
print(spearmanr([1/x[5] for x in X],[1/x[2] for x in X]))
# by dataset
for key in ['docvqa','infovqa','scifact']:
    print(key,np.median([p[2] for p in P if key in p[0]]),sum(key in p[0] for p in P))
