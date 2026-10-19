import json,glob,random
import numpy as np
from scipy.stats import fisher_exact
B="C:/Users/Lenovo/projects/optivision-package/"
M=json.load(open(B+"src/optivision/presets/measured.json"))["models"]
ret={}
for m,ds in M.items():
    v=[r["retention"] for d in ds.values() for r in d["rows"] if r["label"]=="int4"]
    txt='BeIR/scifact' in ds
    if v: ret[m]=(min(v),txt)
print("missing:",[m for m in M if m not in ret])
# raw files for missing
for m in [m for m in M if m not in ret]:
    fs=glob.glob(B+"reports/universal/**/*"+m.split('/')[1]+"*.json",recursive=True)
    for f in fs:
        if "/wide/" in f.replace("\\","/") or "campaign" in f:
            try:j=json.load(open(f))
            except: continue
            for r in j.get("rows",[]):
                if r.get("label")=="int4": print(m,f.split('/')[-1],r["retention:ndcg@5"])
print("n",len(ret))
col={m for m,(x,t) in ret.items() if x<0.9}
txt={m for m,(x,t) in ret.items() if t}
print("text n",len(txt),"collapsed text",len(col&txt),"image n",len(ret)-len(txt),"collapsed image",len(col-txt))
print("text vs image fisher",fisher_exact([[len(col&txt),len(txt)-len(col&txt)],[len(col-txt),len(ret)-len(txt)-len(col-txt)]]))
# text-only list
for m in sorted(txt,key=lambda m:ret[m][0]): print(round(ret[m][0],3),m)
# lineage groups (own): by org+release line
def grp(m):
    n=m.lower()
    if "nemotron-colembed" in n: return "nemotron"
    if "mxbai-edge" in n: return "mxbai-edge"
    if n.endswith("lateon"): return "lateon"
    if "tomoro" in n: return "tomoro"
    if "sauerkraut" in n: return "sauerkraut"
    if "colvec" in n.lower(): return "colvec"
    if "evie" in n: return "evie"
    if "vultron" in n: return "vultron"
    if "neomme" in n: return "neomme"
    if "colpali" in n: return "colpali"
    if "colqwen2.5-3b-multi" in n: return "colqwen25multi"
    if "colnomic" in n: return "colnomic"
    if "lfm2" in n: return "lfm2"
    if "colbertv2" in n: return "colbertv2"
    return m
G={}
for m in ret: G.setdefault(grp(m),[]).append(m)
multi={g:v for g,v in G.items() if len(v)>1}
print({g:[ (m.split('/')[1],round(ret[m][0],2)) for m in v] for g,v in multi.items() if any(m in col for m in v)})
# sibling test, own groups: models with >=1 sibling
rows=[]
for g,v in multi.items():
    for m in v:
        sib=any(s in col for s in v if s!=m)
        rows.append((m in col,sib,g))
t=np.array([[sum(a and b for a,b,_ in rows),sum(a and not b for a,b,_ in rows)],[sum((not a) and b for a,b,_ in rows),sum((not a) and not b for a,b,_ in rows)]])
print(t,fisher_exact(t))
# cluster-level: groups with >=1 collapsed member; fraction fully collapsed
hit=[g for g,v in multi.items() if any(m in col for m in v)]
print("groups multi",len(multi),"with collapse",hit,[all(m in col for m in multi[g]) for g in hit])
# permutation: permute collapse labels across models in multi, count groups fully collapsed(size2)
allm=[m for v in multi.values() for m in v]; k=sum(m in col for m in allm)
obs=sum(all(m in col for m in v) for v in multi.values() if any(m in col for m in v))
rng=random.Random(0);cnt=0;N=20000
for _ in range(N):
    s=set(rng.sample(allm,k)); 
    o=sum(all(m in s for m in v) for v in multi.values() if any(m in s for m in v))
    cnt+= o>=obs
print("perm: obs fully-collapsed groups",obs,"k",k,"of",len(allm),"p",cnt/N)
# lightonai text, ModernBERT family
for m in sorted(ret):
    if m.startswith(("lightonai","mixedbread")): print(m,round(ret[m][0],3))
