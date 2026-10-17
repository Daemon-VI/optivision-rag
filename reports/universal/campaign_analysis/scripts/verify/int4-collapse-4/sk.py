import json, re, numpy as np, pandas as pd
from scipy import stats
from optivision.adapters import MODELS
M=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
hub={h["id"]:h for h in json.load(open("C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/hubrank.json"))}
rows=[]
for mid,ds in M.items():
    mi=MODELS.get(mid) if hasattr(MODELS,'get') else None
    rets={};bases={};vp={}
    for d,v in ds.items():
        r=[x for x in v["rows"] if x["label"]=="int4"]
        if not r: continue
        rets[d]=r[0]["retention"];bases[d]=v["baseline"];vp[d]=r[0]["vectors_per_doc"]
    if not rets: continue
    rows.append(dict(id=mid,ret=np.mean(list(rets.values())),mn=min(rets.values()),nds=len(rets),base=np.mean(list(bases.values())),vpp=np.mean(list(vp.values())),
      mod=getattr(mi,'modality',None),dim=getattr(mi,'dimension',None),ev=getattr(mi,'evidence',''),fam=getattr(mi,'family',None),dl=hub.get(mid,{}).get("downloads",0)))
df=pd.DataFrame(df:=rows) if False else pd.DataFrame(rows)
df["f16"]=df.ev.str.contains("float16|fp16|bfloat16",case=False)
print(len(df),df["mod"].value_counts().to_dict(),df.dim.isna().sum())
print(df.sort_values("ret").head(14)[["id","ret","mn","mod","dim","f16","fam"]].to_string())
rng=np.random.default_rng(0)
def sp(x,y,B=4000):
    r=stats.spearmanr(x,y)[0];n=len(x);bs=[]
    for _ in range(B):
        i=rng.integers(0,n,n);bs.append(stats.spearmanr(x[i],y[i])[0])
    return r,np.nanpercentile(bs,[2.5,97.5])
print("dim",sp(np.log2(df.dim.values.astype(float)),df.ret.values))
print("vpp",sp(df.vpp.values,df.ret.values))
for m in ["image","text"]:
    s=df[df["mod"]==m];print(m,len(s),"base",sp(s.base.values,s.ret.values), stats.spearmanr(s.base,s.ret)[1])
print("all base",sp(df.base.values,df.ret.values))
im=df[df["mod"]=="image"];tx=df[df["mod"]=="text"]
print("n deg<0.95", (im.ret<.95).sum(),(tx.ret<.95).sum(), "coll<.5",(im.ret<.5).sum(),(tx.ret<.5).sum())
for th in [.9,.95,.98]:
    a=(tx.ret<th).sum();b=(im.ret<th).sum();print(th,a,len(tx),b,len(im),stats.fisher_exact([[a,len(tx)-a],[b,len(im)-b]]))
print("overall <0.5",(df.ret<.5).sum())
print(im[im.f16].shape,im[~im.f16].shape)
print(stats.mannwhitneyu(im[im.f16].ret,im[~im.f16].ret), (im[im.f16].ret<.95).sum(),(im[~im.f16].ret<.95).sum())
s=im[(im.dim<=640)];print("128-640 img",len(s),(s.ret<.95).sum(),s.ret.min())
print(df[df.dim>=2048][["id","ret","dim"]])
print(df.sort_values("ret").head(10)[["id","ret","dl"]]); 
print("dl share deg",df[df.ret<.95].dl.sum()/df.dl.sum(),"coll",df[df.ret<.5].dl.sum()/df.dl.sum(), df.dl.sum())
# lighton
print(df[df.id.str.contains("lighton",case=False)][["id","ret","mod"]])
# per-dataset image corr
p={}
for mid in im.id:
    for d,v in M[mid].items():
        r=[x for x in v["rows"] if x["label"]=="int4"]
        if r:p.setdefault(d,{})[mid]=r[0]["retention"]
pp=pd.DataFrame(p).dropna();print(pp.shape,stats.spearmanr(pp.iloc[:,0],pp.iloc[:,1]))
q=pp.drop(index=[i for i in pp.index if "nemotron" in i.lower()]);print(q.shape,stats.spearmanr(q.iloc[:,0],q.iloc[:,1]))
print(pp.sort_values(pp.columns[0]).head(5))
# loo spearman dim
x=np.log2(df.dim.values.astype(float))
print("loo-no-nemo dim",stats.spearmanr(x[~df.id.str.contains('nemotron',case=False)],df.ret[~df.id.str.contains('nemotron',case=False)]))
