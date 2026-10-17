import pandas as pd,numpy as np
from scipy.stats import spearmanr
df=pd.read_pickle("h.pkl")
df=df[df.label.str.startswith("hierarchical_merge(")].copy()
print(len(df), df.groupby("ds").agg(n=("loss","size"),m=("model","nunique")))
print(df.groupby("model").size().value_counts())
df.to_pickle("h4.pkl")
def sp(d,x):
    return spearmanr(d.loss,d[x])[0]
for name,d in [("pooled",df)]+[(k,g) for k,g in df.groupby("ds")]:
    print(name,len(d),"vpd %.3f keep %.3f"%(sp(d,"vpd"),sp(d,"keep")))
# without scifact-> image pooled, and loss from raw ret clipped?
im=df[df.ds!="scifact"]; print("image pooled",sp(im,"vpd"),sp(im,"keep"))
# model-cluster bootstrap of difference |rho_vpd|-|rho_keep|
rng=np.random.default_rng(0)
ms=df.model.unique(); g={m:df[df.model==m] for m in ms}
def diff(d): return abs(sp(d,"vpd"))-abs(sp(d,"keep"))
for name,sub in [("pooled",ms),("docvqa",None),("infovqa",None),("scifact",None)]:
    d0=df if name=="pooled" else df[df.ds==name]
    mm=d0.model.unique(); gg={m:d0[d0.model==m] for m in mm}
    bs=[]
    for _ in range(1000):
        s=rng.choice(mm,len(mm)); bs.append(diff(pd.concat([gg[m] for m in s])))
    print(name,"diff %.3f"%diff(d0),np.percentile(bs,[2.5,97.5]).round(3), "P(diff>0)=%.2f"%np.mean(np.array(bs)>0))
