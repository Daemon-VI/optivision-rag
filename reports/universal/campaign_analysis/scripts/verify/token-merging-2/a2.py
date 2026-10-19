import pandas as pd,numpy as np
from scipy.stats import spearmanr
df=pd.read_pickle("h4.pkl"); rng=np.random.default_rng(1)
print(df.groupby("ds").bv.describe().round(0))
bins=[0,30,50,75,100,150,250,500,5000]
df["bin"]=pd.cut(df.vpd,bins)
print(df.groupby(["bin","ds"],observed=True).loss.agg(["size","mean"]).unstack().round(4))
# fixed ratio: does bv predict loss within image datasets?
print("within ratio, image, spearman(loss,bv) [model-cluster bootstrap]")
for ds in ["docvqa","infovqa","scifact"]:
  for r in [0.5,0.33,0.25,0.1]:
    d=df[(df.ds==ds)&(df.ratio==r)]
    rho=spearmanr(d.loss,d.bv)[0]
    bs=[spearmanr(*(lambda s:(s.loss,s.bv))(d.iloc[rng.integers(0,len(d),len(d))]))[0] for _ in range(1000)]
    print(ds,r,len(d),"%.2f"%rho,np.nanpercentile(bs,[2.5,97.5]).round(2), "loss mean %.4f"%d.loss.mean())
# partial: rank-based within-ratio demeaned, pooled image
im=df[df.ds!="scifact"].copy()
for c in ["loss","bv"]:
    im[c+"r"]=im.groupby(["ds","ratio"])[c].rank()
print("pooled-image within ds,ratio rank corr", spearmanr(im.lossr,im.bvr))
# dedup siblings: one per org family? drop ColNetraEmbed and extreme
for drop in ["Cognitive-Lab/ColNetraEmbed"]:
    d=df[df.model!=drop]
    print("drop",drop,spearmanr(d.loss,d.vpd)[0],spearmanr(d.loss,d.keep)[0])
# text vs image at equal keep
for r in [0.5,0.33,0.25,0.1]:
    t=df[(df.ds=="scifact")&(df.ratio==r)]; i=df[(df.ds!="scifact")&(df.ratio==r)]
    print(r,"text %.4f image %.4f medvpd %.0f vs %.0f"%(t.loss.mean(),i.loss.mean(),t.vpd.median(),i.vpd.median()))
# matched bins w/ model-cluster bootstrap
def md(d,lo,hi):
    t=d[(d.ds=="scifact")&d.vpd.between(lo,hi,inclusive="left")];i=d[(d.ds!="scifact")&d.vpd.between(lo,hi,inclusive="left")]
    return t.loss.mean()-i.loss.mean(),len(t),len(i)
ms=df.model.unique(); g={m:df[df.model==m] for m in ms}
for lo,hi in [(50,100),(100,150),(75,150),(150,250),(30,75)]:
    est=md(df,lo,hi); bs=[]
    for _ in range(2000):
        s=rng.choice(ms,len(ms)); x=md(pd.concat([g[m] for m in s]),lo,hi)[0]; bs.append(x)
    print(lo,hi,"diff %.4f n=%d/%d"%est,np.nanpercentile(bs,[2.5,97.5]).round(4))
# regression: loss ~ ds + log(vpd) vs log(keep), image only/pooled
def fit(X,y):
    X=np.column_stack([np.ones(len(y))]+X); b=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@b; return b,1-r.var()/y.var()
y=df.loss.values; ist=(df.ds=="scifact").values.astype(float)
for nm,x in [("log keep",np.log2(1/df.keep)),("log vpd",-np.log2(df.vpd))]:
    for nm2,pp in [("lin",lambda z:z),("sq",lambda z:z**2)]:
        b,r2=fit([ist,pp(x.values)],y); print(nm,nm2,"R2 %.3f"%r2,b.round(4))
b,r2=fit([ist,((np.log2(1/df.keep))**2).values,np.log2(df.bv).values],y); print("claimed OLS",b.round(4),r2)
# dataset coefficient with vpd model
for nm,x in [("vpd",(-np.log2(df.vpd)).values),("keep",np.log2(1/df.keep).values)]:
    for ex in [True]:
        b,r2=fit([ist,x,x**2],y); print(nm,"quad+ds",b.round(4),"R2 %.3f"%r2)
