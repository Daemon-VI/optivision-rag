import pandas as pd,numpy as np
df=pd.read_pickle("h4.pkl"); rng=np.random.default_rng(2)
ms=df.model.unique(); g={m:df[df.model==m] for m in ms}
def fit(d,kind):
    x=np.log2(1/d.keep) if kind=="keep" else -np.log2(d.vpd)
    X=np.column_stack([np.ones(len(d)),(d.ds=="scifact").astype(float),x,x**2]); y=d.loss.values
    b=np.linalg.lstsq(X,y,rcond=None)[0]; return b[1], 1-((y-X@b)**2).sum()/((y-y.mean())**2).sum()
for kind in ["keep","vpd"]:
    est=fit(df,kind); bs=[fit(pd.concat([g[m] for m in rng.choice(ms,len(ms))]),kind) for _ in range(1000)]
    bs=np.array(bs); print(kind,"text coef %.4f R2 %.3f"%est,"coefCI",np.percentile(bs[:,0],[2.5,97.5]).round(4),"R2CI",np.percentile(bs[:,1],[2.5,97.5]).round(3))
# leave-one-model-out CV RMSE
for kind in ["keep","vpd"]:
    e=[]
    for m in ms:
        tr=df[df.model!=m];te=df[df.model==m]
        f=lambda d:(np.column_stack([np.ones(len(d)),(d.ds=="scifact").astype(float),(np.log2(1/d.keep) if kind=="keep" else -np.log2(d.vpd)),(np.log2(1/d.keep) if kind=="keep" else -np.log2(d.vpd))**2]))
        b=np.linalg.lstsq(f(tr),tr.loss.values,rcond=None)[0]; e+=list(te.loss.values-f(te)@b)
    print(kind,"LOMO RMSE %.4f"%np.sqrt(np.mean(np.square(e))))
