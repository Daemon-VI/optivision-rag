import itertools, numpy as np, pandas as pd
from scipy import stats
from prep import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
rng=np.random.default_rng(0)
w=wide()
print("units",len(w),"models",w.model.nunique())
print(w[CODECS].describe().round(4))
# quantiles
print(w[CODECS].quantile([0,.05,.25,.5,.75,.95,1]).round(4))
# Kendall W with ties on 6 codecs. Treat a difference < eps as tie (eps=0 and 0.002 and 0.005).
def kendallW(M):
    # M: units x codecs values, higher = better rank
    n,k=M.shape
    R=np.array([stats.rankdata(r) for r in M])
    Rj=R.sum(0); S=((Rj-Rj.mean())**2).sum()
    T=0
    for r in M:
        _,c=np.unique(r,return_counts=True); T+=(c**3-c).sum()
    W=12*S/(n**2*(k**3-k)-n*T)
    return W,R.mean(0)
def quant(M,eps):
    # snap values into bins so near-ties tie: use cumulative clustering within a row
    out=np.zeros_like(M)
    for i,r in enumerate(M):
        o=np.argsort(-r); lvl=0; prev=None
        for j in o:
            if prev is not None and prev-r[j]>eps: lvl+=1
            out[i,j]=-lvl; prev=r[j] if prev is None or prev-r[j]>eps else prev
    return out
M=w[CODECS].values
for eps in [0,0.002,0.005,0.01]:
    Q=quant(M,eps) if eps>0 else M
    Wv,mr=kendallW(Q)
    # bootstrap over MODELS (cluster) 
    models=w.model.unique(); bs=[]
    for b in range(1000):
        pick=rng.choice(models,len(models)); idx=np.concatenate([np.where(w.model.values==m)[0] for m in pick])
        bs.append(kendallW(Q[idx])[0])
    n,k=M.shape; chi=n*(k-1)*Wv
    print(f"eps={eps}: W={Wv:.3f} CI[{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}] chi2={chi:.1f} p={stats.chi2.sf(chi,k-1):.2g} mean ranks(6=best)", dict(zip(CODECS,mr.round(2))))
# by subgroup
for name,mask in [('fp16-inference',w.fp16),('float32-inference',~w.fp16&(w.ds!='scifact')),('scifact',w.ds=='scifact'),('docvqa',w.ds=='docvqa'),('infovqa',w.ds=='infovqa')]:
    Wv,mr=kendallW(quant(w.loc[mask,CODECS].values,0.002)); print(name,mask.sum(),round(Wv,3),dict(zip(CODECS,mr.round(2))))
# pairwise
print("PAIRWISE (eps 0.002): share A>B / tie / A<B, sign test p (ties dropped), n units")
rows=[]
for a,b in itertools.combinations(CODECS,2):
    d=w[a]-w[b]; gt=(d>0.002).sum(); lt=(d<-0.002).sum(); tie=len(d)-gt-lt
    p=stats.binomtest(gt,gt+lt,0.5).pvalue if gt+lt>0 else np.nan
    rows.append((a,b,gt,tie,lt,round(p,4),round(d.mean(),4),np.round(np.percentile([d.sample(len(d),replace=True,random_state=int(s)).mean() for s in range(500)],[2.5,97.5]),4)))
print(pd.DataFrame(rows,columns=['A','B','A>B','tie','A<B','p_sign','meandiff','boot_ci_units']).to_string())
print("n tests pairwise:",len(rows))
# expected-order violation counts
exp=[('float16','int8(per_vector)'),('int8(per_vector)','int4'),('int8(per_vector)','int4(mean)'),('int4(mean)','lloyd2(7)'),('lloyd2(7)','binary'),('int4','lloyd2(7)'),('int4','binary')]
print("violations (A worse than B by >0.005):")
for a,b in exp: print(a,'<',b,((w[a]-w[b])<-0.005).sum(),'/',len(w))
w.to_csv('wide_ret.csv',index=False)
