import json,numpy as np,pandas as pd,re
M=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
rows=[]
for m,dss in M.items():
    for ds,d in dss.items():
        R={r['label']:r for r in d['rows']}
        for l,r in R.items():
            parts=l.split(' > ')
            if len(parts)<2: continue
            if re.search(r'ward|adaptive|none >|\[|random|int8/vec',l): continue
            if not all(p in R for p in parts): continue
            ps=[R[p]['retention'] for p in parts]
            rows.append(dict(model=m,ds=ds,label=l,n=len(parts),act=r['retention'],pred=float(np.prod(ps)),maxloss=1-min(ps),f16='float16 inf' in d.get('note',''),text='scifact' in ds.lower()))
A=pd.DataFrame(rows);A['err']=A.act-A.pred
print(len(A),A.model.nunique())
A['bin']=pd.cut(A.maxloss,[-9,0.01,0.02,0.03,0.05,0.1,1.01])
def summ(g):
    return pd.Series(dict(n=len(g),mae=g.err.abs().mean(),bias=g.err.mean(),p95=g.err.abs().quantile(.95),worst=g.err.min(),frac_lt01=(g.err<-0.01).mean(),nmod=g.model.nunique()))
print(A.groupby('bin',observed=True).apply(summ).round(4))
big=(A.err<-0.02); rule=A.maxloss<=0.03
print('rule share',rule.mean(),'big',big.sum(),'in rule',(big&rule).sum())
# by text
print(A.groupby(['text','bin'],observed=True).apply(summ).round(4))
# bootstrap by model-cluster of MAE for <=1% and >10%
rng=np.random.default_rng(0)
mods=A.model.unique()
def mae(g,lo,hi): 
    x=g[(g.maxloss>lo)&(g.maxloss<=hi)].err.abs(); return x.mean() if len(x) else np.nan
res=[]
for _ in range(2000):
    s=rng.choice(mods,len(mods)); g=pd.concat([A[A.model==k] for k in s])
    res.append((mae(g,-9,.01),mae(g,.1,1.01)))
res=np.array(res);print('cluster boot MAE<=1%',np.nanpercentile(res[:,0],[2.5,97.5]),'>10%',np.nanpercentile(res[:,1],[2.5,97.5]))
# by model for >10%
h=A[A.maxloss>0.1];print(h.groupby('model').err.agg(['count','mean',lambda x:x.abs().mean()]).round(3))
print(h.groupby('ds').err.agg(['count','mean',lambda x:x.abs().mean()]))
print(h.sort_values('err').head(5)[['model','label','act','pred','err']])
# drop most extreme model
worst=h.groupby('model').err.apply(lambda x:x.abs().sum()).idxmax();print('drop',worst)
B=A[A.model!=worst];print(B[B.maxloss>0.1].err.abs().mean())
# f16 excluded
C=A[~A.f16];print('no f16',len(C),C.groupby('bin',observed=True).apply(summ).round(4)[['n','mae','bias']])
# rule within non-text, and big within rule by dataset
print(A[big&rule].groupby(['text']).size(), A[big&rule].model.nunique(), A[big&rule].groupby('model').size().sort_values().tail(5))
# alt: threshold err<-0.02 within each; precision
print('P(err<-.02 | rule)',(big&rule).sum()/rule.sum(),' P(.|not rule)',(big&~rule).sum()/(~rule).sum())
