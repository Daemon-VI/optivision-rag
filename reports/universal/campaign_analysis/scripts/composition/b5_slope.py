import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
P=pd.read_pickle(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/pairs.pkl")
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',300)
A=P[(P.pred_type=='all_singles')&(P.nst==2)].copy()
A['codec']=A.stages.map(lambda s: s[-1] if kind(s[-1])=='codec' else '-')
A=A[A.codec.isin(['int4','binary','lloyd2(7)','int4(mean)','int8(per_vector)'])|A.kinds.eq('merge>project')]
A['first']=A.parts.map(lambda p:p[0]); A['second']=A.parts.map(lambda p:p[1])
A['y']=np.log(np.clip(A.act,1e-3,None)/A['first'])   # conditional retention of 2nd stage given 1st
A['x']=np.log(np.clip(A.second,1e-3,None))
A['grp']=np.where(A.ds=='scifact','text','image')
rng=np.random.default_rng(1)
def fit(g):  # through origin
    x=g.x.values;y=g.y.values
    return (x*y).sum()/(x*x).sum()
def boot(g,B=2000):
    models=g.model.unique(); sub={m:g[g.model==m] for m in models}
    out=[]
    for _ in range(B):
        ms=rng.choice(models,len(models)); gg=pd.concat([sub[m] for m in ms]); out.append(fit(gg))
    return np.percentile(out,[2.5,97.5])
rows=[]
for (k,c),g in A.groupby(['kinds','codec']):
    g=g[g.x<-0.005]
    if len(g)<10: continue
    for gr,gg in [('all',g)]+[(a,b) for a,b in g.groupby('grp')]:
        if len(gg)<10 or gg.model.nunique()<5: continue
        b=fit(gg);lo,hi=boot(gg,1000)
        rows.append(dict(kinds=k,codec=c,grp=gr,n=len(gg),models=gg.model.nunique(),beta=b,lo=lo,hi=hi,median_x=gg.x.median()))
print(pd.DataFrame(rows).round(3).to_string())
# merge>project pooled slope: second=project, and first=merge
