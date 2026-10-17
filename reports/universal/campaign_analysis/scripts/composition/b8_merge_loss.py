import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
P=pd.read_pickle(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/pairs.pkl")
df=load()
vpd=df[df.label=='float32'].set_index(['model','ds']).vpd
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',300)
A=P[(P.pred_type=='all_singles')&(P.kinds=='merge>codec')].copy(); A['err']=A.act-A.pred
A['codec']=A.stages.map(lambda s:s[1]); A=A[A.codec.isin(['binary','int4','lloyd2(7)','int4(mean)'])]
A['mloss']=1-A.parts.map(lambda p:p[0]); A['closs']=1-A.parts.map(lambda p:p[1])
A['vpd']=[vpd[(m,d)] for m,d in zip(A.model,A.ds)]
A['grp']=np.where(A.ds=='scifact','text','image')
A['mb']=pd.cut(A.mloss,[-1,0.01,0.03,0.08,0.2,1])
print(A.groupby(['mb','grp']).apply(lambda x:pd.Series(dict(n=len(x),models=x.model.nunique(),bias=x.err.mean(),med=x.err.median(),mean_vpd=x.vpd.mean()))).round(4))
# regression of err on mloss, closs, group (OLS with cluster-robust-ish by bootstrap)
def ols(X,y): return np.linalg.lstsq(X,y,rcond=None)[0]
for cdc in ['binary','int4','lloyd2(7)','int4(mean)']:
    g=A[A.codec==cdc]
    X=np.c_[np.ones(len(g)),g.mloss,g.closs,(g.grp=='text').astype(float)]
    b=ols(X,g.err.values)
    models=g.model.unique();sub={m:g[g.model==m] for m in models};rng=np.random.default_rng(3);bs=[]
    for _ in range(1000):
        gg=pd.concat([sub[m] for m in rng.choice(models,len(models))])
        Xb=np.c_[np.ones(len(gg)),gg.mloss,gg.closs,(gg.grp=='text').astype(float)]
        bs.append(ols(Xb,gg.err.values))
    ci=np.percentile(bs,[2.5,97.5],axis=0)
    print(cdc,'n',len(g),'coef [const, mloss, closs, text]',b.round(3),'\n   CI lo',ci[0].round(3),'\n   CI hi',ci[1].round(3))
# image-only high merge-loss cases
print(A[(A.grp=='image')&(A.mloss>0.08)][['model','ds','label','act','pred','err']].head(20).to_string())
print('image rows mloss>0.05:',((A.grp=='image')&(A.mloss>0.05)).sum(),' text rows mloss>0.05:',((A.grp=='text')&(A.mloss>0.05)).sum())
