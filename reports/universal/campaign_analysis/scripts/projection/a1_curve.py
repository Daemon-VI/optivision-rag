"""A1: retention vs k, k/D for PCA projection. Pure project(k) rows and merge-controlled ratio rows."""
from common import *
pd.set_option('display.width',300,'display.max_rows',500,'display.max_columns',40)
# sanity: project(128) on D=128 is a rotation => rel should be 1
s=mp[(mp.D==128)&(mp.k==128)]
print('sanity rotation D=128,k=128 rel: min %.4f max %.4f mean %.5f n=%d'%(s.rel.min(),s.rel.max(),s.rel.mean(),len(s)))
# pure PCA
pure=pp[pp.method=='pca'].copy()
pure['Dc']=pd.cut(pure.D,[0,128,640,5000],labels=['D<=128','320-640','2048-4096'])
rows=[]
for (D,k),g in pure.groupby(['D','k']):
    m=g.groupby('model').ret.mean()
    e,lo,hi=cluster_boot(g.ret,g.model,B=3000)
    rows.append(dict(D=D,k=k,frac=k/D,n_units=len(g),n_models=g.model.nunique(),mean=e,lo=lo,hi=hi,min=g.ret.min(),share98=(g.ret>=0.98).mean(),share95=(g.ret>=.95).mean()))
print('PURE PCA'); print(pd.DataFrame(rows).round(4).to_string())
# merge-controlled
rows=[]
x=mp[mp.k<mp.D]
for (D,k),g in x.groupby(['D','k']):
    e,lo,hi=cluster_boot(g.rel,g.model,B=3000)
    rows.append(dict(D=D,k=k,frac=round(k/D,4),n=len(g),n_models=g.model.nunique(),rel_mean=e,lo=lo,hi=hi,min=g.rel.min()))
print('MERGE-CONTROLLED rel = ret(merge>project k)/ret(merge)'); print(pd.DataFrame(rows).round(4).to_string())
