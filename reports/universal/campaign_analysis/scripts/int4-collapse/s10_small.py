import pandas as pd,numpy as np
from scipy import stats
g=pd.read_pickle('model_level.pkl')
tx=g[g['mod']=='text'].copy()
lo=tx[tx.dim<128]; hi=tx[tx.dim==128]
tab=[[int((lo.int4<.95).sum()),int((lo.int4>=.95).sum())],[int((hi.int4<.95).sum()),int((hi.int4>=.95).sum())]]
print('text dim<128 vs ==128 degraded table',tab,'Fisher',stats.fisher_exact(tab))
print(lo[['model','dim','int4','bin','int4m']].round(3).to_string())
# Clopper-Pearson
def cp(k,n,a=0.05):
    lo=0 if k==0 else stats.beta.ppf(a/2,k,n-k+1); hi=1 if k==n else stats.beta.ppf(1-a/2,k+1,n-k); return round(lo,3),round(hi,3)
for lab,k,n in [('text degraded',6,15),('image degraded',2,40),('image 128-640 degraded',0,36),('text 128-d degraded',3,12),('image >=2048 degraded',2,4)]: print(lab,k,n,cp(k,n))
# family spreads
tx['org']=tx.model.str.split('/').str[0]
print(tx.groupby('org').int4.agg(['count','min','max']).round(3))
print(tx[tx.org.isin(['lightonai','mixedbread-ai','LiquidAI'])][['model','dim','vpd','base','int4','bin','int4m']].sort_values('int4').round(3).to_string())
# dense/sparse signature: for the degraded models, ratio of binary deficit to int4 deficit
d=g[g.deg].copy(); d['def4']=1-d.int4; d['defb']=1-d.bin; d['ratio_b_over_4']=d.defb/d.def4; print(d[['model','def4','defb','ratio_b_over_4','int8','int4m']].round(3).to_string())
dense=d[d.bin<=0.95]; print('dense-type n',len(dense),'Pearson def4~defb',np.round(stats.pearsonr(dense.def4,dense.defb),3), 'Spearman',np.round(stats.spearmanr(dense.def4,dense.defb),3))
# non-degraded: binary deficit exists (sign code) but int4 not
nd=g[~g.deg&(g['mod']=='text')]; print('text nondegraded n',len(nd),'bin median',nd.bin.median().round(3),'min',nd.bin.min().round(3))
# binary deficit of text models vs image 128-d
im=g[(g['mod']=='image')&(g.dim==128)]; print('binary median image128',im.bin.median().round(3),'text',g[g['mod']=='text'].bin.median().round(3))
u=stats.mannwhitneyu(im.bin,g[g['mod']=='text'].bin); print('MWU binary image128 vs text p',u.pvalue)
