import pandas as pd,numpy as np
from scipy.stats import spearmanr
pd.set_option('display.width',250)
df=pd.read_csv('table.csv')
W=df[df.codec=='int4'].set_index('tag'); R=df.pivot_table(index='tag',columns='codec',values='ret'); L=df.pivot_table(index='tag',columns='codec',values='lo')
U=W.join(R); U['model']=W['model']; Lo=L.add_prefix('lo_'); U=U.join(Lo)
U['gain']=U.int4m-U.int4
print('int4 min/mean',U.int4.min().round(3),U.int4.mean().round(3),'| int4m min/mean/ #<0.97',U.int4m.min().round(3),U.int4m.mean().round(3),(U.int4m<0.97).sum(),' #lo<0.95',(U.lo_int4m<0.95).sum())
print('gain stats: units int4m<int4-0.01:',(U.gain<-0.01).sum(),' min gain',U.gain.min().round(3), U.gain.idxmin())
print(U[U.gain<-0.01][['int4','int4m','lo_int4','lo_int4m']].round(3))
for name,flag in [('always int4',U.sm4<-1),('always int4m',U.sm4>-1),('sm4>0.15',U.sm4>0.15),('sm4>0.18',U.sm4>0.18),('cos>0.9',U.cos_mean>0.9),('cos>0.95',U.cos_mean>0.95),('sm1>0.05',U.sm1>0.05),('int4_sp<0.99',U.int4_sp<0.99),('int4_sp<0.97',U.int4_sp<0.97)]:
    ret=np.where(flag,U.int4m,U.int4)
    print('%-14s flagged %2d/36 models flagged %2d | achieved min %.3f mean %.4f | units<0.9: %d, <0.95: %d | cost vs best-of-both mean %.4f'%(name,flag.sum(),U.model[flag].nunique(),ret.min(),ret.mean(),(ret<0.9).sum(),(ret<0.95).sum(),np.maximum(U.int4,U.int4m).mean()-ret.mean()))
print('flagged models by sm4>0.15:',sorted(U.model[U.sm4>0.15].unique()))
# gain predictors
for s in ['cos_mean','sm1','sm4','maxabs','qcos','p2rc','int4_sp']:
    print('gain vs',s,round(spearmanr(U[s],U.gain).correlation,3))
# within collapse set severity
C=U[U.int4<0.9]
print('within 10 collapse units spearman int4 vs:',{s:round(spearmanr(C[s],C.int4).correlation,2) for s in ['cos_mean','sm4','qcos','int4_sp']})
# dataset effect
print(U[U.model.isin(['LateOn','mLateOn','mxbai-edge-colbert-v0-17m','answerai-colbert-small-v1'])][['sm4','cos_mean','int4_sp','int4']].round(3).sort_index().to_string())
# binary
print('corr int4 vs bin retention spearman',spearmanr(U.int4,U.bin).correlation.round(3))
U['lg']=np.log2(U.dim)
M=U.groupby('model').agg(bin=('bin','mean'),lg=('lg','first'),cos=('cos_mean','mean'),sm1=('sm1','mean'),binsp=('bin_sp','mean'),kind=('kind','first'))
print(M.sort_values('bin').round(3).to_string())
def loo(cols):
    err=[]
    for m in M.index:
        tr=M.drop(m); X=np.c_[np.ones(len(tr)),tr[cols].values]; b=np.linalg.lstsq(X,tr.bin.values,rcond=None)[0]
        err.append(abs(np.r_[1,M.loc[m,cols].values.astype(float)]@b-M.loc[m,'bin']))
    return np.mean(err)
print('binary LOMO MAE: mean-only',np.mean([abs(M.drop(m).bin.mean()-M.bin[m]) for m in M.index]).round(3))
for cols in [['lg'],['cos'],['sm1'],['binsp'],['lg','cos'],['lg','sm1'],['lg','binsp']]: print(cols,round(loo(cols),3))
