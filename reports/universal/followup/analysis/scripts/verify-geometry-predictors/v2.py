import pandas as pd,numpy as np
from scipy.stats import spearmanr
pd.set_option('display.width',250)
df=pd.read_csv('../geometry_predictors/table.csv')
U=df[df.codec=='int4'].set_index('tag').drop(columns=['codec','ret','lo']).join(df.pivot_table(index='tag',columns='codec',values='ret'))
U['gain']=U.int4m-U.int4
def auc(x,y):
    p=x[y==1];n=x[y==0];a=np.mean([(i>j)+.5*(i==j) for i in p for j in n]);return max(a,1-a)
for thr in [0.85,0.9,0.95]:
    y=(U.int4<thr).astype(int).values
    print(thr,y.sum(),{s:round(auc(U[s].values,y),3) for s in ['sm4','sm1','maxabs','cos_mean','qcos','p2r','p2rc','int4_sp','int4_vc','int4_rr']})
for s in ['sm4','sm1','maxabs','cos_mean','qcos','p2r','p2rc','int4_sp','int4_vc','int4_rr']:
    print(s,round(spearmanr(U[s],U.int4).correlation,3),'gain',round(spearmanr(U[s],U.gain).correlation,3))
# pages only
P=U[U.kind=='img'];T=U[U.kind=='text']
print('pages n',len(P),'int4m-int4 mean',P.gain.mean(),'min',P.gain.min(),'| text',T.gain.mean())
print('pages int4 min',P.int4.min(),'text non-collapse gain mean',T[T.int4>=.9].gain.mean())
# model-level (one per model, mean) rather than units
M=U.groupby('model').agg(int4=('int4','min'),int4m=('int4m','min'),g=('gain','mean'),sm4=('sm4','mean'))
print(M.round(3).sort_values('int4').to_string())
# gain correlation within non-collapse only
N=U[U.int4>=.9]; print('gain~sm4 non-collapse',spearmanr(N.sm4,N.gain).correlation,'n',len(N))
# gain correlation at model level
print('gain~sm4 model-level',spearmanr(M.sm4,M.g).correlation)
# dedupe-free gate: independent units (drop plus*)
D=U[~U.index.str.contains('plus')]; r=np.where(D.sm4>.15,D.int4m,D.int4); print('dedupe n',len(D),'gate mean',r.mean(),'orc',np.maximum(D.int4,D.int4m).mean(),'always m',D.int4m.mean(),'int4',D.int4.mean())
# oracle with noise: shrunk oracle: cost of always-m vs best fixed choice
print('always-int4 mean',U.int4.mean())
