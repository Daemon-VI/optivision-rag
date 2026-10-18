import pandas as pd,numpy as np
pd.set_option('display.width',250)
df=pd.read_csv('../geometry_predictors/table.csv')
print(df.groupby('codec').size().to_dict())
d=df[df.codec.isin(['int4','int4m'])]
print('dups',d.duplicated(['tag','codec']).sum())
U=df[df.codec=='int4'].set_index('tag')[['model','batch','wide','kind','sm4','cos_mean']].join(df.pivot_table(index='tag',columns='codec',values='ret')).join(df.pivot_table(index='tag',columns='codec',values='lo').add_prefix('lo_'))
print(len(U),U.model.nunique(), U.batch.value_counts().to_dict())
print(U[['model','kind','sm4','cos_mean','int4','int4m','lo_int4','lo_int4m']].sort_values('int4').round(3).to_string())
print('int4<0.9',(U.int4<0.9).sum(),U.model[U.int4<0.9].unique(),U.int4.min(),U.int4.mean())
print('int4m',U.int4m.min(),U.int4m.mean(),(U.int4m<0.97).sum(),(U.int4m<0.95).sum())
orc=np.maximum(U.int4,U.int4m).mean(); print('oracle cost always m',orc-U.int4m.mean())
g=U.sm4>0.15; r=np.where(g,U.int4m,U.int4); print('gate',g.sum(),U.model[g].nunique(),r.min(),r.mean(),orc-r.mean())
# oracle vs int4 only
print('mean int4m-int4 on non-collapse', (U.int4m-U.int4)[U.int4>=0.9].describe().round(4).to_dict())
# gate cost using lower bounds? ci overlap for 3 losers
print(U[(U.int4m<U.int4-0.01)][['int4','int4m','lo_int4','lo_int4m']])
# sm4 non-collapse top, collapse bottom
print(U[U.int4>=0.9].sm4.max(),U[U.int4<0.9].sm4.min())
a=U[U.model.str.contains('answerai')];print(a[['sm4','cos_mean','int4','int4m']])
