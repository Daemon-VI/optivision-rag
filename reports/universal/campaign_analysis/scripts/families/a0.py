from common import *
df=long_df()
P=mat(df)
n=P.notna().sum()
core=[l for l in P.columns if n[l]>=88]
print(len(core))
miss=P[core].isna().sum(axis=1)
print(miss[miss>0])
print(P.index[P.loc[:, 'none > float32'].notna()].tolist())
