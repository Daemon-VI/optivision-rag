from common import *
df=long_df()
A=labelsA(df); print(len(A),A)
P=mat(df)[A].dropna()
print(P.shape)
print(P.groupby(level='ds').agg(['mean','std']).T.round(3).to_string())
