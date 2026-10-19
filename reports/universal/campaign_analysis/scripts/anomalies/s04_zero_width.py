from common import *
import pandas as pd
df=pd.read_pickle(OUT+'/wide_long.pkl')
z=df[(df.w<1e-9)&(abs(df.ret-1)>1e-9)]
print(z.groupby(['m']).size().to_dict())
print(z.sort_values('ret').to_string())
