import pandas as pd,numpy as np
pd.set_option('display.width',250,'display.max_rows',300)
df=pd.read_pickle('master2.pkl')
g=df.groupby('model').agg(mod=('modality','first'),fam=('family','first'),dim=('dim','first'),dt=('dtype','first'),vpd=('vpd','mean'),n=('int4','size'),int4=('int4','mean'),int4m=('int4m','mean'),bin=('bin','mean'),int8=('int8','mean'),l2=('l2','mean'),dl=('downloads','first'))
print(g.sort_values('int4').round(3).to_string())
