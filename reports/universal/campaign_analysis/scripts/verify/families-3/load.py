import json,sys
from optivision.adapters import MODELS
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
rows=[]
for m,ds in d.items():
    mi=MODELS.get(m)
    for dn,v in ds.items():
        b=[r for r in v['rows'] if r['label']=='binary']
        if not b: continue
        rows.append(dict(model=m,ds=dn.split('/')[-1],ret=b[0]['retention'],lo=b[0]['retention_interval'][0],hi=b[0]['retention_interval'][1],
          dim=getattr(mi,'dimension',None),fam=getattr(mi,'family',None),mod=getattr(mi,'modality',None),ev=getattr(mi,'evidence','') if mi else '',vpd=b[0].get('vectors_per_doc'),nq=v['n_queries']))
import pandas as pd
df=pd.DataFrame(rows); df['loss']=(1-df.ret)*100
df.to_pickle('df.pkl')
print(len(df), df.model.nunique(), df.ds.value_counts().to_dict(), df[df.dim.isna()].model.unique())
print(df[df['mod']=='image'].groupby('dim').loss.agg(['mean','count']))
print(df.groupby('fam').model.nunique())
print(df.drop_duplicates('model')[['model','fam','dim']].to_string())
