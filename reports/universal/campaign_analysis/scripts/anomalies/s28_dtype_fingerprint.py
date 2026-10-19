from common import *
import pandas as pd
m=pd.read_pickle(OUT+'/models.pkl'); m=m[m.unit.notna()].copy()
img=m[m.dtype.isin(['float32','float16'])]
print('known-dtype models with check data:',len(img))
print(img.groupby('dtype').unit.agg(['count','min','max']).to_string())
def cls(u): return 'fp32-like' if u<1e-5 else ('fp16-like' if u<2e-3 else 'bf16-like')
print('image: fingerprint vs recorded', pd.crosstab(img.dtype, img.unit.map(cls)).to_string())
t=m[m.dtype=='auto'].copy(); t['inferred']=t.unit.map(cls)
print(t[['model','unit','inferred']].sort_values('unit').to_string(index=False))
