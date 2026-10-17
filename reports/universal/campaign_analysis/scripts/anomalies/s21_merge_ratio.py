from common import *
import pandas as pd, re
c=pd.read_pickle(OUT+'/comp.pkl')
m=c[c.label.str.match(r'^hierarchical_merge\(0\.\d+\)$')].copy()
m['nom']=m.label.str.extract(r'\((0\.\d+)\)').astype(float)
m['ratio']=m.vpd/m.b_vpd; m['dev']=m.ratio/m.nom-1
m['kind']=np.where(m.model.isin(c[c.fn.str.contains('scifact')].model.unique()),'text','image')
print(m.groupby(['label','kind']).dev.describe().round(4)[['count','mean','min','max']].to_string())
print(m.sort_values('dev').head(5)[['model','fn','label','vpd','b_vpd','ratio','dev']].to_string())
a=c[c.label=='adaptive_merge(0.6)']; a=a.assign(ratio=a.vpd/a.b_vpd, kind=np.where(a.fn.str.contains('scifact'),'text','image'))
print(a.groupby('kind').ratio.describe().round(3).to_string())
print(a.sort_values('ratio').head(3)[['model','fn','ratio']].to_string(), a.sort_values('ratio').tail(3)[['model','fn','ratio']].to_string())
print('----')
big=m[m.dev.abs()>0.05].copy(); big['short']=big.model.str.split('/').str[1]
print(len(big),'of',len(m),'rows with |dev|>5%'); print(big.groupby(['short','label']).agg(ratio=('ratio','mean'),dev=('dev','mean'),vpd=('b_vpd','mean')).round(3).to_string())
