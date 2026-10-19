from common import *
import pandas as pd
H=json.load(open("C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/hubrank.json"))
h=pd.DataFrame(H); print(h.columns.tolist(), len(h))
O=outcomes(); M=measured()
o=pd.DataFrame([dict(model=k,outcome=v['outcome'],kind=v['kind'],dtype=v['dtype'],gpu_h=v['gpu_hours'],natt=len(v.get('attempts',[]))+1,batch=v['batch']) for k,v in O.items()])
o['measured']=o.outcome.eq('measured')
f=o[~o.measured].merge(h,left_on='model',right_on='id',how='left')
print(f[['model','outcome','batch','dtype','gpu_h','natt','downloads','likes','status']].to_string())
# failed vs measured downloads
x=o.merge(h,left_on='model',right_on='id',how='left')
print(x.groupby('measured').downloads.agg(['count','median','sum']))
# download-weighted coverage over registry
print('catalogue (measured.json) models:',len(M),' with hubrank row',h.id.isin(M.keys()).sum())
h['in_cat']=h.id.isin(M.keys())
print(h.sort_values('downloads',ascending=False).head(25)[['id','downloads','in_registry','status','in_cat']].to_string())
tot=h.downloads.sum(); print('share of listed 30d downloads covered by catalogue',round(h[h.in_cat].downloads.sum()/tot,3),' of',len(h),'models')
print('retry successes:',o[(o.natt>1)&o.measured].shape[0],' of', (o.natt>1).sum(),'models with >1 attempt')
print(o.groupby('batch').measured.agg(['sum','count']).T.to_string())
