import json
from load import *
df=load()
m=json.load(open(R+"/src/optivision/presets/measured.json"))['models']
have=set(df.model)
miss=[k for k in m if k not in have]
print(len(m),miss)
for k in miss:
    for ds,v in m[k].items():
        print(k,ds,v['n_docs'],v['baseline'],[r['label'] for r in v['rows']][:10],str(v.get('note'))[:100])
extra=have-set(m); print(extra)
# compare values
import math
bad=0
for _,r in df[df.codec!='float32'].iterrows():
    if r.model in m:
        for dsn,v in m[r.model].items():
            if r.dataset.startswith(dsn) or dsn in r.dataset:
                for x in v['rows']:
                    if x['label']==r.codec and abs(x['retention']-r.ret)>1e-6: bad+=1
print('mismatch',bad)
print(list(m[next(iter(m))].keys()))
