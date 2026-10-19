import json,collections,re
R='C:/Users/Lenovo/projects/optivision-package/'
M=json.load(open(R+'src/optivision/presets/measured.json'))['models']
cs=collections.Counter(); labs=[]
for m,ds in M.items():
    for k,d in ds.items():
        cs[len(d['rows'])]+=1; labs.append(set(r['label'] for r in d['rows']))
print(sorted(cs.items()), len(set.union(*labs)), len(set.intersection(*labs)))
per=collections.Counter(); 
for m,ds in M.items(): per[len(set().union(*[set(r['label'] for r in d['rows']) for d in ds.values()]))]+=1
print('per-model label sets',sorted(per.items()))
# merge ratio all hierarchical_merge(0.1) rows in measured: vectors ratio relative to float32 row
out=[]
for m,ds in M.items():
    for k,d in ds.items():
        base=[r for r in d['rows'] if r['label']=='float32']
        hm=[r for r in d['rows'] if r['label']=='hierarchical_merge(0.1)']
        if base and hm: out.append((hm[0]['vectors_per_doc']/base[0]['vectors_per_doc'],m))
out.sort(); print(out[-8:])
import os,sys
sys.path.insert(0,R+'src')
from optivision.adapters import MODELS
ev=collections.Counter()
for mid,i in MODELS.items():
    if mid in M:
        e=i.evidence.lower()
        ev[('float16' if 'float16' in e or 'fp16' in e else 'bf16' if 'bfloat16' in e or 'bf16' in e else 'float32' if 'float32' in e else 'other')]+=1
print(ev)
for mid,i in MODELS.items():
    if mid in M and i.modality=='text': print(mid,i.evidence[:110])
