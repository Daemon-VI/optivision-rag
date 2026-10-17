import json,collections,numpy as np
m=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
import sys
sys.path.insert(0,'C:/Users/Lenovo/projects/optivision-package/src')
from optivision.adapters import MODELS
dimof={k:v.dimension for k,v in MODELS.items()}
units=[]
for mod,ds in m.items():
  for dn,v in ds.items():
    pure={};prox=collections.defaultdict(list);hm={}
    for r in v['rows']:
        st=r['pipeline']['stages']
        if len(st)==1 and st[0]['stage']=='project' and st[0].get('method')=='pca': pure[st[0]['dim']]=r['retention']
        if len(st)==1 and st[0]['stage']=='hierarchical_merge': hm[st[0]['ratio']]=r['retention']
    for r in v['rows']:
        st=r['pipeline']['stages']
        if len(st)==2 and st[0]['stage']=='hierarchical_merge' and st[1]['stage']=='project' and st[1].get('method')=='pca':
            prox[st[1]['dim']].append((st[0]['ratio'],r['retention']))
    curve={}
    for k,l in prox.items():
        q=[ret/hm[ra] for ra,ret in l if ra in hm and hm[ra]>0]
        if q: curve[k]=('p',float(np.mean(q)))
    for k,x in pure.items(): curve[k]=('P',x)
    d=dimof.get(mod)
    if d is None: continue
    units.append(dict(model=mod,ds=dn,d=d,curve=curve))
print(len(units),len({u['model'] for u in units}))
print(collections.Counter(u['d'] for u in units))
# compare pure vs proxy at same k
diffs=[]
for u in units:
    pass
json.dump(units,open('C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/verify/projection-1/units.json','w'))
# proxy vs pure at k=64 / 128 where both exist
for u in units:
    c=u['curve']
