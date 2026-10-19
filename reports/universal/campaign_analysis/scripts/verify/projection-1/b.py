import json,numpy as np,collections,sys
sys.path.insert(0,'C:/Users/Lenovo/projects/optivision-package/src')
from optivision.adapters import MODELS
m=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
# proxy vs pure at same k
out=[]
for mod,ds in m.items():
  for dn,v in ds.items():
    pure={};hm={};pr=collections.defaultdict(list)
    for r in v['rows']:
        st=r['pipeline']['stages']
        if len(st)==1 and st[0]['stage']=='project' and st[0].get('method')=='pca': pure[st[0]['dim']]=r['retention']
        if len(st)==1 and st[0]['stage']=='hierarchical_merge': hm[st[0]['ratio']]=r['retention']
        if len(st)==2 and st[0]['stage']=='hierarchical_merge' and st[1]['stage']=='project' and st[1].get('method')=='pca' : pr[st[1]['dim']].append(r['retention']/hm.get(st[0]['ratio'],np.nan))
    for k in pure:
        if k in pr: out.append((mod,MODELS[mod].dimension,k,pure[k],float(np.nanmean(pr[k]))))
print(len(out))
for o in out: print(o[0][:40],o[1:3],round(o[3],3),round(o[4],3))
