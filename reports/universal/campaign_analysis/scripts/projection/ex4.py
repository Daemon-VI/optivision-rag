import json,collections
from load import *
M=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
print(len(M))
wm=set(bs.model)
print('in measured not in wide', sorted(set(M)-wm)); print('in wide not measured', sorted(wm-set(M)))
c=collections.Counter()
for mid,ds in M.items():
    for d,v in ds.items():
        for r in v['rows']:
            st=r['pipeline']['stages']
            names=tuple(s['stage']+('R' if s.get('method')=='random' else '') for s in st)
            if any(s['stage']=='project' for s in st): c[names]+=1
for k,v in c.most_common(): print(k,v)
mid='vidore/colpali-v1.3'
for d,v in M[mid].items():
    print(d,v['baseline'],v['n_docs'],v['n_queries'],v['note'])
    for r in v['rows']:
        if any(s['stage']=='project' for s in r['pipeline']['stages']) and len(r['pipeline']['stages'])<=2: print(r['label'],round(r['retention'],4))
