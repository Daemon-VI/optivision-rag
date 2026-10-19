import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/int4norm_analysis/reproducibility")
from load import *
import numpy as np
from itertools import combinations
S=s17()
L=['int8(per_vector)','int4','int4norm','int4(mean)','int4norm(mean)','binary','lloyd2(7)']
D={}
for (b,m,ds),f in S.items():
    if b.startswith(('f9b','f12b')):
        d=json.load(open(f));r=rows(d)
        D[(m.split('--')[1],ds)]=dict(base=d['baseline']['ndcg@5'],nd=d['n_docs'],nq=d['n_queries'],**{l:r[l]['retention:ndcg@5'] for l in L},ci={l:r[l]['retention_ci:ndcg@5'] for l in L})
models=sorted({k[0] for k in D})
print('units',len(D)); 
print('n_docs/queries:',{ds:(v['nd'],v['nq']) for (m,ds),v in D.items() if m=='answerai-colbert-small-v1'})
for l in ['int4','int4norm','int4(mean)','int4norm(mean)']:
    print('\n==',l)
    print(f"{'model':36}{'scifact':>9}{'nfcorp':>9}{'scidocs':>9}  scidocs CI")
    for m in models:
        v=[D.get((m,ds),{}).get(l) for ds in ['scifact','nfcorpus','scidocs']]
        ci=D.get((m,'scidocs'),{}).get('ci',{}).get(l)
        print(f"{m:36}"+''.join(f"{x:9.3f}" if x is not None else f"{'-':>9}" for x in v)+(f"  [{ci[0]:.3f},{ci[1]:.3f}]" if ci else ''))
from scipy.stats import spearmanr,kendalltau
print('\n=== rank agreement of int4 retention')
dss=['scifact','nfcorpus','scidocs']
for l in ['int4','int4norm','int4(mean)','int4norm(mean)']:
  for a,b in combinations(dss,2):
    ms=[m for m in models if (m,a) in D and (m,b) in D]
    x=[D[(m,a)][l] for m in ms];y=[D[(m,b)][l] for m in ms]
    print(l,a,b,'n=',len(ms),'spearman=%.2f'%spearmanr(x,y)[0],'kendall=%.2f'%kendalltau(x,y)[0])
ms=[m for m in models if all((m,ds) in D for ds in dss)]
print('models with all 3:',len(ms))
print('\nbreak (int4 ret<0.90) per dataset:')
for ds in dss:
    print(ds,[m for m in models if (m,ds) in D and D[(m,ds)]['int4']<0.90])
print('int4 base ndcg@5:')
for m in models: print(m,[round(D[(m,ds)]['base'],3) if (m,ds) in D else None for ds in dss])
print('\nmin retention per codec per dataset (int4norm,int4norm(mean))')
for l in ['int4norm','int4norm(mean)']:
  for ds in dss:
    v=[(D[(m,ds)][l],m) for m in models if (m,ds) in D]; print(l,ds,min(v),'n=',len(v))
