import sys; sys.path.insert(0,'C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/analysis/scripts/transfer_pages')
from load import load
import numpy as np
D=load()
models=['vidore--colSmol-500M','vidore--colpali-v1.3','vultr--VultronRetrieverCore-Qwen3.5-4.5B','tencent--EVIE-4.5B']
sets=['docvqa','infovqa','arxivqa','tabfquad']
for m in models:
    print(m,[ (s,D[(m,s)]['nq'],D[(m,s)]['nd'],round(D[(m,s)]['base'],3)) if (m,s) in D else (s,None) for s in sets])
M='retention:ndcg@5'; C='retention_ci:ndcg@5'
labs=None
for m in models:
    L=set.intersection(*[set(D[(m,s)]['rows']) for s in sets])
    print(m,len(L))
    for l in sorted(L):
        lo=[D[(m,s)]['rows'][l][C][0] for s in sets]; pt=[D[(m,s)]['rows'][l][M] for s in sets]
        comp=D[(m,'docvqa')]['rows'][l]['compression_vs_float32']
        if min(lo)>=0.95 and comp>=3: print('  PASS',l,round(comp,1),round(min(lo),3),round(min(pt),3))
    for l in ['binary','int8','int4','hierarchical_merge(0.25) > int4','hierarchical_merge(0.25)']:
        for ll in L:
            if ll.startswith(l) and len(ll)<=len(l)+2 or ll==l:
                pass
