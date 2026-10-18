import sys; sys.path.insert(0,'C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/analysis/scripts/transfer_pages')
from load import load
import numpy as np
D=load()
models=['vidore--colSmol-500M','vidore--colpali-v1.3','vultr--VultronRetrieverCore-Qwen3.5-4.5B','tencent--EVIE-4.5B']
sets=['docvqa','infovqa','arxivqa','tabfquad']
M='retention:ndcg@5'; C='retention_ci:ndcg@5'
for m in models:
    L=sorted(set.intersection(*[set(D[(m,s)]['rows']) for s in sets]))
    print(m,[l for l in L if 'merge(0.5' in l or l.startswith('binary') or l.startswith('lloyd')or l.startswith('adaptive')][:8])
    for l in ['binary','lloyd2(7)','hierarchical_merge(0.25)','hierarchical_merge(0.25) > binary','hierarchical_merge(0.25) > int4','int4','int4(mean)','adaptive_merge(0.6)']:
        r=[D[(m,s)]['rows'][l] for s in sets]
        print('  ',l,[ (round(x[M],3),round(x[C][0],3)) for x in r])
# finding 2: projection
for m in models:
    for l in sorted(D[(m,'docvqa')]['rows']):
        if l.startswith('project(') and ' ' not in l.replace('project(','',1).split(')')[-1] :
            r=[D[(m,s)]['rows'][l] for s in sets if l in D[(m,s)]['rows']]
            print(m[:8],l,[(round(x[M],3),round(x[C][0],3)) for x in r])
# merge 0.25 spread
for m in models:
    out=[]
    for l in ['hierarchical_merge(0.25)','hierarchical_merge(0.33)','hierarchical_merge(0.1)']:
        out.append([round(D[(m,s)]['rows'][l][M],3) for s in sets])
    print(m[:8],out)
# above 1
for s in sets:
    n=k=ci=0
    for m in D_keys if False else []: pass
tot={}
for (m,s),d in D.items():
    for l,r in d['rows'].items():
        t=tot.setdefault(s,[0,0,0]); t[0]+=1
        if r[M]>1.01: t[1]+=1; t[2]+= r[C][0]>1
print(tot)
# EVIE int8 tabfquad
r=D[('tencent--EVIE-4.5B','tabfquad')]['rows']['int8(per_vector)']
print({k:r[k] for k in r if 'retention' in k})
for m in models+['vidore--colqwen2.5-v0.2']:
  if (m,'tabfquad') in D:
    r=D[(m,'tabfquad')]['rows']['int8(per_vector)']; a=D[(m,'arxivqa')]['rows']['int8(per_vector)']
    print(m[:10],r[M],r[C],a[M],a[C], r['retention_ci:recall@5'] if 'retention_ci:recall@5' in r else None)
