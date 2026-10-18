import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from load import *
import numpy as np
D=load()
models=['vidore--colSmol-500M','vidore--colpali-v1.3','vultr--VultronRetrieverCore-Qwen3.5-4.5B','tencent--EVIE-4.5B']
M='retention:ndcg@5';C='retention_ci:ndcg@5'
ds=['docvqa','infovqa','arxivqa','tabfquad']
def fam(l):
    if 'merge' in l and 'project' in l: return 'merge+project(+q)'
    if 'merge' in l: return 'merge(+q)'
    if 'project' in l: return 'project(+q)'
    return 'codec'
# most compressive safe (lo>=.95) per dataset
for m in models:
    print('==',m)
    labs=[l for l in D[(m,'tabfquad')]['rows'] if l!='float32' and all(l in D[(m,d)]['rows'] for d in ds)]
    for d in ds:
        r=D[(m,d)]['rows']
        s=[(r[l]['compression_vs_float32'],l) for l in labs if r[l][C][0]>=0.95]
        s2=[(r[l]['compression_vs_float32'],l) for l in labs if r[l][M]>=0.95]
        print(f' {d}: n_safe_lo={len(s)} best_lo={max(s) if s else None} | n_safe_pt={len(s2)} best_pt={max(s2) if s2 else None}')
    allsafe=[(min(D[(m,d)]['rows'][l]['compression_vs_float32'] for d in ds),l) for l in labs if all(D[(m,d)]['rows'][l][C][0]>=0.95 for d in ds)]
    print(' safe on all 4 (lo):',len(allsafe),'best',sorted(allsafe)[-3:])
    allpt=[(D[(m,'docvqa')]['rows'][l]['compression_vs_float32'],l) for l in labs if all(D[(m,d)]['rows'][l][M]>=0.95 for d in ds)]
    print(' safe on all 4 (pt):',len(allpt),'best',sorted(allpt)[-3:])
    # family spread
    for f in ['codec','merge(+q)','project(+q)','merge+project(+q)']:
        ls=[l for l in labs if fam(l)==f]
        sp=[max(D[(m,d)]['rows'][l][M] for d in ds)-min(D[(m,d)]['rows'][l][M] for d in ds) for l in ls]
        tab_minus=[D[(m,'tabfquad')]['rows'][l][M]-np.mean([D[(m,d)]['rows'][l][M] for d in ds[:3]]) for l in ls]
        print(f'  {f}: n={len(ls)} median spread {np.median(sp):.3f} max {max(sp):.3f}; tab-minus-others median {np.median(tab_minus):+.3f}')
    # pareto fronts
    for d in ds:
        pf=D[(m,d)]['pf']; r=D[(m,d)]['rows']
        print('  front',d,len(pf),'top',[ (l[-45:],round(r[l]['compression_vs_float32'],0),round(r[l][M],3)) for l in pf[-2:]] if pf and pf[0] in r else pf[:1])
