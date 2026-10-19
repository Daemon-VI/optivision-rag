import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from load import *
import numpy as np
D=load()
M='retention:ndcg@5';C='retention_ci:ndcg@5'
ds=['docvqa','infovqa','arxivqa','tabfquad']
print('rows with point>1.01: n, and with lo>1.0 (CI excludes 1)')
for d in ds:
    n=0;ex=0;tot=0;ex_down=0;down=0
    for k,v in D.items():
        if k[1]!=d: continue
        for l,r in v['rows'].items():
            if l=='float32':continue
            tot+=1
            if r[M]>1.01:
                n+=1; ex+= r[C][0]>1.0
            if r[M]<0.99:
                down+=1; ex_down+= r[C][1]<1.0
    print(d,'rows',tot,'pt>1.01',n,'ci excludes 1',ex,'| pt<0.99',down,'ci excludes 1',ex_down)
# per-model tab: upward rows
for m in ['vidore--colSmol-500M','vidore--colpali-v1.3','vidore--colqwen2.5-v0.2','vultr--VultronRetrieverCore-Qwen3.5-4.5B','tencent--EVIE-4.5B']:
    r=D[(m,'tabfquad')]['rows']; v=[x[M] for l,x in r.items() if l!='float32']
    w=[x[C][1]-x[C][0] for l,x in r.items() if l!='float32']
    print(m[:30],'tab: mean ret',round(np.mean(v),3),'max',round(max(v),3),'n>1.01',sum(x>1.01 for x in v),'of',len(v),'median CI width',round(np.median(w),3),'lossy-looking? ndcg5 base',round(D[(m,'tabfquad')]['base'],3))
# CI width vs sqrt(1/nq) : per-model same-pipeline widths ratio tab / arxivqa
for m in ['vidore--colSmol-500M','vidore--colpali-v1.3','vidore--colqwen2.5-v0.2','vultr--VultronRetrieverCore-Qwen3.5-4.5B','tencent--EVIE-4.5B']:
    rt=D[(m,'tabfquad')]['rows'];ra=D[(m,'arxivqa')]['rows']
    labs=[l for l in rt if l!='float32' and l in ra and ra[l][C][1]-ra[l][C][0]>0.005]
    ratio=[(rt[l][C][1]-rt[l][C][0])/(ra[l][C][1]-ra[l][C][0]) for l in labs]
    print(m[:30],'CI width ratio tab/arxiv median',round(np.median(ratio),2),'n',len(labs),'; sqrt(500/280)=',round((500/280)**.5,2))
# EVIE int8 and int4
r=D[('tencent--EVIE-4.5B','tabfquad')]['rows']
for l in ['float16','int8(per_vector)','int4','int4(mean)','binary']:
    print(l,{k:(round(r[l][k],3) if not isinstance(r[l][k],list) else [round(x,3) for x in r[l][k]]) for k in ['retention:ndcg@5','retention_ci:ndcg@5','retention:recall@1','retention:recall@5','retention:ndcg@10']})
