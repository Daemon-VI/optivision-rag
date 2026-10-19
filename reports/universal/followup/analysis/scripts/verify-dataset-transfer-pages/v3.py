import sys; sys.path.insert(0,'C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/analysis/scripts/transfer_pages')
from load import load
import numpy as np
D=load()
models=['vidore--colSmol-500M','vidore--colpali-v1.3','vultr--VultronRetrieverCore-Qwen3.5-4.5B','tencent--EVIE-4.5B']
M='retention:ndcg@5'; C='retention_ci:ndcg@5'
T=[0,0,0]
for src in (['docvqa','infovqa'],['docvqa'],['infovqa']):
  print('src',src)
  for m in models:
    for new in ['arxivqa','tabfquad']:
        n=a=b=0
        for l,r in D[(m,new)]['rows'].items():
            if not all(l in D[(m,s)]['rows'] for s in src): continue
            if l=='float32' or r['compression_vs_float32']<1.5: continue
            if all(D[(m,s)]['rows'][l][C][0]>=0.95 for s in src):
                n+=1; a+= r[C][0]<0.95; b+= r[M]<0.95
        print(' ',m[:8],new,n,a,b)
        if len(src)==2: T[0]+=n;T[1]+=a;T[2]+=b
print(T)
# f5: most compressive lower>=0.95 / point >=0.95 per set
for m in models:
  for crit in (C,M):
    out=[]
    for s in ['docvqa','infovqa','arxivqa','tabfquad']:
        best=None
        for l,r in D[(m,s)]['rows'].items():
            v=r[C][0] if crit==C else r[M]
            if v>=0.95 and (best is None or r['compression_vs_float32']>best[1]): best=(l,round(r['compression_vs_float32']))
        out.append(best)
    print(m[:8],'lo' if crit==C else 'pt',out)
