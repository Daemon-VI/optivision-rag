import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from load import *
import numpy as np
D=load()
M='retention:ndcg@5';C='retention_ci:ndcg@5'
ds=['docvqa','infovqa','arxivqa','tabfquad']
for k in sorted(D):
    r=D[k]['rows']
    f=lambda l:(round(r[l][M],3),round(r[l][C][0],3),round(r[l][C][1],3)) if l in r else None
    print(k[0][:22],k[1],'nq',D[k]['nq'],'f16',f('float16'),'int8',f('int8(per_vector)'),'int4m',f('int4(mean)'))
print('CI width distribution (hi-lo) across non-float32 rows, ndcg@5')
for d in ds:
    w=[];above=[];n=0
    for k,v in D.items():
        if k[1]!=d: continue
        for l,r in v['rows'].items():
            if l=='float32':continue
            w.append(r[C][1]-r[C][0]); n+=1; above.append(r[M]>1.0)
    print(d,'n',n,'median width',round(np.median(w),4),'frac point>1',round(np.mean(above),3),'max pt',round(max(r[M] for k,v in D.items() if k[1]==d for l,r in v['rows'].items()),3))
# CI half width vs baseline noise: retention rows that are near-lossless (int8): width
for d in ds:
    w=[D[k]['rows']['int8(per_vector)'][C][1]-D[k]['rows']['int8(per_vector)'][C][0] for k in D if k[1]==d]
    print('int8 width',d,[round(x,4) for x in w])
