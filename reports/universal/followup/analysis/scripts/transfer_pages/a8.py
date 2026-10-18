import sys,os,json,glob;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from load import *
import numpy as np
from scipy.stats import spearmanr
D=load()
M='retention:ndcg@5';C='retention_ci:ndcg@5'
ds=['docvqa','infovqa','arxivqa','tabfquad']
ms=['vidore--colSmol-500M','vidore--colpali-v1.3','vultr--VultronRetrieverCore-Qwen3.5-4.5B','tencent--EVIE-4.5B']
labs=['float16','int8(per_vector)','int4','int4(mean)','lloyd2(7)','binary','hierarchical_merge(0.5)','hierarchical_merge(0.33)','hierarchical_merge(0.25)','hierarchical_merge(0.1)','adaptive_merge(0.6)',
 'hierarchical_merge(0.5) > int8(per_vector)','hierarchical_merge(0.33) > int8(per_vector)','hierarchical_merge(0.25) > int8(per_vector)','hierarchical_merge(0.33) > int4(mean)','hierarchical_merge(0.25) > int4(mean)','hierarchical_merge(0.25) > int4','hierarchical_merge(0.33) > binary','hierarchical_merge(0.25) > binary','hierarchical_merge(0.25) > lloyd2(7)']
print('min over 4 sets of CI lower bound / min point  (compression)')
print('%-45s'%'pipeline',*[m.split('--')[1][:9].ljust(16) for m in ms], 'comp')
for l in labs:
    row=[]
    for m in ms:
        if all(l in D[(m,d)]['rows'] for d in ds):
            lo=min(D[(m,d)]['rows'][l][C][0] for d in ds); pt=min(D[(m,d)]['rows'][l][M] for d in ds)
            argw=min(ds,key=lambda d:D[(m,d)]['rows'][l][C][0])
            row.append(f'{lo:.3f}/{pt:.3f} {argw[:3]}'.ljust(16))
        else: row.append('-'.ljust(16))
    c=D[(ms[1],'docvqa')]['rows'].get(l,{}).get('compression_vs_float32')
    print('%-45s'%l,*row,round(c,1) if c else '')
# colqwen2.5 new-only
m='vidore--colqwen2.5-v0.2'
print('colqwen2.5 (arxiv,tab) min lo:',{l:(round(min(D[(m,d)]['rows'][l][C][0] for d in ['arxivqa','tabfquad']),3)) for l in labs if l in D[(m,'arxivqa')]['rows']})
# CI width vs baseline top-1 across all page units (int4(mean))
xs=[];ys=[];zs=[]
for (m,d),v in D.items():
    r=v['rows']
    if 'int4(mean)' not in r: continue
    xs.append(r['float32']['recall@1']);zs.append(v['nq']);ys.append(r['int4(mean)'][C][1]-r['int4(mean)'][C][0])
print('int4(mean) CI width vs baseline recall@1: spearman',spearmanr(xs,ys),'n',len(xs))
print('vs nq', spearmanr(zs,ys))
# tabfquad, per-model CI width and baseline r1
for d in ds:
    w=[(D[k]['rows']['int4(mean)'][C][1]-D[k]['rows']['int4(mean)'][C][0],D[k]['rows']['float32']['recall@1']) for k in D if k[1]==d]
    print(d,'median int4(mean) width',round(np.median([a for a,b in w]),3),'median base r1',round(np.median([b for a,b in w]),3),'n',len(w))
