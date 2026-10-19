import sys,os,json,glob;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from load import *
import numpy as np
D=load()
M='retention:ndcg@5';C='retention_ci:ndcg@5'
ds=['docvqa','infovqa','arxivqa','tabfquad']
for m in ['vidore--colSmol-500M','vidore--colpali-v1.3','vultr--VultronRetrieverCore-Qwen3.5-4.5B','tencent--EVIE-4.5B']:
    print('==',m)
    for l in D[(m,'tabfquad')]['rows']:
        if 'project' in l and 'merge' not in l and all(l in D[(m,d)]['rows'] for d in ds):
            print('  %-34s'%l,'comp %.1f'%D[(m,'docvqa')]['rows'][l]['compression_vs_float32'],' '.join(f"{d[:3]} {D[(m,d)]['rows'][l][M]:.3f}[{D[(m,d)]['rows'][l][C][0]:.3f}]" for d in ds))
# merge fractions for all four
print('merge consistency (point retention, comp) per fraction')
for m in ['vidore--colSmol-500M','vidore--colpali-v1.3','vultr--VultronRetrieverCore-Qwen3.5-4.5B','tencent--EVIE-4.5B']:
    for l in ['hierarchical_merge(0.33)','hierarchical_merge(0.25)','hierarchical_merge(0.1)']:
        print(m[-18:],l,' '.join(f"{d[:3]} {D[(m,d)]['rows'][l][M]:.3f}" for d in ds))
