import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from load import *
import numpy as np
D=load()
models=['vidore--colSmol-500M','vidore--colpali-v1.3','vultr--VultronRetrieverCore-Qwen3.5-4.5B','tencent--EVIE-4.5B']
M='retention:ndcg@5';C='retention_ci:ndcg@5'
old=['docvqa','infovqa'];new=['arxivqa','tabfquad']
tot=dict(safe=0,fs_lo=0,fs_pt=0)
fs_list=[]
for m in models:
    labs=set.intersection(*[set(D[(m,d)]['rows']) for d in old+new])
    labs=[l for l in D[(m,'docvqa')]['rows'] if l in labs and l!='float32']
    print('==',m,len(labs),'common pipelines')
    for nd in new:
        for od in old:
            safe=[l for l in labs if D[(m,od)]['rows'][l][C][0]>=0.95]
            fl=[l for l in safe if D[(m,nd)]['rows'][l][C][0]<0.95]
            fp=[l for l in safe if D[(m,nd)]['rows'][l][M]<0.95]
            print(f' {od}->{nd}: safe_on_src={len(safe)} false-safe(lo<.95)={len(fl)} pt<.95={len(fp)}')
        # safe on BOTH old
        both=[l for l in labs if all(D[(m,o)]['rows'][l][C][0]>=0.95 for o in old)]
        fl=[l for l in both if D[(m,nd)]['rows'][l][C][0]<0.95]
        fp=[l for l in both if D[(m,nd)]['rows'][l][M]<0.95]
        print(f' BOTH old->{nd}: safe_both={len(both)} fs_lo={len(fl)} fs_pt={len(fp)}')
        for l in fl: print('    ',l,[round(D[(m,d)]['rows'][l][M],3) for d in old+[nd]], round(D[(m,nd)]['rows'][l][C][0],3),round(D[(m,nd)]['rows'][l]['compression_vs_float32'],1))
