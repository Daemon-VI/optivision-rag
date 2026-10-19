import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/int4norm_analysis/reproducibility")
from load import *
import numpy as np
S=s17();T=s13()
pq=0;pqn=0;ck=collections.Counter();envs=collections.Counter()
for (b,m,ds),f in sorted(S.items()):
    d=json.load(open(f))
    for src,g in T.get((m,ds),[]):
        e=json.load(open(g))
        if 'f3_text' in src: continue
        pqn+=1
        a=d['per_query_ndcg@10'];c=e.get('per_query_ndcg@10',{})
        mx=max([np.max(np.abs(np.array(a[l])-np.array(c[l]))) for l in a if l in c] or [-1])
        if mx!=0: print('pq diff',b,m,ds,src,mx)
        r=rows(d);er=rows(e)
        for l in [x for x in er if x in r]:
            for k in ['compress_seconds','scan_ms_per_query','retention_ci:ndcg@5']:
                if k in r[l] and r[l][k]!=er[l][k]: ck[k]+=1
        ck['n']+=len(er)
        envs[(e['environment'].get('gpus') and tuple(e['environment']['gpus']),tuple(d['environment'].get('gpus') or ()))]+=1
        ck['sec_ratio_n']+=1
print(pqn,ck);print(envs)
# s13 file keys
e=json.load(open(T[('colbert-ir--colbertv2.0','scifact')][0][1])); print(list(e.keys()),e.get('seconds'))
# f3
for src,g in T[('colbert-ir--colbertv2.0','scifact_plus10000')]:
    e=json.load(open(g));print(src,e['n_docs'],e['n_queries'],e['baseline']['ndcg@5'],e['seconds'])
d=json.load(open(S[('f13b_text_scale_same_domain','colbert-ir--colbertv2.0','scifact_plus10000')]));print(d['n_docs'],d['baseline']['ndcg@5'])
