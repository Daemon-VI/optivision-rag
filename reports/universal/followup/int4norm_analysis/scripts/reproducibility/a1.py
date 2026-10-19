import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/int4norm_analysis/reproducibility")
import json
from load import *
S=s17();T=s13()
print(len(S),'s17 units')
cols=['ndcg@5','ndcg@10','recall@1','recall@5','retention:ndcg@5','bytes_per_doc','vectors_per_doc']
tot=0;maxd=collections.defaultdict(float);cnt=0
for (b,m,ds),f in sorted(S.items()):
    d=json.load(open(f)); 
    if (m,ds) not in T: print('NO s13',b,m,ds,d['n_docs'],d['n_queries']); continue
    r=rows(d)
    for src,g in T[(m,ds)]:
        e=json.load(open(g)); er=rows(e)
        diffs={}
        diffs['n']=(d['n_docs']-e['n_docs'],d['n_queries']-e['n_queries'])
        bd=max(abs(d['baseline'][k]-e['baseline'][k]) for k in ['ndcg@5','ndcg@10','recall@1','recall@5'])
        sh=[l for l in r if l in er]
        worst=(0,None,None)
        for l in sh:
            for c in cols:
                if c in r[l] and c in er[l] and isinstance(r[l][c],float):
                    x=abs(r[l][c]-er[l][c])
                    if x>worst[0]: worst=(x,l,c)
        ndiff=sum(1 for l in sh if abs(r[l]['ndcg@5']-er[l]['ndcg@5'])>0)
        print(f"{b[:4]} {m[:28]:28} {ds[:18]:18} vs {src[:14]:14} dn={diffs['n']} basediff={bd:.2e} shared={len(sh)} rows_ndcg5_differ={ndiff} worst={worst[0]:.2e} {worst[1]} {worst[2]}")
