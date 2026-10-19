import sys,os,json,glob;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from load import *
import numpy as np
D=load()
ds=['docvqa','infovqa','arxivqa','tabfquad']
ms=['vidore--colSmol-500M','vidore--colpali-v1.3','vidore--colqwen2.5-v0.2','vultr--VultronRetrieverCore-Qwen3.5-4.5B','tencent--EVIE-4.5B']
print('model dataset base_ndcg5 base_r1 | int8 r1 ret [CI] | int8 ndcg5 ret CI')
for m in ms:
  for d in ds:
    if (m,d) not in D: continue
    r=D[(m,d)]['rows']; b=r['float32']; i=r['int8(per_vector)']
    print(m[:22],d,round(b['ndcg@5'],3),round(b['recall@1'],3),'|',round(i['retention:recall@1'],3),[round(x,3) for x in i['retention_ci:recall@1']],'|',round(i['retention:ndcg@5'],3),[round(x,3) for x in i['retention_ci:ndcg@5']])
# qrels structure
q=glob.glob(R+'followup/f5_pages_wide/tencent--EVIE-4.5B/*tabfquad*qrels.json')
print(q)
if q:
    j=json.load(open(q[0])); print(type(j),len(j)); 
    import collections
    vals=list(j.items())[:2] if isinstance(j,dict) else j[:2]; print(str(vals)[:300])
    if isinstance(j,dict):
        c=collections.Counter()
        for qq,dd in j.items():
            for doc in (dd if isinstance(dd,dict) else dd): c[doc]+=1
        print('docs relevant to >1 query:',sum(1 for v in c.values() if v>1),'of',len(c))
