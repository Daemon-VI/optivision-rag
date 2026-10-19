from common import *
import pandas as pd
M=measured()
sets={}; rows={}
for mid,d in M.items():
    for ds,v in d.items():
        rows[(mid,ds)]={r['label']:r for r in v['rows']}
common=set.intersection(*[set(r) for r in rows.values()]); print('labels common to all',len(common),' union',len(set.union(*[set(r) for r in rows.values()])))
import collections
cnt=collections.Counter(len(r) for r in rows.values()); print('row counts per (model,ds):',dict(cnt))
# which models have extra rows
extra=collections.defaultdict(set)
for k,r in rows.items(): extra[len(r)].add(k[0].split('/')[1])
for n,s in sorted(extra.items()): print(n,sorted(s)[:12])
res=[]
for t in (0.95,0.99):
    for (mid,ds),r in rows.items():
        def best(labels,field='retention'):
            c=[r[l]['compression'] for l in labels if r[l]['retention']>=t]
            return max(c) if c else 1.0
        full=best(list(r)); cm=best([l for l in r if l in common])
        # lower CI version
        def bestlo(labels):
            c=[r[l]['compression'] for l in labels if r[l]['retention_interval'][0]>=t]; return max(c) if c else 1.0
        res.append(dict(model=mid,ds=ds[-7:],t=t,full=full,common=cm,fullLo=bestlo(list(r)),commonLo=bestlo([l for l in r if l in common]),n=len(r)))
x=pd.DataFrame(res)
for t in (0.95,0.99):
    s=x[x.t==t]; ch=s[s.full>s.common*1.01]
    print('target',t,'(model,ds) where full row-set gives >1% more compression than common set:',len(ch),'of',len(s),' median gain x',(ch.full/ch.common).median().round(2),'max',(ch.full/ch.common).max().round(1))
    print(ch.groupby('n').size().to_dict())
    chl=s[s.fullLo>s.commonLo*1.01]; print('  lower-CI version:',len(chl),'of',len(s))
