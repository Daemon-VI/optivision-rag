import json,numpy as np
exec(open('followup/analysis/scripts/corpus_size/pairs.py').read().split("M='ndcg@5'")[0])
M='ndcg@5'
import collections
for n,a,b in P:
    A=json.load(open(R+a));B=json.load(open(R+b))
    ra={r['label']:r for r in A['rows']}
    X=[(o['retention:'+M],r['retention:'+M],r['compression_vs_float32'],l) for l,o in ra.items() for r in B['rows'] if r['label']==l]
    hi=[x for x in X if x[0]>=.9]
    d=np.array([x[1]-x[0] for x in hi])
    print(n,len(hi),'mean',d.mean().round(3),'n<-.03',(d<-.03).sum(),'n<-.04',(d<-.04).sum())
    # same-ret band comparison: do non-proj rows lose less than proj rows at similar orig retention?
    import re
    for lo,hi_ in [(.3,.7),(.7,.9)]:
        P_=[x[1]-x[0] for x in X if lo<=x[0]<hi_ and 'project' in x[3]]
        O_=[x[1]-x[0] for x in X if lo<=x[0]<hi_ and 'project' not in x[3]]
        print('  band',lo,hi_,'proj n',len(P_),np.mean(P_).round(3) if P_ else None,'nonproj n',len(O_),np.mean(O_).round(3) if O_ else None)
    # text baseline ret ndcg5 vs dim
    print('  lossy nonproj',[(round(x[0],2),round(x[1]-x[0],3),x[3][:40]) for x in X if x[0]<.9 and 'project' not in x[3]][:6])
