import numpy as np, itertools
from scipy import stats
B={'17m':(.765,.715),'32m':(.220,.252),'mLateOn':(.248,.289),'GTE':(.073,.118),'answerai':(.103,.059)}
def r(ks):
    a=np.array([B[k] for k in ks]);return stats.pearsonr(*a.T)[0],stats.spearmanr(*a.T)[0]
print(r(list(B)))
for k in B: print("drop",k,r([x for x in B if x!=k]))
print("drop both mxbai",r(['mLateOn','GTE','answerai']))
# exact permutation p for Spearman n=5 observed rho .9
rs=[stats.spearmanr(range(5),p)[0] for p in itertools.permutations(range(5))]
print(np.mean(np.array(rs)>=0.9-1e-9))
# excluding 17m: ratio int4/binary deficits
for k,(a,b) in B.items(): print(k,round(a/b,2),round(a-b,3))
