import json,numpy as np
from scipy import stats
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
D='vidore/docvqa_test_subsampled';I='vidore/infovqa_test_subsampled'
both=[m for m,v in d.items() if D in v and I in v]
R={m:{ds:{r['label']:r for r in d[m][ds]['rows']} for ds in(D,I)} for m in both}
common=set.intersection(*[set(R[m][D])&set(R[m][I]) for m in both])-{'float32'}
print(len(common))
for m in both: break
import collections
cnt=collections.Counter(len(set(R[m][D])&set(R[m][I])) for m in both);print(cnt)
# models with the most common label set
lab=collections.Counter(frozenset(set(R[m][D])&set(R[m][I])) for m in both);print([len(k) for k in lab],list(lab.values()))
big=max(lab,key=lambda k:(lab[k],len(k)));ms=[m for m in both if frozenset(set(R[m][D])&set(R[m][I]))==big]
print(len(ms),len(big))
def comp(m,l):return R[m][D][l]['compression']
L32=[l for l in big if l!='float32' and comp(ms[0],l)>=32];print(len(L32))
bD=np.array([d[m][D]['baseline'] for m in ms]);bI=np.array([d[m][I]['baseline'] for m in ms])
f=lambda ds,L:np.array([np.mean([R[m][ds][l]['retention'] for l in L]) for m in ms])
La=[l for l in big if l!='float32']
for nm,L in(('all',La),('32',L32)):
    g=f(I,L)-f(D,L);print(nm,len(L),g.mean(),(g>0).sum(),len(g),stats.spearmanr(bI-bD,g))
    zs=lambda v:(v-v.mean())/v.std()
    print(' pooled std',stats.spearmanr(np.r_[zs(bD),zs(bI)],np.r_[zs(f(D,L)),zs(f(I,L))]))
    print(' within D',stats.spearmanr(bD,f(D,L)),'within I',stats.spearmanr(bI,f(I,L)))
