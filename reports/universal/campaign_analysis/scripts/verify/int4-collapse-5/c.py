import numpy as np
from scipy import stats
names=['LFM2','LFM2.5','tiny','colbertv2','jina2','CBZero','GTE','LateOn','Reason','lighton-cv2','mLateOn','mxbai-L','mxbai17','mxbai32','answerai']
dim=np.array([128]*12+[48,64,96]);i4=np.array([.9918,.9828,1.0053,.9973,.9881,.9941,.9361,.2967,.9963,1.0026,.7653,.9855,.2678,.7918,.8991])
for th in(.9,.95,.97,.99):
    deg=i4<th; a=deg[dim<128].sum();b=deg[dim==128].sum()
    print(th,a,b,stats.fisher_exact([[a,3-a],[b,12-b]]))
rho=stats.spearmanr(i4,np.log(dim));print(rho)
rng=np.random.default_rng(0);bs=[]
for _ in range(5000):
    i=rng.integers(0,15,15)
    if len(set(dim[i]))<2 or len(set(i4[i]))<2: continue
    bs.append(stats.spearmanr(i4[i],np.log(dim[i]))[0])
print(np.percentile(bs,[2.5,97.5]),len(bs))
# drop mxbai siblings: keep one
for drop in([12],[13],[12,13],[14],[7,10,6]):
    k=np.ones(15,bool);k[drop]=False
    print('drop',drop,stats.spearmanr(i4[k],np.log(dim[k])))
# exact permutation for fisher alt: low-dim mean rank
print(stats.mannwhitneyu(i4[dim<128],i4[dim==128],alternative='less'))
# dimension vs nothing: among 128-d, collapse also; int4(mean) all ~>0.98
