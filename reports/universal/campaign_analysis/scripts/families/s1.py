from common import *
import itertools
rng=np.random.default_rng(0)
df=long_df()
L=L1b(df); print(len(L),L)
P=mat(df)[L].dropna()
SE=se_mat(df,L).loc[P.index]
loss=1-P
out=[]
def rmsd(a,b): return float(np.sqrt(np.mean((a-b)**2)))
res={}
for ds in ['doc','info','sci']:
    idx=[i for i in P.index if i[1]==ds]
    ms=[i[0] for i in idx]
    X=loss.loc[idx].values; S=SE.loc[idx].values
    n=len(ms)
    D=np.array([[rmsd(X[i],X[j]) for j in range(n)] for i in range(n)])
    N=np.array([[np.sqrt(np.mean(S[i]**2+S[j]**2)) for j in range(n)] for i in range(n)])  # independent-noise expectation
    g=np.array([GROUP2.get(m,'single:'+m) for m in ms])
    iu=np.triu_indices(n,1)
    same=(g[:,None]==g[None,:])[iu]
    d=D[iu]; nz=N[iu]
    print(ds,'n models',n,'pairs',len(d),'same-group pairs',same.sum())
    print('  mean RMSD within %.4f between %.4f ; noise-expect within %.4f between %.4f'%(d[same].mean(),d[~same].mean(),nz[same].mean(),nz[~same].mean()))
    # permutation: shuffle group labels among models
    obs=d[same].mean()-d[~same].mean()
    cnt=0; B=5000
    for _ in range(B):
        gp=rng.permutation(g)
        s=(gp[:,None]==gp[None,:])[iu]
        if s.sum()==0: continue
        if d[s].mean()-d[~s].mean()<=obs: cnt+=1
    print('  within-between diff %.4f  perm p(one-sided) %.4f'%(obs,(cnt+1)/(B+1)))
    # excess over noise
    ex=np.sqrt(np.maximum(d**2-nz**2,0))
    print('  excess-over-noise within %.4f between %.4f'%(ex[same].mean(),ex[~same].mean()))
    # per group pairs
    rows=[]
    for i,j in zip(*iu):
        if g[i]==g[j]: rows.append((g[i],ms[i].split('/')[1],ms[j].split('/')[1],D[i,j],N[i,j]))
    for r in sorted(rows): print('   %-18s %-45s %-45s %.4f noise %.4f'%r)
    res[ds]=(D,N,g,ms)
import pickle; pickle.dump(res,open('s1.pkl','wb'))
