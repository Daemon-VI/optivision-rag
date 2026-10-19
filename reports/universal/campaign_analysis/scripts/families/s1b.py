from common import *
rng=np.random.default_rng(11)
df=long_df(); L=L1b(df)
P=mat(df)[L].dropna(); loss=1-P
def pref(m): return (('-hf' in m) or ('-merged' in m), m)
res=[]
for ds in ['doc','info','sci']:
    ms=sorted([m for (m,d) in loss.index if d==ds],key=pref)
    keep={}
    for m in ms: keep.setdefault(twin(m),m)
    ms=sorted(keep.values())
    if ds!='sci': ms=[m for m in ms if True]
    X=np.stack([loss.loc[(m,ds)].values for m in ms]); n=len(ms)
    D=np.sqrt(((X[:,None,:]-X[None,:,:])**2).mean(-1))
    g=np.array([GROUP2.get(m,'single:'+m) for m in ms]); iu=np.triu_indices(n,1)
    d=D[iu]
    def stat(gg):
        s=(gg[:,None]==gg[None,:])[iu]; return d[s].mean()-d[~s].mean(), s.sum()
    obs,ns=stat(g); null=[stat(rng.permutation(g))[0] for _ in range(5000)]
    s=(g[:,None]==g[None,:])[iu]
    # AUC: P(random within pair closer than random between pair)
    from scipy.stats import mannwhitneyu
    u=mannwhitneyu(d[s],d[~s],alternative='less'); auc=1-u.statistic/(s.sum()*(~s).sum())
    # nearest neighbour: share of models whose NN (excluding self) is a same-group model, among models with >=1 sibling
    hit=[];base=[]
    for i in range(n):
        sib=[j for j in range(n) if j!=i and g[j]==g[i]]
        if not sib: continue
        D2=D[i].copy(); D2[i]=np.inf; nn=np.argmin(D2); hit.append(g[nn]==g[i]); base.append(len(sib)/(n-1))
    print(f'{ds}: models {n} (twin-deduped) same-group pairs {ns}/{len(d)}; mean RMSD within {d[s].mean():.4f} between {d[~s].mean():.4f}; diff {obs:.4f} perm p {(np.sum(np.array(null)<=obs)+1)/5001:.4f}; P(within<between)={auc:.3f}; 1-NN same group {np.mean(hit):.2f} (chance {np.mean(base):.2f}, n={len(hit)})')
