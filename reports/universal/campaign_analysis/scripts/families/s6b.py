from common import *
rng=np.random.default_rng(66)
df=long_df(); L=L1b(df)
P=mat(df)[L].dropna(); loss=1-P
ims=[m for (m,d) in loss.index if d=='doc' and (m,'info') in loss.index]
# dedupe twins: keep first by preference (no -hf, no -merged)
def pref(m): return (('-hf' in m) or ('-merged' in m), m)
keep={}
for m in sorted(ims,key=pref): keep.setdefault(twin(m),m)
ims=sorted(keep.values())
print('after twin de-dup',len(ims))
G={m:GROUP2.get(m,'single') for m in ims}
cnt=pd.Series(G).value_counts()
ims=[m for m in ims if cnt[G[m]]>=2]   # grouped models only
G={m:G[m] for m in ims}
print('grouped models',len(ims),dict(pd.Series(G).value_counts()))
Y=np.stack([np.stack([loss.loc[(m,'doc')].values for m in ims]),np.stack([loss.loc[(m,'info')].values for m in ims])],axis=1)
def comp(Y,g,std=False):
    if std: Y=Y/Y.reshape(-1,Y.shape[-1]).std(axis=0)
    n,d,l=Y.shape
    grand=Y.mean(axis=(0,1)); ds=Y.mean(axis=0)-grand; me=Y.mean(axis=1)-grand
    inter=Y-grand-ds[None]-me[:,None]
    ge=np.zeros_like(me)
    for u in set(g):
        i=g==u; ge[i]=me[i].mean(axis=0)
    w=me-ge
    ss=np.array([(ds**2).sum()*n,(ge**2).sum()*d,(w**2).sum()*d,(inter**2).sum()])
    # df-adjusted mean squares: group df = k-1, model-in-group df = n-k, interaction df=(n-1)(d-1)
    return ss/ss.sum(), ss
g=np.array([G[m] for m in ims]); k=len(set(g)); n=len(ims)
for std in (False,True):
    f,ss=comp(Y,g,std)
    print('std',std,'shares dataset/group/model_in_group/model_x_dataset',np.round(f,3))
    # df-adjusted: mean squares per df (variance components)
    l=Y.shape[2]
    ms_g=ss[1]/(k-1); ms_w=ss[2]/(n-k); ms_i=ss[3]/((n-1)); 
    print('   mean squares  group %.4g within-group-model %.4g  model x dataset %.4g ; F(group/within)=%.2f'%(ms_g,ms_w,ms_i,ms_g/ms_w))
    # permutation null on group labels
    obs=f[1]; null=[]; Fnull=[]
    for _ in range(3000):
        gp=rng.permutation(g); ff,s2=comp(Y,gp,std); null.append(ff[1]); Fnull.append((s2[1]/(k-1))/(s2[2]/(n-k)))
    null=np.array(null)
    print('   group share obs %.3f null mean %.3f 95th %.3f p=%.4f ; F obs %.2f p=%.4f'%(obs,null.mean(),np.percentile(null,95),(np.sum(null>=obs)+1)/3001,ms_g/ms_w,(np.sum(np.array(Fnull)>=ms_g/ms_w)+1)/3001))
    # cluster bootstrap on models
    bs=[]
    for _ in range(2000):
        idx=rng.integers(0,n,n)
        gb=np.array([f"{g[i]}" for i in idx])  # groups maintained
        ff,_=comp(Y[idx],gb,std); bs.append(ff)
    bs=np.array(bs); print('   boot 95%% CI shares:',[tuple(np.round(np.percentile(bs[:,j],[2.5,97.5]),3)) for j in range(4)])
# also: persistence (model effect consistent across datasets) test: correlation of model effects between doc and info after removing group
Yd=Y[:,0,:]; Yi=Y[:,1,:]
rd=Yd-Yd.mean(0); ri=Yi-Yi.mean(0)
print('mean corr of per-label deviations doc vs info across models (pearson of raw deviations flat): %.3f'%np.corrcoef(rd.ravel(),ri.ravel())[0,1])
# after subtracting group mean
def demean_g(R):
    R=R.copy()
    for u in set(g):
        i=g==u; R[i]-=R[i].mean(0)
    return R
a=demean_g(rd); b=demean_g(ri)
print('same, after removing group means: %.3f'%np.corrcoef(a.ravel(),b.ravel())[0,1])
# permutation p for that: shuffle model order of info
nul=[]
for _ in range(3000):
    p=rng.permutation(n); nul.append(np.corrcoef(a.ravel(),demean_g(ri)[p].ravel())[0,1])
print(' perm p (model-shuffle) for within-group persistence: ', (np.sum(np.array(nul)>=np.corrcoef(a.ravel(),b.ravel())[0,1])+1)/3001)
