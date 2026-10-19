from common import *
rng=np.random.default_rng(6)
df=long_df()
L=L1b(df)
P=mat(df)[L].dropna(); loss=1-P
def decomp(models, std=False, labels=L, groupfn=lambda m:GROUP2.get(m,'single:'+m)):
    # image models with both datasets
    Y=np.stack([loss.loc[(m,'doc')][labels].values for m in models]),np.stack([loss.loc[(m,'info')][labels].values for m in models])
    Y=np.stack(Y,axis=1) # model x ds x label
    if std: Y=Y/Y.reshape(-1,Y.shape[-1]).std(axis=0)
    grand=Y.mean(axis=(0,1))
    ds_eff=Y.mean(axis=0)-grand  # ds x label
    mod_eff=Y.mean(axis=1)-grand # model x label
    inter=Y-grand-ds_eff[None]-mod_eff[:,None]
    g=np.array([groupfn(m) for m in models])
    grp_eff=np.zeros_like(mod_eff)
    for u in set(g):
        i=g==u; grp_eff[i]=mod_eff[i].mean(axis=0)
    within=mod_eff-grp_eff
    n,d,l=Y.shape
    ss=dict(dataset=(ds_eff**2).sum()*n, group=(grp_eff**2).sum()*d, model_in_group=(within**2).sum()*d, model_x_dataset=(inter**2).sum())
    tot=sum(ss.values()); return {k:v/tot for k,v in ss.items()}, tot
ims=[m for (m,d) in loss.index if d=='doc' and (m,'info') in loss.index]
print('image models with both ds',len(ims))
for std in (False,True):
    f,tot=decomp(ims,std)
    print('std',std,{k:round(v,3) for k,v in f.items()})
    # model bootstrap
    bs=[]
    for _ in range(1000):
        s=list(rng.choice(ims,len(ims)))
        # relabel duplicates as separate singletons for groups: keep group but tag
        mm=[f"{m}#{i}" for i,m in enumerate(s)]
        gm={f"{m}#{i}":GROUP2.get(m,'single:'+m) for i,m in enumerate(s)}
        # build matrices manually
        loss_b=pd.concat({ (f"{m}#{i}",'doc'):loss.loc[(m,'doc')] for i,m in enumerate(s)},axis=1).T
        loss_i=pd.concat({ (f"{m}#{i}",'info'):loss.loc[(m,'info')] for i,m in enumerate(s)},axis=1).T
        Lb=pd.concat([loss_b,loss_i]); Lb.index=pd.MultiIndex.from_tuples(Lb.index)
        globals()['loss_tmp']=Lb
        Y=np.stack([np.stack([Lb.loc[(m,'doc')].values for m in mm]),np.stack([Lb.loc[(m,'info')].values for m in mm])],axis=1)
        if std: Y=Y/Y.reshape(-1,Y.shape[-1]).std(axis=0)
        grand=Y.mean(axis=(0,1)); ds_eff=Y.mean(axis=0)-grand; mod_eff=Y.mean(axis=1)-grand
        inter=Y-grand-ds_eff[None]-mod_eff[:,None]
        g=np.array([gm[m] for m in mm]); grp_eff=np.zeros_like(mod_eff)
        # group = original group (duplicates of same model share group)
        for u in set(g):
            i=g==u; grp_eff[i]=mod_eff[i].mean(axis=0)
        within=mod_eff-grp_eff; n,d,l=Y.shape
        ss=[(ds_eff**2).sum()*n,(grp_eff**2).sum()*d,(within**2).sum()*d,(inter**2).sum()]
        t=sum(ss); bs.append([x/t for x in ss])
    bs=np.array(bs)
    for k,name in enumerate(['dataset','group','model_in_group','model_x_dataset']):
        print('   ',name,np.round(np.percentile(bs[:,k],[2.5,50,97.5]),3))
# strict: groups of size>1 only? also ICC: share of model-level variance explained by group among between-model variance
# permutation null for group share: permute group labels across models
f,_=decomp(ims,False)
obs=f['group']; null=[]
gl=[GROUP2.get(m,'single:'+m) for m in ims]
for _ in range(2000):
    pg=dict(zip(ims,rng.permutation(gl)))
    null.append(decomp(ims,False,groupfn=lambda m:pg[m])[0]['group'])
null=np.array(null); print('group share obs %.3f null mean %.3f 95th %.3f p=%.4f'%(obs,null.mean(),np.percentile(null,95),(np.sum(null>=obs)+1)/2001))
f,_=decomp(ims,True); obs=f['group']; null=[]
for _ in range(2000):
    pg=dict(zip(ims,rng.permutation(gl)))
    null.append(decomp(ims,True,groupfn=lambda m:pg[m])[0]['group'])
null=np.array(null); print('std: group share obs %.3f null mean %.3f 95th %.3f p=%.4f'%(obs,null.mean(),np.percentile(null,95),(np.sum(null>=obs)+1)/2001))
# twin noise floor: interaction share for twin pairs vs overall
