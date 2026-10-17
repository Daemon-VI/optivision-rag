from common import *
rng=np.random.default_rng(1)
df=long_df()
L=L1b(df)
P=mat(df)[L].dropna(); loss=1-P
sd=loss.groupby(level='ds').std()
def run(ds_list, strict, scale, groupfn=lambda m:GROUP2.get(m)):
    errs=[]
    for ds in ds_list:
        sub=loss.xs(ds,level='ds')
        W=1/sd.loc[ds].values if scale else np.ones(len(L))
        for m in sub.index:
            g=groupfn(m)
            others=[o for o in sub.index if o!=m and twin(o)!=twin(m)] if strict else [o for o in sub.index if o!=m]
            glob=sub.loc[others].mean().values
            sib=[o for o in others if groupfn(o)==g and g is not None]
            x=sub.loc[m].values
            e0=np.sqrt(np.mean(((x-glob)*W)**2))
            if sib:
                gm=sub.loc[sib].mean().values
                e1=np.sqrt(np.mean(((x-gm)*W)**2))
                # shrunk: 0.5 group + 0.5 global
                e2=np.sqrt(np.mean(((x-(0.5*gm+0.5*glob))*W)**2))
            else: e1=e2=np.nan
            # own profile on other dataset (image only)
            e3=np.nan
            if ds in('doc','info'):
                od='info' if ds=='doc' else 'doc'
                if (m,od) in loss.index:
                    oth=loss.xs(od,level='ds')
                    xo=oth.loc[m].values
                    oglob=oth.drop(index=[o for o in oth.index if o==m or (strict and twin(o)==twin(m))]).mean().values
                    # model-specific deviation from other-ds global mean, transplanted
                    pred=glob+(xo-oglob)
                    e3=np.sqrt(np.mean(((x-pred)*W)**2))
            errs.append((m,ds,g,len(sib),e0,e1,e2,e3))
    return pd.DataFrame(errs,columns=['model','ds','group','nsib','e_glob','e_grp','e_shr','e_xds'])
def boot_ratio(d,a,b,B=4000):
    d=d.dropna(subset=[a,b]); ms=d.model.unique()
    g={m:d[d.model==m] for m in ms}
    r=[]
    for _ in range(B):
        s=rng.choice(ms,len(ms)); x=pd.concat([g[m] for m in s])
        r.append(np.log(x[a]**2).mean()-np.log(x[b]**2).mean())
    return np.exp(np.percentile(r,[2.5,50,97.5])/1)
for strict in (False,True):
  for scale in (False,True):
    e=run(['doc','info'],strict,scale)
    ee=e.dropna(subset=['e_grp'])
    print(f'strict={strict} scale={scale}: n profiles {len(e)}, with sibling {len(ee)} ({ee.model.nunique()} models)')
    print('  RMSE glob %.4f grp %.4f shrunk %.4f ; (same-subset)'%(np.sqrt((ee.e_glob**2).mean()),np.sqrt((ee.e_grp**2).mean()),np.sqrt((ee.e_shr**2).mean())))
    print('  geo-mean ratio of sq err grp/glob (model bootstrap 95%%): %s ; frac profiles grp better %.2f'%(np.round(boot_ratio(ee,'e_grp','e_glob'),3),(ee.e_grp<ee.e_glob).mean()))
    ex=e.dropna(subset=['e_xds'])
    print('  cross-dataset own-profile n=%d: RMSE glob %.4f xds %.4f ; ratio %s; frac better %.2f'%(len(ex),np.sqrt((ex.e_glob**2).mean()),np.sqrt((ex.e_xds**2).mean()),np.round(boot_ratio(ex,'e_xds','e_glob'),3),(ex.e_xds<ex.e_glob).mean()))
    b=ee.dropna(subset=['e_xds'])
    print('  both available n=%d: grp %.4f xds %.4f'%(len(b),np.sqrt((b.e_grp**2).mean()),np.sqrt((b.e_xds**2).mean())))
    if strict and not scale:
        e.to_csv('s2_loo_strict.csv',index=False)
        print(ee.groupby('group').apply(lambda x: pd.Series(dict(n=len(x),glob=np.sqrt((x.e_glob**2).mean()),grp=np.sqrt((x.e_grp**2).mean())))).round(4))
