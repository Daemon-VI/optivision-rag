from common import *
from scipy.stats import binomtest, wilcoxon
rng=np.random.default_rng(7)
df=long_df()
L=L1b(df)
P=mat(df)[L].dropna(); loss=1-P
SEm=se_mat(df,L).loc[P.index]
def rm(a,b): return float(np.sqrt(np.mean((a-b)**2)))
print('=== A. identical-weight replicate floor vs dtype/packaging pairs (RMSD of loss profile, L1b labels:',len(L),')')
pairs=[('floor f32-f32 adapter vs merged','vidore/colpali-v1.2','vidore/colpali-v1.2-merged'),
 ('floor f32-f32 adapter vs merged','tsystems/colqwen2.5-3b-multilingual-v1.0','tsystems/colqwen2.5-3b-multilingual-v1.0-merged'),
 ('f32 vs f16(-hf)','vidore/colpali-v1.2','vidore/colpali-v1.2-hf'),
 ('f32 vs f16(-hf)','vidore/colpali-v1.2-merged','vidore/colpali-v1.2-hf'),
 ('f32 vs f16(-hf)','vidore/colpali-v1.3','vidore/colpali-v1.3-hf'),
 ('f16-hf vs f16-hf (different versions)','vidore/colpali-v1.2-hf','vidore/colpali-v1.3-hf'),
 ('f32 v1.2 vs f32 v1.3 (versions)','vidore/colpali-v1.2','vidore/colpali-v1.3'),
 ('f32 v1.1 vs f32 v1.2','vidore/colpali-v1.1','vidore/colpali-v1.2'),
 ('f32 v1.1 vs f32 v1.3','vidore/colpali-v1.1','vidore/colpali-v1.3')]
rows=[]
for kind,a,b in pairs:
    for ds in ['doc','info']:
        x=loss.loc[(a,ds)].values; y=loss.loc[(b,ds)].values
        diff=(y-x)
        n_se=np.sqrt(np.mean(SEm.loc[(a,ds)].values**2+SEm.loc[(b,ds)].values**2))
        base=df[(df.model==a)&(df.ds==ds)].base.iloc[0],df[(df.model==b)&(df.ds==ds)].base.iloc[0]
        rows.append((kind,a.split('/')[1],b.split('/')[1],ds,rm(x,y),diff.mean(),n_se,base[0],base[1]))
R=pd.DataFrame(rows,columns=['kind','a','b','ds','rmsd','mean_loss_b_minus_a','indep_noise','base_a','base_b'])
pd.set_option('display.width',250)
print(R.round(4).to_string(index=False))
R.to_csv('s7_twins.csv',index=False)

print('\n=== B. float16-inference effect, group fixed-effects regression of mean loss over L1b labels')
# outcome: per-profile mean loss; predictors: group FE (GROUP2), ds FE, f16 ; cluster bootstrap by model. Image only.
img=[i for i in loss.index if i[1]!='sci']
def feats(sel, labs):
    rows=[]
    for (m,ds) in sel:
        rows.append(dict(model=m,ds=ds,grp=GROUP2.get(m,'single'),f16=int(info(m)['f16']),y=loss.loc[(m,ds)][labs].mean(),
                         lvpd=np.log10(df[(df.model==m)&(df.ds==ds)].vpd.iloc[0]),ldim=np.log10(info(m)['dim'])))
    return pd.DataFrame(rows)
def fe_coef(d, cols):
    # within-transform on group (groups with variation only), ds dummies
    d=d.copy()
    X=d[cols+['ds_info']].astype(float)
    for c in cols+['ds_info','y']:
        d[c]=d[c].astype(float)
    # demean by group
    for c in cols+['ds_info','y']:
        d[c]=d[c]-d.groupby('grp')[c].transform('mean')
    Xm=d[cols+['ds_info']].values; ym=d['y'].values
    b,*_=np.linalg.lstsq(Xm,ym,rcond=None); return b
subsets={'all L1b':L,'merge-only':[l for l in L if '>' not in l and 'merge' in l],'codec-only':['binary','int4(mean)','int8(per_vector)','lloyd2(7)']}
for name,labs in subsets.items():
    d=feats(img,labs); d['ds_info']=(d.ds=='info').astype(int)
    # keep only groups with >=2 distinct models
    gm=d.groupby('grp').model.nunique(); d=d[d.grp.isin(gm[gm>1].index)]
    b=fe_coef(d,['f16']); 
    ms=d.model.unique(); bs=[]
    gidx={m:d[d.model==m] for m in ms}
    for _ in range(2000):
        s=rng.choice(ms,len(ms)); dd=pd.concat([gidx[m] for m in s]); 
        if dd.f16.nunique()<2: continue
        try: bs.append(fe_coef(dd,['f16'])[0])
        except Exception: pass
    print('%-12s n_profiles=%d n_models=%d f16 coef (loss, i.e. 1-retention) = %+.5f  95%% CI [%+.5f, %+.5f]  (n f16 models %d)'%(name,len(d),len(ms),b[0],*np.percentile(bs,[2.5,97.5]),d[d.f16==1].model.nunique()))
    # with log vpd and ldim controls
    b2=fe_coef(d,['f16','lvpd','ldim']); print('      with lvpd+ldim controls: f16 %+.5f  lvpd %+.5f ldim %+.5f'%tuple(b2[:3]))
