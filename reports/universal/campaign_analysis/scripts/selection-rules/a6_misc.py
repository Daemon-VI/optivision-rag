import sys; sys.path.insert(0,'.')
from load import *
from scipy import stats
pd.set_option('display.width',250); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',500)
O=pd.read_pickle(OUT+'/O.pkl'); F=pd.read_pickle(OUT+'/F.pkl'); K=pd.read_pickle(OUT+'/K.pkl')
rng=np.random.default_rng(2)
L=O[O.reference=='labels'].copy(); L['fail']=(L.hold<L.target).astype(float)
# 1. default vs lower 0.999, paired by (model,dataset,target,seed)
a=L[L.rule=='calibrate() default'].set_index(['model','dataset','target','seed']); b=L[L.rule=='lower 0.999'].set_index(['model','dataset','target','seed'])
j=a[['sel','comp','hold','fail','sel_ok']].join(b[['sel','comp','hold','fail','sel_ok']],rsuffix='_b',how='inner')
both=j[j.sel_ok&j.sel_ok_b]
same=both.sel==both.sel_b
print('splits',len(both),'same choice',same.mean())
dif=both[~same]
print('differing splits',len(dif),'default comp gm',np.exp(np.log(dif.comp).mean()),'lower0.999 gm',np.exp(np.log(dif.comp_b).mean()),'fail default',dif.fail.mean(),'fail lower0.999',dif.fail_b.mean())
print('lower0.999 more compressed in diff:',(dif.comp_b>dif.comp).mean(),' fewer:',(dif.comp_b<dif.comp).mean())
# McNemar on fails in discordant
n01=((dif.fail==0)&(dif.fail_b==1)).sum(); n10=((dif.fail==1)&(dif.fail_b==0)).sum(); print('discordant fail: default ok & l999 fail',n01,'default fail & l999 ok',n10,'exact p',stats.binomtest(n01,n01+n10,0.5).pvalue)
# 2. fail rate by compression gap: among diff where l999 deeper
# 3. overdispersion: default & l999 per-cell k fails
for r in ['calibrate() default','lower 0.999','bonferroni 0.975','lower 0.99']:
    x=L[(L.rule==r)&L.sel_ok].groupby(['model','dataset','target']).fail.agg(['sum','count'])
    p=x['sum'].sum()/x['count'].sum(); exp_var=(x['count']*p*(1-p)); disp=((x['sum']-x['count']*p)**2).sum()/exp_var.sum()
    print(r,'pooled fail',round(p,4),'dispersion index (1=binomial)',round(disp,2),'cells with zero fails',(x['sum']==0).mean().round(3),'binomial expected zero',(((1-p)**x['count']).mean()).round(3))
# 4. baseline@1 reference
B=O[O.reference=='baseline@1'].copy(); B['fail']=(B.hold_labels<B.target).astype(float)
print('baseline@1 label-fail counts by rule'); print(B[B.sel_ok].groupby('rule').fail.agg(['sum','count']))
# paired compression label-ref vs baseline@1-ref per rule
for r in ['calibrate() default','lower 0.999','point']:
    cl=L[L.rule==r].groupby(['model','dataset','target']).apply(lambda g: np.log(g.comp).mean() if g.sel_ok.any() else np.nan)
    cb=B[B.rule==r].groupby(['model','dataset','target']).apply(lambda g: np.log(g.comp).mean() if g.sel_ok.any() else np.nan)
    dd=(cb-cl).dropna(); g=dd.groupby(level=0).agg(['sum','count'])
    bs=[np.exp(g['sum'].values[p].sum()/g['count'].values[p].sum()) for p in (rng.integers(0,len(g),len(g)) for _ in range(1000))]
    print(r,'baseline@1 / labels compression ratio',np.exp(dd.mean()).round(3),np.percentile(bs,[2.5,97.5]).round(3),'cells',len(dd),'lower in',(dd<-1e-9).mean().round(3))
# abstain detail
ab=O[~O.sel_ok].groupby(['reference','rule','target']).size().unstack().fillna(0)
print(ab)
ab2=O[(~O.sel_ok)&(O.reference=='labels')].groupby(['model','dataset','target']).size().sort_values(ascending=False).head(10); print(ab2)
