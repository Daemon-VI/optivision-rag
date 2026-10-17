import sys; sys.path.insert(0,'.')
from load import *
from scipy import stats
pd.set_option('display.width',250); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',500)
O=pd.read_pickle(OUT+'/O.pkl'); F=pd.read_pickle(OUT+'/F.pkl'); u2=pd.read_pickle(OUT+'/u2.pkl')
L=O[(O.reference=='labels')&O.sel_ok].copy(); L['fail']=(L.hold<L.target).astype(float)
pool=L.groupby(['model','dataset','rule']).agg(fail=('fail','mean'),n=('fail','size'),gm=('comp',lambda s:np.exp(np.log(s).mean()))).reset_index().merge(F[['model','dataset','n_queries','baseline_ndcg5','vpd','dim','n_docs']],on=['model','dataset'])
tests=0
print('== pool-level (model x dataset) Spearman of failure rate vs features, within dataset')
for r in ['calibrate() default','lower 0.999','bonferroni 0.975','lower 0.99','lower 0.975','point']:
    for ds in ['docvqa','infovqa']:
        z=pool[(pool.rule==r)&(pool.dataset==ds)]
        out=[]
        for f in ['baseline_ndcg5','dim','vpd']:
            rho,p=stats.spearmanr(z.fail,z[f]); out.append(f"{f}: rho={rho:+.2f} p={p:.3f}"); tests+=1
        print(f"{r:20s} {ds:8s} n={len(z)} "+' | '.join(out))
print('tests so far',tests)
print('== docvqa vs infovqa fail rate paired by model (27 models), Wilcoxon')
for r in ['calibrate() default','lower 0.999','bonferroni 0.975','lower 0.99','lower 0.975','point']:
    w=pool[pool.rule==r].pivot(index='model',columns='dataset',values='fail').dropna(subset=['docvqa','infovqa'])
    diff=w.docvqa-w.infovqa
    res=stats.wilcoxon(diff) ; tests+=1
    rng=np.random.default_rng(3); bs=[diff.values[rng.integers(0,len(diff),len(diff))].mean() for _ in range(2000)]
    print(f"{r:20s} docvqa-infovqa mean diff={diff.mean():+.4f} CI[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] wilcoxon p={res.pvalue:.4f}")
print('tests',tests)
# hold noise sd vs baseline
nz=u2.groupby(['model','dataset','sel']).res.agg(['size','std']).reset_index(); nz=nz[nz['size']>=4]
pm=nz.groupby(['model','dataset'])['std'].median().reset_index().merge(F[['model','dataset','baseline_ndcg5','n_queries']],on=['model','dataset'])
print('== median candidate-level hold-residual sd per pool vs baseline ndcg@5')
for ds in ['docvqa','infovqa','scifact']:
    z=pm[pm.dataset==ds]; 
    if len(z)>5: print(ds,len(z),stats.spearmanr(z['std'],z.baseline_ndcg5))
    print(ds,'median sd',z['std'].median().round(4),'median baseline',z.baseline_ndcg5.median().round(3))
z=pm; print('all pools: corr sd vs (1-baseline)',stats.spearmanr(z['std'],1-z.baseline_ndcg5))
print(pm.groupby('dataset').apply(lambda g: pd.Series(dict(sd=g['std'].median(),base=g.baseline_ndcg5.median()))))
# does the noise sd rise with failure rate of loose rules? pool-level sd vs fail(lower 0.975)
x=pool[pool.rule=='lower 0.975'].merge(pm[['model','dataset','std']],on=['model','dataset'])
print('lower0.975 fail vs noise sd, by dataset');
for ds in ['docvqa','infovqa']:
    zz=x[x.dataset==ds]; print(ds,stats.spearmanr(zz.fail,zz['std']))
# pool-level rank consistency of fail between docvqa and infovqa for same model (model effect)
for r in ['lower 0.975','lower 0.99','calibrate() default']:
    w=pool[pool.rule==r].pivot(index='model',columns='dataset',values='fail').dropna(subset=['docvqa','infovqa'])
    print(r,'model-level consistency docvqa vs infovqa fail, spearman',stats.spearmanr(w.docvqa,w.infovqa))
pool.to_pickle(OUT+'/pool.pkl')
