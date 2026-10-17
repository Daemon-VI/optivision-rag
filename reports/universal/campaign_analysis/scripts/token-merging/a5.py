import warnings;warnings.filterwarnings("ignore")
import pandas as pd,numpy as np
from scipy import stats
rng=np.random.default_rng(5)
W=pd.read_pickle('W.pkl')
R=[0.5,0.33,0.25,0.1]
for r in R: W[f'L{r}']=1-W[f'r{r}']
W['La']=1-W.a_ret;W['ak']=W.a_v/W.bv
im=W[W.modality=='image']
def rk(g,c): return g[c].rank()
print('--- redundancy re-test excluding ColNetraEmbed (only image model with <500 vpd)')
ex=im[im.model!='Cognitive-Lab/ColNetraEmbed']
for c in ['L0.5','L0.33','L0.25','L0.1','La']:
    out=[]
    for ds,g in ex.groupby('ds'):
        h=g[['bv',c]].dropna();rho,p=stats.spearmanr(h.bv,h[c]);out.append('%s rho=%.2f p=%.2g n=%d'%(ds,rho,p,len(h)))
    print(c,' | '.join(out))
print('\n--- pooled image, dataset-demeaned ranks: Spearman(bv, loss) within-dataset ranks')
for c in ['L0.25','L0.1','La']:
    h=im[['ds','bv',c]].dropna().copy()
    h['rb']=h.groupby('ds').bv.rank();h['rl']=h.groupby('ds')[c].rank()
    rho,p=stats.pearsonr(h.rb,h.rl);print(c,'rho=%.2f p=%.2g n=%d'%(rho,p,len(h)),end=' ; ')
    # model-cluster permutation: permute bv across models (keep both ds rows with same model)
    m=h.reset_index(drop=False).copy()
    models=m.model.unique() if 'model' in m else None
    h2=im[['model','ds','bv',c]].dropna().copy();h2['rb']=h2.groupby('ds').bv.rank();h2['rl']=h2.groupby('ds')[c].rank()
    # cluster bootstrap by model
    groups={k:d for k,d in h2.groupby('model')};ms=list(groups)
    bs=[]
    for _ in range(1000):
        gg=pd.concat([groups[k] for k in rng.choice(ms,len(ms))]);bs.append(np.corrcoef(gg.rb,gg.rl)[0,1])
    print('model-bootstrap 95%% CI %s'%np.round(np.percentile(bs,[2.5,97.5]),2))
print('\n--- adaptive keep fraction (a redundancy proxy: how few reps cover the page at cos .6) vs ward loss')
for c in ['L0.5','L0.25','L0.1']:
    out=[]
    for ds,g in W.groupby('ds'):
        h=g[['ak',c]].dropna();rho,p=stats.spearmanr(h.ak,h[c]);out.append('%s rho=%.2f p=%.2g n=%d'%(ds,rho,p,len(h)))
    print(c,' | '.join(out))
print('\n--- cross-dataset reliability of tolerance (docvqa vs infovqa) with model-bootstrap CI')
D=W[W.ds=='docvqa'].set_index('model');I=W[W.ds=='infovqa'].set_index('model');common=D.index.intersection(I.index)
for c in ['L0.33','L0.25','L0.1','La']:
    h=pd.concat([D.loc[common,c],I.loc[common,c]],axis=1).dropna()
    rho=stats.spearmanr(h.iloc[:,0],h.iloc[:,1])[0]
    bs=[stats.spearmanr(*h.iloc[rng.integers(0,len(h),len(h))].values.T)[0] for _ in range(2000)]
    print(c,'rho=%.2f CI %s n=%d'%(rho,np.round(np.nanpercentile(bs,[2.5,97.5]),2),len(h)))
print('\n--- replicate/sibling pairs (near-identical checkpoints): loss difference as an empirical noise scale')
pairs=[('tsystems/colqwen2.5-3b-multilingual-v1.0','tsystems/colqwen2.5-3b-multilingual-v1.0-merged'),('vidore/colpali-v1.2','vidore/colpali-v1.2-merged'),('vidore/colpali-v1.2','vidore/colpali-v1.2-hf'),('vidore/colpali-v1.3','vidore/colpali-v1.3-hf'),('vidore/colpali-v1.3','vidore/colpali-v1.3-merged'),('vidore/colqwen2-v1.0-hf','vidore/colqwen2-v1.0-merged'),('lightonai/colbertv2.0','colbert-ir/colbertv2.0')]
rows=[]
for a,b in pairs:
    for ds in ['docvqa','infovqa','scifact']:
        x=W[(W.model==a)&(W.ds==ds)];y=W[(W.model==b)&(W.ds==ds)]
        if len(x) and len(y):
            for c in ['L0.5','L0.33','L0.25','L0.1','La']:
                rows.append(dict(a=a,b=b,ds=ds,c=c,diff=x[c].iloc[0]-y[c].iloc[0]))
P=pd.DataFrame(rows)
print(P.pivot_table(index=['a','b','ds'],columns='c',values='diff').round(4).to_string())
print('RMS diff by metric (pp):',(P.groupby('c')['diff'].apply(lambda s:np.sqrt((s**2).mean()))*100).round(2).to_dict())
print('\n--- float16-inference baseline confound (image): loss by f16 flag')
for c in ['L0.25','L0.1','La']:
    a=im[im.f16][c].dropna();b=im[~im.f16][c].dropna()
    print(c,'f16 n=%d mean %.4f | other n=%d mean %.4f | MWU p=%.2g'%(len(a),a.mean(),len(b),b.mean(),stats.mannwhitneyu(a,b).pvalue))
print('\n--- family means of loss at 0.1 (image, docvqa+infovqa averaged per model first)')
mm=im.groupby(['family','model'])[['L0.25','L0.1','La','bv','dim']].mean().reset_index()
print(mm.groupby('family').agg(n=('model','count'),L25=('L0.25','mean'),L10=('L0.1','mean'),La=('La','mean'),bv=('bv','mean')).round(4))
# eta^2 family at L0.1
h=mm.dropna(subset=['L0.1']);gm=h['L0.1'].mean();ssb=sum(len(g)*(g['L0.1'].mean()-gm)**2 for _,g in h.groupby('family'));sst=((h['L0.1']-gm)**2).sum()
obs=ssb/sst
perm=[]
for _ in range(5000):
    p=h.copy();p['family']=rng.permutation(p.family.values);gm=p['L0.1'].mean();perm.append(sum(len(g)*(g['L0.1'].mean()-gm)**2 for _,g in p.groupby('family'))/sst)
print('eta^2(family) for L0.1 = %.2f, permutation p=%.3g (families of size 1 inflate eta^2)'%(obs,(np.sum(np.array(perm)>=obs)+1)/5001))
print('\nworst/best 5 image models by mean L0.1')
print(mm.sort_values('L0.1').iloc[[0,1,2,3,4,-5,-4,-3,-2,-1]].round(4).to_string())
